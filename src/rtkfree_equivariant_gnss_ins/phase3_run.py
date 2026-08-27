"""Deterministic deployable replay for the Phase 3 navigation baselines."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .phase3_config import FixedEskfConfig, make_initial_covariance
from .phase3_eskf import (
    POSITION_NIS_CHI_SQUARE_95,
    POSITION_NIS_CHI_SQUARE_99,
    POSITION_NIS_DOF,
)
from .phase3_forward import (
    ForwardEskfState,
    GnssOutage,
    apply_forward_position_event,
    gnss_is_masked,
    propagate_forward_eskf,
)
from .phase3_geodesy import (
    earth_rotation_n_radps,
    ecef_covariance_to_ned,
    ecef_position_to_ned,
    normal_gravity_n_mps2,
)
from .phase3_initialization import InitializationResult, initialize_from_causal_window
from .phase3_io import (
    ImuRecord,
    PvtRecord,
    iter_standardized_imu,
    read_selected_extrinsic,
    read_standardized_pvt,
    sha256_file,
)


@dataclass
class _Scenario:
    name: str
    filter_state: ForwardEskfState
    updates_enabled: bool
    outages: tuple[GnssOutage, ...]
    records: list[dict[str, object]]
    update_count: int = 0
    masked_count: int = 0


def _copy_filter_state(value: ForwardEskfState) -> ForwardEskfState:
    nominal = value.nominal
    return ForwardEskfState(
        timestamp_ns_utc=value.timestamp_ns_utc,
        nominal=type(nominal)(
            position_n_m=nominal.position_n_m.copy(),
            velocity_n_mps=nominal.velocity_n_mps.copy(),
            rotation_n_from_b=nominal.rotation_n_from_b.copy(),
            accelerometer_bias_b_mps2=nominal.accelerometer_bias_b_mps2.copy(),
            gyroscope_bias_b_radps=nominal.gyroscope_bias_b_radps.copy(),
        ),
        covariance=value.covariance.copy(),
    )


def _write_jsonl(path: Path, records: Sequence[dict[str, object]]) -> str:
    content = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for record in records
    )
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _state_record(
    scenario: _Scenario,
    pvt: PvtRecord,
    update_applied: bool,
    masked: bool,
    innovation: np.ndarray | None,
    nis: float | None,
    pre_update_state: ForwardEskfState,
) -> dict[str, object]:
    state = scenario.filter_state
    covariance = state.covariance
    nominal = state.nominal
    return {
        "schema_version": 1,
        "scenario": scenario.name,
        "timestamp_ns_utc": pvt.timestamp_ns_utc,
        "pvt_solution_valid": pvt.solution_valid,
        "pvt_solution_status": pvt.solution_status,
        "gnss_masked": masked,
        "position_update_applied": update_applied,
        "innovation_n_m": None if innovation is None else innovation.tolist(),
        "nis": nis,
        "pre_update_position_n_m": pre_update_state.nominal.position_n_m.tolist(),
        "position_n_m": nominal.position_n_m.tolist(),
        "velocity_n_mps": nominal.velocity_n_mps.tolist(),
        "rotation_n_from_b_row_major": nominal.rotation_n_from_b.reshape(-1).tolist(),
        "accelerometer_bias_b_mps2": nominal.accelerometer_bias_b_mps2.tolist(),
        "gyroscope_bias_b_radps": nominal.gyroscope_bias_b_radps.tolist(),
        "covariance_15x15_row_major": covariance.reshape(-1).tolist(),
        "covariance_symmetry_max_abs": float(np.max(np.abs(covariance - covariance.T))),
        "covariance_min_eigenvalue": float(np.min(np.linalg.eigvalsh(covariance))),
    }


def _spp_record(record: PvtRecord) -> dict[str, object]:
    return {
        "schema_version": 1,
        "timestamp_ns_utc": record.timestamp_ns_utc,
        "solution_valid": record.solution_valid,
        "solution_status": record.solution_status,
        "within_common_interval": record.within_common_interval,
        "position_ecef_m": (
            None if record.position_ecef_m is None else record.position_ecef_m.tolist()
        ),
        "covariance_ecef_m2": (
            None
            if record.covariance_ecef_m2 is None
            else record.covariance_ecef_m2.reshape(-1).tolist()
        ),
    }


def _initialization_inputs(
    pvt_records: Sequence[PvtRecord],
    imu_path: Path,
    config: FixedEskfConfig,
    lever_imu_m: np.ndarray,
) -> tuple[InitializationResult, PvtRecord, ImuRecord]:
    usable = [
        record
        for record in pvt_records
        if record.within_common_interval
        and record.solution_valid
        and record.position_ecef_m is not None
        and record.covariance_ecef_m2 is not None
    ]
    if not usable:
        raise ValueError("no usable PVT exists in the common interval")
    start_ns = usable[0].timestamp_ns_utc
    requested_end_ns = start_ns + round(config.initialization_window_s * 1e9)
    pvt_window = [
        record for record in usable if start_ns <= record.timestamp_ns_utc <= requested_end_ns
    ]
    if not pvt_window or pvt_window[-1].timestamp_ns_utc != requested_end_ns:
        raise ValueError("fixed initialization endpoint lacks a valid PVT")
    imu_window: list[ImuRecord] = []
    held_imu: ImuRecord | None = None
    for imu in iter_standardized_imu(imu_path):
        if imu.timestamp_ns_utc <= pvt_window[-1].timestamp_ns_utc:
            held_imu = imu
        if start_ns <= imu.timestamp_ns_utc <= pvt_window[-1].timestamp_ns_utc:
            imu_window.append(imu)
        if imu.timestamp_ns_utc > pvt_window[-1].timestamp_ns_utc:
            break
    if held_imu is None or len(imu_window) < 2:
        raise ValueError("fixed initialization window lacks IMU samples")
    result = initialize_from_causal_window(
        [record.timestamp_ns_utc for record in pvt_window],
        [record.position_ecef_m for record in pvt_window],
        [record.timestamp_ns_utc for record in imu_window],
        [record.linear_acceleration_b_mps2 for record in imu_window],
        lever_imu_m,
        config.initialization_window_s,
        config.minimum_horizontal_speed_mps,
    )
    return result, pvt_window[-1], held_imu


def run_deployable_baselines(
    config: FixedEskfConfig,
    pvt_path: Path,
    imu_path: Path,
    extrinsic_path: Path,
    imu_noise_parameter_path: Path,
    output_directory: Path,
) -> Path:
    """Run SPP, pure INS, fixed ESKF, and fixed 20/30-second mask scenarios."""

    if output_directory.exists() or not output_directory.is_absolute():
        raise ValueError("output directory must be a new absolute path")
    if sha256_file(pvt_path) != config.pvt_sha256:
        raise ValueError("PVT input differs from the fixed Phase 3 hash")
    if sha256_file(imu_path) != config.imu_sha256:
        raise ValueError("IMU input differs from the fixed Phase 3 hash")
    if (
        imu_noise_parameter_path.name != "xsens_imu_param.yaml"
        or imu_noise_parameter_path.is_symlink()
        or not imu_noise_parameter_path.is_file()
        or sha256_file(imu_noise_parameter_path)
        != config.imu_noise_parameter_sha256
    ):
        raise ValueError("IMU noise parameters differ from the fixed Phase 3 hash")
    pvt_records = read_standardized_pvt(pvt_path)
    extrinsic = read_selected_extrinsic(extrinsic_path, config.extrinsic_sha256)
    initialization, initialization_pvt, _ = _initialization_inputs(
        pvt_records,
        imu_path,
        config,
        extrinsic.antenna_lever_imu_m,
    )
    if initialization_pvt.covariance_ecef_m2 is None:
        raise ValueError("initialization PVT covariance is unavailable")
    position_covariance_n = ecef_covariance_to_ned(
        initialization.local_frame,
        initialization_pvt.covariance_ecef_m2,
    )
    initial_filter = ForwardEskfState(
        timestamp_ns_utc=initialization.timestamp_ns_utc,
        nominal=initialization.state,
        covariance=make_initial_covariance(config, position_covariance_n),
    )
    outage_start_ns = (
        initialization.timestamp_ns_utc
        + config.outage_start_after_initialization_s * 1_000_000_000
    )
    outages = {
        duration: GnssOutage(outage_start_ns, duration * 1_000_000_000)
        for duration in config.outage_durations_s
    }
    scenarios = [
        _Scenario("ins_only", _copy_filter_state(initial_filter), False, (), []),
        _Scenario("fixed_eskf", _copy_filter_state(initial_filter), True, (), []),
        *[
            _Scenario(
                f"fixed_eskf_outage_{duration}s",
                _copy_filter_state(initial_filter),
                True,
                (outages[duration],),
                [],
            )
            for duration in config.outage_durations_s
        ],
    ]
    replay_pvt = [
        record
        for record in pvt_records
        if record.within_common_interval
        and record.timestamp_ns_utc > initialization.timestamp_ns_utc
    ]
    if not replay_pvt:
        raise ValueError("no PVT events remain after initialization")
    last_pvt_ns = replay_pvt[-1].timestamp_ns_utc
    pvt_index = 0
    held_imu: ImuRecord | None = None
    timing_gap_count = 0
    gravity_n = normal_gravity_n_mps2(initialization.local_frame)
    earth_rate_n = earth_rotation_n_radps(initialization.local_frame)

    def propagate_scenarios(target_ns: int) -> None:
        nonlocal scenarios
        if held_imu is None:
            raise ValueError("causal replay has no arrived IMU sample")
        for scenario in scenarios:
            scenario.filter_state = propagate_forward_eskf(
                scenario.filter_state,
                target_ns,
                held_imu.linear_acceleration_b_mps2,
                held_imu.angular_velocity_b_radps,
                gravity_n,
                config.noise,
                earth_rate_n,
            )

    def process_pvt(record: PvtRecord) -> None:
        if record.position_ecef_m is not None:
            measured_position_n = ecef_position_to_ned(
                initialization.local_frame,
                record.position_ecef_m,
            )
        else:
            measured_position_n = None
        measurement_covariance_n = (
            None
            if record.covariance_ecef_m2 is None
            else ecef_covariance_to_ned(
                initialization.local_frame,
                record.covariance_ecef_m2,
            )
        )
        for scenario in scenarios:
            pre_update = scenario.filter_state
            masked = gnss_is_masked(record.timestamp_ns_utc, scenario.outages)
            update_applied = False
            innovation: np.ndarray | None = None
            nis: float | None = None
            if (
                scenario.updates_enabled
                and record.solution_valid
                and measured_position_n is not None
                and measurement_covariance_n is not None
            ):
                event = apply_forward_position_event(
                    scenario.filter_state,
                    measured_position_n,
                    measurement_covariance_n,
                    extrinsic.antenna_lever_imu_m,
                    scenario.outages,
                )
                scenario.filter_state = event.filter_state
                update_applied = event.update_applied
                masked = event.masked
                if event.diagnostics is not None:
                    innovation = event.diagnostics.innovation_n_m
                    nis = event.diagnostics.nis
            scenario.update_count += int(update_applied)
            scenario.masked_count += int(masked and record.solution_valid)
            scenario.records.append(
                _state_record(
                    scenario,
                    record,
                    update_applied,
                    masked,
                    innovation,
                    nis,
                    pre_update,
                )
            )

    for imu in iter_standardized_imu(imu_path):
        if imu.timestamp_ns_utc <= initialization.timestamp_ns_utc:
            held_imu = imu
            continue
        if held_imu is None:
            raise ValueError("no IMU sample exists at or before initialization")
        while (
            pvt_index < len(replay_pvt)
            and replay_pvt[pvt_index].timestamp_ns_utc < imu.timestamp_ns_utc
        ):
            event = replay_pvt[pvt_index]
            if event.timestamp_ns_utc > scenarios[0].filter_state.timestamp_ns_utc:
                propagate_scenarios(event.timestamp_ns_utc)
            process_pvt(event)
            pvt_index += 1
        if pvt_index >= len(replay_pvt):
            break
        if imu.timestamp_ns_utc <= last_pvt_ns:
            if imu.timestamp_ns_utc > scenarios[0].filter_state.timestamp_ns_utc:
                propagate_scenarios(imu.timestamp_ns_utc)
            held_imu = imu
            timing_gap_count += int(imu.timing_gap)
            while (
                pvt_index < len(replay_pvt)
                and replay_pvt[pvt_index].timestamp_ns_utc == imu.timestamp_ns_utc
            ):
                process_pvt(replay_pvt[pvt_index])
                pvt_index += 1
    if pvt_index != len(replay_pvt):
        raise ValueError("replay ended before the last PVT event")

    output_directory.mkdir(parents=True, mode=0o700)
    output_hashes: dict[str, str] = {}
    spp_name = "spp_only.jsonl"
    output_hashes[spp_name] = _write_jsonl(
        output_directory / spp_name,
        [_spp_record(record) for record in pvt_records],
    )
    for scenario in scenarios:
        name = f"{scenario.name}.jsonl"
        output_hashes[name] = _write_jsonl(output_directory / name, scenario.records)
    nis_values = [
        float(record["nis"])
        for scenario in scenarios
        if scenario.name == "fixed_eskf"
        for record in scenario.records
        if record["nis"] is not None
    ]
    nis_above_95_count = sum(value > POSITION_NIS_CHI_SQUARE_95 for value in nis_values)
    nis_above_99_count = sum(value > POSITION_NIS_CHI_SQUARE_99 for value in nis_values)
    summary = {
        "schema_version": 1,
        "profile_id": config.profile_id,
        "initialization_timestamp_ns_utc": initialization.timestamp_ns_utc,
        "initialization_fitted_pvt_velocity_n_mps": (
            initialization.fitted_pvt_velocity_n_mps.tolist()
        ),
        "initialization_nominal_velocity_n_mps": initialization.state.velocity_n_mps.tolist(),
        "gravity_n_mps2": gravity_n.tolist(),
        "earth_rotation_n_radps": earth_rate_n.tolist(),
        "timing_gap_count_after_initialization": timing_gap_count,
        "scenario_counts": {
            scenario.name: {
                "record_count": len(scenario.records),
                "position_update_count": scenario.update_count,
                "masked_valid_pvt_count": scenario.masked_count,
            }
            for scenario in scenarios
        },
        "fixed_eskf_nis": {
            "degrees_of_freedom": POSITION_NIS_DOF,
            "count": len(nis_values),
            "minimum": min(nis_values),
            "median": float(np.median(nis_values)),
            "maximum": max(nis_values),
            "chi_square_95_threshold": POSITION_NIS_CHI_SQUARE_95,
            "above_chi_square_95_count": nis_above_95_count,
            "above_chi_square_95_fraction": nis_above_95_count / len(nis_values),
            "chi_square_99_threshold": POSITION_NIS_CHI_SQUARE_99,
            "above_chi_square_99_count": nis_above_99_count,
            "above_chi_square_99_fraction": nis_above_99_count / len(nis_values),
        },
        "output_sha256": dict(sorted(output_hashes.items())),
        "input_sha256": {
            "pvt": config.pvt_sha256,
            "imu": config.imu_sha256,
            "imu_noise_parameter": config.imu_noise_parameter_sha256,
            "extrinsic": config.extrinsic_sha256,
        },
        "high_precision_reference_used": False,
        "receiver_native_pvt_used_as_truth": False,
    }
    summary_path = output_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path
