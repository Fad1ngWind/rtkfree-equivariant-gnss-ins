"""Reference-isolated assembly of the frozen Medium Phase 4 development sequence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from .phase3_config import FixedEskfConfig
from .phase3_geodesy import (
    earth_rotation_n_radps,
    ecef_position_to_ned,
    normal_gravity_n_mps2,
)
from .phase3_io import ImuRecord, iter_standardized_imu, read_selected_extrinsic, read_standardized_pvt, sha256_file
from .phase3_run import _initialization_inputs
from .phase4_config import Phase4Config


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MeanStateArray:
    position_n_m: FloatArray
    velocity_n_mps: FloatArray
    rotation_n_from_b: FloatArray


@dataclass(frozen=True)
class Phase4StepData:
    """One causal event; timestamp is scheduler metadata, never a model feature."""

    timestamp_ns_utc: int
    linear_acceleration_b_mps2: FloatArray
    angular_velocity_b_radps: FloatArray
    dt_s: FloatArray
    imu_timing_gap: bool
    pvt_solution_valid: bool
    pvt_antenna_position_n_m: FloatArray | None
    pvt_quality: FloatArray
    weak_pseudo_label: MeanStateArray


@dataclass(frozen=True)
class Phase4Sequence:
    initialization_timestamp_ns_utc: int
    initial_student_state: MeanStateArray
    antenna_lever_imu_m: FloatArray
    gravity_n_mps2: FloatArray
    earth_rotation_n_radps: FloatArray
    steps: tuple[Phase4StepData, ...]


@dataclass(frozen=True)
class _WeakLabel:
    timestamp_ns_utc: int
    pvt_solution_valid: bool
    state: MeanStateArray


def _regular_file(path: Path, basename: str) -> None:
    if not path.is_absolute() or path.name != basename or path.is_symlink() or not path.is_file():
        raise ValueError(f"expected absolute regular frozen interface file: {basename}")


def _finite_vector(values: object, size: int, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def _read_weak_labels(path: Path, config: Phase4Config) -> tuple[_WeakLabel, ...]:
    _regular_file(path, "fixed_eskf.jsonl")
    if sha256_file(path) != config.teacher_sha256:
        raise ValueError("weak pseudo-label input differs from the frozen Phase 3 hash")
    labels: list[_WeakLabel] = []
    previous_timestamp: int | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                timestamp = int(raw["timestamp_ns_utc"])
                if previous_timestamp is not None and timestamp <= previous_timestamp:
                    raise ValueError("weak pseudo-label timestamps must be strictly increasing")
                if raw["scenario"] != "fixed_eskf":
                    raise ValueError("only the frozen unmasked forward ESKF may supply weak labels")
                rotation = _finite_vector(
                    raw["rotation_n_from_b_row_major"],
                    9,
                    "weak-label attitude",
                ).reshape(3, 3)
                if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.0):
                    raise ValueError("weak-label attitude must be orthonormal")
                if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0.0):
                    raise ValueError("weak-label attitude must be proper")
                labels.append(
                    _WeakLabel(
                        timestamp_ns_utc=timestamp,
                        pvt_solution_valid=raw["pvt_solution_valid"] is True,
                        state=MeanStateArray(
                            position_n_m=_finite_vector(raw["position_n_m"], 3, "weak-label position"),
                            velocity_n_mps=_finite_vector(raw["velocity_n_mps"], 3, "weak-label velocity"),
                            rotation_n_from_b=rotation,
                        ),
                    )
                )
                previous_timestamp = timestamp
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("cannot read frozen Phase 3 weak pseudo-labels") from exc
    if len(labels) != config.record_count:
        raise ValueError("weak pseudo-label count differs from the Phase 4 freeze")
    return tuple(labels)


def _read_pvt_quality(path: Path, config: Phase4Config) -> dict[int, FloatArray]:
    _regular_file(path, "pvt.jsonl")
    if sha256_file(path) != config.pvt_sha256:
        raise ValueError("PVT input differs from the frozen Phase 2 hash")
    quality_by_timestamp: dict[int, FloatArray] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                timestamp = int(raw["timestamp_ns_utc"])
                if raw["solution_valid"] is True:
                    dop = raw["dop"]
                    quality = _finite_vector(
                        (
                            raw["used_satellite_count"],
                            raw["postfit_residual_rms_m"],
                            dop["pdop"],
                            dop["hdop"],
                            dop["vdop"],
                        ),
                        5,
                        "deployable PVT quality",
                    )
                else:
                    quality = np.zeros(5, dtype=np.float64)
                quality_by_timestamp[timestamp] = quality
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("cannot read frozen deployable PVT quality") from exc
    return quality_by_timestamp


def _next_or_none(iterator: Iterator[ImuRecord]) -> ImuRecord | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _partition_imu_intervals(
    imu_records: Iterator[ImuRecord],
    start_timestamp_ns_utc: int,
    target_timestamps_ns_utc: Sequence[int],
) -> tuple[tuple[FloatArray, FloatArray, FloatArray, bool], ...]:
    held: ImuRecord | None = None
    following = _next_or_none(imu_records)
    while following is not None and following.timestamp_ns_utc <= start_timestamp_ns_utc:
        held = following
        following = _next_or_none(imu_records)
    if held is None:
        raise ValueError("no arrived IMU sample exists at Phase 4 initialization")
    cursor = start_timestamp_ns_utc
    intervals: list[tuple[FloatArray, FloatArray, FloatArray, bool]] = []
    for target in target_timestamps_ns_utc:
        if target <= cursor:
            raise ValueError("Phase 4 target timestamps must be strictly increasing")
        accelerations: list[FloatArray] = []
        angular_velocities: list[FloatArray] = []
        durations: list[float] = []
        timing_gap = False
        while following is not None and following.timestamp_ns_utc <= target:
            if following.timestamp_ns_utc > cursor:
                accelerations.append(held.linear_acceleration_b_mps2)
                angular_velocities.append(held.angular_velocity_b_radps)
                durations.append((following.timestamp_ns_utc - cursor) * 1e-9)
                cursor = following.timestamp_ns_utc
            timing_gap = timing_gap or following.timing_gap
            held = following
            following = _next_or_none(imu_records)
        if cursor < target:
            accelerations.append(held.linear_acceleration_b_mps2)
            angular_velocities.append(held.angular_velocity_b_radps)
            durations.append((target - cursor) * 1e-9)
            cursor = target
        if not durations or not np.isclose(sum(durations), 1.0, atol=1e-9, rtol=0.0):
            raise ValueError("each frozen Medium student step must contain one second of IMU time")
        intervals.append(
            (
                np.vstack(accelerations),
                np.vstack(angular_velocities),
                np.asarray(durations, dtype=np.float64),
                timing_gap,
            )
        )
    return tuple(intervals)


def assemble_phase4_sequence(
    config: Phase4Config,
    phase3_config: FixedEskfConfig,
    pvt_path: Path,
    imu_path: Path,
    extrinsic_path: Path,
    teacher_path: Path,
) -> Phase4Sequence:
    """Assemble only deployable inputs and posterior weak pseudo-label means."""

    _regular_file(imu_path, "imu.csv")
    if sha256_file(imu_path) != config.imu_sha256:
        raise ValueError("IMU input differs from the frozen Phase 2 hash")
    pvt_records = read_standardized_pvt(pvt_path)
    quality_by_timestamp = _read_pvt_quality(pvt_path, config)
    extrinsic = read_selected_extrinsic(extrinsic_path, phase3_config.extrinsic_sha256)
    initialization, _, _ = _initialization_inputs(
        pvt_records,
        imu_path,
        phase3_config,
        extrinsic.antenna_lever_imu_m,
    )
    if initialization.timestamp_ns_utc != config.initialization_timestamp_ns_utc:
        raise ValueError("Phase 4 initialization differs from the frozen Phase 3 time")
    labels = _read_weak_labels(teacher_path, config)
    timestamps = tuple(label.timestamp_ns_utc for label in labels)
    if timestamps[0] != initialization.timestamp_ns_utc + 1_000_000_000:
        raise ValueError("first Phase 4 output must be one second after initialization")
    pvt_by_timestamp = {record.timestamp_ns_utc: record for record in pvt_records}
    imu_intervals = _partition_imu_intervals(
        iter_standardized_imu(imu_path),
        initialization.timestamp_ns_utc,
        timestamps,
    )
    steps: list[Phase4StepData] = []
    for label, interval in zip(labels, imu_intervals, strict=True):
        pvt = pvt_by_timestamp.get(label.timestamp_ns_utc)
        if pvt is None or pvt.solution_valid != label.pvt_solution_valid:
            raise ValueError("PVT and weak-label validity do not align")
        measured_position = (
            None
            if pvt.position_ecef_m is None
            else ecef_position_to_ned(initialization.local_frame, pvt.position_ecef_m)
        )
        acceleration, angular_velocity, duration, timing_gap = interval
        steps.append(
            Phase4StepData(
                timestamp_ns_utc=label.timestamp_ns_utc,
                linear_acceleration_b_mps2=acceleration,
                angular_velocity_b_radps=angular_velocity,
                dt_s=duration,
                imu_timing_gap=timing_gap,
                pvt_solution_valid=pvt.solution_valid,
                pvt_antenna_position_n_m=measured_position,
                pvt_quality=quality_by_timestamp[label.timestamp_ns_utc],
                weak_pseudo_label=label.state,
            )
        )
    initial = initialization.state
    if np.any(initial.accelerometer_bias_b_mps2 != 0.0) or np.any(initial.gyroscope_bias_b_radps != 0.0):
        raise ValueError("Phase 4 zero-bias physics reference requires zero initialization biases")
    return Phase4Sequence(
        initialization_timestamp_ns_utc=initialization.timestamp_ns_utc,
        initial_student_state=MeanStateArray(
            position_n_m=initial.position_n_m,
            velocity_n_mps=initial.velocity_n_mps,
            rotation_n_from_b=initial.rotation_n_from_b,
        ),
        antenna_lever_imu_m=extrinsic.antenna_lever_imu_m,
        gravity_n_mps2=normal_gravity_n_mps2(initialization.local_frame),
        earth_rotation_n_radps=earth_rotation_n_radps(initialization.local_frame),
        steps=tuple(steps),
    )
