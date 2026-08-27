"""Independent structural checks for fixed Phase 3 baseline outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .phase3_config import FixedEskfConfig
from .phase3_eskf import (
    POSITION_NIS_CHI_SQUARE_95,
    POSITION_NIS_CHI_SQUARE_99,
    POSITION_NIS_DOF,
)


SCENARIOS = (
    "ins_only",
    "fixed_eskf",
    "fixed_eskf_outage_20s",
    "fixed_eskf_outage_30s",
)
OUTPUT_FILES = ("spp_only.jsonl",) + tuple(f"{name}.jsonl" for name in SCENARIOS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Phase 3 output: {path.name}") from exc
    if not records:
        raise ValueError(f"Phase 3 output is empty: {path.name}")
    return records


def _vector(record: dict[str, object], key: str, size: int) -> np.ndarray:
    value = np.asarray(record[key], dtype=np.float64)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must contain {size} finite values")
    return value


def _summary(directory: Path) -> dict[str, object]:
    try:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read Phase 3 summary") from exc
    if not isinstance(summary, dict):
        raise ValueError("Phase 3 summary must be an object")
    return summary


def validate_baseline_outputs(
    config: FixedEskfConfig,
    run_directory: Path,
    comparison_directory: Path | None = None,
) -> dict[str, object]:
    """Validate one run and optionally require a second run to be byte-identical."""

    if not run_directory.is_absolute() or not run_directory.is_dir():
        raise ValueError("run directory must be an existing absolute directory")
    summary = _summary(run_directory)
    if summary.get("profile_id") != config.profile_id:
        raise ValueError("output profile differs from the fixed configuration")
    if summary.get("high_precision_reference_used") is not False:
        raise ValueError("output does not preserve the high-precision-reference boundary")
    if summary.get("receiver_native_pvt_used_as_truth") is not False:
        raise ValueError("output incorrectly treats receiver-native PVT as truth")
    expected_inputs = {
        "pvt": config.pvt_sha256,
        "imu": config.imu_sha256,
        "imu_noise_parameter": config.imu_noise_parameter_sha256,
        "extrinsic": config.extrinsic_sha256,
    }
    if summary.get("input_sha256") != expected_inputs:
        raise ValueError("output input hashes differ from the fixed configuration")

    output_hashes = summary.get("output_sha256")
    if not isinstance(output_hashes, dict) or set(output_hashes) != set(OUTPUT_FILES):
        raise ValueError("summary output hash set is incomplete")
    for name in OUTPUT_FILES:
        if _sha256(run_directory / name) != output_hashes[name]:
            raise ValueError(f"output hash mismatch: {name}")

    deterministic = comparison_directory is not None
    if comparison_directory is not None:
        if not comparison_directory.is_absolute() or not comparison_directory.is_dir():
            raise ValueError("comparison directory must be an existing absolute directory")
        for name in ("summary.json",) + OUTPUT_FILES:
            if (run_directory / name).read_bytes() != (
                comparison_directory / name
            ).read_bytes():
                raise ValueError(f"independent runs differ: {name}")

    spp_records = _read_jsonl(run_directory / "spp_only.jsonl")
    spp_times = [int(record["timestamp_ns_utc"]) for record in spp_records]
    if any(right <= left for left, right in zip(spp_times, spp_times[1:])):
        raise ValueError("SPP-only output timestamps are not strictly increasing")

    initialization_ns = int(summary["initialization_timestamp_ns_utc"])
    outage_start_ns = (
        initialization_ns
        + config.outage_start_after_initialization_s * 1_000_000_000
    )
    summary_counts = summary.get("scenario_counts")
    if not isinstance(summary_counts, dict) or set(summary_counts) != set(SCENARIOS):
        raise ValueError("scenario count set is incomplete")
    fixed_nis: list[float] = []
    minimum_covariance_eigenvalue = float("inf")

    for scenario in SCENARIOS:
        records = _read_jsonl(run_directory / f"{scenario}.jsonl")
        timestamps = [int(record["timestamp_ns_utc"]) for record in records]
        if timestamps[0] <= initialization_ns or any(
            right <= left for left, right in zip(timestamps, timestamps[1:])
        ):
            raise ValueError(f"{scenario} output time is noncausal or nonmonotonic")
        updates = 0
        masked_valid = 0
        duration_s = 0
        if scenario.endswith("20s"):
            duration_s = 20
        elif scenario.endswith("30s"):
            duration_s = 30
        outage_end_ns = outage_start_ns + duration_s * 1_000_000_000

        for record in records:
            if record.get("scenario") != scenario:
                raise ValueError(f"scenario label mismatch in {scenario}")
            timestamp = int(record["timestamp_ns_utc"])
            expected_mask = duration_s > 0 and outage_start_ns <= timestamp < outage_end_ns
            masked = record.get("gnss_masked") is True
            updated = record.get("position_update_applied") is True
            if masked != expected_mask or (masked and updated):
                raise ValueError(f"controlled outage boundary mismatch in {scenario}")
            if scenario == "ins_only" and updated:
                raise ValueError("INS-only output cannot contain a PVT update")
            updates += int(updated)
            masked_valid += int(masked and record.get("pvt_solution_valid") is True)

            _vector(record, "position_n_m", 3)
            _vector(record, "velocity_n_mps", 3)
            rotation = _vector(record, "rotation_n_from_b_row_major", 9).reshape(3, 3)
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.0):
                raise ValueError(f"rotation lost orthonormality in {scenario}")
            if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0.0):
                raise ValueError(f"rotation became improper in {scenario}")
            covariance = _vector(record, "covariance_15x15_row_major", 225).reshape(15, 15)
            if not np.array_equal(covariance, covariance.T):
                raise ValueError(f"covariance is not exactly symmetric in {scenario}")
            eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
            minimum_covariance_eigenvalue = min(
                minimum_covariance_eigenvalue, eigenvalue
            )
            if eigenvalue < -1e-10:
                raise ValueError(f"covariance is not positive semidefinite in {scenario}")
            nis = record.get("nis")
            if nis is not None:
                nis_value = float(nis)
                if not np.isfinite(nis_value) or nis_value < 0.0 or not updated:
                    raise ValueError(f"invalid NIS diagnostic in {scenario}")
                if scenario == "fixed_eskf":
                    fixed_nis.append(nis_value)

        expected = summary_counts[scenario]
        actual = {
            "record_count": len(records),
            "position_update_count": updates,
            "masked_valid_pvt_count": masked_valid,
        }
        if expected != actual:
            raise ValueError(f"summary counts differ for {scenario}")

    if not fixed_nis:
        raise ValueError("fixed ESKF output has no NIS diagnostics")
    above_95_count = sum(value > POSITION_NIS_CHI_SQUARE_95 for value in fixed_nis)
    above_99_count = sum(value > POSITION_NIS_CHI_SQUARE_99 for value in fixed_nis)
    nis_summary = {
        "degrees_of_freedom": POSITION_NIS_DOF,
        "count": len(fixed_nis),
        "minimum": min(fixed_nis),
        "median": float(np.median(fixed_nis)),
        "maximum": max(fixed_nis),
        "chi_square_95_threshold": POSITION_NIS_CHI_SQUARE_95,
        "above_chi_square_95_count": above_95_count,
        "above_chi_square_95_fraction": above_95_count / len(fixed_nis),
        "chi_square_99_threshold": POSITION_NIS_CHI_SQUARE_99,
        "above_chi_square_99_count": above_99_count,
        "above_chi_square_99_fraction": above_99_count / len(fixed_nis),
    }
    if summary.get("fixed_eskf_nis") != nis_summary:
        raise ValueError("fixed ESKF NIS summary differs from records")

    return {
        "profile_id": config.profile_id,
        "deterministic_comparison": deterministic,
        "spp_record_count": len(spp_records),
        "fixed_eskf_nis": nis_summary,
        "minimum_covariance_eigenvalue": minimum_covariance_eigenvalue,
        "high_precision_reference_used": False,
    }
