"""Deterministic Phase 2 standardization of the selected ROS IMU CSV."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path


class ImuStandardizationError(RuntimeError):
    """Raised when the selected IMU cannot satisfy the frozen record contract."""


_ORIENTATION = [f"field.orientation.{axis}" for axis in ("x", "y", "z", "w")]
_ANGULAR_VELOCITY = [
    f"field.angular_velocity.{axis}" for axis in ("x", "y", "z")
]
_LINEAR_ACCELERATION = [
    f"field.linear_acceleration.{axis}" for axis in ("x", "y", "z")
]
_SOURCE_COVARIANCES = {
    "orientation_covariance": [f"field.orientation_covariance{index}" for index in range(9)],
    "angular_velocity_covariance": [
        f"field.angular_velocity_covariance{index}" for index in range(9)
    ],
    "linear_acceleration_covariance": [
        f"field.linear_acceleration_covariance{index}" for index in range(9)
    ],
}
_REQUIRED_SOURCE_FIELDS = {
    "%time",
    "field.header.seq",
    "field.header.stamp",
    "field.header.frame_id",
    *_ORIENTATION,
    *_ANGULAR_VELOCITY,
    *_LINEAR_ACCELERATION,
    *(field for fields in _SOURCE_COVARIANCES.values() for field in fields),
}


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _finite_float(raw: str, field: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ImuStandardizationError(f"invalid IMU numeric field: {field}") from exc
    if not math.isfinite(value):
        raise ImuStandardizationError(f"non-finite IMU numeric field: {field}")
    return value


def _standardized_header() -> list[str]:
    fields = [
        "timestamp_ns_utc",
        "receipt_timestamp_ns",
        "sequence",
        "frame_id",
        *[f"orientation_{axis}" for axis in ("x", "y", "z", "w")],
        *[f"orientation_covariance_{index}" for index in range(9)],
        *[f"angular_velocity_radps_{axis}" for axis in ("x", "y", "z")],
        *[f"angular_velocity_covariance_{index}" for index in range(9)],
        *[f"linear_acceleration_mps2_{axis}" for axis in ("x", "y", "z")],
        *[f"linear_acceleration_covariance_{index}" for index in range(9)],
        "valid",
        "timing_gap",
        "within_common_interval",
    ]
    return fields


def _pvt_time_bounds(pvt_path: Path) -> tuple[int, int]:
    first: int | None = None
    last: int | None = None
    try:
        with pvt_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                timestamp = int(json.loads(line)["timestamp_ns_utc"])
                if first is None:
                    first = timestamp
                last = timestamp
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ImuStandardizationError("cannot read PVT time boundary") from exc
    if first is None or last is None or last < first:
        raise ImuStandardizationError("invalid PVT time boundary")
    return first, last


def standardize_medium_imu(
    source_lock_path: Path,
    profile_path: Path,
    data_root: Path,
) -> tuple[Path, Path]:
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    artifacts = {item["id"]: item for item in source_lock["artifacts"]}
    imu_artifact = artifacts["low_cost_imu"]
    validation = imu_artifact["deployable_validation"]
    session_root = (
        data_root
        / "deployable"
        / source_lock["dataset"]
        / source_lock["repository_commit"]
        / source_lock["session"]
    )
    source_path = session_root / "imu" / imu_artifact["local_name"]
    source_size, source_digest = _sha256_file(source_path)
    if (source_size, source_digest) != (
        imu_artifact["observed_byte_size"],
        imu_artifact["observed_sha256"],
    ):
        raise ImuStandardizationError("IMU source differs from acquisition record")
    pvt_path = (
        data_root
        / "standardized"
        / source_lock["dataset"]
        / source_lock["repository_commit"]
        / source_lock["session"]
        / profile["profile_id"]
        / "pvt.jsonl"
    )
    overlap_first, overlap_last = _pvt_time_bounds(pvt_path)
    median_interval_ns = int(validation["median_interval_ns"])
    gap_threshold_ns = 2 * median_interval_ns
    output_root = (
        data_root
        / "standardized"
        / source_lock["dataset"]
        / source_lock["repository_commit"]
        / source_lock["session"]
        / "imu_contract_v1"
    )
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_path = output_root / "imu.csv"
    temporary = output_root / ".imu.csv.tmp"
    temporary.unlink(missing_ok=True)
    row_count = 0
    valid_count = 0
    gap_count = 0
    within_count = 0
    previous_timestamp: int | None = None
    try:
        with source_path.open("r", encoding="utf-8", newline="") as source_handle, temporary.open(
            "x", encoding="utf-8", newline=""
        ) as output_handle:
            reader = csv.DictReader(source_handle)
            if reader.fieldnames is None:
                raise ImuStandardizationError("IMU CSV has no header")
            missing = sorted(_REQUIRED_SOURCE_FIELDS - set(reader.fieldnames))
            if missing:
                raise ImuStandardizationError(
                    f"IMU CSV lacks required standardization fields: {', '.join(missing)}"
                )
            header = _standardized_header()
            writer = csv.DictWriter(output_handle, fieldnames=header, lineterminator="\n")
            writer.writeheader()
            for source_row in reader:
                timestamp = int(source_row["field.header.stamp"])
                receipt = int(source_row["%time"])
                sequence = int(source_row["field.header.seq"])
                interval = (
                    timestamp - previous_timestamp if previous_timestamp is not None else None
                )
                valid = previous_timestamp is None or timestamp > previous_timestamp
                timing_gap = interval is not None and interval > gap_threshold_ns
                within = overlap_first <= timestamp <= overlap_last
                values: dict[str, float] = {}
                for field in (
                    *_ORIENTATION,
                    *_ANGULAR_VELOCITY,
                    *_LINEAR_ACCELERATION,
                    *(item for fields in _SOURCE_COVARIANCES.values() for item in fields),
                ):
                    values[field] = _finite_float(source_row[field], field)
                output_row: dict[str, object] = {
                    "timestamp_ns_utc": timestamp,
                    "receipt_timestamp_ns": receipt,
                    "sequence": sequence,
                    "frame_id": source_row["field.header.frame_id"],
                    "valid": str(valid).lower(),
                    "timing_gap": str(timing_gap).lower(),
                    "within_common_interval": str(within).lower(),
                }
                for axis, field in zip(("x", "y", "z", "w"), _ORIENTATION):
                    output_row[f"orientation_{axis}"] = repr(values[field])
                for index, field in enumerate(_SOURCE_COVARIANCES["orientation_covariance"]):
                    output_row[f"orientation_covariance_{index}"] = repr(values[field])
                for axis, field in zip(("x", "y", "z"), _ANGULAR_VELOCITY):
                    output_row[f"angular_velocity_radps_{axis}"] = repr(values[field])
                for index, field in enumerate(
                    _SOURCE_COVARIANCES["angular_velocity_covariance"]
                ):
                    output_row[f"angular_velocity_covariance_{index}"] = repr(values[field])
                for axis, field in zip(("x", "y", "z"), _LINEAR_ACCELERATION):
                    output_row[f"linear_acceleration_mps2_{axis}"] = repr(values[field])
                for index, field in enumerate(
                    _SOURCE_COVARIANCES["linear_acceleration_covariance"]
                ):
                    output_row[f"linear_acceleration_covariance_{index}"] = repr(values[field])
                writer.writerow(output_row)
                row_count += 1
                valid_count += valid
                gap_count += timing_gap
                within_count += within
                previous_timestamp = timestamp
        temporary.chmod(0o600)
        os.replace(temporary, output_path)
    except (OSError, ValueError, csv.Error, ImuStandardizationError) as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, ImuStandardizationError):
            raise
        raise ImuStandardizationError("cannot standardize IMU CSV") from exc
    if row_count != int(validation["row_count"]):
        raise ImuStandardizationError("standardized IMU row count differs from validation")
    if gap_count != int(validation["intervals_over_2x_median_count"]):
        raise ImuStandardizationError("standardized IMU gap flags differ from validation")
    output_size, output_digest = _sha256_file(output_path)
    summary = {
        "schema_version": 1,
        "dataset": source_lock["dataset"],
        "session": source_lock["session"],
        "repository_commit": source_lock["repository_commit"],
        "contract_id": "imu_contract_v1",
        "source_sha256": source_digest,
        "row_count": row_count,
        "valid_count": valid_count,
        "timing_gap_count": gap_count,
        "within_common_interval_count": within_count,
        "common_interval_first_ns_utc": overlap_first,
        "common_interval_last_ns_utc": overlap_last,
        "output_path": output_path.name,
        "output_size": output_size,
        "output_sha256": output_digest,
        "interpolation_applied": False,
        "ins_or_filter_applied": False,
        "high_precision_reference_used": False,
    }
    summary_path = output_root / "imu_summary.json"
    summary_temporary = output_root / ".imu_summary.json.tmp"
    summary_temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_temporary.chmod(0o600)
    os.replace(summary_temporary, summary_path)
    return output_path, summary_path
