"""Read-only diagnostics for the selected ROS-exported IMU CSV."""

from __future__ import annotations

import calendar
import csv
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


class ImuDiagnosticsError(RuntimeError):
    """Raised when the selected IMU CSV violates its deployable contract."""


_TIME_FIELDS = ("%time", "field.header.stamp")
_VECTOR_FIELDS = (
    "field.orientation.x",
    "field.orientation.y",
    "field.orientation.z",
    "field.orientation.w",
    "field.angular_velocity.x",
    "field.angular_velocity.y",
    "field.angular_velocity.z",
    "field.linear_acceleration.x",
    "field.linear_acceleration.y",
    "field.linear_acceleration.z",
)
_REQUIRED_FIELDS = (
    *_TIME_FIELDS,
    "field.header.seq",
    "field.header.frame_id",
    *_VECTOR_FIELDS,
)


def _integer_summary(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _unix_ns_iso(value: int) -> str:
    seconds, nanoseconds = divmod(value, 1_000_000_000)
    timestamp = datetime.fromtimestamp(seconds, timezone.utc)
    return f"{timestamp:%Y-%m-%dT%H:%M:%S}.{nanoseconds:09d}+00:00"


def inspect_imu_csv(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise ImuDiagnosticsError("IMU CSV path cannot be a symbolic link")
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise ImuDiagnosticsError("cannot open IMU CSV") from exc

    row_count = 0
    first_sensor_ns: int | None = None
    last_sensor_ns: int | None = None
    first_receipt_ns: int | None = None
    last_receipt_ns: int | None = None
    previous_sensor_ns: int | None = None
    previous_receipt_ns: int | None = None
    previous_sequence: int | None = None
    first_sequence: int | None = None
    last_sequence: int | None = None
    sensor_intervals: list[int] = []
    receipt_intervals: list[int] = []
    receipt_minus_sensor: list[int] = []
    nonpositive_sensor_intervals = 0
    nonpositive_receipt_intervals = 0
    sequence_discontinuities = 0
    missing_sequence_ids = 0
    missing_values = 0
    nonfinite_values = 0
    quaternion_norm_error_max = 0.0
    frame_ids: set[str] = set()

    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ImuDiagnosticsError("IMU CSV has no header")
        missing_fields = sorted(set(_REQUIRED_FIELDS) - set(reader.fieldnames))
        if missing_fields:
            raise ImuDiagnosticsError(
                f"IMU CSV lacks required fields: {', '.join(missing_fields)}"
            )
        for row in reader:
            row_count += 1
            try:
                sensor_ns = int(row["field.header.stamp"])
                receipt_ns = int(row["%time"])
                sequence = int(row["field.header.seq"])
            except (TypeError, ValueError) as exc:
                raise ImuDiagnosticsError("invalid IMU time or sequence field") from exc
            if first_sensor_ns is None:
                first_sensor_ns = sensor_ns
                first_receipt_ns = receipt_ns
                first_sequence = sequence
            last_sensor_ns = sensor_ns
            last_receipt_ns = receipt_ns
            last_sequence = sequence
            frame_ids.add(row["field.header.frame_id"])

            if previous_sensor_ns is not None:
                delta = sensor_ns - previous_sensor_ns
                sensor_intervals.append(delta)
                nonpositive_sensor_intervals += delta <= 0
            if previous_receipt_ns is not None:
                delta = receipt_ns - previous_receipt_ns
                receipt_intervals.append(delta)
                nonpositive_receipt_intervals += delta <= 0
            if previous_sequence is not None and sequence != previous_sequence + 1:
                sequence_discontinuities += 1
                if sequence > previous_sequence + 1:
                    missing_sequence_ids += sequence - previous_sequence - 1
            previous_sensor_ns = sensor_ns
            previous_receipt_ns = receipt_ns
            previous_sequence = sequence
            receipt_minus_sensor.append(receipt_ns - sensor_ns)

            values: dict[str, float] = {}
            for field in _VECTOR_FIELDS:
                raw = row[field]
                if raw == "":
                    missing_values += 1
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    nonfinite_values += 1
                    continue
                if not math.isfinite(value):
                    nonfinite_values += 1
                    continue
                values[field] = value
            orientation_fields = _VECTOR_FIELDS[:4]
            if all(field in values for field in orientation_fields):
                norm = math.sqrt(sum(values[field] ** 2 for field in orientation_fields))
                quaternion_norm_error_max = max(quaternion_norm_error_max, abs(norm - 1.0))

    if row_count == 0 or first_sensor_ns is None or first_receipt_ns is None:
        raise ImuDiagnosticsError("IMU CSV contains no records")
    assert last_sensor_ns is not None
    assert last_receipt_ns is not None
    assert first_sequence is not None
    assert last_sequence is not None
    nominal_sensor_interval = int(statistics.median(sensor_intervals))
    interval_gap_threshold = nominal_sensor_interval * 2
    return {
        "path": str(path),
        "column_count": len(reader.fieldnames),
        "columns": reader.fieldnames,
        "row_count": row_count,
        "frame_ids": sorted(frame_ids),
        "first_sequence": first_sequence,
        "last_sequence": last_sequence,
        "sequence_discontinuity_count": sequence_discontinuities,
        "missing_sequence_id_count": missing_sequence_ids,
        "sensor_stamp_first_ns": first_sensor_ns,
        "sensor_stamp_last_ns": last_sensor_ns,
        "sensor_stamp_first_epoch_interpretation": _unix_ns_iso(first_sensor_ns),
        "sensor_stamp_last_epoch_interpretation": _unix_ns_iso(last_sensor_ns),
        "sensor_duration_s": (last_sensor_ns - first_sensor_ns) / 1_000_000_000,
        "sensor_interval_ns": _integer_summary(sensor_intervals),
        "nominal_rate_hz": 1_000_000_000 / nominal_sensor_interval,
        "sensor_nonpositive_interval_count": nonpositive_sensor_intervals,
        "sensor_intervals_over_2x_median_count": sum(
            value > interval_gap_threshold for value in sensor_intervals
        ),
        "receipt_stamp_first_ns": first_receipt_ns,
        "receipt_stamp_last_ns": last_receipt_ns,
        "receipt_interval_ns": _integer_summary(receipt_intervals),
        "receipt_nonpositive_interval_count": nonpositive_receipt_intervals,
        "receipt_minus_sensor_ns": _integer_summary(receipt_minus_sensor),
        "required_vector_missing_value_count": missing_values,
        "required_vector_nonfinite_value_count": nonfinite_values,
        "quaternion_norm_error_max": quaternion_norm_error_max,
    }


def align_imu_and_gnss_time(
    imu_report: dict[str, object],
    observation_report: dict[str, object],
    navigation_report: dict[str, object],
) -> dict[str, object]:
    navigation_header = navigation_report["header"]
    assert isinstance(navigation_header, dict)
    leap_seconds = navigation_header.get("leap_seconds")
    if not isinstance(leap_seconds, int):
        raise ImuDiagnosticsError("broadcast navigation lacks leap-second metadata")
    first_gps = datetime.fromisoformat(str(observation_report["first_epoch"]))
    last_gps = datetime.fromisoformat(str(observation_report["last_epoch"]))

    def calendar_ns(value: datetime) -> int:
        return (
            calendar.timegm(value.timetuple()) * 1_000_000_000
            + value.microsecond * 1_000
        )

    first_gnss_utc_ns = calendar_ns(first_gps) - leap_seconds * 1_000_000_000
    last_gnss_utc_ns = calendar_ns(last_gps) - leap_seconds * 1_000_000_000
    first_imu_ns = int(imu_report["sensor_stamp_first_ns"])
    last_imu_ns = int(imu_report["sensor_stamp_last_ns"])
    overlap_start = max(first_gnss_utc_ns, first_imu_ns)
    overlap_end = min(last_gnss_utc_ns, last_imu_ns)
    return {
        "contract_time_scale": "UTC Unix nanoseconds",
        "rinex_native_time_scale": "GPS",
        "gps_minus_utc_s_from_broadcast_navigation": leap_seconds,
        "imu_primary_time_field": "field.header.stamp",
        "imu_receipt_time_field": "%time",
        "imu_stamp_interpretation": (
            "ROS epoch nanoseconds; UTC/system-time interpretation is supported "
            "by the fixed leap-second alignment"
        ),
        "gnss_first_utc_ns": first_gnss_utc_ns,
        "gnss_last_utc_ns": last_gnss_utc_ns,
        "gnss_first_utc": _unix_ns_iso(first_gnss_utc_ns),
        "gnss_last_utc": _unix_ns_iso(last_gnss_utc_ns),
        "imu_starts_after_gnss_s": (first_imu_ns - first_gnss_utc_ns) / 1_000_000_000,
        "imu_ends_after_gnss_s": (last_imu_ns - last_gnss_utc_ns) / 1_000_000_000,
        "overlap_duration_s": max(0, overlap_end - overlap_start) / 1_000_000_000,
    }
