"""Contract validation and receiver-native NMEA cross-check for Phase 2 PVT."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from .gps_spp import WGS84_A, WGS84_F, ecef_to_geodetic


_GGA_PATTERN = re.compile(rb"\$(?:GP|GN)GGA,[\x20-\x7e]{1,200}\*[0-9A-Fa-f]{2}")


class PvtCrosscheckError(RuntimeError):
    """Raised when a standardized PVT or NMEA cross-check input is invalid."""


@dataclass(frozen=True)
class NmeaGga:
    timestamp_ns_utc: int
    latitude_rad: float
    longitude_rad: float
    ellipsoid_height_m: float
    fix_quality: int
    satellite_count: int
    hdop: float | None


def geodetic_to_ecef(
    latitude_rad: float,
    longitude_rad: float,
    height_m: float,
) -> tuple[float, float, float]:
    eccentricity_sq = WGS84_F * (2.0 - WGS84_F)
    sin_latitude = math.sin(latitude_rad)
    normal = WGS84_A / math.sqrt(1.0 - eccentricity_sq * sin_latitude**2)
    cos_latitude = math.cos(latitude_rad)
    return (
        (normal + height_m) * cos_latitude * math.cos(longitude_rad),
        (normal + height_m) * cos_latitude * math.sin(longitude_rad),
        (normal * (1.0 - eccentricity_sq) + height_m) * sin_latitude,
    )


def _checksum_valid(sentence: bytes) -> bool:
    payload, checksum = sentence[1:].rsplit(b"*", 1)
    value = 0
    for byte in payload:
        value ^= byte
    try:
        expected = int(checksum, 16)
    except ValueError:
        return False
    return value == expected


def _nmea_angle(raw: str, hemisphere: str, degree_digits: int) -> float:
    if len(raw) <= degree_digits:
        raise PvtCrosscheckError("invalid NMEA angle")
    degrees = float(raw[:degree_digits])
    minutes = float(raw[degree_digits:])
    value = degrees + minutes / 60.0
    if hemisphere in {"S", "W"}:
        value = -value
    elif hemisphere not in {"N", "E"}:
        raise PvtCrosscheckError("invalid NMEA hemisphere")
    return math.radians(value)


def _nmea_time_ns(session_date: date, raw: str) -> int:
    if len(raw) < 6:
        raise PvtCrosscheckError("invalid NMEA UTC field")
    hour = int(raw[0:2])
    minute = int(raw[2:4])
    seconds = float(raw[4:])
    whole_seconds = int(seconds)
    microseconds = int(round((seconds - whole_seconds) * 1_000_000))
    value = datetime(
        session_date.year,
        session_date.month,
        session_date.day,
        hour,
        minute,
        whole_seconds,
        microseconds,
        tzinfo=timezone.utc,
    )
    return int(value.timestamp()) * 1_000_000_000 + value.microsecond * 1_000


def read_nmea_gga(path: Path, session_date: date) -> tuple[NmeaGga, ...]:
    """Positive-extract checksum-valid GGA sentences from a mixed binary file."""

    if path.is_symlink():
        raise PvtCrosscheckError("NMEA path cannot be a symbolic link")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PvtCrosscheckError("cannot read receiver-native NMEA") from exc
    records: list[NmeaGga] = []
    for match in _GGA_PATTERN.finditer(payload):
        sentence = match.group(0)
        if not _checksum_valid(sentence):
            continue
        try:
            fields = sentence.decode("ascii").split("*")[0].split(",")
            if len(fields) < 15 or not fields[1] or not fields[2] or not fields[4]:
                continue
            fix_quality = int(fields[6] or 0)
            if fix_quality <= 0:
                continue
            altitude = float(fields[9])
            geoid_separation = float(fields[11])
            if fields[10] != "M" or fields[12] != "M":
                continue
            records.append(
                NmeaGga(
                    timestamp_ns_utc=_nmea_time_ns(session_date, fields[1]),
                    latitude_rad=_nmea_angle(fields[2], fields[3], 2),
                    longitude_rad=_nmea_angle(fields[4], fields[5], 3),
                    ellipsoid_height_m=altitude + geoid_separation,
                    fix_quality=fix_quality,
                    satellite_count=int(fields[7]),
                    hdop=float(fields[8]) if fields[8] else None,
                )
            )
        except (ValueError, OverflowError, PvtCrosscheckError):
            continue
    records.sort(key=lambda item: item.timestamp_ns_utc)
    return tuple(records)


def validate_pvt_records(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PvtCrosscheckError("cannot read standardized PVT JSONL") from exc
    timestamps = [int(item["timestamp_ns_utc"]) for item in records]
    strictly_increasing = all(
        later > earlier for earlier, later in zip(timestamps, timestamps[1:])
    )
    valid_records = [item for item in records if item.get("solution_valid") is True]
    invalid_records = [item for item in records if item.get("solution_valid") is not True]
    valid_contract = True
    for item in valid_records:
        position = item.get("ecef_position_m")
        covariance = item.get("position_covariance_ecef_m2")
        if not (
            isinstance(position, list)
            and len(position) == 3
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in position)
            and 6.0e6 < math.sqrt(sum(float(value) ** 2 for value in position)) < 7.0e6
            and isinstance(covariance, list)
            and len(covariance) == 9
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in covariance)
            and all(float(covariance[index]) > 0.0 for index in (0, 4, 8))
            and item.get("coordinate_frame") == "WGS84_ECEF_broadcast"
            and item.get("velocity_valid") is False
            and item.get("ecef_velocity_mps") is None
        ):
            valid_contract = False
            break
    invalid_contract = all(
        item.get("ecef_position_m") is None
        and item.get("receiver_clock_bias_m") is None
        and item.get("covariance_valid") is False
        for item in invalid_records
    )
    report = {
        "epoch_count": len(records),
        "valid_count": len(valid_records),
        "invalid_count": len(invalid_records),
        "timestamps_strictly_increasing": strictly_increasing,
        "valid_record_contract": valid_contract,
        "invalid_record_contract": invalid_contract,
        "contract_status": (
            "PASS"
            if records and strictly_increasing and valid_contract and invalid_contract
            else "FAIL"
        ),
    }
    return records, report


def _percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def crosscheck_pvt_against_nmea(
    pvt_path: Path,
    nmea_path: Path,
    session_date: date,
) -> dict[str, object]:
    pvt_records, contract = validate_pvt_records(pvt_path)
    nmea_records = read_nmea_gga(nmea_path, session_date)
    nmea_by_second: dict[int, NmeaGga] = {}
    for record in nmea_records:
        second = (record.timestamp_ns_utc + 500_000_000) // 1_000_000_000
        nmea_by_second.setdefault(second, record)
    horizontal: list[float] = []
    vertical: list[float] = []
    distance_3d: list[float] = []
    for record in pvt_records:
        if record.get("solution_valid") is not True:
            continue
        second = (int(record["timestamp_ns_utc"]) + 500_000_000) // 1_000_000_000
        nmea = nmea_by_second.get(second)
        if nmea is None:
            continue
        native_ecef = geodetic_to_ecef(
            nmea.latitude_rad,
            nmea.longitude_rad,
            nmea.ellipsoid_height_m,
        )
        position = record["ecef_position_m"]
        assert isinstance(position, list)
        delta = tuple(float(position[index]) - native_ecef[index] for index in range(3))
        latitude, longitude, _ = ecef_to_geodetic(native_ecef)
        east = -math.sin(longitude) * delta[0] + math.cos(longitude) * delta[1]
        north = (
            -math.sin(latitude) * math.cos(longitude) * delta[0]
            - math.sin(latitude) * math.sin(longitude) * delta[1]
            + math.cos(latitude) * delta[2]
        )
        up = (
            math.cos(latitude) * math.cos(longitude) * delta[0]
            + math.cos(latitude) * math.sin(longitude) * delta[1]
            + math.sin(latitude) * delta[2]
        )
        horizontal.append(math.hypot(east, north))
        vertical.append(up)
        distance_3d.append(math.sqrt(sum(value * value for value in delta)))
    minimum_matches = 100
    gross_median_limit_m = 1000.0
    gross_status = (
        "PASS"
        if len(distance_3d) >= minimum_matches
        and statistics.median(distance_3d) < gross_median_limit_m
        else "FAIL"
    )
    return {
        "schema_version": 1,
        "comparison_role": "receiver-native non-high-precision cross-check only",
        "pvt_contract": contract,
        "checksum_valid_fixed_gga_count": len(nmea_records),
        "matched_valid_epoch_count": len(distance_3d),
        "time_match_rule": "nearest integer UTC second; source formats differ by 0.006 s",
        "gross_check_preregistered_minimum_matches": minimum_matches,
        "gross_check_preregistered_median_3d_limit_m": gross_median_limit_m,
        "gross_check_status": gross_status,
        "difference_m": (
            {
                "horizontal_median": statistics.median(horizontal),
                "horizontal_p95": _percentile_95(horizontal),
                "horizontal_max": max(horizontal),
                "vertical_median_signed": statistics.median(vertical),
                "vertical_absolute_p95": _percentile_95([abs(item) for item in vertical]),
                "distance_3d_median": statistics.median(distance_3d),
                "distance_3d_p95": _percentile_95(distance_3d),
                "distance_3d_max": max(distance_3d),
            }
            if distance_3d
            else None
        ),
        "configuration_changed_from_crosscheck": False,
        "high_precision_reference_used": False,
    }
