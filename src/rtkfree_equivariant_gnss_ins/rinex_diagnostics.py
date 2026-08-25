"""Read-only deployable diagnostics for RINEX 3 observation and navigation data."""

from __future__ import annotations

import gzip
import math
import statistics
from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TextIO


class RinexDiagnosticsError(RuntimeError):
    """Raised when a selected RINEX file cannot be diagnosed safely."""



def _satellite_id(raw: str) -> str | None:
    if len(raw) != 3 or not ("A" <= raw[0] <= "Z"):
        return None
    number_text = raw[1:]
    if not (
        number_text.isdigit()
        or (number_text[0] == " " and number_text[1].isdigit())
    ):
        return None
    number = int(number_text)
    if not 1 <= number <= 99:
        return None
    return f"{raw[0]}{number:02d}"


def _timestamp(parts: list[str]) -> datetime:
    if len(parts) < 6:
        raise RinexDiagnosticsError("RINEX timestamp has too few fields")
    try:
        seconds = Decimal(parts[5])
        whole_seconds = int(seconds)
        microseconds = int((seconds - whole_seconds) * Decimal(1_000_000))
        return datetime(
            int(parts[0]),
            int(parts[1]),
            int(parts[2]),
            int(parts[3]),
            int(parts[4]),
            whole_seconds,
        ) + timedelta(microseconds=microseconds)
    except (ValueError, InvalidOperation) as exc:
        raise RinexDiagnosticsError("invalid RINEX timestamp") from exc


def _iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="microseconds") if value is not None else None


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _read_header(handle: TextIO) -> tuple[dict[str, object], list[str]]:
    header: dict[str, object] = {
        "observation_types": {},
        "header_labels": [],
    }
    lines: list[str] = []
    current_system: str | None = None
    expected_observation_types: dict[str, int] = {}
    for _ in range(10_000):
        line = handle.readline()
        if not line:
            raise RinexDiagnosticsError("RINEX header has no END OF HEADER")
        lines.append(line)
        label = line[60:80].strip() if len(line) >= 60 else ""
        if label:
            cast_labels = header["header_labels"]
            assert isinstance(cast_labels, list)
            cast_labels.append(label)
        if label == "RINEX VERSION" + " / TYPE":
            try:
                header["version"] = float(line[:9].strip())
            except ValueError as exc:
                raise RinexDiagnosticsError("invalid RINEX version") from exc
            header["file_type"] = line[20:21].strip()
            header["satellite_system"] = line[40:41].strip()
        elif label == "MARKER NAME":
            header["marker_name"] = line[:60].strip()
        elif label == "REC # / TYPE / VERS":
            header["receiver"] = line[:60].rstrip()
        elif label == "ANT # / TYPE":
            header["antenna"] = line[:60].rstrip()
        elif label == "APPROX POSITION XYZ":
            header["approx_position_xyz_m"] = [float(item) for item in line[:60].split()]
        elif label == "ANTENNA: DELTA H/E/N":
            header["antenna_delta_hen_m"] = [float(item) for item in line[:60].split()]
        elif label == "INTERVAL":
            header["header_interval_s"] = float(line[:10].strip())
        elif label == "LEAP SECONDS":
            tokens = line[:60].split()
            if tokens:
                header["leap_seconds"] = int(tokens[0])
        elif label in {"TIME OF FIRST OBS", "TIME OF LAST OBS"}:
            tokens = line[:60].split()
            value = _timestamp(tokens[:6])
            key = "header_first_epoch" if label.startswith("TIME OF FIRST") else "header_last_epoch"
            header[key] = _iso(value)
            if len(tokens) > 6:
                header["time_system"] = tokens[6]
        elif label == "SYS / # / OBS TYPES":
            content = line[:60]
            if content[0:1].strip():
                current_system = content[0]
                try:
                    expected_observation_types[current_system] = int(content[3:6])
                except ValueError as exc:
                    raise RinexDiagnosticsError("invalid observation-type count") from exc
            if current_system is None:
                raise RinexDiagnosticsError("orphan observation-type continuation")
            observation_types = header["observation_types"]
            assert isinstance(observation_types, dict)
            observation_types.setdefault(current_system, []).extend(content[7:60].split())
        if label == "END OF HEADER":
            break

    if "version" not in header or "file_type" not in header:
        raise RinexDiagnosticsError("RINEX identity header is missing")
    observation_types = header["observation_types"]
    assert isinstance(observation_types, dict)
    for system, expected in expected_observation_types.items():
        actual = observation_types.get(system, [])
        if len(actual) < expected:
            raise RinexDiagnosticsError(f"incomplete observation types for system {system}")
        observation_types[system] = actual[:expected]
    return header, lines


def _parse_observation_block(
    lines: list[str], observation_types: dict[str, list[str]]
) -> tuple[set[str], set[str], dict[str, set[str]]]:
    satellites: set[str] = set()
    pseudorange_satellites: set[str] = set()
    pseudorange_signals: dict[str, set[str]] = {}
    index = 0
    while index < len(lines):
        line = lines[index].rstrip("\r\n")
        satellite = _satellite_id(line[:3])
        if satellite is None:
            index += 1
            continue
        satellites.add(satellite)
        system_types = observation_types.get(satellite[0], [])
        payload = line[3:]
        while len(payload) < 16 * len(system_types) and index + 1 < len(lines):
            continuation = lines[index + 1].rstrip("\r\n")
            if continuation[:3] != "   ":
                break
            index += 1
            payload += continuation[3:]
        for field_index, observation_type in enumerate(system_types):
            if not observation_type.startswith("C"):
                continue
            raw_value = payload[16 * field_index : 16 * field_index + 14].strip()
            if not raw_value:
                continue
            try:
                value = float(raw_value.replace("D", "E"))
            except ValueError:
                continue
            if math.isfinite(value) and value > 0.0:
                pseudorange_satellites.add(satellite)
                signal_key = f"{satellite[0]}:{observation_type}"
                pseudorange_signals.setdefault(signal_key, set()).add(satellite)
        index += 1
    return satellites, pseudorange_satellites, pseudorange_signals


def inspect_observation(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise RinexDiagnosticsError("observation path cannot be a symbolic link")
    try:
        handle = path.open("r", encoding="ascii", errors="strict")
    except OSError as exc:
        raise RinexDiagnosticsError("cannot open observation RINEX") from exc

    epochs: list[dict[str, object]] = []
    with handle:
        header, _ = _read_header(handle)
        if header["file_type"] != "O":
            raise RinexDiagnosticsError("selected file is not RINEX observation data")
        observation_types = header["observation_types"]
        assert isinstance(observation_types, dict)
        current_epoch: dict[str, object] | None = None
        block: list[str] = []

        def finish_epoch() -> None:
            if current_epoch is None:
                return
            satellites, pseudorange_satellites, pseudorange_signals = (
                _parse_observation_block(
                block, observation_types
                )
            )
            current_epoch["satellites"] = satellites
            current_epoch["pseudorange_satellites"] = pseudorange_satellites
            current_epoch["pseudorange_signals"] = pseudorange_signals
            epochs.append(current_epoch)

        for line in handle:
            if not line.startswith(">"):
                block.append(line)
                continue
            finish_epoch()
            block = []
            parts = line[1:].split()
            if len(parts) < 8:
                raise RinexDiagnosticsError("invalid RINEX epoch record")
            current_epoch = {
                "time": _timestamp(parts[:6]),
                "flag": int(parts[6]),
                "declared_satellites": int(parts[7]),
            }
        finish_epoch()

    observation_epochs = [epoch for epoch in epochs if epoch["flag"] in {0, 1}]
    times = [epoch["time"] for epoch in observation_epochs]
    assert all(isinstance(item, datetime) for item in times)
    deltas = [
        int((later - earlier).total_seconds() * 1_000_000)
        for earlier, later in zip(times, times[1:])
    ]
    positive_deltas = [value for value in deltas if value > 0]
    nominal_us = int(statistics.median(positive_deltas)) if positive_deltas else None
    gap_threshold_us = int(nominal_us * 1.5) if nominal_us is not None else None
    gaps = [value for value in positive_deltas if value > gap_threshold_us] if gap_threshold_us else []

    all_satellites: set[str] = set()
    declared_counts: list[int] = []
    parsed_counts: list[int] = []
    pseudorange_counts: list[int] = []
    system_pseudorange_counts: dict[str, list[int]] = {
        system: [] for system in observation_types
    }
    signal_pseudorange_counts: dict[str, list[int]] = {
        f"{system}:{observation_type}": []
        for system, system_types in observation_types.items()
        for observation_type in system_types
        if observation_type.startswith("C")
    }
    declared_mismatches = 0
    flags: Counter[int] = Counter()
    for epoch in epochs:
        flags[int(epoch["flag"])] += 1
    for epoch in observation_epochs:
        satellites = epoch["satellites"]
        pseudorange_satellites = epoch["pseudorange_satellites"]
        pseudorange_signals = epoch["pseudorange_signals"]
        assert isinstance(satellites, set)
        assert isinstance(pseudorange_satellites, set)
        assert isinstance(pseudorange_signals, dict)
        all_satellites.update(satellites)
        declared = int(epoch["declared_satellites"])
        declared_counts.append(declared)
        parsed_counts.append(len(satellites))
        pseudorange_counts.append(len(pseudorange_satellites))
        for system, counts in system_pseudorange_counts.items():
            counts.append(sum(satellite.startswith(system) for satellite in pseudorange_satellites))
        for signal_key, counts in signal_pseudorange_counts.items():
            counts.append(len(pseudorange_signals.get(signal_key, set())))
        if declared != len(satellites):
            declared_mismatches += 1

    constellation_satellites: Counter[str] = Counter(item[0] for item in all_satellites)
    return {
        "path": str(path),
        "header": header,
        "epoch_count": len(epochs),
        "observation_epoch_count": len(observation_epochs),
        "epoch_flags": {str(key): value for key, value in sorted(flags.items())},
        "first_epoch": _iso(times[0] if times else None),
        "last_epoch": _iso(times[-1] if times else None),
        "nominal_interval_s": nominal_us / 1_000_000 if nominal_us is not None else None,
        "nonpositive_interval_count": sum(value <= 0 for value in deltas),
        "obvious_gap_threshold_s": (
            gap_threshold_us / 1_000_000 if gap_threshold_us is not None else None
        ),
        "obvious_gap_count": len(gaps),
        "largest_interval_s": max(positive_deltas) / 1_000_000 if positive_deltas else None,
        "declared_satellites_per_epoch": _summary(declared_counts),
        "parsed_satellites_per_epoch": _summary(parsed_counts),
        "pseudorange_satellites_per_epoch": _summary(pseudorange_counts),
        "pseudorange_satellites_per_epoch_by_system": {
            key: _summary(values)
            for key, values in sorted(system_pseudorange_counts.items())
        },
        "pseudorange_satellites_per_epoch_by_signal": {
            key: _summary(values)
            for key, values in sorted(signal_pseudorange_counts.items())
        },
        "epochs_with_at_least_4_pseudorange_satellites": sum(
            value >= 4 for value in pseudorange_counts
        ),
        "epochs_with_at_least_4_pseudorange_satellites_by_signal": {
            key: sum(value >= 4 for value in values)
            for key, values in sorted(signal_pseudorange_counts.items())
        },
        "declared_vs_parsed_satellite_mismatch_epochs": declared_mismatches,
        "unique_satellite_count": len(all_satellites),
        "unique_satellites_by_constellation": dict(sorted(constellation_satellites.items())),
    }


def inspect_navigation_gzip(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise RinexDiagnosticsError("navigation path cannot be a symbolic link")
    record_counts: Counter[str] = Counter()
    message_times: list[datetime] = []
    try:
        with gzip.open(path, "rt", encoding="ascii", errors="strict") as handle:
            header, _ = _read_header(handle)
            if header["file_type"] != "N":
                raise RinexDiagnosticsError("selected file is not RINEX navigation data")
            for line in handle:
                satellite = _satellite_id(line[:3])
                if satellite is None:
                    continue
                record_counts[satellite[0]] += 1
                parts = line[3:].split()
                try:
                    message_times.append(_timestamp(parts[:6]))
                except RinexDiagnosticsError:
                    continue
    except (OSError, EOFError, UnicodeError) as exc:
        raise RinexDiagnosticsError("cannot read gzip navigation RINEX") from exc
    return {
        "path": str(path),
        "header": header,
        "navigation_record_count": sum(record_counts.values()),
        "records_by_constellation": dict(sorted(record_counts.items())),
        "first_message_epoch": _iso(min(message_times) if message_times else None),
        "last_message_epoch": _iso(max(message_times) if message_times else None),
    }
