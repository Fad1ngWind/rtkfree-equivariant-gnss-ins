"""Deterministic GPS L1 C/A broadcast-ephemeris weighted least-squares SPP."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, TextIO

from .rinex_diagnostics import _read_header, _satellite_id, _timestamp


CLIGHT = 299_792_458.0
GPS_MU = 3.986005e14
GPS_OMEGA_E = 7.2921151467e-5
GPS_RELATIVITY_F = -4.442807633e-10
GPS_WEEK_SECONDS = 604_800.0
GPS_HALF_WEEK_SECONDS = GPS_WEEK_SECONDS / 2.0
GPS_EPOCH = datetime(1980, 1, 6)
UNIX_EPOCH = datetime(1970, 1, 1)
WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563


class SppError(RuntimeError):
    """Raised when the frozen conventional SPP contract cannot be executed."""


@dataclass(frozen=True)
class GpsObservation:
    satellite: str
    pseudorange_m: float
    signal_strength_dbhz: float | None
    doppler_hz: float | None


@dataclass(frozen=True)
class GpsObservationEpoch:
    time_gps: datetime
    flag: int
    observations: tuple[GpsObservation, ...]


@dataclass(frozen=True)
class GpsEphemeris:
    satellite: str
    toc_gps_s: float
    toe_gps_s: float
    toe_sow: float
    transmission_gps_s: float
    af0_s: float
    af1_sps: float
    af2_sps2: float
    crs_m: float
    delta_n_radps: float
    m0_rad: float
    cuc_rad: float
    eccentricity: float
    cus_rad: float
    sqrt_a_sqrt_m: float
    cic_rad: float
    omega0_rad: float
    cis_rad: float
    i0_rad: float
    crc_m: float
    omega_rad: float
    omega_dot_radps: float
    idot_radps: float
    ura_m: float
    health: int
    tgd_s: float


@dataclass(frozen=True)
class GpsNavigation:
    leap_seconds: int
    ionosphere_alpha: tuple[float, float, float, float]
    ionosphere_beta: tuple[float, float, float, float]
    ephemerides: dict[str, tuple[GpsEphemeris, ...]]


@dataclass(frozen=True)
class SatelliteState:
    position_ecef_m: tuple[float, float, float]
    clock_bias_s: float


def _rinex_float(raw: str) -> float:
    text = raw.strip().replace("D", "E").replace("d", "E")
    if not text:
        return 0.0
    try:
        value = float(text)
    except ValueError as exc:
        raise SppError("invalid RINEX numeric field") from exc
    if not math.isfinite(value):
        raise SppError("non-finite RINEX numeric field")
    return value


def _continuation_values(line: str) -> list[float]:
    padded = line.rstrip("\r\n").ljust(80)
    return [_rinex_float(padded[start : start + 19]) for start in (4, 23, 42, 61)]


def _gps_seconds(value: datetime) -> float:
    return (value - GPS_EPOCH).total_seconds()


def _wrap_week(seconds: float) -> float:
    while seconds > GPS_HALF_WEEK_SECONDS:
        seconds -= GPS_WEEK_SECONDS
    while seconds < -GPS_HALF_WEEK_SECONDS:
        seconds += GPS_WEEK_SECONDS
    return seconds


def _align_week_seconds(seconds: float, reference: float) -> float:
    while seconds - reference > GPS_HALF_WEEK_SECONDS:
        seconds -= GPS_WEEK_SECONDS
    while seconds - reference < -GPS_HALF_WEEK_SECONDS:
        seconds += GPS_WEEK_SECONDS
    return seconds


def read_gps_navigation(path: Path) -> GpsNavigation:
    """Read the GPS LNAV subset needed by the frozen SPP profile."""

    if path.is_symlink():
        raise SppError("navigation path cannot be a symbolic link")
    try:
        handle: TextIO = gzip.open(path, "rt", encoding="ascii", errors="strict")
    except OSError as exc:
        raise SppError("cannot open broadcast navigation RINEX") from exc

    header_lines: list[str] = []
    body_lines: list[str] = []
    try:
        with handle:
            while True:
                line = handle.readline()
                if not line:
                    raise SppError("navigation RINEX has no END OF HEADER")
                header_lines.append(line)
                if line[60:80].strip() == "END OF HEADER":
                    break
            body_lines = handle.readlines()
    except (OSError, EOFError, UnicodeError) as exc:
        raise SppError("cannot read broadcast navigation RINEX") from exc

    leap_seconds: int | None = None
    ionosphere: dict[str, tuple[float, float, float, float]] = {}
    for line in header_lines:
        label = line[60:80].strip() if len(line) >= 60 else ""
        if label == "LEAP SECONDS":
            try:
                leap_seconds = int(line[:6].strip())
            except ValueError as exc:
                raise SppError("invalid navigation leap-second field") from exc
        elif label == "IONOSPHERIC CORR":
            parts = line[:60].replace("D", "E").split()
            if len(parts) >= 5 and parts[0] in {"GPSA", "GPSB"}:
                ionosphere[parts[0]] = tuple(float(item) for item in parts[1:5])

    if leap_seconds is None:
        raise SppError("navigation RINEX lacks LEAP SECONDS")
    if "GPSA" not in ionosphere or "GPSB" not in ionosphere:
        raise SppError("navigation RINEX lacks GPS Klobuchar coefficients")

    ephemerides: dict[str, list[GpsEphemeris]] = {}
    index = 0
    while index < len(body_lines):
        line = body_lines[index]
        satellite = _satellite_id(line[:3])
        if satellite is None:
            index += 1
            continue
        record_length = 4 if satellite[0] in {"R", "S"} else 8
        if index + record_length > len(body_lines):
            raise SppError("truncated navigation record")
        record = body_lines[index : index + record_length]
        index += record_length
        if satellite[0] != "G":
            continue
        if len(record) != 8:
            raise SppError("GPS navigation record is not eight lines")
        first = record[0].rstrip("\r\n").ljust(80)
        toc = _timestamp(first[3:23].split())
        clock = [_rinex_float(first[start : start + 19]) for start in (23, 42, 61)]
        values = [_continuation_values(item) for item in record[1:]]
        week = int(round(values[4][2]))
        toe_sow = values[2][0]
        toe_gps_s = _align_week_seconds(
            week * GPS_WEEK_SECONDS + toe_sow,
            _gps_seconds(toc),
        )
        transmission_gps_s = _align_week_seconds(
            week * GPS_WEEK_SECONDS + values[6][0],
            toe_gps_s,
        )
        ephemeris = GpsEphemeris(
            satellite=satellite,
            toc_gps_s=_gps_seconds(toc),
            toe_gps_s=toe_gps_s,
            toe_sow=toe_sow,
            transmission_gps_s=transmission_gps_s,
            af0_s=clock[0],
            af1_sps=clock[1],
            af2_sps2=clock[2],
            crs_m=values[0][1],
            delta_n_radps=values[0][2],
            m0_rad=values[0][3],
            cuc_rad=values[1][0],
            eccentricity=values[1][1],
            cus_rad=values[1][2],
            sqrt_a_sqrt_m=values[1][3],
            cic_rad=values[2][1],
            omega0_rad=values[2][2],
            cis_rad=values[2][3],
            i0_rad=values[3][0],
            crc_m=values[3][1],
            omega_rad=values[3][2],
            omega_dot_radps=values[3][3],
            idot_radps=values[4][0],
            ura_m=max(values[5][0], 0.0),
            health=int(round(values[5][1])),
            tgd_s=values[5][2],
        )
        ephemerides.setdefault(satellite, []).append(ephemeris)

    if not ephemerides:
        raise SppError("navigation RINEX contains no GPS ephemeris")
    return GpsNavigation(
        leap_seconds=leap_seconds,
        ionosphere_alpha=ionosphere["GPSA"],
        ionosphere_beta=ionosphere["GPSB"],
        ephemerides={
            key: tuple(sorted(items, key=lambda item: item.transmission_gps_s))
            for key, items in sorted(ephemerides.items())
        },
    )


def _observation_value(payload: str, field_index: int) -> float | None:
    raw = payload[16 * field_index : 16 * field_index + 14].strip()
    if not raw:
        return None
    try:
        value = float(raw.replace("D", "E"))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def read_gps_c1c_observations(
    path: Path,
) -> tuple[dict[str, object], tuple[GpsObservationEpoch, ...]]:
    """Read only GPS C1C/S1C/D1C from RINEX 3 observation data."""

    if path.is_symlink():
        raise SppError("observation path cannot be a symbolic link")
    try:
        handle = path.open("r", encoding="ascii", errors="strict")
    except OSError as exc:
        raise SppError("cannot open observation RINEX") from exc
    epochs: list[GpsObservationEpoch] = []
    with handle:
        header, _ = _read_header(handle)
        if header.get("file_type") != "O" or header.get("time_system") != "GPS":
            raise SppError("SPP profile requires GPS-time observation RINEX")
        observation_types = header.get("observation_types", {})
        if not isinstance(observation_types, dict):
            raise SppError("invalid observation-type header")
        gps_types = observation_types.get("G", [])
        if "C1C" not in gps_types:
            raise SppError("observation RINEX lacks GPS C1C")
        indices = {
            name: gps_types.index(name) if name in gps_types else None
            for name in ("C1C", "S1C", "D1C")
        }
        current_time: datetime | None = None
        current_flag = 0
        block: list[str] = []

        def finish_epoch() -> None:
            if current_time is None:
                return
            observations: list[GpsObservation] = []
            index = 0
            while index < len(block):
                line = block[index].rstrip("\r\n")
                satellite = _satellite_id(line[:3])
                if satellite is None or satellite[0] != "G":
                    index += 1
                    continue
                payload = line[3:]
                while len(payload) < 16 * len(gps_types) and index + 1 < len(block):
                    continuation = block[index + 1].rstrip("\r\n")
                    if continuation[:3] != "   ":
                        break
                    index += 1
                    payload += continuation[3:]
                pseudorange = _observation_value(payload, int(indices["C1C"]))
                if pseudorange is not None and pseudorange > 0.0:
                    strength_index = indices["S1C"]
                    doppler_index = indices["D1C"]
                    observations.append(
                        GpsObservation(
                            satellite=satellite,
                            pseudorange_m=pseudorange,
                            signal_strength_dbhz=(
                                _observation_value(payload, strength_index)
                                if isinstance(strength_index, int)
                                else None
                            ),
                            doppler_hz=(
                                _observation_value(payload, doppler_index)
                                if isinstance(doppler_index, int)
                                else None
                            ),
                        )
                    )
                index += 1
            epochs.append(
                GpsObservationEpoch(
                    time_gps=current_time,
                    flag=current_flag,
                    observations=tuple(observations),
                )
            )

        for line in handle:
            if not line.startswith(">"):
                block.append(line)
                continue
            finish_epoch()
            block = []
            parts = line[1:].split()
            if len(parts) < 8:
                raise SppError("invalid observation epoch line")
            current_time = _timestamp(parts[:6])
            current_flag = int(parts[6])
        finish_epoch()
    return header, tuple(epochs)


def select_ephemeris(
    navigation: GpsNavigation,
    satellite: str,
    receive_gps_s: float,
    maximum_age_s: float,
) -> GpsEphemeris | None:
    candidates = [
        item
        for item in navigation.ephemerides.get(satellite, ())
        if item.health == 0
        and item.transmission_gps_s <= receive_gps_s + 1e-6
        and abs(_wrap_week(receive_gps_s - item.toe_gps_s)) <= maximum_age_s
    ]
    return max(candidates, key=lambda item: item.transmission_gps_s, default=None)


def satellite_position_clock(
    ephemeris: GpsEphemeris,
    transmit_gps_s: float,
) -> SatelliteState:
    semi_major_axis = ephemeris.sqrt_a_sqrt_m**2
    if semi_major_axis <= 0.0 or not 0.0 <= ephemeris.eccentricity < 1.0:
        raise SppError("invalid GPS broadcast orbit")
    tk = _wrap_week(transmit_gps_s - ephemeris.toe_gps_s)
    mean_motion = math.sqrt(GPS_MU / semi_major_axis**3) + ephemeris.delta_n_radps
    mean_anomaly = ephemeris.m0_rad + mean_motion * tk
    eccentric_anomaly = mean_anomaly
    for _ in range(30):
        updated = mean_anomaly + ephemeris.eccentricity * math.sin(eccentric_anomaly)
        if abs(updated - eccentric_anomaly) < 1e-14:
            eccentric_anomaly = updated
            break
        eccentric_anomaly = updated
    else:
        raise SppError("GPS Kepler equation did not converge")

    sin_e = math.sin(eccentric_anomaly)
    cos_e = math.cos(eccentric_anomaly)
    true_anomaly = math.atan2(
        math.sqrt(1.0 - ephemeris.eccentricity**2) * sin_e,
        cos_e - ephemeris.eccentricity,
    )
    argument = true_anomaly + ephemeris.omega_rad
    sin_2u = math.sin(2.0 * argument)
    cos_2u = math.cos(2.0 * argument)
    corrected_argument = (
        argument + ephemeris.cus_rad * sin_2u + ephemeris.cuc_rad * cos_2u
    )
    radius = (
        semi_major_axis * (1.0 - ephemeris.eccentricity * cos_e)
        + ephemeris.crs_m * sin_2u
        + ephemeris.crc_m * cos_2u
    )
    inclination = (
        ephemeris.i0_rad
        + ephemeris.idot_radps * tk
        + ephemeris.cis_rad * sin_2u
        + ephemeris.cic_rad * cos_2u
    )
    x_orbit = radius * math.cos(corrected_argument)
    y_orbit = radius * math.sin(corrected_argument)
    ascending_node = (
        ephemeris.omega0_rad
        + (ephemeris.omega_dot_radps - GPS_OMEGA_E) * tk
        - GPS_OMEGA_E * ephemeris.toe_sow
    )
    cos_node = math.cos(ascending_node)
    sin_node = math.sin(ascending_node)
    cos_i = math.cos(inclination)
    sin_i = math.sin(inclination)
    position = (
        x_orbit * cos_node - y_orbit * cos_i * sin_node,
        x_orbit * sin_node + y_orbit * cos_i * cos_node,
        y_orbit * sin_i,
    )
    clock_dt = _wrap_week(transmit_gps_s - ephemeris.toc_gps_s)
    relativity = (
        GPS_RELATIVITY_F
        * ephemeris.eccentricity
        * ephemeris.sqrt_a_sqrt_m
        * sin_e
    )
    clock = (
        ephemeris.af0_s
        + ephemeris.af1_sps * clock_dt
        + ephemeris.af2_sps2 * clock_dt**2
        + relativity
        - ephemeris.tgd_s
    )
    return SatelliteState(position_ecef_m=position, clock_bias_s=clock)


def satellite_state_at_transmission(
    ephemeris: GpsEphemeris,
    receive_gps_s: float,
    pseudorange_m: float,
) -> SatelliteState:
    transmit = receive_gps_s - pseudorange_m / CLIGHT
    state = satellite_position_clock(ephemeris, transmit)
    for _ in range(2):
        transmit = receive_gps_s - pseudorange_m / CLIGHT - state.clock_bias_s
        state = satellite_position_clock(ephemeris, transmit)
    return state


def ecef_to_geodetic(position: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = position
    eccentricity_sq = WGS84_F * (2.0 - WGS84_F)
    longitude = math.atan2(y, x)
    radius_xy = math.hypot(x, y)
    latitude = math.atan2(z, radius_xy * (1.0 - eccentricity_sq))
    height = 0.0
    for _ in range(20):
        sin_latitude = math.sin(latitude)
        normal = WGS84_A / math.sqrt(1.0 - eccentricity_sq * sin_latitude**2)
        cos_latitude = math.cos(latitude)
        if abs(cos_latitude) < 1e-12:
            height = abs(z) - normal * (1.0 - eccentricity_sq)
        else:
            height = radius_xy / cos_latitude - normal
        updated = math.atan2(
            z,
            radius_xy * (1.0 - eccentricity_sq * normal / (normal + height)),
        )
        if abs(updated - latitude) < 1e-13:
            latitude = updated
            break
        latitude = updated
    return latitude, longitude, height


def azimuth_elevation(
    receiver_ecef_m: Iterable[float],
    satellite_ecef_m: Iterable[float],
) -> tuple[float, float, tuple[float, float, float], float]:
    receiver = tuple(receiver_ecef_m)
    satellite = tuple(satellite_ecef_m)
    delta = tuple(satellite[index] - receiver[index] for index in range(3))
    distance = math.sqrt(sum(value * value for value in delta))
    if distance <= 0.0:
        raise SppError("nonpositive satellite range")
    line_of_sight = tuple(value / distance for value in delta)
    latitude, longitude, _ = ecef_to_geodetic(receiver)
    sin_lat = math.sin(latitude)
    cos_lat = math.cos(latitude)
    sin_lon = math.sin(longitude)
    cos_lon = math.cos(longitude)
    east = -sin_lon * line_of_sight[0] + cos_lon * line_of_sight[1]
    north = (
        -sin_lat * cos_lon * line_of_sight[0]
        - sin_lat * sin_lon * line_of_sight[1]
        + cos_lat * line_of_sight[2]
    )
    up = (
        cos_lat * cos_lon * line_of_sight[0]
        + cos_lat * sin_lon * line_of_sight[1]
        + sin_lat * line_of_sight[2]
    )
    azimuth = math.atan2(east, north) % (2.0 * math.pi)
    elevation = math.asin(max(-1.0, min(1.0, up)))
    return azimuth, elevation, line_of_sight, distance


def klobuchar_delay_m(
    gps_tow_s: float,
    latitude_rad: float,
    longitude_rad: float,
    azimuth_rad: float,
    elevation_rad: float,
    alpha: tuple[float, float, float, float],
    beta: tuple[float, float, float, float],
) -> float:
    elevation_sc = elevation_rad / math.pi
    latitude_sc = latitude_rad / math.pi
    longitude_sc = longitude_rad / math.pi
    psi = 0.0137 / (elevation_sc + 0.11) - 0.022
    subion_lat = latitude_sc + psi * math.cos(azimuth_rad)
    subion_lat = max(-0.416, min(0.416, subion_lat))
    subion_lon = longitude_sc + psi * math.sin(azimuth_rad) / math.cos(
        subion_lat * math.pi
    )
    geomagnetic_lat = subion_lat + 0.064 * math.cos(
        (subion_lon - 1.617) * math.pi
    )
    local_time = (43_200.0 * subion_lon + gps_tow_s) % 86_400.0
    amplitude = sum(alpha[index] * geomagnetic_lat**index for index in range(4))
    amplitude = max(amplitude, 0.0)
    period = sum(beta[index] * geomagnetic_lat**index for index in range(4))
    period = max(period, 72_000.0)
    phase = 2.0 * math.pi * (local_time - 50_400.0) / period
    mapping = 1.0 + 16.0 * (0.53 - elevation_sc) ** 3
    if abs(phase) < 1.57:
        delay_s = 5e-9 + amplitude * (1.0 - phase**2 / 2.0 + phase**4 / 24.0)
    else:
        delay_s = 5e-9
    return CLIGHT * mapping * delay_s


def saastamoinen_delay_m(
    latitude_rad: float,
    height_m: float,
    elevation_rad: float,
    relative_humidity: float,
) -> float:
    if height_m < -100.0 or height_m > 10_000.0 or elevation_rad <= 0.0:
        return 0.0
    pressure_hpa = 1013.25 * (1.0 - 2.2557e-5 * height_m) ** 5.2568
    temperature_k = 15.0 - 6.5e-3 * height_m + 273.16
    water_vapor_hpa = 6.108 * relative_humidity * math.exp(
        (17.15 * temperature_k - 4684.0) / (temperature_k - 38.45)
    )
    cosine_zenith = math.sin(elevation_rad)
    hydrostatic = (
        0.0022768
        * pressure_hpa
        / (1.0 - 0.00266 * math.cos(2.0 * latitude_rad) - 0.00028 * height_m / 1000.0)
        / cosine_zenith
    )
    wet = (
        0.002277
        * (1255.0 / temperature_k + 0.05)
        * water_vapor_hpa
        / cosine_zenith
    )
    return hydrostatic + wet


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-16:
            raise SppError("rank-deficient normal matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                augmented[row][item] - factor * augmented[column][item]
                for item in range(size + 1)
            ]
    result = [augmented[row][-1] for row in range(size)]
    if not all(math.isfinite(value) for value in result):
        raise SppError("non-finite linear solution")
    return result


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    columns = []
    for column in range(size):
        basis = [1.0 if row == column else 0.0 for row in range(size)]
        columns.append(_solve_linear(matrix, basis))
    return [[columns[column][row] for column in range(size)] for row in range(size)]


def _normal_matrix(
    rows: list[list[float]],
    residuals: list[float],
    variances: list[float],
) -> tuple[list[list[float]], list[float]]:
    size = len(rows[0])
    normal = [[0.0 for _ in range(size)] for _ in range(size)]
    right = [0.0 for _ in range(size)]
    for row, residual, variance in zip(rows, residuals, variances):
        weight = 1.0 / variance
        for first in range(size):
            right[first] += weight * row[first] * residual
            for second in range(size):
                normal[first][second] += weight * row[first] * row[second]
    return normal, right


def solve_static_weighted_position(
    satellite_positions_ecef_m: list[tuple[float, float, float]],
    corrected_pseudoranges_m: list[float],
    variances_m2: list[float],
    initial_position_ecef_m: tuple[float, float, float],
    maximum_iterations: int = 10,
) -> tuple[tuple[float, float, float, float], list[list[float]]]:
    """Solve a correction-free synthetic WLS fixture using the production algebra."""

    if not (
        len(satellite_positions_ecef_m)
        == len(corrected_pseudoranges_m)
        == len(variances_m2)
    ) or len(corrected_pseudoranges_m) < 4:
        raise SppError("static WLS requires at least four matched measurements")
    state = [*initial_position_ecef_m, 0.0]
    normal: list[list[float]] | None = None
    for _ in range(maximum_iterations):
        rows: list[list[float]] = []
        residuals: list[float] = []
        for satellite, pseudorange in zip(
            satellite_positions_ecef_m, corrected_pseudoranges_m
        ):
            delta = [satellite[index] - state[index] for index in range(3)]
            distance = math.sqrt(sum(value * value for value in delta))
            rows.append([-value / distance for value in delta] + [1.0])
            residuals.append(pseudorange - (distance + state[3]))
        normal, right = _normal_matrix(rows, residuals, variances_m2)
        update = _solve_linear(normal, right)
        state = [state[index] + update[index] for index in range(4)]
        if math.sqrt(sum(value * value for value in update[:3])) < 1e-4 and abs(
            update[3]
        ) < 1e-3:
            return tuple(state), _inverse(normal)
    raise SppError("static WLS did not converge")


def _build_epoch_rows(
    epoch: GpsObservationEpoch,
    navigation: GpsNavigation,
    profile: dict[str, object],
    state: list[float],
) -> tuple[list[list[float]], list[float], list[float], list[dict[str, object]]]:
    receive_gps_s = _gps_seconds(epoch.time_gps)
    latitude, longitude, height = ecef_to_geodetic(state[:3])
    mask = math.radians(float(profile["elevation_mask_deg"]))
    sigma_zenith = float(profile["measurement_sigma_zenith_m"])
    rows: list[list[float]] = []
    residuals: list[float] = []
    variances: list[float] = []
    details: list[dict[str, object]] = []
    for observation in epoch.observations:
        ephemeris = select_ephemeris(
            navigation,
            observation.satellite,
            receive_gps_s,
            float(profile["maximum_ephemeris_age_s"]),
        )
        if ephemeris is None:
            continue
        satellite_state = satellite_state_at_transmission(
            ephemeris, receive_gps_s, observation.pseudorange_m
        )
        azimuth, elevation, line_of_sight, geometric_distance = azimuth_elevation(
            state[:3], satellite_state.position_ecef_m
        )
        if elevation < mask:
            continue
        sagnac = GPS_OMEGA_E / CLIGHT * (
            satellite_state.position_ecef_m[0] * state[1]
            - satellite_state.position_ecef_m[1] * state[0]
        )
        ionosphere = klobuchar_delay_m(
            receive_gps_s % GPS_WEEK_SECONDS,
            latitude,
            longitude,
            azimuth,
            elevation,
            navigation.ionosphere_alpha,
            navigation.ionosphere_beta,
        )
        troposphere = saastamoinen_delay_m(
            latitude,
            height,
            elevation,
            float(profile["relative_humidity"]),
        )
        predicted = (
            geometric_distance
            + sagnac
            + state[3]
            - CLIGHT * satellite_state.clock_bias_s
            + ionosphere
            + troposphere
        )
        sine_elevation = max(math.sin(elevation), math.sin(mask))
        code_variance = (sigma_zenith / sine_elevation) ** 2
        ionosphere_variance = (0.5 * ionosphere) ** 2
        troposphere_variance = (0.3 / (sine_elevation + 0.1)) ** 2
        variance = (
            code_variance
            + ephemeris.ura_m**2
            + ionosphere_variance
            + troposphere_variance
        )
        rows.append([-value for value in line_of_sight] + [1.0])
        residuals.append(observation.pseudorange_m - predicted)
        variances.append(variance)
        details.append(
            {
                "satellite": observation.satellite,
                "elevation_deg": math.degrees(elevation),
                "signal_strength_dbhz": observation.signal_strength_dbhz,
                "residual_m": observation.pseudorange_m - predicted,
            }
        )
    return rows, residuals, variances, details


def _dop_values(
    rows: list[list[float]],
    receiver_ecef_m: tuple[float, float, float],
) -> dict[str, float] | None:
    if len(rows) < 4:
        return None
    geometry, _ = _normal_matrix(rows, [0.0] * len(rows), [1.0] * len(rows))
    try:
        inverse = _inverse(geometry)
    except SppError:
        return None
    latitude, longitude, _ = ecef_to_geodetic(receiver_ecef_m)
    east = (-math.sin(longitude), math.cos(longitude), 0.0)
    north = (
        -math.sin(latitude) * math.cos(longitude),
        -math.sin(latitude) * math.sin(longitude),
        math.cos(latitude),
    )
    up = (
        math.cos(latitude) * math.cos(longitude),
        math.cos(latitude) * math.sin(longitude),
        math.sin(latitude),
    )
    position_covariance = [row[:3] for row in inverse[:3]]

    def projected(axis: tuple[float, float, float]) -> float:
        return sum(
            axis[first] * position_covariance[first][second] * axis[second]
            for first in range(3)
            for second in range(3)
        )

    q_east = projected(east)
    q_north = projected(north)
    q_up = projected(up)
    values = {
        "gdop": math.sqrt(max(sum(inverse[index][index] for index in range(4)), 0.0)),
        "pdop": math.sqrt(max(sum(inverse[index][index] for index in range(3)), 0.0)),
        "hdop": math.sqrt(max(q_east + q_north, 0.0)),
        "vdop": math.sqrt(max(q_up, 0.0)),
    }
    return values if all(math.isfinite(value) for value in values.values()) else None


def _invalid_solution(epoch: GpsObservationEpoch, status: str) -> dict[str, object]:
    return {
        "native_gps_epoch": epoch.time_gps.isoformat(timespec="microseconds"),
        "solution_status": status,
        "solution_valid": False,
        "observed_c1c_satellite_count": len(epoch.observations),
        "used_satellite_count": 0,
        "used_satellites": [],
        "ecef_position_m": None,
        "ecef_velocity_mps": None,
        "velocity_valid": False,
        "receiver_clock_bias_m": None,
        "receiver_clock_drift_mps": None,
        "position_covariance_ecef_m2": None,
        "covariance_valid": False,
        "dop": None,
        "postfit_residual_rms_m": None,
    }


def solve_epoch(
    epoch: GpsObservationEpoch,
    navigation: GpsNavigation,
    profile: dict[str, object],
    initial_position_ecef_m: tuple[float, float, float],
) -> dict[str, object]:
    if epoch.flag not in {0, 1}:
        return _invalid_solution(epoch, "unsupported_epoch_flag")
    if len(epoch.observations) < int(profile["minimum_satellites"]):
        return _invalid_solution(epoch, "insufficient_c1c_observations")
    state = [*initial_position_ecef_m, 0.0]
    final_normal: list[list[float]] | None = None
    final_rows: list[list[float]] = []
    final_residuals: list[float] = []
    final_details: list[dict[str, object]] = []
    try:
        for _ in range(int(profile["maximum_iterations"])):
            rows, residuals, variances, _ = _build_epoch_rows(
                epoch, navigation, profile, state
            )
            if len(rows) < int(profile["minimum_satellites"]):
                return _invalid_solution(epoch, "insufficient_satellites_after_screening")
            normal, right = _normal_matrix(rows, residuals, variances)
            update = _solve_linear(normal, right)
            state = [state[index] + update[index] for index in range(4)]
            if not all(math.isfinite(value) for value in state):
                return _invalid_solution(epoch, "nonfinite_state")
            if math.sqrt(sum(value * value for value in update[:3])) < float(
                profile["position_convergence_m"]
            ) and abs(update[3]) < float(profile["clock_convergence_m"]):
                final_rows, final_residuals, final_variances, final_details = (
                    _build_epoch_rows(epoch, navigation, profile, state)
                )
                if len(final_rows) < int(profile["minimum_satellites"]):
                    return _invalid_solution(
                        epoch, "insufficient_satellites_after_screening"
                    )
                final_normal, _ = _normal_matrix(
                    final_rows, final_residuals, final_variances
                )
                break
        else:
            return _invalid_solution(epoch, "iteration_not_converged")
        assert final_normal is not None
        covariance = _inverse(final_normal)
    except SppError as exc:
        return _invalid_solution(epoch, str(exc).replace(" ", "_"))

    position_covariance = [row[:3] for row in covariance[:3]]
    residual_rms = math.sqrt(
        sum(value * value for value in final_residuals) / len(final_residuals)
    )
    result = _invalid_solution(epoch, "valid")
    result.update(
        {
            "solution_valid": True,
            "used_satellite_count": len(final_details),
            "used_satellites": [item["satellite"] for item in final_details],
            "ecef_position_m": state[:3],
            "receiver_clock_bias_m": state[3],
            "position_covariance_ecef_m2": [
                value for row in position_covariance for value in row
            ],
            "covariance_valid": True,
            "dop": _dop_values(final_rows, tuple(state[:3])),
            "postfit_residual_rms_m": residual_rms,
            "satellite_quality": final_details,
        }
    )
    return result


def _datetime_to_unix_ns(value: datetime) -> int:
    delta = value - UNIX_EPOCH
    return (
        delta.days * 86_400_000_000_000
        + delta.seconds * 1_000_000_000
        + delta.microseconds * 1_000
    )


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _write_atomic(path: Path, content: bytes) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.chmod(0o600)
    os.replace(temporary, path)
    return _sha256_file(path)


def run_medium_spp(
    source_lock_path: Path,
    profile_path: Path,
    data_root: Path,
) -> tuple[Path, Path]:
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("profile_id") != "gps_l1ca_broadcast_spp_v1":
        raise SppError("unsupported SPP profile")
    artifacts = {item["id"]: item for item in source_lock["artifacts"]}
    session_root = (
        data_root
        / "deployable"
        / source_lock["dataset"]
        / source_lock["repository_commit"]
        / source_lock["session"]
    )
    observation_name = next(
        name
        for name in artifacts["gnss_observation_archive"]["selected_member_basenames"]
        if name.endswith(".obs")
    )
    observation_path = session_root / "gnss" / observation_name
    navigation_path = session_root / "gnss" / artifacts["broadcast_navigation"]["local_name"]
    navigation = read_gps_navigation(navigation_path)
    header, epochs = read_gps_c1c_observations(observation_path)
    approximate = header.get("approx_position_xyz_m")
    if not isinstance(approximate, list) or len(approximate) != 3:
        raise SppError("observation RINEX lacks approximate ECEF initialization")
    initial = tuple(float(item) for item in approximate)
    imu_artifact = artifacts["low_cost_imu"]
    validation = imu_artifact.get("deployable_validation", {})
    imu_first = int(validation["sensor_stamp_first_ns"])
    imu_last = int(validation["sensor_stamp_last_ns"])

    records: list[dict[str, object]] = []
    valid_count = 0
    status_counts: dict[str, int] = {}
    for epoch in epochs:
        solution = solve_epoch(epoch, navigation, profile, initial)
        utc_time = epoch.time_gps - timedelta(seconds=navigation.leap_seconds)
        timestamp_ns_utc = _datetime_to_unix_ns(utc_time)
        gps_seconds = _gps_seconds(epoch.time_gps)
        solution.update(
            {
                "schema_version": 1,
                "profile_id": profile["profile_id"],
                "timestamp_ns_utc": timestamp_ns_utc,
                "gps_week": int(gps_seconds // GPS_WEEK_SECONDS),
                "gps_tow_s": gps_seconds % GPS_WEEK_SECONDS,
                "within_common_interval": imu_first <= timestamp_ns_utc <= imu_last,
                "coordinate_frame": profile["coordinate_frame"],
                "position_unit": "m",
                "velocity_unit": "m/s",
                "clock_bias_unit": "m",
                "correction_flags": profile["corrections"],
                "covariance_method": profile["covariance_method"],
                "used_constellations": ["G"] if solution["solution_valid"] else [],
                "used_signals": ["G:C1C"] if solution["solution_valid"] else [],
                "inter_system_bias_states": [],
            }
        )
        status = str(solution["solution_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if solution["solution_valid"]:
            valid_count += 1
            position = solution["ecef_position_m"]
            assert isinstance(position, list)
            initial = tuple(float(item) for item in position)
        records.append(solution)

    output_root = (
        data_root
        / "standardized"
        / source_lock["dataset"]
        / source_lock["repository_commit"]
        / source_lock["session"]
        / profile["profile_id"]
    )
    pvt_path = output_root / "pvt.jsonl"
    pvt_bytes = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for record in records
    )
    pvt_size, pvt_digest = _write_atomic(pvt_path, pvt_bytes)
    _, profile_digest = _sha256_file(profile_path)
    summary = {
        "schema_version": 1,
        "dataset": source_lock["dataset"],
        "session": source_lock["session"],
        "repository_commit": source_lock["repository_commit"],
        "profile_id": profile["profile_id"],
        "profile_sha256": profile_digest,
        "input_observation_sha256": next(
            item["sha256"]
            for item in json.loads((session_root / "provenance_manifest.json").read_text())["selected_outputs"]
            if item["path"] == f"gnss/{observation_name}"
        ),
        "input_navigation_sha256": artifacts["broadcast_navigation"]["observed_sha256"],
        "epoch_count": len(records),
        "valid_solution_count": valid_count,
        "invalid_solution_count": len(records) - valid_count,
        "status_counts": dict(sorted(status_counts.items())),
        "pvt_path": pvt_path.name,
        "pvt_size": pvt_size,
        "pvt_sha256": pvt_digest,
        "high_precision_reference_used": False,
        "receiver_native_pvt_used_as_input": False,
    }
    summary_path = output_root / "spp_summary.json"
    _write_atomic(
        summary_path,
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return pvt_path, summary_path
