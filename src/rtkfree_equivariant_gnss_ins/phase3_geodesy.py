"""Fixed local-NED conversions for Phase 3 standardized PVT."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .gps_spp import GPS_OMEGA_E, WGS84_F, ecef_to_geodetic


FloatArray = NDArray[np.float64]


def _vector3(values: Sequence[float], name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite three-component vector")
    return array


def _covariance3(values: Sequence[Sequence[float]]) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError("ECEF covariance must be a finite 3 by 3 matrix")
    if not np.allclose(array, array.T, atol=1e-10, rtol=0.0):
        raise ValueError("ECEF covariance must be symmetric")
    return array


@dataclass(frozen=True)
class LocalNedFrame:
    """Per-sequence fixed NED frame established from one causal ECEF PVT."""

    origin_ecef_m: FloatArray
    latitude_rad: float
    longitude_rad: float
    height_m: float
    rotation_n_from_e: FloatArray


def make_local_ned_frame(origin_ecef_m: Sequence[float]) -> LocalNedFrame:
    origin = _vector3(origin_ecef_m, "origin_ecef_m")
    latitude, longitude, height = ecef_to_geodetic(origin)
    sin_latitude = math.sin(latitude)
    cos_latitude = math.cos(latitude)
    sin_longitude = math.sin(longitude)
    cos_longitude = math.cos(longitude)
    rotation = np.array(
        (
            (
                -sin_latitude * cos_longitude,
                -sin_latitude * sin_longitude,
                cos_latitude,
            ),
            (-sin_longitude, cos_longitude, 0.0),
            (
                -cos_latitude * cos_longitude,
                -cos_latitude * sin_longitude,
                -sin_latitude,
            ),
        ),
        dtype=np.float64,
    )
    return LocalNedFrame(
        origin_ecef_m=origin.copy(),
        latitude_rad=latitude,
        longitude_rad=longitude,
        height_m=height,
        rotation_n_from_e=rotation,
    )


def ecef_position_to_ned(
    frame: LocalNedFrame,
    position_ecef_m: Sequence[float],
) -> FloatArray:
    delta_ecef = _vector3(position_ecef_m, "position_ecef_m") - frame.origin_ecef_m
    return frame.rotation_n_from_e @ delta_ecef


def ned_position_to_ecef(
    frame: LocalNedFrame,
    position_ned_m: Sequence[float],
) -> FloatArray:
    position_ned = _vector3(position_ned_m, "position_ned_m")
    return frame.origin_ecef_m + frame.rotation_n_from_e.T @ position_ned


def ecef_covariance_to_ned(
    frame: LocalNedFrame,
    covariance_ecef_m2: Sequence[Sequence[float]],
) -> FloatArray:
    covariance = _covariance3(covariance_ecef_m2)
    transformed = frame.rotation_n_from_e @ covariance @ frame.rotation_n_from_e.T
    return 0.5 * (transformed + transformed.T)


def earth_rotation_n_radps(frame: LocalNedFrame) -> FloatArray:
    """Earth rotation expressed in the fixed local NED frame."""

    return np.array(
        (
            GPS_OMEGA_E * math.cos(frame.latitude_rad),
            0.0,
            -GPS_OMEGA_E * math.sin(frame.latitude_rad),
        ),
        dtype=np.float64,
    )


def normal_gravity_n_mps2(frame: LocalNedFrame) -> FloatArray:
    """Somigliana normal gravity with a first-order free-air height correction."""

    eccentricity_squared = WGS84_F * (2.0 - WGS84_F)
    sin_latitude_squared = math.sin(frame.latitude_rad) ** 2
    gravity_surface = (
        9.7803253359
        * (1.0 + 0.00193185265241 * sin_latitude_squared)
        / math.sqrt(1.0 - eccentricity_squared * sin_latitude_squared)
    )
    gravity = gravity_surface - 3.086e-6 * frame.height_m
    if not math.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("normal gravity is invalid at the local-frame origin")
    return np.array((0.0, 0.0, gravity), dtype=np.float64)
