"""Causal deployable-only initialization for the Phase 3 INS state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .phase3_geodesy import LocalNedFrame, ecef_position_to_ned, make_local_ned_frame
from .phase3_ins import InsState


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class InitializationResult:
    """First causal state, valid only at the end of its initialization window."""

    timestamp_ns_utc: int
    local_frame: LocalNedFrame
    state: InsState
    fitted_pvt_velocity_n_mps: FloatArray


def _timestamps(values: Sequence[int], name: str) -> NDArray[np.int64]:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1 or array.size < 2 or np.any(np.diff(array) <= 0):
        raise ValueError(f"{name} must contain at least two strictly increasing times")
    return array


def _samples3(values: Sequence[Sequence[float]], count: int, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite three-component samples")
    return array


def _initial_rotation_n_from_b(mean_specific_force_b: FloatArray, yaw_rad: float) -> FloatArray:
    up_b = mean_specific_force_b / np.linalg.norm(mean_specific_force_b)
    nominal_forward_b = np.array((0.0, 1.0, 0.0))
    forward_b = nominal_forward_b - up_b * float(nominal_forward_b @ up_b)
    forward_norm = float(np.linalg.norm(forward_b))
    if forward_norm < 1e-6:
        raise ValueError("mean specific force is parallel to the body forward axis")
    forward_b /= forward_norm
    right_b = np.cross(forward_b, up_b)

    forward_n = np.array((np.cos(yaw_rad), np.sin(yaw_rad), 0.0))
    up_n = np.array((0.0, 0.0, -1.0))
    right_n = np.cross(forward_n, up_n)
    body_triad = np.column_stack((right_b, forward_b, up_b))
    navigation_triad = np.column_stack((right_n, forward_n, up_n))
    return navigation_triad @ body_triad.T


def initialize_from_causal_window(
    pvt_timestamps_ns_utc: Sequence[int],
    pvt_positions_ecef_m: Sequence[Sequence[float]],
    imu_timestamps_ns_utc: Sequence[int],
    imu_linear_acceleration_b_mps2: Sequence[Sequence[float]],
    antenna_lever_imu_m: Sequence[float],
    minimum_window_s: float,
    minimum_horizontal_speed_mps: float,
) -> InitializationResult:
    """Initialize at the last PVT time without back-dating the fitted state."""

    pvt_times = _timestamps(pvt_timestamps_ns_utc, "PVT timestamps")
    pvt_positions = _samples3(
        pvt_positions_ecef_m,
        pvt_times.size,
        "PVT positions",
    )
    imu_times = _timestamps(imu_timestamps_ns_utc, "IMU timestamps")
    imu_acceleration = _samples3(
        imu_linear_acceleration_b_mps2,
        imu_times.size,
        "IMU linear acceleration",
    )
    if not np.isfinite(minimum_window_s) or minimum_window_s <= 0.0:
        raise ValueError("minimum_window_s must be finite and positive")
    if not np.isfinite(minimum_horizontal_speed_mps) or minimum_horizontal_speed_mps <= 0.0:
        raise ValueError("minimum_horizontal_speed_mps must be finite and positive")

    start_ns = int(pvt_times[0])
    end_ns = int(pvt_times[-1])
    duration_s = (end_ns - start_ns) * 1e-9
    if duration_s < minimum_window_s:
        raise ValueError("PVT initialization window is too short")
    if int(imu_times[0]) < start_ns or int(imu_times[-1]) > end_ns:
        raise ValueError("IMU initialization samples must stay inside the PVT window")

    local_frame = make_local_ned_frame(pvt_positions[0])
    positions_ned = np.vstack(
        [ecef_position_to_ned(local_frame, position) for position in pvt_positions]
    )
    elapsed_s = (pvt_times - pvt_times[0]).astype(np.float64) * 1e-9
    design = np.column_stack((np.ones(pvt_times.size), elapsed_s))
    coefficients, _, _, _ = np.linalg.lstsq(design, positions_ned, rcond=None)
    fitted_pvt_velocity_n = coefficients[1]
    horizontal_speed = float(np.linalg.norm(fitted_pvt_velocity_n[:2]))
    if horizontal_speed < minimum_horizontal_speed_mps:
        raise ValueError("PVT motion is insufficient for causal heading initialization")
    yaw_rad = float(
        np.arctan2(fitted_pvt_velocity_n[1], fitted_pvt_velocity_n[0])
    )

    mean_specific_force_b = np.mean(imu_acceleration, axis=0)
    if float(np.linalg.norm(mean_specific_force_b)) < 1e-6:
        raise ValueError("mean IMU specific force cannot establish vertical direction")
    rotation_n_from_b = _initial_rotation_n_from_b(mean_specific_force_b, yaw_rad)
    lever_imu = np.asarray(antenna_lever_imu_m, dtype=np.float64)
    if lever_imu.shape != (3,) or not np.all(np.isfinite(lever_imu)):
        raise ValueError("antenna lever must be a finite three-component vector")
    imu_position_n = positions_ned[-1] - rotation_n_from_b @ lever_imu

    return InitializationResult(
        timestamp_ns_utc=end_ns,
        local_frame=local_frame,
        state=InsState(
            position_n_m=imu_position_n,
            velocity_n_mps=np.array(
                (
                    fitted_pvt_velocity_n[0],
                    fitted_pvt_velocity_n[1],
                    0.0,
                ),
                dtype=np.float64,
            ),
            rotation_n_from_b=rotation_n_from_b,
            accelerometer_bias_b_mps2=np.zeros(3),
            gyroscope_bias_b_radps=np.zeros(3),
        ),
        fitted_pvt_velocity_n_mps=fitted_pvt_velocity_n.copy(),
    )
