"""Minimal fixed-local-NED inertial mechanization for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class InsState:
    """Nominal INS state at the IMU origin in a fixed local NED frame."""

    position_n_m: FloatArray
    velocity_n_mps: FloatArray
    rotation_n_from_b: FloatArray
    accelerometer_bias_b_mps2: FloatArray
    gyroscope_bias_b_radps: FloatArray


def _vector3(values: FloatArray, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite three-component vector")
    return array


def _rotation3(values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError("rotation_n_from_b must be a finite 3 by 3 matrix")
    if not np.allclose(array.T @ array, np.eye(3), atol=1e-10, rtol=0.0):
        raise ValueError("rotation_n_from_b must be orthonormal")
    if not np.isclose(np.linalg.det(array), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("rotation_n_from_b must be a proper rotation")
    return array


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = vector
    return np.array(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)),
        dtype=np.float64,
    )


def rotation_increment(rotation_vector_rad: FloatArray) -> FloatArray:
    """SO(3) exponential with a stable small-angle branch."""

    angle = float(np.linalg.norm(rotation_vector_rad))
    skew = _skew(rotation_vector_rad)
    skew_squared = skew @ skew
    if angle < 1e-8:
        return np.eye(3) + skew + 0.5 * skew_squared
    return (
        np.eye(3)
        + (np.sin(angle) / angle) * skew
        + ((1.0 - np.cos(angle)) / (angle * angle)) * skew_squared
    )


def propagate_ins(
    state: InsState,
    linear_acceleration_b_mps2: FloatArray,
    angular_velocity_b_radps: FloatArray,
    gravity_n_mps2: FloatArray,
    dt_s: float,
    earth_rotation_n_radps: FloatArray | None = None,
) -> InsState:
    """Propagate one causal IMU interval using midpoint attitude for specific force."""

    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    position = _vector3(state.position_n_m, "position_n_m")
    velocity = _vector3(state.velocity_n_mps, "velocity_n_mps")
    rotation = _rotation3(state.rotation_n_from_b)
    accelerometer_bias = _vector3(
        state.accelerometer_bias_b_mps2,
        "accelerometer_bias_b_mps2",
    )
    gyroscope_bias = _vector3(
        state.gyroscope_bias_b_radps,
        "gyroscope_bias_b_radps",
    )
    specific_force_b = (
        _vector3(linear_acceleration_b_mps2, "linear_acceleration_b_mps2")
        - accelerometer_bias
    )
    measured_angular_rate_b = (
        _vector3(angular_velocity_b_radps, "angular_velocity_b_radps")
        - gyroscope_bias
    )
    gravity = _vector3(gravity_n_mps2, "gravity_n_mps2")
    earth_rate_n = (
        np.zeros(3)
        if earth_rotation_n_radps is None
        else _vector3(earth_rotation_n_radps, "earth_rotation_n_radps")
    )
    angular_rate_b = measured_angular_rate_b - rotation.T @ earth_rate_n

    midpoint_rotation = rotation @ rotation_increment(0.5 * angular_rate_b * dt_s)
    coriolis_acceleration_n = -2.0 * np.cross(earth_rate_n, velocity)
    acceleration_n = (
        midpoint_rotation @ specific_force_b
        + gravity
        + coriolis_acceleration_n
    )
    next_position = position + velocity * dt_s + 0.5 * acceleration_n * dt_s * dt_s
    next_velocity = velocity + acceleration_n * dt_s
    next_rotation = rotation @ rotation_increment(angular_rate_b * dt_s)

    return InsState(
        position_n_m=next_position,
        velocity_n_mps=next_velocity,
        rotation_n_from_b=next_rotation,
        accelerometer_bias_b_mps2=accelerometer_bias.copy(),
        gyroscope_bias_b_radps=gyroscope_bias.copy(),
    )
