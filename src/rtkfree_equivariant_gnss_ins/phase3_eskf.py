"""Fixed 15-state error dynamics and covariance operations for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .phase3_ins import InsState, rotation_increment


FloatArray = NDArray[np.float64]
POSITION = slice(0, 3)
VELOCITY = slice(3, 6)
ATTITUDE = slice(6, 9)
ACCELEROMETER_BIAS = slice(9, 12)
GYROSCOPE_BIAS = slice(12, 15)
ERROR_STATE_SIZE = 15
DRIVING_NOISE_SIZE = 12
POSITION_NIS_DOF = 3
POSITION_NIS_CHI_SQUARE_95 = 7.814727903251179
POSITION_NIS_CHI_SQUARE_99 = 11.344866730144373


@dataclass(frozen=True)
class EskfNoiseDensities:
    """Continuous white-noise densities used by the fixed ESKF."""

    accelerometer_white_noise_mps2_sqrt_s: float
    gyroscope_white_noise_radps_sqrt_s: float
    accelerometer_bias_random_walk_mps2_per_sqrt_s: float
    gyroscope_bias_random_walk_radps_per_sqrt_s: float


@dataclass(frozen=True)
class PositionUpdateResult:
    """Posterior state, covariance, and diagnostics for one PVT update."""

    state: InsState
    covariance: FloatArray
    innovation_n_m: FloatArray
    innovation_covariance_n_m2: FloatArray
    kalman_gain: FloatArray
    nis: float


def _vector3(values: Sequence[float], name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite three-component vector")
    return array


def _rotation3(values: Sequence[Sequence[float]]) -> FloatArray:
    rotation = np.asarray(values, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_n_from_b must be a finite 3 by 3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.0):
        raise ValueError("rotation_n_from_b must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("rotation_n_from_b must be a proper rotation")
    return rotation


def _covariance15(values: Sequence[Sequence[float]]) -> FloatArray:
    covariance = np.asarray(values, dtype=np.float64)
    if covariance.shape != (ERROR_STATE_SIZE, ERROR_STATE_SIZE) or not np.all(
        np.isfinite(covariance)
    ):
        raise ValueError("ESKF covariance must be a finite 15 by 15 matrix")
    if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=0.0):
        raise ValueError("ESKF covariance must be symmetric")
    if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-10:
        raise ValueError("ESKF covariance must be positive semidefinite")
    return covariance


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = vector
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def continuous_error_model(
    rotation_n_from_b: Sequence[Sequence[float]],
    corrected_specific_force_b_mps2: Sequence[float],
    bias_corrected_inertial_angular_velocity_b_radps: Sequence[float],
    earth_rotation_n_radps: Sequence[float] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return F and L for right-multiplicative attitude error.

    The error convention is true minus nominal, with
    ``R_true = R_nominal @ Exp(delta_theta_b)``.
    """

    rotation = _rotation3(rotation_n_from_b)
    specific_force = _vector3(
        corrected_specific_force_b_mps2,
        "corrected_specific_force_b_mps2",
    )
    angular_velocity = _vector3(
        bias_corrected_inertial_angular_velocity_b_radps,
        "bias_corrected_inertial_angular_velocity_b_radps",
    )
    earth_rate_n = (
        np.zeros(3)
        if earth_rotation_n_radps is None
        else _vector3(earth_rotation_n_radps, "earth_rotation_n_radps")
    )
    dynamics = np.zeros((ERROR_STATE_SIZE, ERROR_STATE_SIZE), dtype=np.float64)
    dynamics[POSITION, VELOCITY] = np.eye(3)
    dynamics[VELOCITY, VELOCITY] = -2.0 * _skew(earth_rate_n)
    dynamics[VELOCITY, ATTITUDE] = -rotation @ _skew(specific_force)
    dynamics[VELOCITY, ACCELEROMETER_BIAS] = -rotation
    dynamics[ATTITUDE, ATTITUDE] = -_skew(angular_velocity)
    dynamics[ATTITUDE, GYROSCOPE_BIAS] = -np.eye(3)

    noise_map = np.zeros((ERROR_STATE_SIZE, DRIVING_NOISE_SIZE), dtype=np.float64)
    noise_map[VELOCITY, 0:3] = -rotation
    noise_map[ATTITUDE, 3:6] = -np.eye(3)
    noise_map[ACCELEROMETER_BIAS, 6:9] = np.eye(3)
    noise_map[GYROSCOPE_BIAS, 9:12] = np.eye(3)
    return dynamics, noise_map


def _noise_covariance(noise: EskfNoiseDensities) -> FloatArray:
    values = np.array(
        (
            noise.accelerometer_white_noise_mps2_sqrt_s,
            noise.gyroscope_white_noise_radps_sqrt_s,
            noise.accelerometer_bias_random_walk_mps2_per_sqrt_s,
            noise.gyroscope_bias_random_walk_radps_per_sqrt_s,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("ESKF noise densities must be finite and nonnegative")
    return np.diag(np.repeat(values * values, 3))


def propagate_error_covariance(
    covariance: Sequence[Sequence[float]],
    rotation_n_from_b: Sequence[Sequence[float]],
    corrected_specific_force_b_mps2: Sequence[float],
    bias_corrected_inertial_angular_velocity_b_radps: Sequence[float],
    noise: EskfNoiseDensities,
    dt_s: float,
    earth_rotation_n_radps: Sequence[float] | None = None,
) -> FloatArray:
    """Propagate covariance with a second-order transition and PSD noise quadrature."""

    prior = _covariance15(covariance)
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")

    dynamics, noise_map = continuous_error_model(
        rotation_n_from_b,
        corrected_specific_force_b_mps2,
        bias_corrected_inertial_angular_velocity_b_radps,
        earth_rotation_n_radps,
    )
    dynamics_dt = dynamics * dt_s
    transition = np.eye(ERROR_STATE_SIZE) + dynamics_dt + 0.5 * dynamics_dt @ dynamics_dt
    continuous_noise = noise_map @ _noise_covariance(noise) @ noise_map.T
    discrete_noise = 0.5 * (
        continuous_noise + transition @ continuous_noise @ transition.T
    ) * dt_s
    propagated = transition @ prior @ transition.T + discrete_noise
    propagated = 0.5 * (propagated + propagated.T)
    if not np.all(np.isfinite(propagated)):
        raise ValueError("propagated ESKF covariance is non-finite")
    if float(np.min(np.linalg.eigvalsh(propagated))) < -1e-10:
        raise ValueError("propagated ESKF covariance lost positive semidefiniteness")
    return propagated


def position_measurement_model(
    state: InsState,
    antenna_lever_imu_m: Sequence[float],
) -> tuple[FloatArray, FloatArray]:
    """Predict antenna position and its right-error measurement Jacobian."""

    position = _vector3(state.position_n_m, "position_n_m")
    rotation = _rotation3(state.rotation_n_from_b)
    lever = _vector3(antenna_lever_imu_m, "antenna_lever_imu_m")
    predicted_antenna_position = position + rotation @ lever
    measurement_jacobian = np.zeros((3, ERROR_STATE_SIZE), dtype=np.float64)
    measurement_jacobian[:, POSITION] = np.eye(3)
    measurement_jacobian[:, ATTITUDE] = -rotation @ _skew(lever)
    return predicted_antenna_position, measurement_jacobian


def update_position(
    state: InsState,
    covariance: Sequence[Sequence[float]],
    measured_antenna_position_n_m: Sequence[float],
    measurement_covariance_n_m2: Sequence[Sequence[float]],
    antenna_lever_imu_m: Sequence[float],
) -> PositionUpdateResult:
    """Apply one valid loose-coupled PVT position update without NIS gating."""

    prior = _covariance15(covariance)
    measurement = _vector3(
        measured_antenna_position_n_m,
        "measured_antenna_position_n_m",
    )
    measurement_covariance = np.asarray(measurement_covariance_n_m2, dtype=np.float64)
    if measurement_covariance.shape != (3, 3) or not np.all(
        np.isfinite(measurement_covariance)
    ):
        raise ValueError("PVT measurement covariance must be a finite 3 by 3 matrix")
    if not np.allclose(
        measurement_covariance,
        measurement_covariance.T,
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("PVT measurement covariance must be symmetric")
    if float(np.min(np.linalg.eigvalsh(measurement_covariance))) <= 0.0:
        raise ValueError("PVT measurement covariance must be positive definite")

    predicted, measurement_jacobian = position_measurement_model(
        state,
        antenna_lever_imu_m,
    )
    innovation = measurement - predicted
    innovation_covariance = (
        measurement_jacobian @ prior @ measurement_jacobian.T
        + measurement_covariance
    )
    innovation_covariance = 0.5 * (
        innovation_covariance + innovation_covariance.T
    )
    solved_innovation = np.linalg.solve(innovation_covariance, innovation)
    kalman_gain = np.linalg.solve(
        innovation_covariance,
        measurement_jacobian @ prior,
    ).T
    correction = kalman_gain @ innovation

    rotation = _rotation3(state.rotation_n_from_b)
    corrected_state = InsState(
        position_n_m=_vector3(state.position_n_m, "position_n_m") + correction[POSITION],
        velocity_n_mps=_vector3(state.velocity_n_mps, "velocity_n_mps")
        + correction[VELOCITY],
        rotation_n_from_b=rotation @ rotation_increment(correction[ATTITUDE]),
        accelerometer_bias_b_mps2=_vector3(
            state.accelerometer_bias_b_mps2,
            "accelerometer_bias_b_mps2",
        )
        + correction[ACCELEROMETER_BIAS],
        gyroscope_bias_b_radps=_vector3(
            state.gyroscope_bias_b_radps,
            "gyroscope_bias_b_radps",
        )
        + correction[GYROSCOPE_BIAS],
    )

    identity = np.eye(ERROR_STATE_SIZE)
    residual_map = identity - kalman_gain @ measurement_jacobian
    posterior = (
        residual_map @ prior @ residual_map.T
        + kalman_gain @ measurement_covariance @ kalman_gain.T
    )
    reset_jacobian = identity.copy()
    reset_jacobian[ATTITUDE, ATTITUDE] = np.eye(3) - 0.5 * _skew(
        correction[ATTITUDE]
    )
    posterior = reset_jacobian @ posterior @ reset_jacobian.T
    posterior = 0.5 * (posterior + posterior.T)
    if not np.all(np.isfinite(posterior)):
        raise ValueError("posterior ESKF covariance is non-finite")
    if float(np.min(np.linalg.eigvalsh(posterior))) < -1e-10:
        raise ValueError("posterior ESKF covariance lost positive semidefiniteness")

    return PositionUpdateResult(
        state=corrected_state,
        covariance=posterior,
        innovation_n_m=innovation,
        innovation_covariance_n_m2=innovation_covariance,
        kalman_gain=kalman_gain,
        nis=float(innovation @ solved_innovation),
    )
