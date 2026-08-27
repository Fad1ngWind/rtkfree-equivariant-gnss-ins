"""Validation and construction for the one fixed Phase 3 ESKF profile."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .phase3_eskf import EskfNoiseDensities


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FixedEskfConfig:
    profile_id: str
    pvt_sha256: str
    imu_sha256: str
    imu_noise_parameter_sha256: str
    extrinsic_sha256: str
    initialization_window_s: float
    minimum_horizontal_speed_mps: float
    initial_velocity_std_ned_mps: FloatArray
    initial_attitude_std_body_rad: FloatArray
    initial_accelerometer_bias_std_b_mps2: FloatArray
    initial_gyroscope_bias_std_b_radps: FloatArray
    noise: EskfNoiseDensities
    outage_start_after_initialization_s: int
    outage_durations_s: tuple[int, ...]


def _positive_vector3(values: object, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain three finite positive values")
    return array


def load_fixed_eskf_config(path: Path) -> FixedEskfConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        inputs = raw["inputs"]
        initialization = raw["initialization"]
        standard_deviations = initialization["initial_standard_deviations"]
        process_noise = raw["process_noise"]
        position_update = raw["position_update"]
        outages = raw["controlled_outages"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("cannot read fixed Phase 3 ESKF configuration") from exc
    if raw.get("schema_version") != 1 or raw.get("profile_id") != "fixed_forward_eskf_v1":
        raise ValueError("unsupported fixed Phase 3 ESKF profile")
    if inputs.get("source_transform_interpretation") != "antenna_from_imu":
        raise ValueError("fixed profile must use the tested antenna-from-IMU transform")
    if initialization.get("nominal_vertical_velocity_mps") != 0.0:
        raise ValueError("fixed profile nominal vertical initialization must be zero")
    if standard_deviations.get("position") != "rotated_standardized_pvt_covariance":
        raise ValueError("fixed profile must initialize position from PVT covariance")
    mechanization = raw.get("mechanization", {})
    if not all(
        mechanization.get(key) is True
        for key in ("earth_rotation_compensation", "coriolis_compensation")
    ):
        raise ValueError("fixed profile requires Earth rotation and Coriolis compensation")
    if process_noise.get("source") != "official_xsens_imu_param_avg_axis":
        raise ValueError("fixed profile must use the official Xsens noise parameters")
    if process_noise.get("source_group") != "avg-axis":
        raise ValueError("fixed profile must use the preselected average-axis scalars")
    if process_noise.get("bias_model") != "continuous_random_walk":
        raise ValueError("fixed profile must use the official bias random walks")
    if position_update.get("nis_gate_enabled") is not False:
        raise ValueError("fixed profile cannot gate updates with NIS")
    if position_update.get("adaptive_enabled") is not False:
        raise ValueError("adaptive ESKF is outside the fixed profile")
    if outages.get("interval") != "left_closed_right_open":
        raise ValueError("controlled outages must be left-closed and right-open")

    official_noise = process_noise["official_avg_axis"]
    interpretation = process_noise["continuous_time_interpretation"]
    required_interpretation = {
        "acc_n_unit": "m/s^2/sqrt(Hz)",
        "gyr_n_unit": "rad/s/sqrt(Hz)",
        "acc_w_unit": "(m/s^2)/sqrt(s)",
        "gyr_w_unit": "(rad/s)/sqrt(s)",
        "qc_diagonal_order": [
            "acc_n^2_I3",
            "gyr_n^2_I3",
            "acc_w^2_I3",
            "gyr_w^2_I3",
        ],
        "qd_mapping": "C=L_Qc_Lt; Qd=0.5*(C+Phi_C_Phit)*dt",
    }
    if interpretation != required_interpretation:
        raise ValueError("fixed profile has an unsupported continuous-time noise interpretation")
    noise = EskfNoiseDensities(
        accelerometer_white_noise_mps2_sqrt_s=float(official_noise["acc_n"]),
        gyroscope_white_noise_radps_sqrt_s=float(official_noise["gyr_n"]),
        accelerometer_bias_random_walk_mps2_per_sqrt_s=float(
            official_noise["acc_w"]
        ),
        gyroscope_bias_random_walk_radps_per_sqrt_s=float(official_noise["gyr_w"]),
    )
    durations = tuple(int(value) for value in outages["durations_s"])
    if durations != (20, 30):
        raise ValueError("fixed profile requires exactly 20 and 30 second outages")
    return FixedEskfConfig(
        profile_id=raw["profile_id"],
        pvt_sha256=str(inputs["pvt_sha256"]),
        imu_sha256=str(inputs["imu_sha256"]),
        imu_noise_parameter_sha256=str(inputs["imu_noise_parameter_sha256"]),
        extrinsic_sha256=str(inputs["extrinsic_sha256"]),
        initialization_window_s=float(initialization["window_s"]),
        minimum_horizontal_speed_mps=float(
            initialization["minimum_horizontal_speed_mps"]
        ),
        initial_velocity_std_ned_mps=_positive_vector3(
            standard_deviations["velocity_ned_mps"],
            "initial velocity standard deviations",
        ),
        initial_attitude_std_body_rad=_positive_vector3(
            standard_deviations["attitude_body_rad"],
            "initial attitude standard deviations",
        ),
        initial_accelerometer_bias_std_b_mps2=_positive_vector3(
            standard_deviations["accelerometer_bias_b_mps2"],
            "initial accelerometer-bias standard deviations",
        ),
        initial_gyroscope_bias_std_b_radps=_positive_vector3(
            standard_deviations["gyroscope_bias_b_radps"],
            "initial gyroscope-bias standard deviations",
        ),
        noise=noise,
        outage_start_after_initialization_s=int(
            outages["start_after_initialization_s"]
        ),
        outage_durations_s=durations,
    )


def make_initial_covariance(
    config: FixedEskfConfig,
    position_covariance_n_m2: FloatArray,
) -> FloatArray:
    position_covariance = np.asarray(position_covariance_n_m2, dtype=np.float64)
    if position_covariance.shape != (3, 3) or not np.all(np.isfinite(position_covariance)):
        raise ValueError("initial position covariance must be finite and 3 by 3")
    covariance = np.zeros((15, 15), dtype=np.float64)
    covariance[0:3, 0:3] = position_covariance
    covariance[3:6, 3:6] = np.diag(config.initial_velocity_std_ned_mps**2)
    covariance[6:9, 6:9] = np.diag(config.initial_attitude_std_body_rad**2)
    covariance[9:12, 9:12] = np.diag(
        config.initial_accelerometer_bias_std_b_mps2**2
    )
    covariance[12:15, 12:15] = np.diag(
        config.initial_gyroscope_bias_std_b_radps**2
    )
    covariance = 0.5 * (covariance + covariance.T)
    if float(np.min(np.linalg.eigvalsh(covariance))) <= 0.0:
        raise ValueError("initial ESKF covariance must be positive definite")
    return covariance
