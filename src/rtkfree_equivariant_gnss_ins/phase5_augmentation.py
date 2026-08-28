"""Deterministic Phase 5 yaw augmentation of the frozen Phase 4 interface."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .phase4_data import MeanStateArray, Phase4Sequence, Phase4StepData


def augmentation_yaw_rad(seed: int, training_pass_index: int) -> float:
    """Return one reproducible yaw without consuming a training RNG stream."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("augmentation seed must be an integer")
    if isinstance(training_pass_index, bool) or not isinstance(training_pass_index, int):
        raise TypeError("training pass index must be an integer")
    if training_pass_index < 0:
        raise ValueError("training pass index must be nonnegative")
    digest = hashlib.sha256(
        f"phase5-yaw-v1:{seed}:{training_pass_index}".encode("ascii")
    ).digest()
    unit_interval = (int.from_bytes(digest[:8], "big") + 0.5) / float(1 << 64)
    return (2.0 * unit_interval - 1.0) * math.pi


def horizontal_yaw_rotation_numpy(yaw_rad: float) -> np.ndarray:
    """Return the float64 active navigation-frame yaw action."""

    yaw = float(yaw_rad)
    if not math.isfinite(yaw):
        raise ValueError("augmentation yaw must be finite")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _rotate_state(state: MeanStateArray, group: np.ndarray) -> MeanStateArray:
    return MeanStateArray(
        position_n_m=group @ state.position_n_m,
        velocity_n_mps=group @ state.velocity_n_mps,
        rotation_n_from_b=group @ state.rotation_n_from_b,
    )


def rotate_phase4_sequence(sequence: Phase4Sequence, yaw_rad: float) -> Phase4Sequence:
    """Rotate eligible navigation quantities and copy invariant deployable fields."""

    group = horizontal_yaw_rotation_numpy(yaw_rad)
    steps: list[Phase4StepData] = []
    for step in sequence.steps:
        steps.append(
            Phase4StepData(
                timestamp_ns_utc=step.timestamp_ns_utc,
                linear_acceleration_b_mps2=step.linear_acceleration_b_mps2.copy(),
                angular_velocity_b_radps=step.angular_velocity_b_radps.copy(),
                dt_s=step.dt_s.copy(),
                imu_timing_gap=step.imu_timing_gap,
                pvt_solution_valid=step.pvt_solution_valid,
                pvt_antenna_position_n_m=(
                    None
                    if step.pvt_antenna_position_n_m is None
                    else group @ step.pvt_antenna_position_n_m
                ),
                pvt_quality=step.pvt_quality.copy(),
                weak_pseudo_label=_rotate_state(step.weak_pseudo_label, group),
            )
        )
    return Phase4Sequence(
        initialization_timestamp_ns_utc=sequence.initialization_timestamp_ns_utc,
        initial_student_state=_rotate_state(sequence.initial_student_state, group),
        antenna_lever_imu_m=sequence.antenna_lever_imu_m.copy(),
        gravity_n_mps2=group @ sequence.gravity_n_mps2,
        earth_rotation_n_radps=group @ sequence.earth_rotation_n_radps,
        steps=tuple(steps),
    )
