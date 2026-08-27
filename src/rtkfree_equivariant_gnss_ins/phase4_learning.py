"""Minimal normalization and matched Phase 4 training loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as functional

from .phase3_forward import GnssOutage
from .phase4_config import Phase4Config
from .phase4_data import MeanStateArray, Phase4Sequence, Phase4StepData
from .phase4_masking import student_gnss_available
from .phase4_state import MeanState, propagate_imu_sequence, state_boxminus
from .phase4_student import OrdinaryCausalStudent, summarize_imu_interval


@dataclass(frozen=True)
class RobustNormalizer:
    imu_center: np.ndarray
    imu_scale: np.ndarray
    quality_center: np.ndarray
    quality_scale: np.ndarray

    def normalize_imu(self, values: np.ndarray) -> np.ndarray:
        return (values - self.imu_center) / self.imu_scale

    def normalize_quality(self, values: np.ndarray) -> np.ndarray:
        return (values - self.quality_center) / self.quality_scale


@dataclass(frozen=True)
class UnrollResult:
    total_loss: Tensor
    weak_label_loss: Tensor
    physics_loss: Tensor
    final_state: MeanState
    final_hidden: Tensor
    physics_valid_step_count: int


@dataclass(frozen=True)
class InferenceStep:
    timestamp_ns_utc: int
    gnss_available: bool
    pre_gnss: MeanState
    posterior: MeanState
    gnss_correction: Tensor


def _numpy_imu_summary(step: Phase4StepData) -> np.ndarray:
    duration = float(np.sum(step.dt_s))
    mean_acceleration = np.sum(
        step.linear_acceleration_b_mps2 * step.dt_s[:, None],
        axis=0,
    ) / duration
    mean_angular_velocity = np.sum(
        step.angular_velocity_b_radps * step.dt_s[:, None],
        axis=0,
    ) / duration
    return np.concatenate((mean_acceleration, mean_angular_velocity, (duration,)))


def _robust_center_scale(values: np.ndarray, minimum_scale: float) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    return center, np.maximum(scale, minimum_scale)


def fit_deployable_normalizer(
    sequence: Phase4Sequence,
    config: Phase4Config,
) -> RobustNormalizer:
    """Fit only deployable fields from the frozen training split."""

    start, stop = config.splits["train"]
    train_steps = sequence.steps[start:stop]
    imu_values = np.vstack([_numpy_imu_summary(step) for step in train_steps])
    quality_values = np.vstack(
        [step.pvt_quality for step in train_steps if step.pvt_solution_valid]
    )
    imu_center, imu_scale = _robust_center_scale(
        imu_values,
        config.normalization_minimum_scale,
    )
    quality_center, quality_scale = _robust_center_scale(
        quality_values,
        config.normalization_minimum_scale,
    )
    return RobustNormalizer(imu_center, imu_scale, quality_center, quality_scale)


def tensor_state(value: MeanStateArray, dtype: torch.dtype) -> MeanState:
    return MeanState(
        position_n_m=torch.as_tensor(value.position_n_m, dtype=dtype),
        velocity_n_mps=torch.as_tensor(value.velocity_n_mps, dtype=dtype),
        rotation_n_from_b=torch.as_tensor(value.rotation_n_from_b, dtype=dtype),
    )


def detach_state(value: MeanState) -> MeanState:
    return MeanState(
        position_n_m=value.position_n_m.detach(),
        velocity_n_mps=value.velocity_n_mps.detach(),
        rotation_n_from_b=value.rotation_n_from_b.detach(),
    )


def _normalized_huber(vector: Tensor, scale: Tensor, delta: float) -> Tensor:
    return functional.huber_loss(
        vector / scale,
        torch.zeros_like(vector),
        reduction="mean",
        delta=delta,
    )


def unroll_loss(
    student: OrdinaryCausalStudent,
    sequence: Phase4Sequence,
    config: Phase4Config,
    normalizer: RobustNormalizer,
    start: int,
    stop: int,
    previous_state: MeanState,
    previous_hidden: Tensor,
    physics_weight: float,
    outages: Sequence[GnssOutage] = (),
    score_start: int | None = None,
    compute_physics: bool = True,
) -> UnrollResult:
    """Run one causal truncated segment and compute matched weak/physics losses."""

    dtype = next(student.parameters()).dtype
    device = next(student.parameters()).device
    loss_scale = torch.tensor(config.state_loss_scale, dtype=dtype, device=device)
    innovation_scale = torch.tensor(
        config.gnss_innovation_scale_n_m,
        dtype=dtype,
        device=device,
    )
    lever = torch.as_tensor(sequence.antenna_lever_imu_m, dtype=dtype, device=device)
    gravity = torch.as_tensor(sequence.gravity_n_mps2, dtype=dtype, device=device)
    earth_rate = torch.as_tensor(sequence.earth_rotation_n_radps, dtype=dtype, device=device)
    state = previous_state
    hidden = previous_hidden
    weak_losses: list[Tensor] = []
    physics_losses: list[Tensor] = []
    effective_score_start = start if score_start is None else score_start
    if effective_score_start < start or effective_score_start >= stop:
        raise ValueError("score start must lie inside the unroll")
    for step_index, data_step in enumerate(sequence.steps[start:stop], start=start):
        imu_summary = torch.as_tensor(
            normalizer.normalize_imu(_numpy_imu_summary(data_step)),
            dtype=dtype,
            device=device,
        )
        prediction = student.predict_pre_gnss(state, hidden, imu_summary)
        available = student_gnss_available(
            data_step.timestamp_ns_utc,
            data_step.pvt_solution_valid,
            outages,
        )
        if available and data_step.pvt_antenna_position_n_m is not None:
            measured_position = torch.as_tensor(
                data_step.pvt_antenna_position_n_m,
                dtype=dtype,
                device=device,
            )
            predicted_antenna_position = (
                prediction.state.position_n_m
                + prediction.state.rotation_n_from_b @ lever
            )
            innovation = (measured_position - predicted_antenna_position) / innovation_scale
        else:
            innovation = torch.zeros(3, dtype=dtype, device=device)
        quality = torch.as_tensor(
            normalizer.normalize_quality(data_step.pvt_quality),
            dtype=dtype,
            device=device,
        )
        model_step = student.apply_current_gnss(
            prediction,
            innovation,
            quality,
            torch.tensor((float(available),), dtype=dtype, device=device),
        )
        if step_index >= effective_score_start:
            weak_label = tensor_state(data_step.weak_pseudo_label, dtype)
            weak_losses.append(
                _normalized_huber(
                    state_boxminus(model_step.posterior, weak_label),
                    loss_scale,
                    config.huber_delta,
                )
            )
        if compute_physics and not data_step.imu_timing_gap:
            acceleration = torch.as_tensor(
                data_step.linear_acceleration_b_mps2,
                dtype=dtype,
                device=device,
            )
            angular_velocity = torch.as_tensor(
                data_step.angular_velocity_b_radps,
                dtype=dtype,
                device=device,
            )
            duration = torch.as_tensor(data_step.dt_s, dtype=dtype, device=device)
            with torch.no_grad():
                reference = propagate_imu_sequence(
                    detach_state(state),
                    acceleration,
                    angular_velocity,
                    duration,
                    gravity,
                    earth_rate,
                )
            physics_losses.append(
                _normalized_huber(
                    state_boxminus(model_step.pre_gnss, reference),
                    loss_scale,
                    config.huber_delta,
                )
            )
        state = model_step.posterior
        hidden = model_step.hidden
    if not weak_losses:
        raise ValueError("training unroll cannot be empty")
    weak_loss = torch.stack(weak_losses).mean()
    physics_loss = (
        torch.stack(physics_losses).mean()
        if physics_losses
        else torch.zeros((), dtype=dtype, device=device)
    )
    return UnrollResult(
        total_loss=weak_loss + physics_weight * physics_loss,
        weak_label_loss=weak_loss,
        physics_loss=physics_loss,
        final_state=state,
        final_hidden=hidden,
        physics_valid_step_count=len(physics_losses),
    )


def inference_rollout(
    student: OrdinaryCausalStudent,
    sequence: Phase4Sequence,
    config: Phase4Config,
    normalizer: RobustNormalizer,
    outages: Sequence[GnssOutage] = (),
    reject_all_gnss: bool = False,
    coordinate_shift_n_m: Sequence[float] = (0.0, 0.0, 0.0),
    stop: int | None = None,
) -> tuple[InferenceStep, ...]:
    """Run deployment inference without any teacher input."""

    dtype = next(student.parameters()).dtype
    device = next(student.parameters()).device
    shift = torch.as_tensor(coordinate_shift_n_m, dtype=dtype, device=device)
    if shift.shape != (3,) or not bool(torch.all(torch.isfinite(shift))):
        raise ValueError("coordinate shift must contain three finite values")
    initial = tensor_state(sequence.initial_student_state, dtype)
    state = MeanState(
        position_n_m=initial.position_n_m + shift,
        velocity_n_mps=initial.velocity_n_mps,
        rotation_n_from_b=initial.rotation_n_from_b,
    )
    hidden = student.initial_hidden(state)
    innovation_scale = torch.tensor(
        config.gnss_innovation_scale_n_m,
        dtype=dtype,
        device=device,
    )
    lever = torch.as_tensor(sequence.antenna_lever_imu_m, dtype=dtype, device=device)
    output: list[InferenceStep] = []
    step_stop = len(sequence.steps) if stop is None else stop
    if step_stop <= 0 or step_stop > len(sequence.steps):
        raise ValueError("inference stop must lie inside the sequence")
    with torch.no_grad():
        for data_step in sequence.steps[:step_stop]:
            imu_summary = torch.as_tensor(
                normalizer.normalize_imu(_numpy_imu_summary(data_step)),
                dtype=dtype,
                device=device,
            )
            prediction = student.predict_pre_gnss(state, hidden, imu_summary)
            available = (
                not reject_all_gnss
                and student_gnss_available(
                    data_step.timestamp_ns_utc,
                    data_step.pvt_solution_valid,
                    outages,
                )
            )
            if available and data_step.pvt_antenna_position_n_m is not None:
                measured_position = (
                    torch.as_tensor(
                        data_step.pvt_antenna_position_n_m,
                        dtype=dtype,
                        device=device,
                    )
                    + shift
                )
                predicted_antenna_position = (
                    prediction.state.position_n_m
                    + prediction.state.rotation_n_from_b @ lever
                )
                innovation = (measured_position - predicted_antenna_position) / innovation_scale
            else:
                innovation = torch.zeros(3, dtype=dtype, device=device)
            quality = torch.as_tensor(
                normalizer.normalize_quality(data_step.pvt_quality),
                dtype=dtype,
                device=device,
            )
            model_step = student.apply_current_gnss(
                prediction,
                innovation,
                quality,
                torch.tensor((float(available),), dtype=dtype, device=device),
            )
            output.append(
                InferenceStep(
                    timestamp_ns_utc=data_step.timestamp_ns_utc,
                    gnss_available=available,
                    pre_gnss=detach_state(model_step.pre_gnss),
                    posterior=detach_state(model_step.posterior),
                    gnss_correction=model_step.gnss_correction.detach(),
                )
            )
            state = model_step.posterior
            hidden = model_step.hidden
    return tuple(output)
