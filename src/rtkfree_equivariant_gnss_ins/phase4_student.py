"""One ordinary causal recurrent mean-state student for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .phase4_state import MeanState, so3_exp


IMU_FEATURE_DIM = 7


@dataclass(frozen=True)
class PreGnssPrediction:
    """Free student prediction produced before current GNSS is visible."""

    state: MeanState
    hidden: Tensor


@dataclass(frozen=True)
class StudentStep:
    """Pre-GNSS prediction and final posterior at one causal event time."""

    pre_gnss: MeanState
    posterior: MeanState
    hidden: Tensor
    gnss_correction: Tensor


def summarize_imu_interval(
    linear_acceleration_b_mps2: Tensor,
    angular_velocity_b_radps: Tensor,
    dt_s: Tensor,
) -> Tensor:
    """Return causal time-weighted mean IMU values and elapsed duration."""

    if linear_acceleration_b_mps2.ndim < 2 or linear_acceleration_b_mps2.shape[-1] != 3:
        raise ValueError("IMU interval must have trailing dimensions time by three")
    if angular_velocity_b_radps.shape != linear_acceleration_b_mps2.shape:
        raise ValueError("angular velocity and acceleration intervals must match")
    if dt_s.shape != linear_acceleration_b_mps2.shape[:-1]:
        raise ValueError("one duration is required for each IMU sample")
    if bool(torch.any(~torch.isfinite(dt_s))) or bool(torch.any(dt_s <= 0.0)):
        raise ValueError("IMU durations must be finite and positive")
    duration = dt_s.sum(dim=-1)
    weights = dt_s.unsqueeze(-1)
    mean_acceleration = (linear_acceleration_b_mps2 * weights).sum(dim=-2) / duration.unsqueeze(-1)
    mean_angular_velocity = (angular_velocity_b_radps * weights).sum(dim=-2) / duration.unsqueeze(-1)
    return torch.cat(
        (mean_acceleration, mean_angular_velocity, duration.unsqueeze(-1)),
        dim=-1,
    )


def _apply_increment(state: MeanState, increment: Tensor) -> MeanState:
    if increment.shape[-1:] != (9,):
        raise ValueError("state increment must have trailing dimension nine")
    return MeanState(
        position_n_m=state.position_n_m + increment[..., 0:3],
        velocity_n_mps=state.velocity_n_mps + increment[..., 3:6],
        rotation_n_from_b=(
            state.rotation_n_from_b @ so3_exp(increment[..., 6:9])
        ),
    )


class OrdinaryCausalStudent(nn.Module):
    """Small non-equivariant GRU student with strict pre/post-GNSS separation."""

    def __init__(
        self,
        quality_dim: int,
        hidden_dim: int = 32,
        previous_velocity_scale_n_mps: Sequence[float] = (20.0, 20.0, 10.0),
        pre_position_scale_n_m: Sequence[float] = (30.0, 30.0, 10.0),
        pre_velocity_scale_n_mps: Sequence[float] = (20.0, 20.0, 10.0),
        pre_attitude_scale_rad: Sequence[float] = (0.1, 0.1, 0.1),
        post_correction_scale: Sequence[float] = (
            20.0,
            20.0,
            10.0,
            5.0,
            5.0,
            2.0,
            0.15,
            0.15,
            0.15,
        ),
    ) -> None:
        super().__init__()
        if quality_dim < 0 or hidden_dim <= 0:
            raise ValueError("student dimensions must be nonnegative and finite")
        self.quality_dim = quality_dim
        self.hidden_dim = hidden_dim
        previous_state_feature_dim = 3 + 9
        self.recurrent = nn.GRUCell(
            previous_state_feature_dim + IMU_FEATURE_DIM,
            hidden_dim,
        )
        self.pre_head = nn.Linear(hidden_dim, 9)
        self.post_head = nn.Sequential(
            nn.Linear(hidden_dim + 3 + quality_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 9),
        )
        self.register_buffer(
            "pre_scale",
            torch.as_tensor(
                tuple(pre_position_scale_n_m)
                + tuple(pre_velocity_scale_n_mps)
                + tuple(pre_attitude_scale_rad),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "post_scale",
            torch.as_tensor(post_correction_scale, dtype=torch.float32),
        )
        if self.pre_scale.shape != (9,) or self.post_scale.shape != (9,):
            raise ValueError("pre and post scales must each contain nine values")
        if not bool(torch.all(torch.isfinite(self.pre_scale))) or not bool(
            torch.all(torch.isfinite(self.post_scale))
        ):
            raise ValueError("pre and post scales must be finite")
        if bool(torch.any(self.pre_scale <= 0.0)) or bool(torch.any(self.post_scale <= 0.0)):
            raise ValueError("pre and post scales must be positive")
        velocity_scale = torch.as_tensor(previous_velocity_scale_n_mps, dtype=torch.float32)
        if velocity_scale.shape != (3,) or not bool(torch.all(torch.isfinite(velocity_scale))):
            raise ValueError("previous velocity scale must contain three finite values")
        if bool(torch.any(velocity_scale <= 0.0)):
            raise ValueError("previous velocity scale must be positive")
        self.register_buffer("previous_velocity_scale", velocity_scale)

    def initial_hidden(self, state: MeanState) -> Tensor:
        return torch.zeros(
            state.position_n_m.shape[:-1] + (self.hidden_dim,),
            dtype=state.position_n_m.dtype,
            device=state.position_n_m.device,
        )

    def predict_pre_gnss(
        self,
        previous_posterior: MeanState,
        previous_hidden: Tensor,
        normalized_imu_features: Tensor,
    ) -> PreGnssPrediction:
        """Predict without accepting any current-GNSS argument."""

        if normalized_imu_features.shape[-1:] != (IMU_FEATURE_DIM,):
            raise ValueError("normalized IMU feature dimension differs from the contract")
        previous_features = torch.cat(
            (
                previous_posterior.velocity_n_mps
                / self.previous_velocity_scale.to(previous_posterior.velocity_n_mps),
                previous_posterior.rotation_n_from_b.flatten(start_dim=-2),
                normalized_imu_features,
            ),
            dim=-1,
        )
        hidden = self.recurrent(previous_features, previous_hidden)
        prediction = torch.tanh(self.pre_head(hidden)) * self.pre_scale.to(hidden)
        return PreGnssPrediction(
            state=MeanState(
                position_n_m=previous_posterior.position_n_m + prediction[..., 0:3],
                velocity_n_mps=prediction[..., 3:6],
                rotation_n_from_b=(
                    previous_posterior.rotation_n_from_b
                    @ so3_exp(prediction[..., 6:9])
                ),
            ),
            hidden=hidden,
        )

    def apply_current_gnss(
        self,
        prediction: PreGnssPrediction,
        normalized_gnss_innovation_n_m: Tensor,
        normalized_gnss_quality: Tensor,
        gnss_available: Tensor,
    ) -> StudentStep:
        """Apply current GNSS only after the free pre-GNSS state exists."""

        if normalized_gnss_innovation_n_m.shape[-1:] != (3,):
            raise ValueError("GNSS innovation must have trailing dimension three")
        if normalized_gnss_quality.shape[-1:] != (self.quality_dim,):
            raise ValueError("GNSS quality dimension differs from the contract")
        if gnss_available.shape != prediction.hidden.shape[:-1] + (1,):
            raise ValueError("GNSS availability must have one scalar per state")
        availability = gnss_available.to(dtype=prediction.hidden.dtype)
        gnss_features = torch.cat(
            (normalized_gnss_innovation_n_m, normalized_gnss_quality),
            dim=-1,
        ) * availability
        correction = (
            torch.tanh(self.post_head(torch.cat((prediction.hidden, gnss_features), dim=-1)))
            * self.post_scale.to(prediction.hidden)
            * availability
        )
        return StudentStep(
            pre_gnss=prediction.state,
            posterior=_apply_increment(prediction.state, correction),
            hidden=prediction.hidden,
            gnss_correction=correction,
        )

    def forward_step(
        self,
        previous_posterior: MeanState,
        previous_hidden: Tensor,
        normalized_imu_features: Tensor,
        normalized_gnss_innovation_n_m: Tensor,
        normalized_gnss_quality: Tensor,
        gnss_available: Tensor,
    ) -> StudentStep:
        prediction = self.predict_pre_gnss(
            previous_posterior,
            previous_hidden,
            normalized_imu_features,
        )
        return self.apply_current_gnss(
            prediction,
            normalized_gnss_innovation_n_m,
            normalized_gnss_quality,
            gnss_available,
        )


def causal_rollout(
    student: OrdinaryCausalStudent,
    initial_state: MeanState,
    imu_features: Tensor,
    gnss_innovations_n_m: Tensor,
    gnss_quality: Tensor,
    gnss_available: Tensor,
) -> tuple[StudentStep, ...]:
    """Run a time-major student sequence without future access."""

    step_count = imu_features.shape[0]
    if not all(
        values.shape[0] == step_count
        for values in (gnss_innovations_n_m, gnss_quality, gnss_available)
    ):
        raise ValueError("all rollout inputs must have the same time dimension")
    state = initial_state
    hidden = student.initial_hidden(initial_state)
    outputs: list[StudentStep] = []
    for index in range(step_count):
        step = student.forward_step(
            state,
            hidden,
            imu_features[index],
            gnss_innovations_n_m[index],
            gnss_quality[index],
            gnss_available[index],
        )
        outputs.append(step)
        state = step.posterior
        hidden = step.hidden
    return tuple(outputs)
