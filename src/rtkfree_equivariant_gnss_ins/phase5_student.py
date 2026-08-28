"""Capacity-controlled strictly SO(2)-equivariant Phase 5 student."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .phase4_state import MeanState, so3_exp
from .phase5_group import transform_horizontal_vector_channels
from .phase5_layers import So2VectorLinear, apply_scalar_vector_gate, radial_tanh


IMU_FEATURE_DIM = 7
PRE_INPUT_SCALAR_DIM = 11
PRE_INPUT_VECTOR_CHANNELS = 4
OUTPUT_SCALAR_DIM = 5
OUTPUT_VECTOR_CHANNELS = 2


@dataclass(frozen=True)
class EquivariantHidden:
    """Typed recurrent state: invariant scalars plus horizontal vectors."""

    scalars: Tensor
    vectors: Tensor


@dataclass(frozen=True)
class EquivariantPreGnssPrediction:
    state: MeanState
    hidden: EquivariantHidden


@dataclass(frozen=True)
class EquivariantStudentStep:
    pre_gnss: MeanState
    posterior: MeanState
    hidden: EquivariantHidden
    gnss_correction: Tensor


def transform_equivariant_hidden(hidden: EquivariantHidden, yaw_rad: Tensor) -> EquivariantHidden:
    """Apply the declared SO(2) representation to a typed hidden state."""

    return EquivariantHidden(
        scalars=hidden.scalars,
        vectors=transform_horizontal_vector_channels(hidden.vectors, yaw_rad),
    )


def _as_positive_vector(values: Sequence[float], length: int, name: str) -> Tensor:
    result = torch.as_tensor(tuple(values), dtype=torch.float32)
    if result.shape != (length,) or not bool(torch.all(torch.isfinite(result))):
        raise ValueError(f"{name} must contain {length} finite values")
    if bool(torch.any(result <= 0.0)):
        raise ValueError(f"{name} must be positive")
    return result


def _require_horizontal_isotropy(values: Tensor, first: int, second: int, name: str) -> None:
    if not bool(values[first] == values[second]):
        raise ValueError(f"{name} horizontal scales must be identical")


def _apply_increment(state: MeanState, increment: Tensor) -> MeanState:
    return MeanState(
        position_n_m=state.position_n_m + increment[..., 0:3],
        velocity_n_mps=state.velocity_n_mps + increment[..., 3:6],
        rotation_n_from_b=state.rotation_n_from_b @ so3_exp(increment[..., 6:9]),
    )


class GravityAwareSo2Student(nn.Module):
    """One causal scalar/vector recurrent student with an exact SO(2) type discipline."""

    def __init__(
        self,
        quality_dim: int,
        scalar_hidden_dim: int = 30,
        vector_hidden_channels: int = 5,
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
        if quality_dim < 0 or scalar_hidden_dim <= 0 or vector_hidden_channels <= 0:
            raise ValueError("student channel counts must be nonnegative and finite")
        self.quality_dim = quality_dim
        self.scalar_hidden_dim = scalar_hidden_dim
        self.vector_hidden_channels = vector_hidden_channels

        velocity_scale = _as_positive_vector(
            previous_velocity_scale_n_mps,
            3,
            "previous velocity scale",
        )
        pre_position_scale = _as_positive_vector(
            pre_position_scale_n_m,
            3,
            "pre position scale",
        )
        pre_velocity_scale = _as_positive_vector(
            pre_velocity_scale_n_mps,
            3,
            "pre velocity scale",
        )
        pre_attitude_scale = _as_positive_vector(
            pre_attitude_scale_rad,
            3,
            "pre attitude scale",
        )
        post_scale = _as_positive_vector(post_correction_scale, 9, "post correction scale")
        _require_horizontal_isotropy(velocity_scale, 0, 1, "previous velocity")
        _require_horizontal_isotropy(pre_position_scale, 0, 1, "pre position")
        _require_horizontal_isotropy(pre_velocity_scale, 0, 1, "pre velocity")
        _require_horizontal_isotropy(post_scale, 0, 1, "post position")
        _require_horizontal_isotropy(post_scale, 3, 4, "post velocity")
        self.register_buffer("previous_velocity_scale", velocity_scale)
        self.register_buffer(
            "pre_scalar_scale",
            torch.cat((pre_position_scale[2:3], pre_velocity_scale[2:3], pre_attitude_scale)),
        )
        self.register_buffer(
            "pre_vector_scale",
            torch.stack((pre_position_scale[0], pre_velocity_scale[0])),
        )
        self.register_buffer(
            "post_scalar_scale",
            torch.cat((post_scale[2:3], post_scale[5:6], post_scale[6:9])),
        )
        self.register_buffer(
            "post_vector_scale",
            torch.stack((post_scale[0], post_scale[3])),
        )

        invariant_recurrent_input_dim = (
            PRE_INPUT_SCALAR_DIM
            + PRE_INPUT_VECTOR_CHANNELS
            + vector_hidden_channels
        )
        self.scalar_recurrent = nn.GRUCell(invariant_recurrent_input_dim, scalar_hidden_dim)
        self.vector_recurrent = So2VectorLinear(
            PRE_INPUT_VECTOR_CHANNELS + vector_hidden_channels,
            vector_hidden_channels,
        )
        self.vector_recurrent_gate = nn.Linear(scalar_hidden_dim, vector_hidden_channels)

        self.pre_scalar_head = nn.Linear(scalar_hidden_dim, OUTPUT_SCALAR_DIM)
        self.pre_vector_head = So2VectorLinear(vector_hidden_channels, OUTPUT_VECTOR_CHANNELS)

        post_vector_input_channels = vector_hidden_channels + 1
        post_invariant_input_dim = (
            scalar_hidden_dim
            + 1
            + quality_dim
            + post_vector_input_channels
        )
        self.post_scalar_hidden = nn.Linear(post_invariant_input_dim, scalar_hidden_dim)
        self.post_vector_hidden = So2VectorLinear(
            post_vector_input_channels,
            vector_hidden_channels,
        )
        self.post_vector_gate = nn.Linear(scalar_hidden_dim, vector_hidden_channels)
        self.post_scalar_head = nn.Linear(scalar_hidden_dim, OUTPUT_SCALAR_DIM)
        self.post_vector_head = So2VectorLinear(
            vector_hidden_channels,
            OUTPUT_VECTOR_CHANNELS,
        )

    def initial_hidden(self, state: MeanState) -> EquivariantHidden:
        leading = state.position_n_m.shape[:-1]
        options = {
            "dtype": state.position_n_m.dtype,
            "device": state.position_n_m.device,
        }
        return EquivariantHidden(
            scalars=torch.zeros(leading + (self.scalar_hidden_dim,), **options),
            vectors=torch.zeros(leading + (self.vector_hidden_channels, 2), **options),
        )

    def _validate_hidden(self, state: MeanState, hidden: EquivariantHidden) -> None:
        leading = state.position_n_m.shape[:-1]
        if hidden.scalars.shape != leading + (self.scalar_hidden_dim,):
            raise ValueError("scalar hidden state shape differs from the contract")
        if hidden.vectors.shape != leading + (self.vector_hidden_channels, 2):
            raise ValueError("vector hidden state shape differs from the contract")

    def _typed_pre_inputs(
        self,
        previous_posterior: MeanState,
        normalized_imu_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if normalized_imu_features.shape != previous_posterior.position_n_m.shape[:-1] + (
            IMU_FEATURE_DIM,
        ):
            raise ValueError("normalized IMU feature dimension differs from the contract")
        scale = self.previous_velocity_scale.to(previous_posterior.velocity_n_mps)
        scalar_inputs = torch.cat(
            (
                previous_posterior.velocity_n_mps[..., 2:3] / scale[2],
                previous_posterior.rotation_n_from_b[..., 2, :],
                normalized_imu_features,
            ),
            dim=-1,
        )
        vector_inputs = torch.cat(
            (
                (previous_posterior.velocity_n_mps[..., :2] / scale[0]).unsqueeze(-2),
                previous_posterior.rotation_n_from_b[..., :2, :].transpose(-1, -2),
            ),
            dim=-2,
        )
        return scalar_inputs, vector_inputs

    def predict_pre_gnss(
        self,
        previous_posterior: MeanState,
        previous_hidden: EquivariantHidden,
        normalized_imu_features: Tensor,
    ) -> EquivariantPreGnssPrediction:
        """Predict freely before accepting any current-GNSS argument."""

        self._validate_hidden(previous_posterior, previous_hidden)
        scalar_inputs, vector_inputs = self._typed_pre_inputs(
            previous_posterior,
            normalized_imu_features,
        )
        invariant_inputs = torch.cat(
            (
                scalar_inputs,
                torch.linalg.vector_norm(vector_inputs, dim=-1),
                torch.linalg.vector_norm(previous_hidden.vectors, dim=-1),
            ),
            dim=-1,
        )
        hidden_scalars = self.scalar_recurrent(invariant_inputs, previous_hidden.scalars)
        vector_candidate = radial_tanh(
            self.vector_recurrent(torch.cat((vector_inputs, previous_hidden.vectors), dim=-2))
        )
        hidden_vectors = apply_scalar_vector_gate(
            vector_candidate,
            self.vector_recurrent_gate(hidden_scalars),
        )
        hidden = EquivariantHidden(hidden_scalars, hidden_vectors)

        scalar_output = (
            torch.tanh(self.pre_scalar_head(hidden_scalars))
            * self.pre_scalar_scale.to(hidden_scalars)
        )
        vector_output = (
            radial_tanh(self.pre_vector_head(hidden_vectors))
            * self.pre_vector_scale.to(hidden_vectors).unsqueeze(-1)
        )
        increment = torch.cat(
            (
                vector_output[..., 0, :],
                scalar_output[..., 0:1],
                vector_output[..., 1, :],
                scalar_output[..., 1:2],
                scalar_output[..., 2:5],
            ),
            dim=-1,
        )
        return EquivariantPreGnssPrediction(
            state=_apply_increment(previous_posterior, increment),
            hidden=hidden,
        )

    def apply_current_gnss(
        self,
        prediction: EquivariantPreGnssPrediction,
        normalized_gnss_innovation_n_m: Tensor,
        normalized_gnss_quality: Tensor,
        gnss_available: Tensor,
    ) -> EquivariantStudentStep:
        """Apply current GNSS after the equivariant free prediction exists."""

        leading = prediction.hidden.scalars.shape[:-1]
        if normalized_gnss_innovation_n_m.shape != leading + (3,):
            raise ValueError("GNSS innovation must have trailing dimension three")
        if normalized_gnss_quality.shape != leading + (self.quality_dim,):
            raise ValueError("GNSS quality dimension differs from the contract")
        if gnss_available.shape != leading + (1,):
            raise ValueError("GNSS availability must have one scalar per state")
        availability = gnss_available.to(dtype=prediction.hidden.scalars.dtype)
        innovation = normalized_gnss_innovation_n_m * availability
        quality = normalized_gnss_quality * availability
        vector_inputs = torch.cat(
            (prediction.hidden.vectors, innovation[..., :2].unsqueeze(-2)),
            dim=-2,
        )
        invariant_inputs = torch.cat(
            (
                prediction.hidden.scalars,
                innovation[..., 2:3],
                quality,
                torch.linalg.vector_norm(vector_inputs, dim=-1),
            ),
            dim=-1,
        )
        scalar_hidden = torch.tanh(self.post_scalar_hidden(invariant_inputs))
        vector_hidden = apply_scalar_vector_gate(
            radial_tanh(self.post_vector_hidden(vector_inputs)),
            self.post_vector_gate(scalar_hidden),
        )
        scalar_output = (
            torch.tanh(self.post_scalar_head(scalar_hidden))
            * self.post_scalar_scale.to(scalar_hidden)
            * availability
        )
        vector_output = (
            radial_tanh(self.post_vector_head(vector_hidden))
            * self.post_vector_scale.to(vector_hidden).unsqueeze(-1)
            * availability.unsqueeze(-1)
        )
        correction = torch.cat(
            (
                vector_output[..., 0, :],
                scalar_output[..., 0:1],
                vector_output[..., 1, :],
                scalar_output[..., 1:2],
                scalar_output[..., 2:5],
            ),
            dim=-1,
        )
        return EquivariantStudentStep(
            pre_gnss=prediction.state,
            posterior=_apply_increment(prediction.state, correction),
            hidden=prediction.hidden,
            gnss_correction=correction,
        )

    def forward_step(
        self,
        previous_posterior: MeanState,
        previous_hidden: EquivariantHidden,
        normalized_imu_features: Tensor,
        normalized_gnss_innovation_n_m: Tensor,
        normalized_gnss_quality: Tensor,
        gnss_available: Tensor,
    ) -> EquivariantStudentStep:
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


def equivariant_causal_rollout(
    student: GravityAwareSo2Student,
    initial_state: MeanState,
    imu_features: Tensor,
    gnss_innovations_n_m: Tensor,
    gnss_quality: Tensor,
    gnss_available: Tensor,
) -> tuple[EquivariantStudentStep, ...]:
    """Run the complete time-major strict student without future access."""

    if imu_features.ndim < 2 or imu_features.shape[-1] != IMU_FEATURE_DIM:
        raise ValueError("rollout IMU features must end in time by seven")
    step_count = imu_features.shape[0]
    if not all(
        values.shape[0] == step_count
        for values in (gnss_innovations_n_m, gnss_quality, gnss_available)
    ):
        raise ValueError("all rollout inputs must have the same time dimension")
    state = initial_state
    hidden = student.initial_hidden(initial_state)
    outputs: list[EquivariantStudentStep] = []
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
