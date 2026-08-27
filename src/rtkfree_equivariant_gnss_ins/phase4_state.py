"""Differentiable mean-state geometry and independent inertial reference for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class MeanState:
    """Position, velocity, and body attitude in the fixed local NED frame."""

    position_n_m: Tensor
    velocity_n_mps: Tensor
    rotation_n_from_b: Tensor


@dataclass(frozen=True)
class PhysicsResidual:
    """State residual and the independently propagated inertial reference."""

    vector: Tensor
    inertial_reference: MeanState


def skew(vector: Tensor) -> Tensor:
    """Return the skew-symmetric matrix for vectors with trailing dimension three."""

    if vector.shape[-1:] != (3,):
        raise ValueError("vector must have trailing dimension three")
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def so3_exp(rotation_vector_rad: Tensor) -> Tensor:
    """Differentiable exponential map with stable coefficients at zero."""

    if rotation_vector_rad.shape[-1:] != (3,):
        raise ValueError("rotation vector must have trailing dimension three")
    angle = torch.linalg.vector_norm(rotation_vector_rad, dim=-1, keepdim=True)
    first = torch.sinc(angle / torch.pi)
    second = 0.5 * torch.sinc(angle / (2.0 * torch.pi)).square()
    generator = skew(rotation_vector_rad)
    identity = torch.eye(
        3,
        dtype=rotation_vector_rad.dtype,
        device=rotation_vector_rad.device,
    ).expand(generator.shape)
    return (
        identity
        + first.unsqueeze(-1) * generator
        + second.unsqueeze(-1) * (generator @ generator)
    )


def so3_log(rotation: Tensor) -> Tensor:
    """Principal logarithm for proper rotations away from the pi singularity."""

    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must have trailing dimensions 3 by 3")
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    antisymmetric = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(antisymmetric, dim=-1)
    angle = torch.atan2(sine, cosine)
    safe_sine = sine.clamp_min(torch.finfo(rotation.dtype).eps)
    regular_scale = angle / (2.0 * safe_sine)
    small_scale = 0.5 + angle.square() / 12.0
    scale = torch.where(sine < 1e-7, small_scale, regular_scale)
    return scale.unsqueeze(-1) * antisymmetric


def state_boxminus(prediction: MeanState, reference: MeanState) -> Tensor:
    """Return NED position, velocity, and right attitude error."""

    relative_rotation = (
        reference.rotation_n_from_b.transpose(-1, -2)
        @ prediction.rotation_n_from_b
    )
    return torch.cat(
        (
            prediction.position_n_m - reference.position_n_m,
            prediction.velocity_n_mps - reference.velocity_n_mps,
            so3_log(relative_rotation),
        ),
        dim=-1,
    )


def propagate_zero_bias_reference(
    state: MeanState,
    linear_acceleration_b_mps2: Tensor,
    angular_velocity_b_radps: Tensor,
    gravity_n_mps2: Tensor,
    dt_s: Tensor,
    earth_rotation_n_radps: Tensor,
) -> MeanState:
    """Match the Phase 3 midpoint mechanization for one zero-bias IMU interval."""

    if linear_acceleration_b_mps2.shape[-1:] != (3,):
        raise ValueError("linear acceleration must have trailing dimension three")
    if angular_velocity_b_radps.shape != linear_acceleration_b_mps2.shape:
        raise ValueError("angular velocity and linear acceleration shapes must match")
    if bool(torch.any(~torch.isfinite(dt_s))) or bool(torch.any(dt_s <= 0.0)):
        raise ValueError("IMU intervals must be finite and positive")
    interval = dt_s.unsqueeze(-1)
    body_rate = (
        angular_velocity_b_radps
        - (state.rotation_n_from_b.transpose(-1, -2) @ earth_rotation_n_radps.unsqueeze(-1)).squeeze(-1)
    )
    midpoint_rotation = state.rotation_n_from_b @ so3_exp(0.5 * body_rate * interval)
    coriolis_n = -2.0 * torch.linalg.cross(
        earth_rotation_n_radps,
        state.velocity_n_mps,
        dim=-1,
    )
    acceleration_n = (
        (midpoint_rotation @ linear_acceleration_b_mps2.unsqueeze(-1)).squeeze(-1)
        + gravity_n_mps2
        + coriolis_n
    )
    return MeanState(
        position_n_m=(
            state.position_n_m
            + state.velocity_n_mps * interval
            + 0.5 * acceleration_n * interval.square()
        ),
        velocity_n_mps=state.velocity_n_mps + acceleration_n * interval,
        rotation_n_from_b=state.rotation_n_from_b @ so3_exp(body_rate * interval),
    )


def propagate_imu_sequence(
    state: MeanState,
    linear_acceleration_b_mps2: Tensor,
    angular_velocity_b_radps: Tensor,
    dt_s: Tensor,
    gravity_n_mps2: Tensor,
    earth_rotation_n_radps: Tensor,
) -> MeanState:
    """Causally propagate all IMU samples in a nonempty interval."""

    if linear_acceleration_b_mps2.ndim < 2 or linear_acceleration_b_mps2.shape[-1] != 3:
        raise ValueError("IMU sequence must have trailing dimensions time by three")
    if angular_velocity_b_radps.shape != linear_acceleration_b_mps2.shape:
        raise ValueError("angular velocity and linear acceleration sequences must match")
    if dt_s.shape != linear_acceleration_b_mps2.shape[:-1]:
        raise ValueError("one duration is required for each IMU sample")
    result = state
    for index in range(linear_acceleration_b_mps2.shape[-2]):
        result = propagate_zero_bias_reference(
            result,
            linear_acceleration_b_mps2[..., index, :],
            angular_velocity_b_radps[..., index, :],
            gravity_n_mps2,
            dt_s[..., index],
            earth_rotation_n_radps,
        )
    return result


def independent_physics_residual(
    predicted_pre_gnss: MeanState,
    previous_student_posterior: MeanState,
    linear_acceleration_b_mps2: Tensor,
    angular_velocity_b_radps: Tensor,
    dt_s: Tensor,
    gravity_n_mps2: Tensor,
    earth_rotation_n_radps: Tensor,
) -> PhysicsResidual:
    """Compare a free pre-GNSS prediction with an independent IMU propagation."""

    inertial_reference = propagate_imu_sequence(
        previous_student_posterior,
        linear_acceleration_b_mps2,
        angular_velocity_b_radps,
        dt_s,
        gravity_n_mps2,
        earth_rotation_n_radps,
    )
    return PhysicsResidual(
        vector=state_boxminus(predicted_pre_gnss, inertial_reference),
        inertial_reference=inertial_reference,
    )
