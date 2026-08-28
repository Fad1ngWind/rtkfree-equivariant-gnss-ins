"""Gravity-preserving active SO(2) actions for Phase 5."""

from __future__ import annotations

import torch
from torch import Tensor

from .phase4_state import MeanState


def horizontal_yaw_rotation(yaw_rad: Tensor) -> Tensor:
    """Return the active navigation-frame rotation ``diag(R_yaw, 1)``."""

    if not isinstance(yaw_rad, Tensor) or not torch.is_floating_point(yaw_rad):
        raise TypeError("yaw must be a floating-point tensor")
    if not bool(torch.all(torch.isfinite(yaw_rad))):
        raise ValueError("yaw must be finite")
    cosine = torch.cos(yaw_rad)
    sine = torch.sin(yaw_rad)
    zero = torch.zeros_like(yaw_rad)
    one = torch.ones_like(yaw_rad)
    return torch.stack(
        (
            torch.stack((cosine, -sine, zero), dim=-1),
            torch.stack((sine, cosine, zero), dim=-1),
            torch.stack((zero, zero, one), dim=-1),
        ),
        dim=-2,
    )


def transform_navigation_vector(vector_n: Tensor, yaw_rad: Tensor) -> Tensor:
    """Apply the same active yaw action to a navigation-frame vector."""

    if vector_n.shape[-1:] != (3,):
        raise ValueError("navigation vector must have trailing dimension three")
    if not torch.is_floating_point(vector_n):
        raise TypeError("navigation vector must be floating point")
    if not bool(torch.all(torch.isfinite(vector_n))):
        raise ValueError("navigation vector must be finite")
    group = horizontal_yaw_rotation(yaw_rad).to(vector_n)
    return (group @ vector_n.unsqueeze(-1)).squeeze(-1)


def transform_horizontal_vector_channels(vectors: Tensor, yaw_rad: Tensor) -> Tensor:
    """Rotate any number of typed horizontal vector channels together."""

    if vectors.ndim < 2 or vectors.shape[-1] != 2:
        raise ValueError("horizontal vector channels must end in channels by two")
    if not torch.is_floating_point(vectors):
        raise TypeError("horizontal vector channels must be floating point")
    if not bool(torch.all(torch.isfinite(vectors))):
        raise ValueError("horizontal vector channels must be finite")
    group_2d = horizontal_yaw_rotation(yaw_rad).to(vectors)[..., :2, :2]
    return (group_2d.unsqueeze(-3) @ vectors.unsqueeze(-1)).squeeze(-1)


def transform_attitude_n_from_b(rotation_n_from_b: Tensor, yaw_rad: Tensor) -> Tensor:
    """Left-multiply a body-to-navigation attitude by the active yaw action."""

    if rotation_n_from_b.shape[-2:] != (3, 3):
        raise ValueError("attitude must have trailing dimensions three by three")
    if not torch.is_floating_point(rotation_n_from_b):
        raise TypeError("attitude must be floating point")
    if not bool(torch.all(torch.isfinite(rotation_n_from_b))):
        raise ValueError("attitude must be finite")
    return horizontal_yaw_rotation(yaw_rad).to(rotation_n_from_b) @ rotation_n_from_b


def transform_mean_state(state: MeanState, yaw_rad: Tensor) -> MeanState:
    """Transform every navigation-frame component of the Phase 4 mean state."""

    return MeanState(
        position_n_m=transform_navigation_vector(state.position_n_m, yaw_rad),
        velocity_n_mps=transform_navigation_vector(state.velocity_n_mps, yaw_rad),
        rotation_n_from_b=transform_attitude_n_from_b(
            state.rotation_n_from_b,
            yaw_rad,
        ),
    )


def transform_state_increment(increment: Tensor, yaw_rad: Tensor) -> Tensor:
    """Transform navigation p/v increments while keeping body attitude increments invariant."""

    if increment.shape[-1:] != (9,):
        raise ValueError("state increment must have trailing dimension nine")
    return torch.cat(
        (
            transform_navigation_vector(increment[..., 0:3], yaw_rad),
            transform_navigation_vector(increment[..., 3:6], yaw_rad),
            increment[..., 6:9],
        ),
        dim=-1,
    )
