"""Thin Phase 5 adapter that reuses the frozen Phase 4 learning chain."""

from __future__ import annotations

from typing import Sequence, cast

import torch
from torch import Tensor, nn

from .phase3_forward import GnssOutage
from .phase4_config import Phase4Config
from .phase4_data import Phase4Sequence
from .phase4_learning import RobustNormalizer, UnrollResult, unroll_loss
from .phase4_state import MeanState
from .phase4_student import OrdinaryCausalStudent
from .phase5_augmentation import augmentation_yaw_rad, rotate_phase4_sequence
from .phase5_config import Phase5Config, Phase5Variant
from .phase5_student import EquivariantHidden, GravityAwareSo2Student


Phase5Student = OrdinaryCausalStudent | GravityAwareSo2Student
Phase5Hidden = Tensor | EquivariantHidden


def new_phase5_student(
    variant: Phase5Variant,
    phase4_config: Phase4Config,
    phase5_config: Phase5Config,
) -> Phase5Student:
    """Create each variant from the same fixed seed and frozen Phase 4 scales."""

    torch.manual_seed(phase4_config.seed)
    common = {
        "quality_dim": len(phase4_config.quality_fields),
        "previous_velocity_scale_n_mps": phase4_config.previous_velocity_scale_n_mps,
        "pre_position_scale_n_m": phase4_config.pre_position_scale_n_m,
        "pre_velocity_scale_n_mps": phase4_config.pre_velocity_scale_n_mps,
        "pre_attitude_scale_rad": phase4_config.pre_attitude_scale_rad,
        "post_correction_scale": phase4_config.post_correction_scale,
    }
    if variant.strict_so2:
        student: Phase5Student = GravityAwareSo2Student(
            **common,
            scalar_hidden_dim=phase5_config.scalar_hidden_dim,
            vector_hidden_channels=phase5_config.vector_hidden_channels,
        )
    else:
        student = OrdinaryCausalStudent(
            **common,
            hidden_dim=phase4_config.hidden_dim,
        )
    parameter_count = sum(parameter.numel() for parameter in student.parameters())
    if parameter_count != variant.trainable_parameter_count:
        raise ValueError("constructed model parameter count differs from Phase 5 config")
    return student


def training_sequence_for_variant(
    sequence: Phase4Sequence,
    variant: Phase5Variant,
    phase5_config: Phase5Config,
    training_pass_index: int,
) -> tuple[Phase4Sequence, float]:
    """Return one internally consistent coordinate view for a complete training pass."""

    if training_pass_index < 0:
        raise ValueError("training pass index must be nonnegative")
    if not variant.rotation_augmentation:
        return sequence, 0.0
    yaw_rad = augmentation_yaw_rad(phase5_config.augmentation_seed, training_pass_index)
    return rotate_phase4_sequence(sequence, yaw_rad), yaw_rad


def detach_phase5_hidden(hidden: Phase5Hidden) -> Phase5Hidden:
    """Detach either ordinary or typed hidden state between truncated segments."""

    if isinstance(hidden, EquivariantHidden):
        return EquivariantHidden(
            scalars=hidden.scalars.detach(),
            vectors=hidden.vectors.detach(),
        )
    return hidden.detach()


def unroll_phase5_no_physics(
    student: Phase5Student,
    sequence: Phase4Sequence,
    phase4_config: Phase4Config,
    normalizer: RobustNormalizer,
    start: int,
    stop: int,
    previous_state: MeanState,
    previous_hidden: Phase5Hidden,
    variant: Phase5Variant,
    outages: Sequence[GnssOutage] = (),
    score_start: int | None = None,
) -> UnrollResult:
    """Use the accepted Phase 4 weak-label loss with physics disabled for all variants."""

    if variant.physics_weight != 0.0:
        raise ValueError("Phase 5 main comparison requires physics weight zero")
    return unroll_loss(
        cast(OrdinaryCausalStudent, student),
        sequence,
        phase4_config,
        normalizer,
        start,
        stop,
        previous_state,
        cast(Tensor, previous_hidden),
        0.0,
        outages,
        score_start=score_start,
        compute_physics=False,
    )
