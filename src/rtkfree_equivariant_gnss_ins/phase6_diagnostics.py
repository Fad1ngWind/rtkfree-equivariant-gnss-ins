"""Small deployable-only diagnostics added by the Phase 6 freeze."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .phase4_learning import RobustNormalizer, inference_rollout, tensor_state
from .phase4_state import propagate_imu_sequence, state_boxminus
from .phase6_config import Phase6Config


def low_quality_stratum_indices(
    sequence: object,
    phase4_config: object,
    phase6_config: Phase6Config,
) -> tuple[int, ...]:
    """Return the frozen union of deployable HDOP and residual worst-quartile conditions."""

    start, stop = phase4_config.splits["diagnostic"]
    valid = [index for index in range(start, stop) if sequence.steps[index].pvt_solution_valid]
    if len(valid) != phase6_config.low_quality_valid_epoch_count:
        raise ValueError("valid diagnostic PVT count differs from the Phase 6 freeze")
    selected = tuple(
        index
        for index in valid
        if (
            float(sequence.steps[index].pvt_quality[3])
            >= phase6_config.low_quality_hdop_threshold
            or float(sequence.steps[index].pvt_quality[1])
            >= phase6_config.low_quality_residual_threshold_m
        )
    )
    if len(selected) != phase6_config.low_quality_union_epoch_count:
        raise ValueError("low-quality union count differs from the Phase 6 freeze")
    return selected


def _component_mean_huber(errors: np.ndarray, scale: np.ndarray, delta: float) -> float:
    normalized = errors / scale
    absolute = np.abs(normalized)
    values = np.where(absolute <= delta, 0.5 * absolute**2, delta * (absolute - 0.5 * delta))
    return float(np.mean(values))


def low_quality_stratum_report(
    student: nn.Module,
    sequence: object,
    phase4_config: object,
    phase6_config: Phase6Config,
    normalizer: RobustNormalizer,
) -> dict[str, object]:
    """Report weak-label and physics proxies without using them for model selection."""

    outputs = inference_rollout(student, sequence, phase4_config, normalizer)
    indices = low_quality_stratum_indices(sequence, phase4_config, phase6_config)
    weak_errors: list[np.ndarray] = []
    physics_errors: list[np.ndarray] = []
    correction_norms: list[float] = []
    excluded_timing_gaps = 0
    gravity = torch.as_tensor(sequence.gravity_n_mps2, dtype=torch.float32)
    earth_rate = torch.as_tensor(sequence.earth_rotation_n_radps, dtype=torch.float32)
    with torch.no_grad():
        for index in indices:
            step = sequence.steps[index]
            output = outputs[index]
            weak_label = tensor_state(step.weak_pseudo_label, torch.float32)
            weak_errors.append(
                state_boxminus(output.posterior, weak_label).detach().cpu().numpy()
            )
            correction_norms.append(float(torch.linalg.vector_norm(output.gnss_correction)))
            if step.imu_timing_gap:
                excluded_timing_gaps += 1
                continue
            previous = (
                tensor_state(sequence.initial_student_state, torch.float32)
                if index == 0
                else outputs[index - 1].posterior
            )
            reference = propagate_imu_sequence(
                previous,
                torch.as_tensor(step.linear_acceleration_b_mps2, dtype=torch.float32),
                torch.as_tensor(step.angular_velocity_b_radps, dtype=torch.float32),
                torch.as_tensor(step.dt_s, dtype=torch.float32),
                gravity,
                earth_rate,
            )
            physics_errors.append(
                state_boxminus(output.pre_gnss, reference).detach().cpu().numpy()
            )
    weak = np.vstack(weak_errors)
    physics = np.vstack(physics_errors)
    scale = np.asarray(phase4_config.state_loss_scale, dtype=np.float64)
    report = {
        "name": "deployable-metadata low-quality stratum",
        "valid_diagnostic_epoch_count": phase6_config.low_quality_valid_epoch_count,
        "union_epoch_count": len(indices),
        "selected_indices": list(indices),
        "hdop_threshold": phase6_config.low_quality_hdop_threshold,
        "postfit_residual_rms_m_threshold": (
            phase6_config.low_quality_residual_threshold_m
        ),
        "weak_label_component_mean_huber": _component_mean_huber(
            weak, scale, phase4_config.huber_delta
        ),
        "independent_physics_component_mean_huber": _component_mean_huber(
            physics, scale, phase4_config.huber_delta
        ),
        "physics_valid_epoch_count": int(physics.shape[0]),
        "excluded_timing_gap_epoch_count": excluded_timing_gaps,
        "all_gnss_corrections_zero_at_1e-8": all(value < 1e-8 for value in correction_norms),
        "all_values_finite": all(
            math.isfinite(value)
            for value in (
                *weak.ravel(),
                *physics.ravel(),
                *correction_norms,
            )
        ),
        "enters_model_selection": False,
        "truth_reference_used": False,
    }
    report["robustness_gate_passed"] = (
        report["all_values_finite"]
        and not report["all_gnss_corrections_zero_at_1e-8"]
    )
    return report
