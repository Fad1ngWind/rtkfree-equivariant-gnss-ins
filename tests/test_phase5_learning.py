from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from rtkfree_equivariant_gnss_ins.phase4_config import load_phase4_config
from rtkfree_equivariant_gnss_ins.phase4_data import (
    MeanStateArray,
    Phase4Sequence,
    Phase4StepData,
)
from rtkfree_equivariant_gnss_ins.phase4_learning import (
    detach_state,
    fit_deployable_normalizer,
    tensor_state,
)
from rtkfree_equivariant_gnss_ins.phase5_config import load_phase5_config
from rtkfree_equivariant_gnss_ins.phase5_learning import (
    detach_phase5_hidden,
    new_phase5_student,
    training_sequence_for_variant,
    unroll_phase5_no_physics,
)
from rtkfree_equivariant_gnss_ins.phase5_student import EquivariantHidden


ROOT = Path(__file__).resolve().parents[1]
PHASE4 = load_phase4_config(ROOT / "config" / "phase4" / "ordinary_student_v1.json")
PHASE5 = load_phase5_config(
    ROOT / "config" / "phase5" / "gravity_aware_so2_v1.json",
    ROOT / "config" / "phase4" / "ordinary_student_v1.json",
)


def _synthetic_sequence() -> Phase4Sequence:
    initial = MeanStateArray(np.zeros(3), np.zeros(3), np.eye(3))
    steps = []
    for index in range(8):
        label = MeanStateArray(
            np.array((0.2 * index, -0.1 * index, 0.02 * index)),
            np.array((0.2, -0.1, 0.02)),
            np.eye(3),
        )
        steps.append(
            Phase4StepData(
                timestamp_ns_utc=(index + 1) * 1_000_000_000,
                linear_acceleration_b_mps2=np.array(((0.1, -0.2, -9.7),)),
                angular_velocity_b_radps=np.array(((0.01, -0.02, 0.03),)),
                dt_s=np.ones(1),
                imu_timing_gap=False,
                pvt_solution_valid=index not in (2, 5),
                pvt_antenna_position_n_m=np.array((0.2 * index, -0.1 * index, 0.0)),
                pvt_quality=np.array((8.0, 2.0, 1.5, 1.0, 1.1)),
                weak_pseudo_label=label,
            )
        )
    return Phase4Sequence(
        initialization_timestamp_ns_utc=0,
        initial_student_state=initial,
        antenna_lever_imu_m=np.array((0.1, 0.0, 0.0)),
        gravity_n_mps2=np.array((0.0, 0.0, 9.81)),
        earth_rotation_n_radps=np.zeros(3),
        steps=tuple(steps),
    )


class Phase5LearningAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sequence = _synthetic_sequence()
        self.normalizer = fit_deployable_normalizer(self.sequence, PHASE4)

    def test_all_three_variants_share_no_physics_loss_and_backpropagate(self) -> None:
        for variant in PHASE5.variants:
            student = new_phase5_student(variant, PHASE4, PHASE5)
            training_sequence, _ = training_sequence_for_variant(
                self.sequence,
                variant,
                PHASE5,
                training_pass_index=0,
            )
            initial = tensor_state(training_sequence.initial_student_state, torch.float32)
            result = unroll_phase5_no_physics(
                student,
                training_sequence,
                PHASE4,
                self.normalizer,
                0,
                len(training_sequence.steps),
                initial,
                student.initial_hidden(initial),
                variant,
            )
            self.assertTrue(torch.equal(result.total_loss, result.weak_label_loss))
            self.assertTrue(torch.equal(result.physics_loss, torch.zeros_like(result.physics_loss)))
            self.assertEqual(result.physics_valid_step_count, 0)
            result.total_loss.backward()
            gradients = [
                parameter.grad
                for parameter in student.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.all(torch.isfinite(gradient)) for gradient in gradients))
            detached_hidden = detach_phase5_hidden(result.final_hidden)
            self.assertFalse(detach_state(result.final_state).position_n_m.requires_grad)
            if isinstance(detached_hidden, EquivariantHidden):
                self.assertFalse(detached_hidden.scalars.requires_grad)
                self.assertFalse(detached_hidden.vectors.requires_grad)
            else:
                self.assertFalse(detached_hidden.requires_grad)

    def test_ordinary_and_augmented_initial_models_are_exactly_identical(self) -> None:
        ordinary = new_phase5_student(PHASE5.variants[0], PHASE4, PHASE5)
        augmented = new_phase5_student(PHASE5.variants[1], PHASE4, PHASE5)
        for name, value in ordinary.state_dict().items():
            self.assertTrue(torch.equal(value, augmented.state_dict()[name]))

    def test_augmented_view_is_constant_within_pass_and_changes_only_between_passes(self) -> None:
        variant = PHASE5.variants[1]
        first, first_yaw = training_sequence_for_variant(self.sequence, variant, PHASE5, 0)
        repeated, repeated_yaw = training_sequence_for_variant(self.sequence, variant, PHASE5, 0)
        second, second_yaw = training_sequence_for_variant(self.sequence, variant, PHASE5, 1)
        self.assertEqual(first_yaw, repeated_yaw)
        self.assertNotEqual(first_yaw, second_yaw)
        np.testing.assert_array_equal(
            first.initial_student_state.position_n_m,
            repeated.initial_student_state.position_n_m,
        )
        self.assertFalse(
            np.array_equal(
                first.initial_student_state.rotation_n_from_b,
                second.initial_student_state.rotation_n_from_b,
            )
        )
        for original_step, rotated_step in zip(self.sequence.steps, first.steps, strict=True):
            np.testing.assert_array_equal(
                original_step.linear_acceleration_b_mps2,
                rotated_step.linear_acceleration_b_mps2,
            )
            np.testing.assert_array_equal(original_step.pvt_quality, rotated_step.pvt_quality)

    def test_non_augmented_variants_keep_the_original_sequence_object(self) -> None:
        for variant in (PHASE5.variants[0], PHASE5.variants[2]):
            selected, yaw = training_sequence_for_variant(self.sequence, variant, PHASE5, 4)
            self.assertIs(selected, self.sequence)
            self.assertEqual(yaw, 0.0)


if __name__ == "__main__":
    unittest.main()
