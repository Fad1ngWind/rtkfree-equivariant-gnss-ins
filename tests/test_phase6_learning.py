from __future__ import annotations

from pathlib import Path
import unittest

import torch

from tests.test_phase5_learning import _synthetic_sequence
from rtkfree_equivariant_gnss_ins.phase4_config import load_phase4_config
from rtkfree_equivariant_gnss_ins.phase4_learning import (
    fit_deployable_normalizer,
    tensor_state,
    unroll_loss,
)
from rtkfree_equivariant_gnss_ins.phase5_config import load_phase5_config
from rtkfree_equivariant_gnss_ins.phase5_learning import new_phase5_student


ROOT = Path(__file__).resolve().parents[1]
PHASE4 = load_phase4_config(ROOT / "config" / "phase4" / "ordinary_student_v1.json")
PHASE5 = load_phase5_config(
    ROOT / "config" / "phase5" / "gravity_aware_so2_v1.json",
    ROOT / "config" / "phase4" / "ordinary_student_v1.json",
)


class Phase6LearningTests(unittest.TestCase):
    def test_custom_seed_is_matched_across_ordinary_controls(self) -> None:
        first = new_phase5_student(PHASE5.variants[0], PHASE4, PHASE5, seed=163736869)
        augmented = new_phase5_student(
            PHASE5.variants[1], PHASE4, PHASE5, seed=163736869
        )
        second = new_phase5_student(PHASE5.variants[0], PHASE4, PHASE5, seed=188003056)
        self.assertTrue(
            all(
                torch.equal(value, augmented.state_dict()[name])
                for name, value in first.state_dict().items()
            )
        )
        self.assertTrue(
            any(
                not torch.equal(value, second.state_dict()[name])
                for name, value in first.state_dict().items()
            )
        )

    def test_physics_loss_backpropagates_for_ordinary_and_strict(self) -> None:
        sequence = _synthetic_sequence()
        normalizer = fit_deployable_normalizer(sequence, PHASE4)
        for variant in (PHASE5.variants[0], PHASE5.variants[2]):
            student = new_phase5_student(variant, PHASE4, PHASE5, seed=163736869)
            initial = tensor_state(sequence.initial_student_state, torch.float32)
            result = unroll_loss(
                student,
                sequence,
                PHASE4,
                normalizer,
                0,
                len(sequence.steps),
                initial,
                student.initial_hidden(initial),
                0.1,
                compute_physics=True,
            )
            self.assertEqual(result.physics_valid_step_count, len(sequence.steps))
            self.assertTrue(torch.isfinite(result.total_loss))
            self.assertGreater(float(result.physics_loss.detach()), 0.0)
            result.total_loss.backward()
            gradients = [
                parameter.grad
                for parameter in student.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(
                all(torch.all(torch.isfinite(gradient)) for gradient in gradients)
            )


if __name__ == "__main__":
    unittest.main()
