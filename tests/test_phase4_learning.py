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
    unroll_loss,
)
from rtkfree_equivariant_gnss_ins.phase4_student import OrdinaryCausalStudent


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_phase4_config(ROOT / "config" / "phase4" / "ordinary_student_v1.json")


def synthetic_sequence() -> Phase4Sequence:
    state = MeanStateArray(np.zeros(3), np.zeros(3), np.eye(3))
    steps = []
    for index in range(8):
        steps.append(
            Phase4StepData(
                timestamp_ns_utc=(index + 1) * 1_000_000_000,
                linear_acceleration_b_mps2=np.array(((0.0, 0.0, -9.81),)),
                angular_velocity_b_radps=np.zeros((1, 3)),
                dt_s=np.ones(1),
                imu_timing_gap=index == 3,
                pvt_solution_valid=True,
                pvt_antenna_position_n_m=np.zeros(3),
                pvt_quality=np.array((8.0 + index, 2.0, 1.5, 1.0, 1.1)),
                weak_pseudo_label=state,
            )
        )
    return Phase4Sequence(
        initialization_timestamp_ns_utc=0,
        initial_student_state=state,
        antenna_lever_imu_m=np.zeros(3),
        gravity_n_mps2=np.array((0.0, 0.0, 9.81)),
        earth_rotation_n_radps=np.zeros(3),
        steps=tuple(steps),
    )


class Phase4LearningTests(unittest.TestCase):
    def test_synthetic_smoke_is_finite_decreases_and_excludes_timing_gap(self) -> None:
        torch.manual_seed(CONFIG.seed)
        sequence = synthetic_sequence()
        normalizer = fit_deployable_normalizer(sequence, CONFIG)
        student = OrdinaryCausalStudent(
            quality_dim=len(CONFIG.quality_fields),
            hidden_dim=CONFIG.hidden_dim,
            previous_velocity_scale_n_mps=CONFIG.previous_velocity_scale_n_mps,
        )
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=CONFIG.learning_rate,
            weight_decay=CONFIG.weight_decay,
        )

        def evaluate() -> float:
            result = unroll_loss(
                student,
                sequence,
                CONFIG,
                normalizer,
                0,
                len(sequence.steps),
                tensor_state(sequence.initial_student_state, torch.float32),
                student.initial_hidden(tensor_state(sequence.initial_student_state, torch.float32)),
                0.0,
            )
            self.assertEqual(result.physics_valid_step_count, 7)
            self.assertTrue(torch.isfinite(result.total_loss))
            return float(result.total_loss.detach())

        before = evaluate()
        for _ in range(20):
            initial = tensor_state(sequence.initial_student_state, torch.float32)
            result = unroll_loss(
                student,
                sequence,
                CONFIG,
                normalizer,
                0,
                len(sequence.steps),
                initial,
                student.initial_hidden(initial),
                0.0,
            )
            optimizer.zero_grad(set_to_none=True)
            result.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), CONFIG.gradient_norm_clip)
            optimizer.step()
        after = evaluate()
        self.assertLess(after, before)

    def test_physics_weight_changes_only_total_loss_composition(self) -> None:
        torch.manual_seed(CONFIG.seed)
        sequence = synthetic_sequence()
        normalizer = fit_deployable_normalizer(sequence, CONFIG)
        student = OrdinaryCausalStudent(5, CONFIG.hidden_dim)
        initial = tensor_state(sequence.initial_student_state, torch.float32)
        without = unroll_loss(
            student,
            sequence,
            CONFIG,
            normalizer,
            0,
            4,
            initial,
            student.initial_hidden(initial),
            0.0,
        )
        with_physics = unroll_loss(
            student,
            sequence,
            CONFIG,
            normalizer,
            0,
            4,
            detach_state(initial),
            student.initial_hidden(initial),
            0.1,
        )
        self.assertTrue(torch.equal(without.weak_label_loss, with_physics.weak_label_loss))
        self.assertTrue(torch.equal(without.physics_loss, with_physics.physics_loss))
        self.assertTrue(
            torch.allclose(
                with_physics.total_loss,
                without.total_loss + 0.1 * without.physics_loss,
                atol=1e-7,
                rtol=0.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
