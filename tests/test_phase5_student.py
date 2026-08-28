from __future__ import annotations

import inspect
import unittest

import torch

from rtkfree_equivariant_gnss_ins.phase4_state import MeanState, so3_exp
from rtkfree_equivariant_gnss_ins.phase4_student import OrdinaryCausalStudent
from rtkfree_equivariant_gnss_ins.phase5_group import (
    transform_mean_state,
    transform_navigation_vector,
    transform_state_increment,
)
from rtkfree_equivariant_gnss_ins.phase5_student import (
    EquivariantHidden,
    GravityAwareSo2Student,
    equivariant_causal_rollout,
    transform_equivariant_hidden,
)


DTYPE = torch.float64
ATOL = 1e-9
RTOL = 1e-8
ANGLES = (0.37, -1.11, 2.23)


def _state(generator: torch.Generator) -> MeanState:
    return MeanState(
        position_n_m=torch.randn(3, generator=generator, dtype=DTYPE),
        velocity_n_mps=torch.randn(3, generator=generator, dtype=DTYPE),
        rotation_n_from_b=so3_exp(
            torch.randn(3, generator=generator, dtype=DTYPE) * 0.4
        ),
    )


def _assert_state_close(test: unittest.TestCase, actual: MeanState, expected: MeanState) -> None:
    for actual_value, expected_value in (
        (actual.position_n_m, expected.position_n_m),
        (actual.velocity_n_mps, expected.velocity_n_mps),
        (actual.rotation_n_from_b, expected.rotation_n_from_b),
    ):
        test.assertTrue(torch.allclose(actual_value, expected_value, atol=ATOL, rtol=RTOL))


class Phase5StrictStudentTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5301)
        self.student = GravityAwareSo2Student(quality_dim=5).to(dtype=DTYPE)
        self.generator = torch.Generator().manual_seed(5302)
        self.state = _state(self.generator)
        self.hidden = EquivariantHidden(
            scalars=torch.randn(30, generator=self.generator, dtype=DTYPE),
            vectors=torch.randn((5, 2), generator=self.generator, dtype=DTYPE),
        )
        self.imu = torch.randn(7, generator=self.generator, dtype=DTYPE)
        self.innovation = torch.randn(3, generator=self.generator, dtype=DTYPE)
        self.quality = torch.randn(5, generator=self.generator, dtype=DTYPE)
        self.available = torch.ones(1, dtype=DTYPE)

    def test_complete_forward_step_commutes_for_non_special_angles(self) -> None:
        baseline = self.student.forward_step(
            self.state,
            self.hidden,
            self.imu,
            self.innovation,
            self.quality,
            self.available,
        )
        for angle_value in ANGLES:
            angle = torch.tensor(angle_value, dtype=DTYPE)
            transformed = self.student.forward_step(
                transform_mean_state(self.state, angle),
                transform_equivariant_hidden(self.hidden, angle),
                self.imu,
                transform_navigation_vector(self.innovation, angle),
                self.quality,
                self.available,
            )
            _assert_state_close(
                self,
                transformed.pre_gnss,
                transform_mean_state(baseline.pre_gnss, angle),
            )
            _assert_state_close(
                self,
                transformed.posterior,
                transform_mean_state(baseline.posterior, angle),
            )
            expected_hidden = transform_equivariant_hidden(baseline.hidden, angle)
            torch.testing.assert_close(
                transformed.hidden.scalars,
                expected_hidden.scalars,
                atol=ATOL,
                rtol=RTOL,
            )
            torch.testing.assert_close(
                transformed.hidden.vectors,
                expected_hidden.vectors,
                atol=ATOL,
                rtol=RTOL,
            )
            torch.testing.assert_close(
                transformed.gnss_correction,
                transform_state_increment(baseline.gnss_correction, angle),
                atol=ATOL,
                rtol=RTOL,
            )

    def test_complete_forward_step_identity_is_exact(self) -> None:
        baseline = self.student.forward_step(
            self.state,
            self.hidden,
            self.imu,
            self.innovation,
            self.quality,
            self.available,
        )
        identity = torch.tensor(0.0, dtype=DTYPE)
        transformed = self.student.forward_step(
            transform_mean_state(self.state, identity),
            transform_equivariant_hidden(self.hidden, identity),
            self.imu,
            transform_navigation_vector(self.innovation, identity),
            self.quality,
            self.available,
        )
        _assert_state_close(self, transformed.pre_gnss, baseline.pre_gnss)
        _assert_state_close(self, transformed.posterior, baseline.posterior)
        torch.testing.assert_close(
            transformed.gnss_correction,
            baseline.gnss_correction,
            atol=ATOL,
            rtol=RTOL,
        )

    def test_masked_gnss_is_zero_and_posterior_equals_pre_gnss(self) -> None:
        masked = torch.zeros(1, dtype=DTYPE)
        first = self.student.forward_step(
            self.state,
            self.hidden,
            self.imu,
            self.innovation,
            self.quality,
            masked,
        )
        second = self.student.forward_step(
            self.state,
            self.hidden,
            self.imu,
            self.innovation * 1e6,
            self.quality * -1e6,
            masked,
        )
        self.assertTrue(torch.equal(first.gnss_correction, torch.zeros(9, dtype=DTYPE)))
        _assert_state_close(self, first.posterior, first.pre_gnss)
        _assert_state_close(self, first.posterior, second.posterior)

    def test_pre_gnss_api_excludes_current_gnss_teacher_time_and_identity(self) -> None:
        argument_names = set(inspect.signature(GravityAwareSo2Student.predict_pre_gnss).parameters)
        prohibited = {
            "teacher",
            "covariance",
            "gnss",
            "timestamp",
            "route",
            "file",
            "device",
        }
        self.assertTrue(argument_names.isdisjoint(prohibited))

    def test_parameter_count_is_predeclared_and_capacity_matched(self) -> None:
        ordinary = OrdinaryCausalStudent(quality_dim=5, hidden_dim=32)
        ordinary_count = sum(parameter.numel() for parameter in ordinary.parameters())
        strict_count = sum(parameter.numel() for parameter in self.student.parameters())
        self.assertEqual(ordinary_count, 6994)
        self.assertEqual(strict_count, 6780)
        self.assertGreaterEqual(strict_count, 0.85 * ordinary_count)
        self.assertLessEqual(strict_count, 1.15 * ordinary_count)

    def test_anisotropic_horizontal_scales_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "horizontal scales"):
            GravityAwareSo2Student(
                quality_dim=5,
                pre_position_scale_n_m=(30.0, 29.0, 10.0),
            )


class Phase5StrictRolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5401)
        self.student = GravityAwareSo2Student(quality_dim=5).to(dtype=DTYPE)
        generator = torch.Generator().manual_seed(5402)
        self.initial = _state(generator)
        self.imu = torch.randn((6, 7), generator=generator, dtype=DTYPE)
        self.innovations = torch.randn((6, 3), generator=generator, dtype=DTYPE)
        self.quality = torch.randn((6, 5), generator=generator, dtype=DTYPE)
        self.available = torch.tensor(((1.0,), (0.0,), (1.0,), (1.0,), (0.0,), (1.0,)), dtype=DTYPE)

    def _rollout(self, initial: MeanState, innovations: torch.Tensor) -> tuple:
        return equivariant_causal_rollout(
            self.student,
            initial,
            self.imu,
            innovations,
            self.quality,
            self.available,
        )

    def _assert_step_action(self, actual: object, baseline: object, angle: torch.Tensor) -> None:
        _assert_state_close(self, actual.pre_gnss, transform_mean_state(baseline.pre_gnss, angle))
        _assert_state_close(self, actual.posterior, transform_mean_state(baseline.posterior, angle))
        expected_hidden = transform_equivariant_hidden(baseline.hidden, angle)
        torch.testing.assert_close(actual.hidden.scalars, expected_hidden.scalars, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(actual.hidden.vectors, expected_hidden.vectors, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(
            actual.gnss_correction,
            transform_state_increment(baseline.gnss_correction, angle),
            atol=ATOL,
            rtol=RTOL,
        )

    def test_complete_rollout_commutes_for_non_special_angles_and_mixed_masks(self) -> None:
        baseline = self._rollout(self.initial, self.innovations)
        for angle_value in ANGLES:
            angle = torch.tensor(angle_value, dtype=DTYPE)
            transformed = self._rollout(
                transform_mean_state(self.initial, angle),
                transform_navigation_vector(self.innovations, angle),
            )
            self.assertEqual(len(transformed), len(baseline))
            for actual_step, baseline_step in zip(transformed, baseline, strict=True):
                self._assert_step_action(actual_step, baseline_step, angle)
        for index in (1, 4):
            self.assertTrue(torch.equal(baseline[index].gnss_correction, torch.zeros(9, dtype=DTYPE)))

    def test_complete_rollout_identity_and_composition(self) -> None:
        baseline = self._rollout(self.initial, self.innovations)
        identity = torch.tensor(0.0, dtype=DTYPE)
        identity_rollout = self._rollout(
            transform_mean_state(self.initial, identity),
            transform_navigation_vector(self.innovations, identity),
        )
        for actual_step, baseline_step in zip(identity_rollout, baseline, strict=True):
            self._assert_step_action(actual_step, baseline_step, identity)

        first = torch.tensor(0.37, dtype=DTYPE)
        second = torch.tensor(-1.11, dtype=DTYPE)
        sequential_initial = transform_mean_state(
            transform_mean_state(self.initial, first),
            second,
        )
        sequential_innovations = transform_navigation_vector(
            transform_navigation_vector(self.innovations, first),
            second,
        )
        sequential = self._rollout(sequential_initial, sequential_innovations)
        composed_angle = first + second
        composed = self._rollout(
            transform_mean_state(self.initial, composed_angle),
            transform_navigation_vector(self.innovations, composed_angle),
        )
        for sequential_step, composed_step, baseline_step in zip(
            sequential,
            composed,
            baseline,
            strict=True,
        ):
            _assert_state_close(self, sequential_step.pre_gnss, composed_step.pre_gnss)
            _assert_state_close(self, sequential_step.posterior, composed_step.posterior)
            self._assert_step_action(composed_step, baseline_step, composed_angle)

    def test_future_perturbation_cannot_change_recursive_prefix(self) -> None:
        baseline = self._rollout(self.initial, self.innovations)
        changed_imu = self.imu.clone()
        changed_innovations = self.innovations.clone()
        changed_quality = self.quality.clone()
        changed_imu[-1] = 1e4
        changed_innovations[-1] = -1e4
        changed_quality[-1] = 1e4
        changed = equivariant_causal_rollout(
            self.student,
            self.initial,
            changed_imu,
            changed_innovations,
            changed_quality,
            self.available,
        )
        for baseline_step, changed_step in zip(baseline[:-1], changed[:-1], strict=True):
            self.assertTrue(torch.equal(baseline_step.posterior.position_n_m, changed_step.posterior.position_n_m))
            self.assertTrue(torch.equal(baseline_step.hidden.scalars, changed_step.hidden.scalars))
            self.assertTrue(torch.equal(baseline_step.hidden.vectors, changed_step.hidden.vectors))
        self.assertFalse(torch.equal(baseline[-1].posterior.position_n_m, changed[-1].posterior.position_n_m))

    def test_rollout_rejects_mismatched_time_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "same time dimension"):
            equivariant_causal_rollout(
                self.student,
                self.initial,
                self.imu,
                self.innovations[:-1],
                self.quality,
                self.available,
            )


if __name__ == "__main__":
    unittest.main()
