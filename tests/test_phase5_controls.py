from __future__ import annotations

import unittest

import torch

from rtkfree_equivariant_gnss_ins.phase4_state import MeanState, so3_exp
from rtkfree_equivariant_gnss_ins.phase4_student import OrdinaryCausalStudent
from rtkfree_equivariant_gnss_ins.phase5_group import (
    transform_mean_state,
    transform_navigation_vector,
    transform_state_increment,
)
from rtkfree_equivariant_gnss_ins.phase5_student import GravityAwareSo2Student


DTYPE = torch.float64
ATOL = 1e-9
RTOL = 1e-8
ANGLES = (0.37, -1.11, 2.23)


def _maximum_state_difference(actual: MeanState, expected: MeanState) -> float:
    return max(
        float(torch.max(torch.abs(actual.position_n_m - expected.position_n_m)).detach()),
        float(torch.max(torch.abs(actual.velocity_n_mps - expected.velocity_n_mps)).detach()),
        float(
            torch.max(
                torch.abs(actual.rotation_n_from_b - expected.rotation_n_from_b)
            ).detach()
        ),
    )


class Phase5ControlClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(5601)
        self.state = MeanState(
            position_n_m=torch.randn(3, generator=generator, dtype=DTYPE),
            velocity_n_mps=torch.randn(3, generator=generator, dtype=DTYPE),
            rotation_n_from_b=so3_exp(
                torch.randn(3, generator=generator, dtype=DTYPE) * 0.4
            ),
        )
        self.imu = torch.randn(7, generator=generator, dtype=DTYPE)
        self.innovation = torch.randn(3, generator=generator, dtype=DTYPE)
        self.quality = torch.randn(5, generator=generator, dtype=DTYPE)
        self.available = torch.ones(1, dtype=DTYPE)

    def test_ordinary_architecture_used_by_plain_and_augmented_has_counterexamples(self) -> None:
        torch.manual_seed(5602)
        ordinary = OrdinaryCausalStudent(quality_dim=5, hidden_dim=32).to(dtype=DTYPE)
        baseline = ordinary.forward_step(
            self.state,
            ordinary.initial_hidden(self.state),
            self.imu,
            self.innovation,
            self.quality,
            self.available,
        )
        residuals = []
        for angle_value in ANGLES:
            angle = torch.tensor(angle_value, dtype=DTYPE)
            transformed_state = transform_mean_state(self.state, angle)
            transformed = ordinary.forward_step(
                transformed_state,
                ordinary.initial_hidden(transformed_state),
                self.imu,
                transform_navigation_vector(self.innovation, angle),
                self.quality,
                self.available,
            )
            residuals.append(
                max(
                    _maximum_state_difference(
                        transformed.pre_gnss,
                        transform_mean_state(baseline.pre_gnss, angle),
                    ),
                    _maximum_state_difference(
                        transformed.posterior,
                        transform_mean_state(baseline.posterior, angle),
                    ),
                    float(
                        torch.max(
                            torch.abs(
                                transformed.gnss_correction
                                - transform_state_increment(baseline.gnss_correction, angle)
                            )
                        ).detach()
                    ),
                )
            )
        self.assertTrue(all(residual > 1e-6 for residual in residuals), residuals)

    def test_strict_architecture_passes_on_the_same_fixture(self) -> None:
        torch.manual_seed(5602)
        strict = GravityAwareSo2Student(quality_dim=5).to(dtype=DTYPE)
        baseline = strict.forward_step(
            self.state,
            strict.initial_hidden(self.state),
            self.imu,
            self.innovation,
            self.quality,
            self.available,
        )
        for angle_value in ANGLES:
            angle = torch.tensor(angle_value, dtype=DTYPE)
            transformed_state = transform_mean_state(self.state, angle)
            transformed = strict.forward_step(
                transformed_state,
                strict.initial_hidden(transformed_state),
                self.imu,
                transform_navigation_vector(self.innovation, angle),
                self.quality,
                self.available,
            )
            torch.testing.assert_close(
                transformed.gnss_correction,
                transform_state_increment(baseline.gnss_correction, angle),
                atol=ATOL,
                rtol=RTOL,
            )
            for actual_state, expected_state in (
                (transformed.pre_gnss, transform_mean_state(baseline.pre_gnss, angle)),
                (transformed.posterior, transform_mean_state(baseline.posterior, angle)),
            ):
                torch.testing.assert_close(
                    actual_state.position_n_m,
                    expected_state.position_n_m,
                    atol=ATOL,
                    rtol=RTOL,
                )
                torch.testing.assert_close(
                    actual_state.velocity_n_mps,
                    expected_state.velocity_n_mps,
                    atol=ATOL,
                    rtol=RTOL,
                )
                torch.testing.assert_close(
                    actual_state.rotation_n_from_b,
                    expected_state.rotation_n_from_b,
                    atol=ATOL,
                    rtol=RTOL,
                )

    def test_rotation_augmented_control_has_exactly_the_ordinary_parameterization(self) -> None:
        torch.manual_seed(5603)
        ordinary = OrdinaryCausalStudent(quality_dim=5, hidden_dim=32)
        torch.manual_seed(5603)
        rotation_augmented = OrdinaryCausalStudent(quality_dim=5, hidden_dim=32)
        self.assertEqual(type(ordinary), type(rotation_augmented))
        self.assertEqual(
            sum(parameter.numel() for parameter in ordinary.parameters()),
            sum(parameter.numel() for parameter in rotation_augmented.parameters()),
        )
        for name, value in ordinary.state_dict().items():
            self.assertTrue(torch.equal(value, rotation_augmented.state_dict()[name]))


if __name__ == "__main__":
    unittest.main()
