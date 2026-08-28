from __future__ import annotations

import math
import unittest

import torch

from rtkfree_equivariant_gnss_ins.phase4_state import MeanState, so3_exp
from rtkfree_equivariant_gnss_ins.phase5_group import (
    horizontal_yaw_rotation,
    transform_attitude_n_from_b,
    transform_mean_state,
    transform_navigation_vector,
)


DTYPE = torch.float64
ATOL = 1e-9
RTOL = 1e-8


class Phase5GroupActionTests(unittest.TestCase):
    def test_active_positive_quarter_turn_and_gravity_axis(self) -> None:
        yaw = torch.tensor(math.pi / 2.0, dtype=DTYPE)
        horizontal = torch.tensor((1.0, 0.0, 0.0), dtype=DTYPE)
        gravity = torch.tensor((0.0, 0.0, 9.81), dtype=DTYPE)
        torch.testing.assert_close(
            transform_navigation_vector(horizontal, yaw),
            torch.tensor((0.0, 1.0, 0.0), dtype=DTYPE),
            atol=ATOL,
            rtol=RTOL,
        )
        torch.testing.assert_close(
            transform_navigation_vector(gravity, yaw),
            gravity,
            atol=ATOL,
            rtol=RTOL,
        )

    def test_identity_preserves_full_mean_state(self) -> None:
        generator = torch.Generator().manual_seed(5101)
        state = MeanState(
            position_n_m=torch.randn(3, generator=generator, dtype=DTYPE),
            velocity_n_mps=torch.randn(3, generator=generator, dtype=DTYPE),
            rotation_n_from_b=so3_exp(
                torch.randn(3, generator=generator, dtype=DTYPE) * 0.4
            ),
        )
        transformed = transform_mean_state(state, torch.tensor(0.0, dtype=DTYPE))
        torch.testing.assert_close(transformed.position_n_m, state.position_n_m, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(transformed.velocity_n_mps, state.velocity_n_mps, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(
            transformed.rotation_n_from_b,
            state.rotation_n_from_b,
            atol=ATOL,
            rtol=RTOL,
        )

    def test_non_special_angles_obey_composition_for_vectors_and_attitudes(self) -> None:
        generator = torch.Generator().manual_seed(5102)
        vector = torch.randn(3, generator=generator, dtype=DTYPE)
        attitude = so3_exp(torch.randn(3, generator=generator, dtype=DTYPE) * 0.3)
        first = torch.tensor(0.37, dtype=DTYPE)
        second = torch.tensor(-1.11, dtype=DTYPE)

        vector_sequential = transform_navigation_vector(
            transform_navigation_vector(vector, first),
            second,
        )
        vector_composed = transform_navigation_vector(vector, first + second)
        torch.testing.assert_close(vector_sequential, vector_composed, atol=ATOL, rtol=RTOL)

        attitude_sequential = transform_attitude_n_from_b(
            transform_attitude_n_from_b(attitude, first),
            second,
        )
        attitude_composed = transform_attitude_n_from_b(attitude, first + second)
        torch.testing.assert_close(
            attitude_sequential,
            attitude_composed,
            atol=ATOL,
            rtol=RTOL,
        )

    def test_batched_random_inputs_support_multiple_non_special_angles(self) -> None:
        generator = torch.Generator().manual_seed(5103)
        angles = torch.tensor((0.37, -1.11, 2.23), dtype=DTYPE)
        vectors = torch.randn((3, 3), generator=generator, dtype=DTYPE)
        rotations = horizontal_yaw_rotation(angles)
        transformed = transform_navigation_vector(vectors, angles)
        expected = (rotations @ vectors.unsqueeze(-1)).squeeze(-1)
        self.assertEqual(rotations.shape, (3, 3, 3))
        torch.testing.assert_close(transformed, expected, atol=ATOL, rtol=RTOL)

    def test_body_frame_imu_and_declared_scalars_are_invariant_inputs(self) -> None:
        body_acceleration = torch.tensor((1.0, 2.0, 3.0), dtype=DTYPE)
        body_angular_rate = torch.tensor((0.1, -0.2, 0.3), dtype=DTYPE)
        elapsed_time = torch.tensor(1.0, dtype=DTYPE)
        gnss_available = torch.tensor(0.0, dtype=DTYPE)
        satellite_count = torch.tensor(7.0, dtype=DTYPE)
        dop = torch.tensor((1.4, 0.8, 1.1), dtype=DTYPE)

        invariant_inputs = (
            body_acceleration,
            body_angular_rate,
            elapsed_time,
            gnss_available,
            satellite_count,
            dop,
        )
        unchanged_inputs = tuple(value.clone() for value in invariant_inputs)
        for before, after in zip(invariant_inputs, unchanged_inputs, strict=True):
            self.assertTrue(torch.equal(before, after))

    def test_invalid_group_inputs_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            horizontal_yaw_rotation(torch.tensor(1))
        with self.assertRaises(ValueError):
            horizontal_yaw_rotation(torch.tensor(float("nan"), dtype=DTYPE))
        with self.assertRaises(ValueError):
            transform_navigation_vector(
                torch.zeros(2, dtype=DTYPE),
                torch.tensor(0.2, dtype=DTYPE),
            )


if __name__ == "__main__":
    unittest.main()
