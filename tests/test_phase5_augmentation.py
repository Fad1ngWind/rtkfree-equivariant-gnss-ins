from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from rtkfree_equivariant_gnss_ins.phase4_data import (
    MeanStateArray,
    Phase4Sequence,
    Phase4StepData,
)
from rtkfree_equivariant_gnss_ins.phase5_augmentation import (
    augmentation_yaw_rad,
    horizontal_yaw_rotation_numpy,
    rotate_phase4_sequence,
)


ATOL = 1e-12


def _state(offset: float) -> MeanStateArray:
    return MeanStateArray(
        position_n_m=np.array((1.0 + offset, 2.0 - offset, 3.0)),
        velocity_n_mps=np.array((-2.0, 0.5 + offset, 0.25)),
        rotation_n_from_b=np.array(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        ),
    )


def _sequence() -> Phase4Sequence:
    steps = []
    for index in range(2):
        steps.append(
            Phase4StepData(
                timestamp_ns_utc=(index + 1) * 1_000_000_000,
                linear_acceleration_b_mps2=np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))),
                angular_velocity_b_radps=np.array(((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))),
                dt_s=np.array((0.25, 0.75)),
                imu_timing_gap=index == 1,
                pvt_solution_valid=index == 0,
                pvt_antenna_position_n_m=(
                    np.array((5.0, -3.0, 1.0)) if index == 0 else None
                ),
                pvt_quality=np.array((7.0, 2.0, 1.4, 0.8, 1.1)),
                weak_pseudo_label=_state(float(index)),
            )
        )
    return Phase4Sequence(
        initialization_timestamp_ns_utc=0,
        initial_student_state=_state(-1.0),
        antenna_lever_imu_m=np.array((0.2, -0.1, 0.3)),
        gravity_n_mps2=np.array((0.0, 0.0, 9.81)),
        earth_rotation_n_radps=np.array((2e-5, -3e-5, -6e-5)),
        steps=tuple(steps),
    )


def _assert_state_close(test: unittest.TestCase, actual: MeanStateArray, expected: MeanStateArray) -> None:
    for actual_value, expected_value in (
        (actual.position_n_m, expected.position_n_m),
        (actual.velocity_n_mps, expected.velocity_n_mps),
        (actual.rotation_n_from_b, expected.rotation_n_from_b),
    ):
        np.testing.assert_allclose(actual_value, expected_value, atol=ATOL, rtol=0.0)


class Phase5AugmentationTests(unittest.TestCase):
    def test_navigation_fields_rotate_and_invariant_fields_are_bit_identical(self) -> None:
        sequence = _sequence()
        yaw = 0.37
        group = horizontal_yaw_rotation_numpy(yaw)
        rotated = rotate_phase4_sequence(sequence, yaw)

        _assert_state_close(
            self,
            rotated.initial_student_state,
            MeanStateArray(
                group @ sequence.initial_student_state.position_n_m,
                group @ sequence.initial_student_state.velocity_n_mps,
                group @ sequence.initial_student_state.rotation_n_from_b,
            ),
        )
        np.testing.assert_allclose(rotated.gravity_n_mps2, sequence.gravity_n_mps2, atol=ATOL, rtol=0.0)
        np.testing.assert_allclose(
            rotated.earth_rotation_n_radps,
            group @ sequence.earth_rotation_n_radps,
            atol=ATOL,
            rtol=0.0,
        )
        self.assertTrue(np.array_equal(rotated.antenna_lever_imu_m, sequence.antenna_lever_imu_m))
        for before, after in zip(sequence.steps, rotated.steps, strict=True):
            self.assertEqual(after.timestamp_ns_utc, before.timestamp_ns_utc)
            self.assertEqual(after.imu_timing_gap, before.imu_timing_gap)
            self.assertEqual(after.pvt_solution_valid, before.pvt_solution_valid)
            self.assertTrue(np.array_equal(after.linear_acceleration_b_mps2, before.linear_acceleration_b_mps2))
            self.assertTrue(np.array_equal(after.angular_velocity_b_radps, before.angular_velocity_b_radps))
            self.assertTrue(np.array_equal(after.dt_s, before.dt_s))
            self.assertTrue(np.array_equal(after.pvt_quality, before.pvt_quality))
            if before.pvt_antenna_position_n_m is None:
                self.assertIsNone(after.pvt_antenna_position_n_m)
            else:
                np.testing.assert_allclose(
                    after.pvt_antenna_position_n_m,
                    group @ before.pvt_antenna_position_n_m,
                    atol=ATOL,
                    rtol=0.0,
                )

    def test_sequence_identity_and_composition(self) -> None:
        sequence = _sequence()
        identity = rotate_phase4_sequence(sequence, 0.0)
        _assert_state_close(self, identity.initial_student_state, sequence.initial_student_state)
        first = 0.37
        second = -1.11
        sequential = rotate_phase4_sequence(rotate_phase4_sequence(sequence, first), second)
        composed = rotate_phase4_sequence(sequence, first + second)
        _assert_state_close(self, sequential.initial_student_state, composed.initial_student_state)
        np.testing.assert_allclose(
            sequential.earth_rotation_n_radps,
            composed.earth_rotation_n_radps,
            atol=ATOL,
            rtol=0.0,
        )
        for sequential_step, composed_step in zip(sequential.steps, composed.steps, strict=True):
            _assert_state_close(self, sequential_step.weak_pseudo_label, composed_step.weak_pseudo_label)

    def test_seeded_schedule_is_repeatable_bounded_and_rng_isolated(self) -> None:
        torch.manual_seed(5501)
        state_before = torch.random.get_rng_state().clone()
        first = tuple(augmentation_yaw_rad(3407, pass_index) for pass_index in range(40))
        repeated = tuple(augmentation_yaw_rad(3407, pass_index) for pass_index in range(40))
        state_after = torch.random.get_rng_state()
        self.assertEqual(first, repeated)
        self.assertEqual(len(set(first)), 40)
        self.assertTrue(all(-math.pi < angle < math.pi for angle in first))
        self.assertTrue(torch.equal(state_before, state_after))

    def test_invalid_yaw_schedule_inputs_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            augmentation_yaw_rad(True, 0)
        with self.assertRaises(ValueError):
            augmentation_yaw_rad(3407, -1)
        with self.assertRaises(ValueError):
            rotate_phase4_sequence(_sequence(), float("nan"))


if __name__ == "__main__":
    unittest.main()
