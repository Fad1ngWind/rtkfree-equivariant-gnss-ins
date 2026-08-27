from __future__ import annotations

import unittest

import torch

import numpy as np

from rtkfree_equivariant_gnss_ins.phase3_ins import (
    InsState,
    propagate_ins,
    rotation_increment,
)
from rtkfree_equivariant_gnss_ins.phase4_state import (
    MeanState,
    independent_physics_residual,
    propagate_imu_sequence,
    so3_exp,
    so3_log,
)


DTYPE = torch.float64


def zero_state() -> MeanState:
    return MeanState(
        position_n_m=torch.zeros(3, dtype=DTYPE),
        velocity_n_mps=torch.zeros(3, dtype=DTYPE),
        rotation_n_from_b=torch.eye(3, dtype=DTYPE),
    )


class Phase4StateTests(unittest.TestCase):
    def test_torch_reference_matches_frozen_phase3_mechanization(self) -> None:
        rotation = rotation_increment(np.array((0.2, -0.1, 0.3), dtype=np.float64))
        phase3_state = InsState(
            position_n_m=np.array((10.0, -4.0, 2.0), dtype=np.float64),
            velocity_n_mps=np.array((3.0, 1.0, -0.5), dtype=np.float64),
            rotation_n_from_b=rotation,
            accelerometer_bias_b_mps2=np.zeros(3),
            gyroscope_bias_b_radps=np.zeros(3),
        )
        acceleration = np.array((0.4, -0.2, -9.7), dtype=np.float64)
        angular_velocity = np.array((0.01, -0.02, 0.03), dtype=np.float64)
        gravity = np.array((0.0, 0.0, 9.81), dtype=np.float64)
        earth_rate = np.array((3e-5, 0.0, -6e-5), dtype=np.float64)
        expected = propagate_ins(
            phase3_state,
            acceleration,
            angular_velocity,
            gravity,
            0.025,
            earth_rate,
        )
        actual = propagate_imu_sequence(
            MeanState(
                position_n_m=torch.from_numpy(phase3_state.position_n_m),
                velocity_n_mps=torch.from_numpy(phase3_state.velocity_n_mps),
                rotation_n_from_b=torch.from_numpy(phase3_state.rotation_n_from_b),
            ),
            torch.from_numpy(acceleration).unsqueeze(0),
            torch.from_numpy(angular_velocity).unsqueeze(0),
            torch.tensor((0.025,), dtype=DTYPE),
            torch.from_numpy(gravity),
            torch.from_numpy(earth_rate),
        )
        self.assertTrue(
            np.allclose(actual.position_n_m.detach().numpy(), expected.position_n_m, atol=1e-12, rtol=0.0)
        )
        self.assertTrue(
            np.allclose(actual.velocity_n_mps.detach().numpy(), expected.velocity_n_mps, atol=1e-12, rtol=0.0)
        )
        self.assertTrue(
            np.allclose(
                actual.rotation_n_from_b.detach().numpy(),
                expected.rotation_n_from_b,
                atol=1e-12,
                rtol=0.0,
            )
        )

    def test_so3_log_recovers_small_and_ordinary_rotation_vectors(self) -> None:
        rotation_vectors = torch.tensor(
            ((1e-10, -2e-10, 3e-10), (0.3, -0.2, 0.1)),
            dtype=DTYPE,
        )
        recovered = so3_log(so3_exp(rotation_vectors))
        self.assertTrue(torch.allclose(recovered, rotation_vectors, atol=1e-12, rtol=1e-12))

    def test_zero_bias_reference_matches_constant_acceleration_motion(self) -> None:
        acceleration = torch.tensor(
            ((1.0, 0.0, -9.81), (1.0, 0.0, -9.81)),
            dtype=DTYPE,
        )
        reference = propagate_imu_sequence(
            zero_state(),
            acceleration,
            torch.zeros_like(acceleration),
            torch.tensor((0.5, 0.5), dtype=DTYPE),
            torch.tensor((0.0, 0.0, 9.81), dtype=DTYPE),
            torch.zeros(3, dtype=DTYPE),
        )
        self.assertTrue(
            torch.allclose(
                reference.position_n_m,
                torch.tensor((0.5, 0.0, 0.0), dtype=DTYPE),
                atol=1e-12,
                rtol=0.0,
            )
        )
        self.assertTrue(
            torch.allclose(
                reference.velocity_n_mps,
                torch.tensor((1.0, 0.0, 0.0), dtype=DTYPE),
                atol=1e-12,
                rtol=0.0,
            )
        )

    def test_physics_residual_compares_free_prediction_with_independent_reference(self) -> None:
        acceleration = torch.tensor(((0.0, 0.0, -9.81),), dtype=DTYPE)
        angular_velocity = torch.zeros_like(acceleration)
        duration = torch.tensor((1.0,), dtype=DTYPE)
        gravity = torch.tensor((0.0, 0.0, 9.81), dtype=DTYPE)
        earth_rate = torch.zeros(3, dtype=DTYPE)
        reference = propagate_imu_sequence(
            zero_state(),
            acceleration,
            angular_velocity,
            duration,
            gravity,
            earth_rate,
        )
        zero_residual = independent_physics_residual(
            reference,
            zero_state(),
            acceleration,
            angular_velocity,
            duration,
            gravity,
            earth_rate,
        )
        self.assertTrue(torch.equal(zero_residual.vector, torch.zeros(9, dtype=DTYPE)))

        attitude_error = torch.tensor((0.03, -0.02, 0.01), dtype=DTYPE)
        free_prediction = MeanState(
            position_n_m=reference.position_n_m + torch.tensor((1.0, 2.0, 3.0), dtype=DTYPE),
            velocity_n_mps=reference.velocity_n_mps + torch.tensor((-1.0, 0.5, 0.25), dtype=DTYPE),
            rotation_n_from_b=reference.rotation_n_from_b @ so3_exp(attitude_error),
        )
        residual = independent_physics_residual(
            free_prediction,
            zero_state(),
            acceleration,
            angular_velocity,
            duration,
            gravity,
            earth_rate,
        )
        expected = torch.cat(
            (
                torch.tensor((1.0, 2.0, 3.0), dtype=DTYPE),
                torch.tensor((-1.0, 0.5, 0.25), dtype=DTYPE),
                attitude_error,
            )
        )
        self.assertTrue(torch.allclose(residual.vector, expected, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.equal(residual.inertial_reference.position_n_m, reference.position_n_m))

    def test_physics_residual_is_differentiable_with_respect_to_free_prediction(self) -> None:
        predicted_position = torch.tensor((0.2, -0.1, 0.3), dtype=DTYPE, requires_grad=True)
        predicted_rotation_vector = torch.tensor(
            (0.02, 0.01, -0.03),
            dtype=DTYPE,
            requires_grad=True,
        )
        predicted = MeanState(
            position_n_m=predicted_position,
            velocity_n_mps=torch.zeros(3, dtype=DTYPE, requires_grad=True),
            rotation_n_from_b=so3_exp(predicted_rotation_vector),
        )
        residual = independent_physics_residual(
            predicted,
            zero_state(),
            torch.tensor(((0.0, 0.0, -9.81),), dtype=DTYPE),
            torch.zeros((1, 3), dtype=DTYPE),
            torch.ones(1, dtype=DTYPE),
            torch.tensor((0.0, 0.0, 9.81), dtype=DTYPE),
            torch.zeros(3, dtype=DTYPE),
        ).vector
        residual.square().sum().backward()
        self.assertTrue(torch.all(torch.isfinite(predicted_position.grad)))
        self.assertTrue(torch.all(torch.isfinite(predicted_rotation_vector.grad)))


if __name__ == "__main__":
    unittest.main()
