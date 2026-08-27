from __future__ import annotations

import unittest

import numpy as np

from rtkfree_equivariant_gnss_ins.gps_spp import GPS_OMEGA_E
from rtkfree_equivariant_gnss_ins.phase3_eskf import (
    ACCELEROMETER_BIAS,
    ATTITUDE,
    EskfNoiseDensities,
    GYROSCOPE_BIAS,
    VELOCITY,
    continuous_error_model,
    position_measurement_model,
    propagate_error_covariance,
    update_position,
)
from rtkfree_equivariant_gnss_ins.phase3_ins import (
    InsState,
    propagate_ins,
    rotation_increment,
)


LEVEL_NORTH_FACING_ROTATION_N_FROM_B = np.array(
    ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    dtype=np.float64,
)
FIXTURE_NOISE = EskfNoiseDensities(
    accelerometer_white_noise_mps2_sqrt_s=0.02,
    gyroscope_white_noise_radps_sqrt_s=0.002,
    accelerometer_bias_random_walk_mps2_per_sqrt_s=0.0002,
    gyroscope_bias_random_walk_radps_per_sqrt_s=0.00002,
)


class Phase3EskfPropagationTests(unittest.TestCase):
    def test_right_attitude_transition_matches_finite_difference_with_earth_rate(
        self,
    ) -> None:
        rotation = rotation_increment(np.array((0.3, -0.2, 0.1)))
        earth_rate_n = np.array((0.8 * GPS_OMEGA_E, 0.0, -0.6 * GPS_OMEGA_E))
        measured_inertial_rate_b = np.array((0.012, -0.007, 0.004))
        dt_s = 1.0e-3
        perturbation = 1.0e-6

        def state_with_rotation(value: np.ndarray) -> InsState:
            return InsState(
                position_n_m=np.zeros(3),
                velocity_n_mps=np.zeros(3),
                rotation_n_from_b=value,
                accelerometer_bias_b_mps2=np.zeros(3),
                gyroscope_bias_b_radps=np.zeros(3),
            )

        nominal_next = propagate_ins(
            state_with_rotation(rotation),
            np.zeros(3),
            measured_inertial_rate_b,
            np.zeros(3),
            dt_s,
            earth_rate_n,
        )
        numerical_transition = np.zeros((3, 3))
        for axis in range(3):
            initial_error = np.zeros(3)
            initial_error[axis] = perturbation
            true_next = propagate_ins(
                state_with_rotation(rotation @ rotation_increment(initial_error)),
                np.zeros(3),
                measured_inertial_rate_b,
                np.zeros(3),
                dt_s,
                earth_rate_n,
            )
            relative_rotation = (
                nominal_next.rotation_n_from_b.T @ true_next.rotation_n_from_b
            )
            numerical_transition[:, axis] = np.array(
                (
                    relative_rotation[2, 1] - relative_rotation[1, 2],
                    relative_rotation[0, 2] - relative_rotation[2, 0],
                    relative_rotation[1, 0] - relative_rotation[0, 1],
                )
            ) / (2.0 * perturbation)

        dynamics, _ = continuous_error_model(
            rotation,
            np.zeros(3),
            measured_inertial_rate_b,
            earth_rate_n,
        )
        numerical_rate = (numerical_transition - np.eye(3)) / dt_s
        np.testing.assert_allclose(
            numerical_rate,
            dynamics[ATTITUDE, ATTITUDE],
            atol=1.0e-7,
            rtol=0.0,
        )

        earth_subtracted_rate = measured_inertial_rate_b - rotation.T @ earth_rate_n
        incorrect_dynamics, _ = continuous_error_model(
            rotation,
            np.zeros(3),
            earth_subtracted_rate,
            earth_rate_n,
        )
        self.assertGreater(
            float(
                np.max(
                    np.abs(
                        numerical_rate - incorrect_dynamics[ATTITUDE, ATTITUDE]
                    )
                )
            ),
            1.0e-5,
        )

    def test_accelerometer_bias_error_has_true_minus_nominal_sign(self) -> None:
        dynamics, _ = continuous_error_model(
            LEVEL_NORTH_FACING_ROTATION_N_FROM_B,
            (0.0, 0.0, 9.81),
            (0.0, 0.0, 0.0),
        )
        error = np.zeros(15)
        error[ACCELEROMETER_BIAS.start + 1] = 0.1

        error_rate = dynamics @ error

        self.assertAlmostEqual(error_rate[VELOCITY.start], -0.1)
        np.testing.assert_allclose(error_rate[VELOCITY][1:], (0.0, 0.0))

    def test_covariance_propagation_is_symmetric_psd_and_deterministic(self) -> None:
        prior = np.diag(
            np.array(
                [10.0] * 3
                + [1.0] * 3
                + [0.1] * 3
                + [0.01] * 3
                + [0.001] * 3
            )
        )
        arguments = (
            prior,
            LEVEL_NORTH_FACING_ROTATION_N_FROM_B,
            (0.0, 0.0, 9.81),
            (0.0, 0.0, 0.0),
            FIXTURE_NOISE,
            0.0025,
        )

        first = propagate_error_covariance(*arguments)
        second = propagate_error_covariance(*arguments)

        self.assertTrue(np.array_equal(first, second))
        np.testing.assert_allclose(first, first.T, atol=0.0)
        self.assertGreater(float(np.min(np.linalg.eigvalsh(first))), 0.0)
        self.assertGreater(float(np.trace(first)), float(np.trace(prior)))

    def test_process_noise_reaches_position_through_interval_dynamics(self) -> None:
        dt_s = 0.01
        propagated = propagate_error_covariance(
            np.zeros((15, 15)),
            LEVEL_NORTH_FACING_ROTATION_N_FROM_B,
            (0.0, 0.0, 9.81),
            (0.0, 0.0, 0.0),
            FIXTURE_NOISE,
            dt_s,
        )

        self.assertTrue(np.all(np.diag(propagated)[0:3] > 0.0))
        np.testing.assert_allclose(
            np.diag(propagated)[ACCELEROMETER_BIAS],
            FIXTURE_NOISE.accelerometer_bias_random_walk_mps2_per_sqrt_s**2
            * dt_s,
        )
        np.testing.assert_allclose(
            np.diag(propagated)[GYROSCOPE_BIAS],
            FIXTURE_NOISE.gyroscope_bias_random_walk_radps_per_sqrt_s**2 * dt_s,
        )
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(propagated))), -1e-12)


class Phase3EskfPositionUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = InsState(
            position_n_m=np.array((10.0, 0.0, 0.0)),
            velocity_n_mps=np.zeros(3),
            rotation_n_from_b=np.eye(3),
            accelerometer_bias_b_mps2=np.zeros(3),
            gyroscope_bias_b_radps=np.zeros(3),
        )

    def test_scalar_equivalent_update_has_expected_gain_and_nis(self) -> None:
        prior = np.eye(15)
        prior[0:3, 0:3] = 4.0 * np.eye(3)

        result = update_position(
            self.state,
            prior,
            measured_antenna_position_n_m=(12.0, 0.0, 0.0),
            measurement_covariance_n_m2=np.eye(3),
            antenna_lever_imu_m=(0.0, 0.0, 0.0),
        )

        self.assertAlmostEqual(result.kalman_gain[0, 0], 0.8)
        self.assertAlmostEqual(result.state.position_n_m[0], 11.6)
        self.assertAlmostEqual(result.nis, 0.8)
        self.assertAlmostEqual(result.covariance[0, 0], 0.8)
        np.testing.assert_allclose(result.covariance, result.covariance.T, atol=0.0)
        self.assertGreater(float(np.min(np.linalg.eigvalsh(result.covariance))), 0.0)

    def test_lever_arm_attitude_jacobian_matches_small_rotation(self) -> None:
        lever = np.array((0.2, -0.5, 0.3))
        predicted, jacobian = position_measurement_model(self.state, lever)
        small_angle = np.array((1e-7, -2e-7, 3e-7))
        perturbed_state = InsState(
            position_n_m=self.state.position_n_m,
            velocity_n_mps=self.state.velocity_n_mps,
            rotation_n_from_b=self.state.rotation_n_from_b
            @ rotation_increment(small_angle),
            accelerometer_bias_b_mps2=self.state.accelerometer_bias_b_mps2,
            gyroscope_bias_b_radps=self.state.gyroscope_bias_b_radps,
        )

        perturbed, _ = position_measurement_model(perturbed_state, lever)

        np.testing.assert_allclose(
            perturbed - predicted,
            jacobian[:, ATTITUDE] @ small_angle,
            atol=1e-13,
        )

    def test_zero_innovation_keeps_nominal_state_and_reduces_uncertainty(self) -> None:
        lever = np.array((0.0, -0.86, 0.31))
        predicted, _ = position_measurement_model(self.state, lever)
        prior = np.eye(15)

        result = update_position(
            self.state,
            prior,
            predicted,
            np.eye(3),
            lever,
        )

        self.assertEqual(result.nis, 0.0)
        np.testing.assert_allclose(result.innovation_n_m, np.zeros(3))
        np.testing.assert_allclose(result.state.position_n_m, self.state.position_n_m)
        np.testing.assert_allclose(result.state.rotation_n_from_b, np.eye(3))
        self.assertLess(float(np.trace(result.covariance)), float(np.trace(prior)))


if __name__ == "__main__":
    unittest.main()
