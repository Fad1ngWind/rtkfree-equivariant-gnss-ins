from __future__ import annotations

import unittest

import numpy as np

from rtkfree_equivariant_gnss_ins.gps_spp import GPS_OMEGA_E
from rtkfree_equivariant_gnss_ins.phase3_ins import InsState, propagate_ins


LEVEL_NORTH_FACING_ROTATION_N_FROM_B = np.array(
    ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    dtype=np.float64,
)
GRAVITY_N_MPS2 = np.array((0.0, 0.0, 9.81), dtype=np.float64)


def zero_state() -> InsState:
    return InsState(
        position_n_m=np.zeros(3),
        velocity_n_mps=np.zeros(3),
        rotation_n_from_b=LEVEL_NORTH_FACING_ROTATION_N_FROM_B.copy(),
        accelerometer_bias_b_mps2=np.zeros(3),
        gyroscope_bias_b_radps=np.zeros(3),
    )


class Phase3InsTests(unittest.TestCase):
    def test_stationary_fixture_cancels_specific_force_and_gravity(self) -> None:
        state = zero_state()
        for _ in range(4_000):
            state = propagate_ins(
                state,
                np.array((0.0, 0.0, 9.81)),
                np.zeros(3),
                GRAVITY_N_MPS2,
                0.0025,
            )

        np.testing.assert_allclose(state.position_n_m, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(state.velocity_n_mps, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(
            state.rotation_n_from_b,
            LEVEL_NORTH_FACING_ROTATION_N_FROM_B,
            atol=1e-12,
        )

    def test_constant_navigation_acceleration_matches_closed_form_motion(self) -> None:
        state = zero_state()
        # Body y is forward and maps to NED north for this level fixture.
        measured_specific_force_b = np.array((0.0, 1.0, 9.81))
        for _ in range(400):
            state = propagate_ins(
                state,
                measured_specific_force_b,
                np.zeros(3),
                GRAVITY_N_MPS2,
                0.01,
            )

        np.testing.assert_allclose(state.velocity_n_mps, (4.0, 0.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(state.position_n_m, (8.0, 0.0, 0.0), atol=1e-12)

    def test_stationary_ground_attitude_cancels_measured_earth_rotation(self) -> None:
        state = zero_state()
        earth_rate_n = np.array((GPS_OMEGA_E, 0.0, 0.0))
        measured_earth_rate_b = state.rotation_n_from_b.T @ earth_rate_n
        for _ in range(1_000):
            state = propagate_ins(
                state,
                np.array((0.0, 0.0, 9.81)),
                measured_earth_rate_b,
                GRAVITY_N_MPS2,
                0.1,
                earth_rate_n,
            )

        np.testing.assert_allclose(
            state.rotation_n_from_b,
            LEVEL_NORTH_FACING_ROTATION_N_FROM_B,
            atol=1e-12,
        )
        np.testing.assert_allclose(state.velocity_n_mps, np.zeros(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
