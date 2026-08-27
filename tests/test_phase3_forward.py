from __future__ import annotations

import unittest

import numpy as np

from rtkfree_equivariant_gnss_ins.phase3_eskf import EskfNoiseDensities
from rtkfree_equivariant_gnss_ins.phase3_forward import (
    ForwardEskfState,
    GnssOutage,
    apply_forward_position_event,
    propagate_forward_eskf,
)
from rtkfree_equivariant_gnss_ins.phase3_ins import InsState


LEVEL_NORTH_FACING_ROTATION_N_FROM_B = np.array(
    ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    dtype=np.float64,
)
NOISE = EskfNoiseDensities(0.02, 0.002, 0.0002, 0.00002)
LEVER_IMU_M = np.array((0.0, -0.86, 0.31))
GRAVITY_N_MPS2 = np.array((0.0, 0.0, 9.81))


def initial_filter() -> ForwardEskfState:
    return ForwardEskfState(
        timestamp_ns_utc=0,
        nominal=InsState(
            position_n_m=np.zeros(3),
            velocity_n_mps=np.zeros(3),
            rotation_n_from_b=LEVEL_NORTH_FACING_ROTATION_N_FROM_B.copy(),
            accelerometer_bias_b_mps2=np.zeros(3),
            gyroscope_bias_b_radps=np.zeros(3),
        ),
        covariance=np.diag([1.0] * 3 + [0.1] * 3 + [0.01] * 9),
    )


def run_stationary_outage(duration_s: int) -> tuple[ForwardEskfState, list[int], float, float]:
    state = initial_filter()
    outage = GnssOutage(10_000_000_000, duration_s * 1_000_000_000)
    masked_seconds: list[int] = []
    trace_before_recovery = 0.0
    trace_after_recovery = 0.0
    true_antenna_position = LEVEL_NORTH_FACING_ROTATION_N_FROM_B @ LEVER_IMU_M
    end_s = 10 + duration_s + 5
    for tenth in range(1, end_s * 10 + 1):
        state = propagate_forward_eskf(
            state,
            tenth * 100_000_000,
            (0.0, 0.0, 9.81),
            (0.0, 0.0, 0.0),
            GRAVITY_N_MPS2,
            NOISE,
        )
        if tenth % 10 != 0:
            continue
        second = tenth // 10
        if second == 10 + duration_s:
            trace_before_recovery = float(np.trace(state.covariance))
        event = apply_forward_position_event(
            state,
            true_antenna_position,
            np.eye(3),
            LEVER_IMU_M,
            (outage,),
        )
        state = event.filter_state
        if event.masked:
            masked_seconds.append(second)
        if second == 10 + duration_s:
            trace_after_recovery = float(np.trace(state.covariance))
            assert event.update_applied
    return state, masked_seconds, trace_before_recovery, trace_after_recovery


class Phase3ForwardTests(unittest.TestCase):
    def test_outage_interval_is_left_closed_right_open(self) -> None:
        outage = GnssOutage(100, 20)

        self.assertFalse(outage.contains(99))
        self.assertTrue(outage.contains(100))
        self.assertTrue(outage.contains(119))
        self.assertFalse(outage.contains(120))

    def test_twenty_and_thirty_second_bridges_are_causal_and_deterministic(self) -> None:
        for duration_s in (20, 30):
            with self.subTest(duration_s=duration_s):
                first = run_stationary_outage(duration_s)
                second = run_stationary_outage(duration_s)

                expected_masked = list(range(10, 10 + duration_s))
                self.assertEqual(first[1], expected_masked)
                self.assertEqual(second[1], expected_masked)
                self.assertTrue(
                    np.array_equal(first[0].nominal.position_n_m, second[0].nominal.position_n_m)
                )
                self.assertTrue(np.array_equal(first[0].covariance, second[0].covariance))
                np.testing.assert_allclose(first[0].nominal.position_n_m, np.zeros(3), atol=1e-12)
                np.testing.assert_allclose(first[0].nominal.velocity_n_mps, np.zeros(3), atol=1e-12)
                self.assertGreater(first[2], first[3])
                self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(first[0].covariance))), -1e-10)

    def test_noncausal_or_duplicate_propagation_time_is_rejected(self) -> None:
        state = initial_filter()
        for invalid_time in (0, -1):
            with self.subTest(invalid_time=invalid_time):
                with self.assertRaisesRegex(ValueError, "strictly later"):
                    propagate_forward_eskf(
                        state,
                        invalid_time,
                        (0.0, 0.0, 9.81),
                        (0.0, 0.0, 0.0),
                        GRAVITY_N_MPS2,
                        NOISE,
                    )


if __name__ == "__main__":
    unittest.main()
