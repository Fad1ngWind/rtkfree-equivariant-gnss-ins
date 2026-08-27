from __future__ import annotations

import unittest

import numpy as np

from rtkfree_equivariant_gnss_ins.gps_spp import WGS84_A
from rtkfree_equivariant_gnss_ins.phase3_geodesy import make_local_ned_frame, ned_position_to_ecef
from rtkfree_equivariant_gnss_ins.phase3_initialization import initialize_from_causal_window


class Phase3InitializationTests(unittest.TestCase):
    def test_straight_motion_initializes_only_at_window_end(self) -> None:
        start_ns = 1_000_000_000_000
        pvt_times = start_ns + np.arange(11, dtype=np.int64) * 1_000_000_000
        origin_ecef = np.array((WGS84_A, 0.0, 0.0))
        frame = make_local_ned_frame(origin_ecef)
        pvt_positions = np.vstack(
            [ned_position_to_ecef(frame, (5.0 * second, 0.0, 0.0)) for second in range(11)]
        )
        imu_times = start_ns + np.arange(4_001, dtype=np.int64) * 2_500_000
        imu_acceleration = np.tile((0.0, 0.0, 9.81), (imu_times.size, 1))

        result = initialize_from_causal_window(
            pvt_times,
            pvt_positions,
            imu_times,
            imu_acceleration,
            antenna_lever_imu_m=(0.0, -0.86, 0.31),
            minimum_window_s=10.0,
            minimum_horizontal_speed_mps=2.0,
        )

        self.assertEqual(result.timestamp_ns_utc, int(pvt_times[-1]))
        np.testing.assert_allclose(result.state.velocity_n_mps, (5.0, 0.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(
            result.fitted_pvt_velocity_n_mps,
            (5.0, 0.0, 0.0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.state.rotation_n_from_b,
            ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.state.position_n_m,
            (50.86, 0.0, 0.31),
            atol=1e-12,
        )

    def test_vertical_pvt_slope_is_diagnostic_not_nominal_velocity(self) -> None:
        start_ns = 1_000_000_000_000
        pvt_times = start_ns + np.arange(11, dtype=np.int64) * 1_000_000_000
        origin_ecef = np.array((WGS84_A, 0.0, 0.0))
        frame = make_local_ned_frame(origin_ecef)
        pvt_positions = np.vstack(
            [
                ned_position_to_ecef(frame, (5.0 * second, 0.0, 2.0 * second))
                for second in range(11)
            ]
        )
        imu_times = start_ns + np.arange(4_001, dtype=np.int64) * 2_500_000
        imu_acceleration = np.tile((0.0, 0.0, 9.81), (imu_times.size, 1))

        result = initialize_from_causal_window(
            pvt_times,
            pvt_positions,
            imu_times,
            imu_acceleration,
            antenna_lever_imu_m=(0.0, -0.86, 0.31),
            minimum_window_s=10.0,
            minimum_horizontal_speed_mps=2.0,
        )

        np.testing.assert_allclose(
            result.fitted_pvt_velocity_n_mps,
            (5.0, 0.0, 2.0),
            atol=1e-12,
        )
        np.testing.assert_allclose(result.state.velocity_n_mps, (5.0, 0.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(
            result.state.position_n_m,
            (50.86, 0.0, 20.31),
            atol=1e-12,
        )

    def test_short_or_slow_window_cannot_claim_initialized_state(self) -> None:
        origin = np.array((WGS84_A, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "too short"):
            initialize_from_causal_window(
                (0, 1_000_000_000),
                (origin, origin),
                (0, 1_000_000_000),
                ((0.0, 0.0, 9.81), (0.0, 0.0, 9.81)),
                (0.0, -0.86, 0.31),
                minimum_window_s=10.0,
                minimum_horizontal_speed_mps=2.0,
            )
        with self.assertRaisesRegex(ValueError, "insufficient"):
            initialize_from_causal_window(
                (0, 10_000_000_000),
                (origin, origin),
                (0, 10_000_000_000),
                ((0.0, 0.0, 9.81), (0.0, 0.0, 9.81)),
                (0.0, -0.86, 0.31),
                minimum_window_s=10.0,
                minimum_horizontal_speed_mps=2.0,
            )


if __name__ == "__main__":
    unittest.main()
