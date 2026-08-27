from __future__ import annotations

import unittest

import numpy as np

from rtkfree_equivariant_gnss_ins.phase3_io import ImuRecord
from rtkfree_equivariant_gnss_ins.phase4_data import _partition_imu_intervals


def imu(timestamp_ns_utc: int, value: float, timing_gap: bool = False) -> ImuRecord:
    return ImuRecord(
        timestamp_ns_utc=timestamp_ns_utc,
        angular_velocity_b_radps=np.full(3, -value, dtype=np.float64),
        linear_acceleration_b_mps2=np.full(3, value, dtype=np.float64),
        timing_gap=timing_gap,
        within_common_interval=True,
    )


class Phase4DataTests(unittest.TestCase):
    def test_imu_partition_uses_only_arrived_zero_order_held_samples(self) -> None:
        records = iter(
            (
                imu(0, 1.0),
                imu(400_000_000, 2.0),
                imu(1_000_000_000, 3.0),
                imu(1_400_000_000, 4.0, timing_gap=True),
                imu(2_000_000_000, 5.0),
            )
        )
        intervals = _partition_imu_intervals(
            records,
            0,
            (1_000_000_000, 2_000_000_000),
        )
        first_acceleration, _, first_dt, first_gap = intervals[0]
        second_acceleration, _, second_dt, second_gap = intervals[1]
        self.assertTrue(np.array_equal(first_acceleration[:, 0], (1.0, 2.0)))
        self.assertTrue(np.allclose(first_dt, (0.4, 0.6), atol=1e-15, rtol=0.0))
        self.assertFalse(first_gap)
        self.assertTrue(np.array_equal(second_acceleration[:, 0], (3.0, 4.0)))
        self.assertTrue(np.allclose(second_dt, (0.4, 0.6), atol=1e-15, rtol=0.0))
        self.assertTrue(second_gap)

    def test_partition_rejects_noncausal_or_incomplete_student_steps(self) -> None:
        with self.assertRaises(ValueError):
            _partition_imu_intervals(iter((imu(0, 1.0),)), 0, (0,))
        with self.assertRaises(ValueError):
            _partition_imu_intervals(
                iter((imu(0, 1.0), imu(500_000_000, 2.0))),
                0,
                (900_000_000,),
            )


if __name__ == "__main__":
    unittest.main()
