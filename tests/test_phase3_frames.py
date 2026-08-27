from __future__ import annotations

import unittest

from rtkfree_equivariant_gnss_ins.phase3_frames import (
    antenna_lever_in_imu,
    antenna_position_from_imu,
    imu_position_from_antenna,
)


IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


class Phase3FrameTests(unittest.TestCase):
    def test_selected_transform_has_expected_inverse_lever_sign(self) -> None:
        lever = antenna_lever_in_imu(IDENTITY, (0.0, 0.86, -0.31))

        self.assertEqual(lever, (0.0, -0.86, 0.31))
        self.assertLess(lever[1], 0.0)  # antenna is behind the IMU
        self.assertGreater(lever[2], 0.0)  # antenna is above the IMU

    def test_lever_is_rotated_before_position_composition_and_round_trips(self) -> None:
        rotation_navigation_from_imu = (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        imu_position = (10.0, 20.0, 30.0)
        lever_imu = (0.0, -0.86, 0.31)

        antenna_position = antenna_position_from_imu(
            imu_position,
            rotation_navigation_from_imu,
            lever_imu,
        )
        recovered_imu_position = imu_position_from_antenna(
            antenna_position,
            rotation_navigation_from_imu,
            lever_imu,
        )

        self.assertEqual(antenna_position, (10.86, 20.0, 30.31))
        self.assertEqual(recovered_imu_position, imu_position)


if __name__ == "__main__":
    unittest.main()
