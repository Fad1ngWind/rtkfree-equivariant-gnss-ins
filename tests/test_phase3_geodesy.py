from __future__ import annotations

import unittest

import numpy as np

from rtkfree_equivariant_gnss_ins.gps_spp import WGS84_A
from rtkfree_equivariant_gnss_ins.phase3_geodesy import (
    earth_rotation_n_radps,
    ecef_covariance_to_ned,
    ecef_position_to_ned,
    make_local_ned_frame,
    ned_position_to_ecef,
    normal_gravity_n_mps2,
)


class Phase3GeodesyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.origin_ecef_m = np.array((WGS84_A, 0.0, 0.0))
        self.frame = make_local_ned_frame(self.origin_ecef_m)

    def test_equator_zero_longitude_axes_and_position_round_trip(self) -> None:
        np.testing.assert_allclose(
            self.frame.rotation_n_from_e,
            ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
            atol=1e-15,
        )
        offset_ecef_m = np.array((2.0, 3.0, 4.0))
        position_ecef_m = self.origin_ecef_m + offset_ecef_m

        position_ned_m = ecef_position_to_ned(self.frame, position_ecef_m)
        recovered_ecef_m = ned_position_to_ecef(self.frame, position_ned_m)

        np.testing.assert_allclose(position_ned_m, (4.0, 3.0, -2.0), atol=1e-12)
        np.testing.assert_allclose(recovered_ecef_m, position_ecef_m, atol=1e-12)
        np.testing.assert_allclose(
            ecef_position_to_ned(
                self.frame,
                self.origin_ecef_m + np.array((-5.0, 0.0, 0.0)),
            ),
            (0.0, 0.0, 5.0),
            atol=1e-12,
        )

    def test_covariance_rotates_with_the_same_frame_mapping(self) -> None:
        covariance_ecef_m2 = np.diag((4.0, 9.0, 16.0))

        covariance_ned_m2 = ecef_covariance_to_ned(
            self.frame,
            covariance_ecef_m2,
        )

        np.testing.assert_allclose(covariance_ned_m2, np.diag((16.0, 9.0, 4.0)))
        np.testing.assert_allclose(covariance_ned_m2, covariance_ned_m2.T)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance_ned_m2) > 0.0))

    def test_equator_earth_rate_and_normal_gravity_have_ned_signs(self) -> None:
        earth_rate = earth_rotation_n_radps(self.frame)
        gravity = normal_gravity_n_mps2(self.frame)

        self.assertGreater(earth_rate[0], 0.0)
        np.testing.assert_allclose(earth_rate[1:], (0.0, 0.0), atol=1e-20)
        np.testing.assert_allclose(gravity, (0.0, 0.0, 9.7803253359), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
