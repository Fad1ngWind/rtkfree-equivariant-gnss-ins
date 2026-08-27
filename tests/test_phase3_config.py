from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from rtkfree_equivariant_gnss_ins.phase3_config import (
    load_fixed_eskf_config,
    make_initial_covariance,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "phase3" / "fixed_eskf_v1.json"


class Phase3ConfigTests(unittest.TestCase):
    def test_fixed_profile_has_one_noise_derivation_and_no_adaptation(self) -> None:
        config = load_fixed_eskf_config(CONFIG_PATH)

        self.assertEqual(config.initialization_window_s, 20.0)
        self.assertEqual(config.minimum_horizontal_speed_mps, 2.0)
        self.assertEqual(config.outage_durations_s, (20, 30))
        self.assertEqual(
            config.imu_noise_parameter_sha256,
            "55da278a30572015d2ab4024ae1dfc11abb1fb54f252498ce318152c6d39ecd5",
        )
        self.assertEqual(
            config.noise.accelerometer_white_noise_mps2_sqrt_s,
            1.1197412605492375e-02,
        )
        self.assertEqual(
            config.noise.gyroscope_white_noise_radps_sqrt_s,
            1.0270904839480961e-02,
        )
        self.assertEqual(
            config.noise.accelerometer_bias_random_walk_mps2_per_sqrt_s,
            1.1751767903346351e-04,
        )
        self.assertEqual(
            config.noise.gyroscope_bias_random_walk_radps_per_sqrt_s,
            9.1355383994881894e-05,
        )

    def test_initial_covariance_uses_pvt_position_and_fixed_other_blocks(self) -> None:
        config = load_fixed_eskf_config(CONFIG_PATH)
        position_covariance = np.diag((4.0, 9.0, 16.0))

        covariance = make_initial_covariance(config, position_covariance)

        np.testing.assert_allclose(covariance[0:3, 0:3], position_covariance)
        np.testing.assert_allclose(np.sqrt(np.diag(covariance)[3:6]), (2.0, 2.0, 5.0))
        np.testing.assert_allclose(covariance, covariance.T, atol=0.0)
        self.assertGreater(float(np.min(np.linalg.eigvalsh(covariance))), 0.0)


if __name__ == "__main__":
    unittest.main()
