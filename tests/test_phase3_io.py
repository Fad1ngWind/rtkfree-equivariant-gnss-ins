from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rtkfree_equivariant_gnss_ins.phase3_io import (
    iter_standardized_imu,
    read_selected_extrinsic,
    read_standardized_pvt,
)


class Phase3IoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="rtkfree-phase3-io-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_pvt_reader_preserves_valid_and_invalid_epochs(self) -> None:
        path = self.root / "pvt.jsonl"
        common = {
            "coordinate_frame": "WGS84_ECEF_broadcast",
            "within_common_interval": True,
        }
        valid = {
            **common,
            "timestamp_ns_utc": 1,
            "solution_valid": True,
            "solution_status": "valid",
            "ecef_position_m": [1.0, 2.0, 3.0],
            "position_covariance_ecef_m2": [
                4.0, 1.0e-10, 0.0, 2.0e-10, 5.0, 0.0, 0.0, 0.0, 6.0
            ],
            "covariance_valid": True,
        }
        invalid = {
            **common,
            "timestamp_ns_utc": 2,
            "solution_valid": False,
            "solution_status": "insufficient_c1c_observations",
            "ecef_position_m": None,
            "position_covariance_ecef_m2": None,
            "covariance_valid": False,
        }
        path.write_text(
            json.dumps(valid) + "\n" + json.dumps(invalid) + "\n",
            encoding="utf-8",
        )

        records = read_standardized_pvt(path)

        self.assertEqual(len(records), 2)
        np.testing.assert_allclose(records[0].position_ecef_m, (1.0, 2.0, 3.0))
        np.testing.assert_array_equal(
            records[0].covariance_ecef_m2,
            records[0].covariance_ecef_m2.T,
        )
        self.assertEqual(records[0].covariance_ecef_m2[0, 1], 1.5e-10)
        self.assertIsNone(records[1].position_ecef_m)
        self.assertEqual(records[1].solution_status, "insufficient_c1c_observations")

    def test_imu_reader_preserves_primary_time_and_gap_flags(self) -> None:
        path = self.root / "imu.csv"
        fields = [
            "timestamp_ns_utc",
            "frame_id",
            "angular_velocity_radps_x",
            "angular_velocity_radps_y",
            "angular_velocity_radps_z",
            "linear_acceleration_mps2_x",
            "linear_acceleration_mps2_y",
            "linear_acceleration_mps2_z",
            "valid",
            "timing_gap",
            "within_common_interval",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for timestamp, gap in ((10, "false"), (20, "true")):
                writer.writerow(
                    {
                        "timestamp_ns_utc": timestamp,
                        "frame_id": "/imu",
                        "angular_velocity_radps_x": 0.1,
                        "angular_velocity_radps_y": 0.2,
                        "angular_velocity_radps_z": 0.3,
                        "linear_acceleration_mps2_x": 1.0,
                        "linear_acceleration_mps2_y": 2.0,
                        "linear_acceleration_mps2_z": 3.0,
                        "valid": "true",
                        "timing_gap": gap,
                        "within_common_interval": "true",
                    }
                )

        records = tuple(iter_standardized_imu(path))

        self.assertEqual([record.timestamp_ns_utc for record in records], [10, 20])
        self.assertEqual([record.timing_gap for record in records], [False, True])
        np.testing.assert_allclose(records[0].angular_velocity_b_radps, (0.1, 0.2, 0.3))

    def test_selected_extrinsic_reader_reproduces_verified_lever(self) -> None:
        path = self.root / "gnss_imu_extrinsic.json"
        record = {
            "selected_key": "ANTENNA_T_IMU",
            "translation_unit": "m",
            "matrix_4x4_row_major": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.86],
                [0.0, 0.0, 1.0, -0.31],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
        content = json.dumps(record).encode("utf-8")
        path.write_bytes(content)

        extrinsic = read_selected_extrinsic(
            path,
            hashlib.sha256(content).hexdigest(),
        )

        np.testing.assert_allclose(extrinsic.antenna_lever_imu_m, (0.0, -0.86, 0.31))


if __name__ == "__main__":
    unittest.main()
