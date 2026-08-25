from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from rtkfree_equivariant_gnss_ins.imu_diagnostics import (
    align_imu_and_gnss_time,
    inspect_imu_csv,
)


class ImuDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="rtkfree-imu-diagnostics-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_scan_checks_sequence_rate_values_and_frame(self) -> None:
        path = self.root / "imu.csv"
        fields = [
            "%time",
            "field.header.seq",
            "field.header.stamp",
            "field.header.frame_id",
            "field.orientation.x",
            "field.orientation.y",
            "field.orientation.z",
            "field.orientation.w",
            "field.angular_velocity.x",
            "field.angular_velocity.y",
            "field.angular_velocity.z",
            "field.linear_acceleration.x",
            "field.linear_acceleration.y",
            "field.linear_acceleration.z",
        ]
        start = 1_621_218_775_000_000_000
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(5):
                stamp = start + index * 2_500_000
                writer.writerow(
                    {
                        "%time": stamp + 100_000,
                        "field.header.seq": 100 + index,
                        "field.header.stamp": stamp,
                        "field.header.frame_id": "/imu",
                        "field.orientation.x": 0.0,
                        "field.orientation.y": 0.0,
                        "field.orientation.z": 0.0,
                        "field.orientation.w": 1.0,
                        "field.angular_velocity.x": 0.1,
                        "field.angular_velocity.y": 0.2,
                        "field.angular_velocity.z": 0.3,
                        "field.linear_acceleration.x": 0.0,
                        "field.linear_acceleration.y": 0.0,
                        "field.linear_acceleration.z": 9.8,
                    }
                )

        report = inspect_imu_csv(path)

        self.assertEqual(report["row_count"], 5)
        self.assertEqual(report["frame_ids"], ["/imu"])
        self.assertEqual(report["sequence_discontinuity_count"], 0)
        self.assertEqual(report["sensor_interval_ns"]["median"], 2_500_000.0)
        self.assertEqual(report["nominal_rate_hz"], 400.0)
        self.assertEqual(report["required_vector_missing_value_count"], 0)
        self.assertEqual(report["required_vector_nonfinite_value_count"], 0)
        self.assertEqual(report["quaternion_norm_error_max"], 0.0)

    def test_alignment_uses_navigation_leap_seconds(self) -> None:
        imu = {
            "sensor_stamp_first_ns": 1_621_218_775_500_000_000,
            "sensor_stamp_last_ns": 1_621_219_561_000_000_000,
        }
        observation = {
            "first_epoch": "2021-05-17T02:33:13.000000",
            "last_epoch": "2021-05-17T02:46:18.000000",
        }
        navigation = {"header": {"leap_seconds": 18}}

        report = align_imu_and_gnss_time(imu, observation, navigation)

        self.assertEqual(report["gps_minus_utc_s_from_broadcast_navigation"], 18)
        self.assertEqual(report["imu_starts_after_gnss_s"], 0.5)
        self.assertEqual(report["imu_ends_after_gnss_s"], 1.0)
        self.assertEqual(report["overlap_duration_s"], 784.5)


if __name__ == "__main__":
    unittest.main()
