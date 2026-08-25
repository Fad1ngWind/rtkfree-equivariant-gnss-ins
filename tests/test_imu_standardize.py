from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rtkfree_equivariant_gnss_ins.imu_standardize import standardize_medium_imu


class ImuStandardizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="rtkfree-imu-standardize-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_standardization_preserves_rows_units_and_quality_flags(self) -> None:
        commit = "a" * 40
        session_root = self.root / "deployable" / "UrbanNav" / commit / "Session"
        source_path = session_root / "imu" / "imu.csv"
        source_path.parent.mkdir(parents=True)
        base_fields = [
            "%time",
            "field.header.seq",
            "field.header.stamp",
            "field.header.frame_id",
            *[f"field.orientation.{axis}" for axis in ("x", "y", "z", "w")],
            *[f"field.orientation_covariance{index}" for index in range(9)],
            *[f"field.angular_velocity.{axis}" for axis in ("x", "y", "z")],
            *[f"field.angular_velocity_covariance{index}" for index in range(9)],
            *[f"field.linear_acceleration.{axis}" for axis in ("x", "y", "z")],
            *[f"field.linear_acceleration_covariance{index}" for index in range(9)],
        ]
        start = 1_621_218_775_000_000_000
        stamps = [start, start + 2_500_000, start + 8_500_000]
        with source_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=base_fields)
            writer.writeheader()
            for index, stamp in enumerate(stamps):
                row = {field: "0" for field in base_fields}
                row.update(
                    {
                        "%time": str(stamp + 100_000),
                        "field.header.seq": str(10 + index),
                        "field.header.stamp": str(stamp),
                        "field.header.frame_id": "/imu",
                        "field.orientation.w": "1",
                        "field.angular_velocity.x": "0.1",
                        "field.linear_acceleration.z": "9.8",
                    }
                )
                writer.writerow(row)
        source_bytes = source_path.read_bytes()
        source_lock = {
            "dataset": "UrbanNav",
            "session": "Session",
            "repository_commit": commit,
            "artifacts": [
                {
                    "id": "low_cost_imu",
                    "local_name": "imu.csv",
                    "observed_byte_size": len(source_bytes),
                    "observed_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "deployable_validation": {
                        "row_count": 3,
                        "median_interval_ns": 2_500_000,
                        "intervals_over_2x_median_count": 1,
                    },
                }
            ],
        }
        source_lock_path = self.root / "source.json"
        source_lock_path.write_text(json.dumps(source_lock), encoding="utf-8")
        profile = {"profile_id": "gps_l1ca_broadcast_spp_v1"}
        profile_path = self.root / "profile.json"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        pvt_path = (
            self.root
            / "standardized"
            / "UrbanNav"
            / commit
            / "Session"
            / profile["profile_id"]
            / "pvt.jsonl"
        )
        pvt_path.parent.mkdir(parents=True)
        pvt_path.write_text(
            json.dumps({"timestamp_ns_utc": start})
            + "\n"
            + json.dumps({"timestamp_ns_utc": start + 10_000_000})
            + "\n",
            encoding="utf-8",
        )

        output_path, summary_path = standardize_medium_imu(
            source_lock_path, profile_path, self.root
        )

        with output_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["angular_velocity_radps_x"], "0.1")
        self.assertEqual(rows[0]["linear_acceleration_mps2_z"], "9.8")
        self.assertEqual([row["timing_gap"] for row in rows], ["false", "false", "true"])
        self.assertTrue(all(row["within_common_interval"] == "true" for row in rows))
        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["timing_gap_count"], 1)
        self.assertFalse(summary["interpolation_applied"])
        self.assertFalse(summary["ins_or_filter_applied"])


if __name__ == "__main__":
    unittest.main()
