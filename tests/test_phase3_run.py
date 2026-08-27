from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rtkfree_equivariant_gnss_ins.phase3_config import FixedEskfConfig
from rtkfree_equivariant_gnss_ins.phase3_eskf import EskfNoiseDensities
from rtkfree_equivariant_gnss_ins.phase3_geodesy import (
    make_local_ned_frame,
    ned_position_to_ecef,
    normal_gravity_n_mps2,
)
from rtkfree_equivariant_gnss_ins.phase3_io import sha256_file
from rtkfree_equivariant_gnss_ins.phase3_run import run_deployable_baselines
from rtkfree_equivariant_gnss_ins.phase3_validate import validate_baseline_outputs


class Phase3DeployableRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="rtkfree-phase3-run-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_fixture(self) -> tuple[Path, Path, Path, Path]:
        pvt_path = self.root / "pvt.jsonl"
        imu_path = self.root / "imu.csv"
        extrinsic_path = self.root / "gnss_imu_extrinsic.json"
        noise_parameter_path = self.root / "xsens_imu_param.yaml"

        origin_ecef = np.array((6_378_137.0, 0.0, 0.0))
        frame = make_local_ned_frame(origin_ecef)
        covariance = np.eye(3).reshape(-1).tolist()
        pvt_lines = []
        for second in range(176):
            antenna_n = np.array((4.0 * second, 0.0, 0.0))
            pvt_lines.append(
                json.dumps(
                    {
                        "timestamp_ns_utc": second * 1_000_000_000,
                        "coordinate_frame": "WGS84_ECEF_broadcast",
                        "within_common_interval": True,
                        "solution_valid": True,
                        "solution_status": "valid",
                        "ecef_position_m": ned_position_to_ecef(
                            frame, antenna_n
                        ).tolist(),
                        "position_covariance_ecef_m2": covariance,
                        "covariance_valid": True,
                    },
                    sort_keys=True,
                )
            )
        pvt_path.write_text("\n".join(pvt_lines) + "\n", encoding="utf-8")

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
        gravity = float(normal_gravity_n_mps2(frame)[2])
        earth_rate = 7.2921151467e-5
        with imu_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(176 * 5):
                writer.writerow(
                    {
                        "timestamp_ns_utc": index * 200_000_000,
                        "frame_id": "/imu",
                        "angular_velocity_radps_x": 0.0,
                        "angular_velocity_radps_y": earth_rate,
                        "angular_velocity_radps_z": 0.0,
                        "linear_acceleration_mps2_x": 0.0,
                        "linear_acceleration_mps2_y": 0.0,
                        "linear_acceleration_mps2_z": gravity,
                        "valid": "true",
                        "timing_gap": "false",
                        "within_common_interval": "true",
                    }
                )

        extrinsic_path.write_text(
            json.dumps(
                {
                    "selected_key": "ANTENNA_T_IMU",
                    "translation_unit": "m",
                    "matrix_4x4_row_major": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.86],
                        [0.0, 0.0, 1.0, -0.31],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        noise_parameter_path.write_text("fixed synthetic noise fixture\n", encoding="utf-8")
        return pvt_path, imu_path, extrinsic_path, noise_parameter_path

    def _config(
        self,
        pvt_path: Path,
        imu_path: Path,
        extrinsic_path: Path,
        noise_parameter_path: Path,
    ) -> FixedEskfConfig:
        return FixedEskfConfig(
            profile_id="fixed_forward_eskf_v1",
            pvt_sha256=sha256_file(pvt_path),
            imu_sha256=sha256_file(imu_path),
            imu_noise_parameter_sha256=sha256_file(noise_parameter_path),
            extrinsic_sha256=sha256_file(extrinsic_path),
            initialization_window_s=20.0,
            minimum_horizontal_speed_mps=2.0,
            initial_velocity_std_ned_mps=np.array((2.0, 2.0, 5.0)),
            initial_attitude_std_body_rad=np.deg2rad((10.0, 10.0, 30.0)),
            initial_accelerometer_bias_std_b_mps2=np.full(3, 0.1),
            initial_gyroscope_bias_std_b_radps=np.full(3, 0.02),
            noise=EskfNoiseDensities(0.011, 0.010, 0.0001, 0.00009),
            outage_start_after_initialization_s=120,
            outage_durations_s=(20, 30),
        )

    def test_full_synthetic_replay_is_deterministic_and_counts_outages(self) -> None:
        pvt_path, imu_path, extrinsic_path, noise_parameter_path = self._write_fixture()
        config = self._config(
            pvt_path, imu_path, extrinsic_path, noise_parameter_path
        )
        first = self.root / "run-a"
        second = self.root / "run-b"

        first_summary_path = run_deployable_baselines(
            config,
            pvt_path,
            imu_path,
            extrinsic_path,
            noise_parameter_path,
            first,
        )
        second_summary_path = run_deployable_baselines(
            config,
            pvt_path,
            imu_path,
            extrinsic_path,
            noise_parameter_path,
            second,
        )

        self.assertEqual(first_summary_path.read_bytes(), second_summary_path.read_bytes())
        summary = json.loads(first_summary_path.read_text(encoding="utf-8"))
        self.assertFalse(summary["high_precision_reference_used"])
        self.assertFalse(summary["receiver_native_pvt_used_as_truth"])
        self.assertEqual(summary["fixed_eskf_nis"]["count"], 155)
        self.assertEqual(summary["fixed_eskf_nis"]["degrees_of_freedom"], 3)
        self.assertGreaterEqual(
            summary["fixed_eskf_nis"]["above_chi_square_95_fraction"], 0.0
        )
        self.assertLessEqual(
            summary["fixed_eskf_nis"]["above_chi_square_99_fraction"], 1.0
        )
        expected_counts = {
            "ins_only": (155, 0, 0),
            "fixed_eskf": (155, 155, 0),
            "fixed_eskf_outage_20s": (155, 135, 20),
            "fixed_eskf_outage_30s": (155, 125, 30),
        }
        for name, (records, updates, masked) in expected_counts.items():
            actual = summary["scenario_counts"][name]
            self.assertEqual(actual["record_count"], records)
            self.assertEqual(actual["position_update_count"], updates)
            self.assertEqual(actual["masked_valid_pvt_count"], masked)
            output_name = f"{name}.jsonl"
            self.assertEqual(
                sha256_file(first / output_name),
                sha256_file(second / output_name),
            )
        self.assertEqual(
            len((first / "spp_only.jsonl").read_text(encoding="utf-8").splitlines()),
            176,
        )
        report = validate_baseline_outputs(config, first, second)
        self.assertTrue(report["deterministic_comparison"])
        self.assertGreater(report["minimum_covariance_eigenvalue"], 0.0)

        with (second / "ins_only.jsonl").open("ab") as handle:
            handle.write(b" ")
        with self.assertRaisesRegex(ValueError, "independent runs differ"):
            validate_baseline_outputs(config, first, second)

        noise_parameter_path.write_text("tampered synthetic noise fixture\n", encoding="utf-8")
        rejected_output = self.root / "run-rejected"
        with self.assertRaisesRegex(ValueError, "IMU noise parameters differ"):
            run_deployable_baselines(
                config,
                pvt_path,
                imu_path,
                extrinsic_path,
                noise_parameter_path,
                rejected_output,
            )
        self.assertFalse(rejected_output.exists())


if __name__ == "__main__":
    unittest.main()
