from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "config" / "phase2" / "urbannav_medium_source_lock.json"


class Phase2SourceLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    def test_required_sources_exist_without_forbidding_later_navigation_source(self) -> None:
        ids = {artifact["id"] for artifact in self.lock["artifacts"]}
        required = {
            "gnss_observation_archive",
            "low_cost_imu",
            "extrinsic_calibration",
            "imu_calibration",
            "broadcast_navigation",
        }
        self.assertTrue(required.issubset(ids))

    def test_repository_version_is_full_hash_and_repository_files_are_pinned(self) -> None:
        commit = self.lock["repository_commit"]
        self.assertRegex(commit, re.compile(r"^[0-9a-f]{40}$"))
        for artifact in self.lock["artifacts"]:
            if urlsplit(artifact["source_url"]).hostname == "raw.githubusercontent.com":
                self.assertIn(f"/{commit}/", artifact["source_url"])

    def test_sources_are_https_and_do_not_name_reference_inputs(self) -> None:
        forbidden = ("groundtruth", "ground_truth", "ground-truth", "reference", "inspvax")
        for artifact in self.lock["artifacts"]:
            url = artifact["source_url"]
            parsed = urlsplit(url)
            with self.subTest(artifact=artifact["id"]):
                self.assertEqual(parsed.scheme, "https")
                self.assertTrue(parsed.hostname)
                self.assertFalse(any(marker in url.casefold() for marker in forbidden))

    def test_each_recorded_acquisition_has_minimal_provenance(self) -> None:
        for artifact in self.lock["artifacts"]:
            with self.subTest(artifact=artifact["id"]):
                if artifact.get("acquisition_status") == "pending":
                    continue
                self.assertRegex(artifact["retrieved_at"], r"^\d{4}-\d{2}-\d{2}")
                self.assertGreater(artifact["observed_byte_size"], 0)
                self.assertRegex(artifact["observed_sha256"], r"^[0-9a-f]{64}$")

    def test_broadcast_navigation_is_fixed_to_the_medium_utc_day(self) -> None:
        navigation = next(
            artifact
            for artifact in self.lock["artifacts"]
            if artifact["id"] == "broadcast_navigation"
        )
        self.assertEqual(navigation["coverage_date_utc"], "2021-05-17")
        self.assertEqual(navigation["acquisition_status"], "acquired")
        self.assertEqual(
            navigation["local_name"],
            "BRDM00DLR_S_20211370000_01D_MN.rnx.gz",
        )

    def test_only_the_selected_gnss_imu_extrinsic_is_deployable(self) -> None:
        extrinsic = next(
            artifact
            for artifact in self.lock["artifacts"]
            if artifact["id"] == "extrinsic_calibration"
        )
        self.assertEqual(extrinsic["selected_key"], "ANTENNA_T_IMU")
        self.assertEqual(
            extrinsic["selection_status"],
            "selected_key_only_deployable",
        )
        self.assertEqual(extrinsic["translation_unit"], "m")
        self.assertIn("ROS REP-103", extrinsic["unit_basis"])
        self.assertIn("not explicitly annotated", extrinsic["unit_status"])
        self.assertEqual(extrinsic["phase2_wls_use"], "antenna_phase_center_only")
        self.assertRegex(extrinsic["selected_output_sha256"], r"^[0-9a-f]{64}$")

    def test_main_receiver_is_confirmed_from_deployable_diagnostics(self) -> None:
        gnss = next(
            artifact for artifact in self.lock["artifacts"]
            if artifact["id"] == "gnss_observation_archive"
        )
        self.assertEqual(
            gnss["selection_status"],
            "confirmed_from_deployable_rinex_diagnostics",
        )
        self.assertEqual(len(gnss["selected_member_basenames"]), 2)
        evidence = gnss["selection_evidence"]
        self.assertEqual(evidence["obvious_gap_count"], 0)
        self.assertGreaterEqual(
            evidence["pseudorange_satellites_per_epoch"]["min"], 4
        )
        self.assertFalse(evidence["receiver_header_identity_available"])


if __name__ == "__main__":
    unittest.main()
