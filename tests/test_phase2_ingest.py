from __future__ import annotations

import json
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from rtkfree_equivariant_gnss_ins.phase2_ingest import (
    Phase2IngestError,
    acquire_broadcast_navigation,
    locate_selected_members,
    prepare_medium,
    refresh_selected_medium_extrinsic,
    restrict_existing_medium_extrinsic,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


class MinimalPhase2IngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="rtkfree-phase2-minimal-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_positive_list_locates_only_selected_members(self) -> None:
        archive = self.root / "data.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("selected.obs", "observation")
            handle.writestr("selected.nmea", "nmea")
            handle.writestr("unselected.bin", "not opened")
        found = locate_selected_members(archive, {"selected.obs", "selected.nmea"})
        self.assertEqual(set(found), {"selected.obs", "selected.nmea"})

    def test_selected_traversal_or_symlink_is_rejected(self) -> None:
        traversal = self.root / "traversal.zip"
        with zipfile.ZipFile(traversal, "w") as handle:
            handle.writestr("../selected.obs", "unsafe")
        with self.assertRaises(Phase2IngestError):
            locate_selected_members(traversal, {"selected.obs"})

        symlink = self.root / "symlink.zip"
        info = zipfile.ZipInfo("selected.obs")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(symlink, "w") as handle:
            handle.writestr(info, "target")
        with self.assertRaises(Phase2IngestError):
            locate_selected_members(symlink, {"selected.obs"})

    def test_prepare_reuses_local_sources_and_writes_one_manifest(self) -> None:
        repository = self.root / "repository"
        data_root = self.root / "external-data"
        data_root.mkdir()
        direct = self.root / "direct"
        direct.mkdir()
        archive = self.root / "gnss.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("route/selected.obs", "observation")
            handle.writestr("route/selected.nmea", "nmea")
            handle.writestr("route/unused.obs", "unused")
        direct_files = {
            "imu.csv": b"time,gyr\n0,0\n",
            "extrinsic.yaml": (
                b"%YAML:1.0\n"
                b"ANTENNA_T_IMU: !!opencv-matrix\n"
                b"   rows: 4\n"
                b"   cols: 4\n"
                b"   dt: d\n"
                b"   data: [ 1, 0, 0, 0, 0, 1, 0, 1, "
                b"0, 0, 1, 2, 0, 0, 0, 1 ]\n"
                b"EXCLUDED_REFERENCE_DEVICE_T_IMU: !!opencv-matrix\n"
                b"   data: [ 9, 9, 9, 9, 9, 9, 9, 9, "
                b"9, 9, 9, 9, 9, 9, 9, 9 ]\n"
            ),
            "imu.yaml": b"%YAML:1.0\n",
        }
        artifacts = []
        archive_size, archive_hash = sha256_file(archive)
        artifacts.append(
            {
                "id": "gnss_observation_archive",
                "kind": "zip",
                "source_url": "https://example.test/gnss.zip",
                "retrieved_at": "2026-08-25",
                "observed_byte_size": archive_size,
                "observed_sha256": archive_hash,
                "selected_member_basenames": ["selected.obs", "selected.nmea"],
                "selection_status": "provisional_pending_rinex_deployable_diagnostics",
            }
        )
        for artifact_id, filename in (
            ("low_cost_imu", "imu.csv"),
            ("extrinsic_calibration", "extrinsic.yaml"),
            ("imu_calibration", "imu.yaml"),
        ):
            path = direct / filename
            path.write_bytes(direct_files[filename])
            size, digest = sha256_file(path)
            artifact = {
                "id": artifact_id,
                "kind": "file",
                "source_url": f"https://example.test/{filename}",
                "local_name": filename,
                "retrieved_at": "2026-08-25",
                "observed_byte_size": size,
                "observed_sha256": digest,
            }
            if artifact_id == "extrinsic_calibration":
                artifact.update(
                    {
                        "selected_key": "ANTENNA_T_IMU",
                        "selected_output_name": "gnss_imu_extrinsic.json",
                    }
                )
            artifacts.append(artifact)
        artifacts.append(
            {
                "id": "broadcast_navigation",
                "kind": "gzip",
                "source_url": "https://example.test/navigation.rnx.gz",
                "local_name": "navigation.rnx.gz",
                "acquisition_status": "pending",
            }
        )
        lock_path = repository / "config" / "phase2" / "lock.json"
        lock_path.parent.mkdir(parents=True)
        lock_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset": "Fixture",
                    "session": "session",
                    "repository_commit": "a" * 40,
                    "artifacts": artifacts,
                }
            ),
            encoding="utf-8",
        )
        manifest_path = prepare_medium(lock_path, data_root, archive, direct)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["sources"]), 4)
        self.assertEqual(len(manifest["selected_outputs"]), 5)
        self.assertFalse(
            any("unused" in item["path"] for item in manifest["selected_outputs"])
        )
        calibration_dir = manifest_path.parent / "calibration"
        selected_extrinsic = calibration_dir / "gnss_imu_extrinsic.json"
        self.assertTrue(selected_extrinsic.is_file())
        self.assertFalse((calibration_dir / "extrinsic.yaml").exists())
        selected_record = json.loads(selected_extrinsic.read_text(encoding="utf-8"))
        self.assertEqual(selected_record["selected_key"], "ANTENNA_T_IMU")
        self.assertEqual(selected_record["translation_unit"], "m")
        self.assertIn("ROS REP-103", selected_record["unit_basis"])
        self.assertIn("not explicitly annotated", selected_record["unit_status"])
        self.assertIn("antenna phase center", selected_record["phase2_use"])
        self.assertNotIn("EXCLUDED_REFERENCE_DEVICE", selected_extrinsic.read_text())

        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        extrinsic_artifact = next(
            item for item in lock["artifacts"]
            if item["id"] == "extrinsic_calibration"
        )
        selected_size, selected_digest = sha256_file(selected_extrinsic)
        extrinsic_artifact["selected_output_byte_size"] = selected_size
        extrinsic_artifact["selected_output_sha256"] = selected_digest
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
        refreshed_path, refreshed_size, refreshed_digest, _ = (
            refresh_selected_medium_extrinsic(
                lock_path,
                data_root,
                direct / "extrinsic.yaml",
            )
        )
        self.assertEqual(refreshed_path, selected_extrinsic)
        self.assertEqual((refreshed_size, refreshed_digest), sha256_file(refreshed_path))

        # Recreate the one legacy deployable state and verify its bounded restriction.
        raw_extrinsic = calibration_dir / "extrinsic.yaml"
        raw_extrinsic.write_bytes(direct_files["extrinsic.yaml"])
        raw_size, raw_digest = sha256_file(raw_extrinsic)
        selected_extrinsic.unlink()
        for item in manifest["selected_outputs"]:
            if item["path"] == "calibration/gnss_imu_extrinsic.json":
                item.clear()
                item.update(
                    {
                        "path": "calibration/extrinsic.yaml",
                        "size": raw_size,
                        "sha256": raw_digest,
                    }
                )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        restricted_path, _, _, _ = restrict_existing_medium_extrinsic(
            lock_path,
            data_root,
            direct / "extrinsic.yaml",
        )
        self.assertEqual(restricted_path, selected_extrinsic)
        self.assertTrue(restricted_path.is_file())
        self.assertFalse(raw_extrinsic.exists())

        navigation_bytes = b"synthetic compressed navigation"

        def write_navigation(_url: str, destination: Path) -> None:
            destination.write_bytes(navigation_bytes)

        with mock.patch(
            "rtkfree_equivariant_gnss_ins.phase2_ingest._download_once",
            side_effect=write_navigation,
        ):
            navigation_path, size, digest, updated_manifest_path = (
                acquire_broadcast_navigation(lock_path, data_root)
            )
        updated_manifest = json.loads(
            updated_manifest_path.read_text(encoding="utf-8")
        )
        self.assertEqual(navigation_path.read_bytes(), navigation_bytes)
        self.assertEqual(size, len(navigation_bytes))
        self.assertEqual(sha256_file(navigation_path)[1], digest)
        self.assertEqual(len(updated_manifest["sources"]), 5)
        self.assertEqual(len(updated_manifest["selected_outputs"]), 6)


if __name__ == "__main__":
    unittest.main()
