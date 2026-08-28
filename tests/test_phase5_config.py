from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rtkfree_equivariant_gnss_ins.phase5_config import load_phase5_config


ROOT = Path(__file__).resolve().parents[1]
PHASE5_PATH = ROOT / "config" / "phase5" / "gravity_aware_so2_v1.json"
PHASE4_PATH = ROOT / "config" / "phase4" / "ordinary_student_v1.json"


class Phase5ConfigTests(unittest.TestCase):
    def test_frozen_profile_loads_exactly_three_no_physics_variants(self) -> None:
        config = load_phase5_config(PHASE5_PATH, PHASE4_PATH)
        self.assertEqual(config.profile_id, "gravity_aware_so2_comparison_v1")
        self.assertEqual(
            tuple(variant.variant_id for variant in config.variants),
            ("ordinary", "rotation_augmented", "strict_so2"),
        )
        self.assertTrue(all(variant.physics_weight == 0.0 for variant in config.variants))
        self.assertEqual(tuple(variant.trainable_parameter_count for variant in config.variants), (6994, 6994, 6780))
        self.assertEqual(config.property_angles_rad, (0.37, -1.11, 2.23))
        self.assertEqual(config.checkpoint_every_steps, 10)

    def test_recorded_phase4_hash_matches_exact_file(self) -> None:
        config = load_phase5_config(PHASE5_PATH, PHASE4_PATH)
        self.assertEqual(hashlib.sha256(PHASE4_PATH.read_bytes()).hexdigest(), config.base_phase4_sha256)

    def test_nonzero_physics_or_fourth_variant_is_rejected(self) -> None:
        raw = json.loads(PHASE5_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase5.json"
            raw["variants"][0]["physics_weight"] = 0.1
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen comparison"):
                load_phase5_config(path, PHASE4_PATH)

            raw = json.loads(PHASE5_PATH.read_text(encoding="utf-8"))
            raw["variants"].append(dict(raw["variants"][0], id="extra"))
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly three"):
                load_phase5_config(path, PHASE4_PATH)

    def test_modified_phase4_base_is_rejected_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ordinary_student_v1.json"
            raw = json.loads(PHASE4_PATH.read_text(encoding="utf-8"))
            raw["optimization"]["development_optimizer_steps"] = 41
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                load_phase5_config(PHASE5_PATH, path)

    def test_capacity_or_scope_expansion_is_rejected(self) -> None:
        raw = json.loads(PHASE5_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase5.json"
            raw["capacity_control"]["relative_tolerance"] = 0.25
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "capacity control"):
                load_phase5_config(path, PHASE4_PATH)

            raw = json.loads(PHASE5_PATH.read_text(encoding="utf-8"))
            raw["scope"]["primary_group"] = "O(2)"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scope"):
                load_phase5_config(path, PHASE4_PATH)


if __name__ == "__main__":
    unittest.main()
