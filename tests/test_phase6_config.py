from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rtkfree_equivariant_gnss_ins.phase6_config import file_sha256, load_phase6_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "phase6" / "final_freeze_v1.json"
PHASE4 = ROOT / "config" / "phase4" / "ordinary_student_v1.json"
PHASE5 = ROOT / "config" / "phase5" / "gravity_aware_so2_v1.json"


class Phase6ConfigTests(unittest.TestCase):
    def test_loads_exact_prospective_contract(self) -> None:
        config = load_phase6_config(CONFIG, PHASE4, PHASE5)
        self.assertEqual(config.seeds, (163736869, 188003056, 1899326961))
        self.assertEqual(config.optimizer_steps, 200)
        self.assertEqual(config.checkpoint_every_steps, 10)
        self.assertEqual(
            config.structure_variants,
            ("ordinary", "rotation_augmented", "strict_so2"),
        )
        self.assertEqual(config.interaction_structures, ("ordinary", "strict_so2"))
        self.assertEqual(config.physics_weights, (0.0, 0.1))
        self.assertEqual(config.low_quality_union_epoch_count, 53)

    def test_rejects_budget_extension(self) -> None:
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        raw["optimization"]["optimizer_steps"] = 201
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "phase6.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_phase6_config(path, PHASE4, PHASE5)

    def test_rejects_rotation_augmented_factorial_expansion(self) -> None:
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        raw["arms"]["interaction_structures"].append("rotation_augmented")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "phase6.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_phase6_config(path, PHASE4, PHASE5)

    def test_exact_config_hash_is_prospectively_registered(self) -> None:
        digest = file_sha256(CONFIG)
        registry = (ROOT / "docs" / "governance" / "EXPERIMENT_REGISTRY.md").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            digest,
            "7d6ec6c2995c63d31a57544d2b9baff59d586807dd5fbddb66772d66fda79de0",
        )
        self.assertIn(digest, registry)


if __name__ == "__main__":
    unittest.main()
