from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rtkfree_equivariant_gnss_ins.phase4_config import load_phase4_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "phase4" / "ordinary_student_v1.json"


class Phase4ConfigTests(unittest.TestCase):
    def test_profile_freezes_one_student_two_matched_variants_and_guarded_split(self) -> None:
        config = load_phase4_config(CONFIG_PATH)
        self.assertEqual(config.hidden_dim, 32)
        self.assertEqual(config.development_optimizer_steps, 40)
        self.assertEqual(config.outage_durations_s, (20, 30))
        self.assertEqual(config.splits["guard_1"], (458, 488))
        self.assertEqual(config.splits["guard_2"], (626, 656))
        self.assertEqual(
            tuple((value.variant_id, value.physics_weight) for value in config.variants),
            (("no_physics", 0.0), ("physics", 0.1)),
        )

    def test_teacher_covariance_or_current_gnss_in_pre_branch_is_rejected(self) -> None:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        fixtures = (
            ("inputs", "teacher_covariance_used", True),
            ("model", "current_gnss_enters_pre_branch", True),
            ("controlled_masks", "planned_duration_visible_to_student", True),
        )
        for section, key, value in fixtures:
            changed = json.loads(json.dumps(raw))
            changed[section][key] = value
            with tempfile.TemporaryDirectory() as temporary_directory:
                path = Path(temporary_directory) / "config.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_phase4_config(path)

    def test_student_quality_fields_exclude_time_identity_and_covariance(self) -> None:
        config = load_phase4_config(CONFIG_PATH)
        joined = " ".join(config.quality_fields).lower()
        for prohibited in ("time", "route", "file", "device", "teacher", "covariance"):
            self.assertNotIn(prohibited, joined)


if __name__ == "__main__":
    unittest.main()
