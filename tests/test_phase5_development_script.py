from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from rtkfree_equivariant_gnss_ins.phase3_forward import GnssOutage


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "phase5" / "run_medium_development.py"


def _load_script() -> object:
    specification = importlib.util.spec_from_file_location("phase5_development_script", SCRIPT_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load Phase 5 development script")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class Phase5DevelopmentScriptTests(unittest.TestCase):
    def test_script_freezes_common_selection_and_pass_level_augmentation(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("RTKFREE_SEALED_REFERENCE_ROOT", source)
        self.assertIn("output_directory.is_relative_to(ROOT.resolve())", source)
        self.assertIn("unroll_phase5_no_physics", source)
        self.assertIn("common_unrotated_validation_weak_label_loss", source)
        self.assertIn("training_pass_index += 1", source)
        self.assertIn("detach_phase5_hidden", source)
        self.assertIn("phase5_config.checkpoint_every_steps", source)
        self.assertNotIn("phase4_config.checkpoint_every_steps", source)
        self.assertIn('"physics_loss": None', source)

    def test_training_outage_matches_phase4_segment_rule(self) -> None:
        module = _load_script()

        class Step:
            def __init__(self, timestamp: int) -> None:
                self.timestamp_ns_utc = timestamp

        class Sequence:
            steps = tuple(Step(index * 1_000_000_000) for index in range(100))

        class Config:
            training_mask_start_step = 16

        self.assertEqual(module.training_outage(Sequence(), Config(), 0, 64, 0), ())
        self.assertEqual(module.training_outage(Sequence(), Config(), 448, 458, 20), ())
        self.assertEqual(
            module.training_outage(Sequence(), Config(), 64, 128, 20),
            (GnssOutage(80_000_000_000, 20_000_000_000),),
        )


if __name__ == "__main__":
    unittest.main()
