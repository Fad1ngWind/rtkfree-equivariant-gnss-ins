from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch

from rtkfree_equivariant_gnss_ins.phase3_forward import GnssOutage
from rtkfree_equivariant_gnss_ins.phase4_student import OrdinaryCausalStudent


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "phase5" / "run_medium_smoke.py"


def _load_script() -> object:
    specification = importlib.util.spec_from_file_location("phase5_smoke_script", SCRIPT_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load Phase 5 smoke script")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class Phase5SmokeScriptTests(unittest.TestCase):
    def test_script_contains_external_output_and_information_guards(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("RTKFREE_SEALED_REFERENCE_ROOT", source)
        self.assertIn("output_directory.is_relative_to(ROOT.resolve())", source)
        self.assertIn("unroll_phase5_no_physics", source)
        self.assertIn('"physics_loss": None', source)
        self.assertNotIn("RTKFREE_SEALED_REFERENCE_ROOT\"]", source)

    def test_model_hash_is_deterministic(self) -> None:
        module = _load_script()
        torch.manual_seed(5701)
        first = OrdinaryCausalStudent(quality_dim=5)
        torch.manual_seed(5701)
        second = OrdinaryCausalStudent(quality_dim=5)
        self.assertEqual(module._model_hash(first), module._model_hash(second))

    def test_outage_helper_retains_left_closed_right_open_phase4_timing(self) -> None:
        module = _load_script()

        class Step:
            timestamp_ns_utc = 17_000_000_000

        class Sequence:
            steps = (Step(),)

        class Config:
            training_mask_start_step = 0

        self.assertEqual(module._outage_for_duration(Sequence(), Config(), 0), ())
        outage = module._outage_for_duration(Sequence(), Config(), 20)
        self.assertEqual(outage, (GnssOutage(17_000_000_000, 20_000_000_000),))


if __name__ == "__main__":
    unittest.main()
