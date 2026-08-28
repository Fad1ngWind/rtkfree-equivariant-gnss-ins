from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "phase5" / "diagnose_medium_models.py"


def _load_script() -> object:
    specification = importlib.util.spec_from_file_location("phase5_diagnostics_script", SCRIPT_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load Phase 5 diagnostics script")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class Phase5DiagnosticsScriptTests(unittest.TestCase):
    def test_script_reuses_phase4_checks_and_adds_complete_mapping_classification(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("from scripts.phase4.diagnose_medium_models import", source)
        self.assertIn("diagnose_variant", source)
        self.assertIn("so2_rollout_diagnostic", source)
        self.assertIn("complete_real_sequence_step_count", source)
        self.assertIn("strict_so2_structural_claim_allowed", source)
        self.assertIn("strict_so2_float32_full_rollout_claim_allowed", source)
        self.assertIn("reference_free_safety_checks_passed", source)
        self.assertIn("float64_structural_so2_checks_passed", source)
        self.assertIn("float32_deployed_rollout_checks_passed", source)
        self.assertNotIn('"diagnostics_passed"', source)
        self.assertIn("phase5_config.float64_atol", source)
        self.assertIn("sensor_remounting_tested", source)
        self.assertIn("RTKFREE_SEALED_REFERENCE_ROOT", source)

    def test_tolerance_comparison_accepts_boundary_and_rejects_counterexample(self) -> None:
        module = _load_script()
        within = module._comparison(
            ((torch.tensor((1.00001,)), torch.tensor((1.0,))),),
            atol=1e-5,
            rtol=1e-5,
        )
        outside = module._comparison(
            ((torch.tensor((1.1,)), torch.tensor((1.0,))),),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertTrue(within["passed"])
        self.assertFalse(outside["passed"])
        self.assertGreater(outside["maximum_tolerance_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
