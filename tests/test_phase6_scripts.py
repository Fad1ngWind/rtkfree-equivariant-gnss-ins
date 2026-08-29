from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "phase6" / "run_medium_freeze.py"
VERIFIER = ROOT / "scripts" / "phase6" / "verify_medium_freeze.py"


class Phase6ScriptBoundaryTests(unittest.TestCase):
    def test_main_entry_exposes_no_tuning_or_reference_arguments(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        for argument in ("--data-root", "--teacher-run-dir", "--output-dir"):
            self.assertIn(f'parser.add_argument("{argument}"', source)
        for prohibited in (
            "--seed",
            "--optimizer-steps",
            "--checkpoint-every",
            "--physics-weight",
            "--variant",
            "--config",
            "--reference",
            "--receiver",
            "--route",
        ):
            self.assertNotIn(f'parser.add_argument("{prohibited}"', source)
        self.assertIn('Arm("ordinary", 0.1)', source)
        self.assertIn('Arm("strict_so2", 0.1)', source)
        self.assertIn('if selected_structure == "rotation_augmented":', source)

    def test_verifier_reads_only_the_external_run_directory(self) -> None:
        source = VERIFIER.read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--run-dir"', source)
        self.assertNotIn('parser.add_argument("--data-root"', source)
        self.assertNotIn('parser.add_argument("--teacher-run-dir"', source)
        self.assertNotIn('parser.add_argument("--reference"', source)


if __name__ == "__main__":
    unittest.main()
