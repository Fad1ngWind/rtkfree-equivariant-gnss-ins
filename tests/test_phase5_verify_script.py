from __future__ import annotations

import unittest
from pathlib import Path


class Phase5VerifyScriptTests(unittest.TestCase):
    def test_verifier_has_no_data_or_reference_arguments(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts" / "phase5" / "verify_medium_artifacts.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("--data-root", script)
        self.assertNotIn("--teacher-run-dir", script)
        self.assertIn("RTKFREE_SEALED_REFERENCE_ROOT", script)
        self.assertIn("sealed-reference environment must be absent", script)


if __name__ == "__main__":
    unittest.main()
