from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Phase3DependencyTests(unittest.TestCase):
    def test_numpy_is_the_only_hashed_phase3_dependency(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]
        self.assertEqual(project["dependencies"], [])
        self.assertEqual(project["optional-dependencies"]["phase3"], ["numpy==2.5.2"])

        lock = (ROOT / "requirements" / "phase3.lock").read_text(encoding="utf-8")
        self.assertIn("--only-binary=:all:", lock)
        self.assertIn("--require-hashes", lock)
        self.assertEqual(
            re.findall(r"^([a-zA-Z0-9_-]+)==([^ ]+) \\$", lock, re.MULTILINE),
            [("numpy", "2.5.2")],
        )
        self.assertIn(
            "sha256:3cdec01fa790a186d430433fdd4d4ffb70eed6f0eeb4bf05c8dbe2dce0a9bcb8",
            lock,
        )

    def test_readme_and_healthcheck_use_the_locked_phase3_environment(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        healthcheck = (ROOT / "scripts" / "healthcheck.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("~/rtkfree-venvs/phase3/bin/activate", readme)
        self.assertIn("requirements/phase3.lock", readme)
        self.assertIn("numpy.__version__", healthcheck)
        self.assertIn("3.12.*\\ 2.5.2", healthcheck)
        self.assertIn("-W error -m unittest discover", healthcheck)


if __name__ == "__main__":
    unittest.main()
