from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Phase4DependencyTests(unittest.TestCase):
    def test_cpu_pytorch_environment_is_fully_hashed(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]
        self.assertEqual(
            project["optional-dependencies"]["phase4"],
            ["numpy==2.5.2", "torch==2.13.0+cpu"],
        )

        lock = (ROOT / "requirements" / "phase4.lock").read_text(encoding="utf-8")
        self.assertIn("--only-binary=:all:", lock)
        self.assertIn("--require-hashes", lock)
        self.assertEqual(
            re.findall(r"^([a-zA-Z0-9_-]+)==([^ ]+) \\$", lock, re.MULTILINE),
            [
                ("filelock", "3.32.3"),
                ("fsspec", "2026.7.0"),
                ("jinja2", "3.1.6"),
                ("markupsafe", "3.0.3"),
                ("mpmath", "1.3.0"),
                ("networkx", "3.6.1"),
                ("numpy", "2.5.2"),
                ("setuptools", "78.1.0"),
                ("sympy", "1.14.0"),
                ("torch", "2.13.0+cpu"),
                ("typing-extensions", "4.16.0"),
            ],
        )
        self.assertIn(
            "sha256:4ca4a9394b0c771238a4f73590fdbbc4debad85ed0fa63d026ae1b085da7d6e2",
            lock,
        )

    def test_readme_and_healthcheck_select_the_locked_cpu_runtime(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        healthcheck = (ROOT / "scripts" / "healthcheck.sh").read_text(encoding="utf-8")
        self.assertIn("requirements/phase4.lock", readme)
        self.assertIn("~/rtkfree-venvs/phase4/bin/activate", readme)
        self.assertIn("torch.__version__", healthcheck)
        self.assertIn("2.13.0+cpu\\ None", healthcheck)
        self.assertIn("source ~/rtkfree-venvs/phase4/bin/activate", healthcheck)


if __name__ == "__main__":
    unittest.main()
