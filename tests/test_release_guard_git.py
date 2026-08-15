from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtkfree_equivariant_gnss_ins.release_guard import scan_repository


ROOT = Path(__file__).resolve().parents[1]
TMP_PARENT = ROOT / ".phase0_tmp"


class GitReleaseGuardIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        TMP_PARENT.mkdir(exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix="git-guard-", dir=TMP_PARENT)
        self.repo = Path(self._temporary.name)
        self.git("init", "-b", "main")
        self.git("config", "user.name", "Phase0 Synthetic Test")
        self.git("config", "user.email", "phase0-test@example.invalid")
        shutil.copyfile(ROOT / ".gitignore", self.repo / ".gitignore")

    def tearDown(self) -> None:
        self._temporary.cleanup()
        try:
            TMP_PARENT.rmdir()
        except OSError:
            pass

    def git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", *args], cwd=self.repo, check=check, capture_output=True
        )

    def write(self, relative: str, content: bytes = b"synthetic\n") -> None:
        target = self.repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def test_force_added_risk_classes_are_blocked_and_safe_file_is_allowed(self) -> None:
        cases = {
            ".env.local": b"SYNTHETIC=true\n",
            ".venv/bin/python": b"synthetic\n",
            "pkg/__pycache__/x.pyc": b"synthetic\n",
            "module.pyc": b"synthetic\n",
            "route.24o.gz": b"synthetic compressed rinex\n",
            "sample.hdf": b"synthetic\n",
            "payload.bin": b"\x89HDF\r\n\x1a\nsynthetic",
            "weights/model.pkl": b"synthetic\n",
            "RTKResults/route.pos": b"synthetic\n",
            "rtk_data/route.txt": b"synthetic\n",
            "ppkPath/route.txt": b"synthetic\n",
            "logs/run.log": b"synthetic\n",
            "chat-exports/session.txt": b"synthetic private-chat path only\n",
            "private_notes/chat.md": b"synthetic private-note path only\n",
            "navigation.txt": b"3.05 " + b"RINEX VERSION" + b" / TYPE",
        }
        for path, content in cases.items():
            self.write(path, content)
        self.write("safe.py", b"value = 1\n")
        ignored = set(
            self.git("check-ignore", "--", *cases.keys()).stdout.decode().splitlines()
        )
        content_only = {"payload.bin", "navigation.txt"}
        self.assertTrue((set(cases) - content_only).issubset(ignored))
        self.git("add", "-f", "--", *cases.keys())
        self.git("add", "--", "safe.py")

        findings = scan_repository(self.repo, staged=True)
        blocked_paths = {finding.path for finding in findings}
        self.assertTrue(set(cases).issubset(blocked_paths))
        self.assertNotIn("safe.py", blocked_paths)

    def test_force_added_root_dotenv_is_blocked(self) -> None:
        self.write(".env", b"SYNTHETIC_NON_SECRET=true\n")
        self.git("check-ignore", ".env")
        self.git("add", "-f", ".env")
        findings = scan_repository(self.repo, staged=True)
        self.assertTrue(any(f.path == ".env" for f in findings))

    def test_force_added_uppercase_dotenv_is_blocked(self) -> None:
        self.write(".ENV", b"SYNTHETIC_NON_SECRET=true\n")
        self.git("check-ignore", ".ENV")
        self.git("add", "-f", ".ENV")
        findings = scan_repository(self.repo, staged=True)
        self.assertTrue(any(f.path == ".ENV" for f in findings))

    def test_staged_type_change_to_symlink_is_blocked(self) -> None:
        self.write("safe.txt", b"initial\n")
        self.git("add", "safe.txt")
        self.git("commit", "-m", "synthetic fixture baseline")
        self.write("target.txt", b"target\n")
        (self.repo / "safe.txt").unlink()
        try:
            os.symlink("target.txt", self.repo / "safe.txt")
        except OSError as exc:
            self.skipTest(f"symlink unsupported: {type(exc).__name__}")
        self.git("add", "safe.txt")
        name_status = self.git("diff", "--cached", "--name-status").stdout.decode()
        self.assertIn("T\tsafe.txt", name_status)
        findings = scan_repository(self.repo, staged=True)
        self.assertTrue(any(f.path == "safe.txt" and "120000" in f.reason for f in findings))

    def test_gitlink_mode_is_blocked(self) -> None:
        self.write("base.txt", b"base\n")
        self.git("add", "base.txt")
        self.git("commit", "-m", "synthetic fixture baseline")
        object_id = self.git("rev-parse", "HEAD").stdout.decode().strip()
        self.git("update-index", "--add", "--cacheinfo", f"160000,{object_id},vendor_module")
        findings = scan_repository(self.repo, staged=True)
        self.assertTrue(any(f.path == "vendor_module" and "160000" in f.reason for f in findings))

    def test_missing_worktree_file_is_a_read_failure_finding(self) -> None:
        self.write("tracked.txt", b"safe\n")
        self.git("add", "tracked.txt")
        self.git("commit", "-m", "synthetic fixture baseline")
        (self.repo / "tracked.txt").unlink()
        findings = scan_repository(self.repo, staged=False)
        self.assertTrue(any(f.path == "tracked.txt" and "cannot inspect" in f.reason for f in findings))

    def test_safe_staged_candidate_passes(self) -> None:
        self.write("safe.py", b"value = 1\n")
        self.git("add", "safe.py")
        self.assertEqual(scan_repository(self.repo, staged=True), [])

    def test_ignore_query_failure_fails_closed(self) -> None:
        self.write("safe.py", b"value = 1\n")
        self.git("add", "safe.py")
        with patch(
            "rtkfree_equivariant_gnss_ins.release_guard._ignored_paths",
            side_effect=OSError("synthetic ignore query failure"),
        ):
            findings = scan_repository(self.repo, staged=True)
        self.assertEqual(findings[0].path, "<repository>")
        self.assertIn("cannot evaluate ignore policy", findings[0].reason)


if __name__ == "__main__":
    unittest.main()
