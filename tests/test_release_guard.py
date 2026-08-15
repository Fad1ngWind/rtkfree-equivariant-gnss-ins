from __future__ import annotations

import unittest

from rtkfree_equivariant_gnss_ins.release_guard import inspect_candidate


class ReleaseGuardClassificationTests(unittest.TestCase):
    def assert_blocked(
        self, path: str, content: bytes = b"synthetic", mode: str = "100644"
    ) -> None:
        self.assertTrue(inspect_candidate(path, content, mode=mode), path)

    def test_rejects_absolute_traversal_dot_and_preserves_dotfile(self) -> None:
        for path in ("/absolute/file.txt", "C:/absolute/file.txt", "../escape.txt", "a/../b.txt", "./safe.txt"):
            with self.subTest(path=path):
                self.assert_blocked(path)
        self.assert_blocked(".env")
        self.assert_blocked(".env.local")
        self.assert_blocked(".ENV")
        self.assert_blocked(".ENV.example")
        self.assertEqual(inspect_candidate(".env.example", b"DOCUMENTED_ONLY=\n"), [])

    def test_environment_and_cache_paths(self) -> None:
        paths = [
            ".venv/bin/python", "venv/pyvenv.cfg", "env/lib/module.py",
            "pkg/site-packages/x.py", "src/__pycache__/x.pyc", ".pytest_cache/x",
            ".mypy-cache/x", ".ruff_cache/x", ".tox/x", ".nox/x",
            "notebooks/.ipynb_checkpoints/work.ipynb",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assert_blocked(path)

    def test_navigation_hdf_and_compressed_rinex(self) -> None:
        paths = [
            "route.24o", "route.24n", "route.rnx", "route.crx", "route.rnx.gz",
            "route.crx.Z", "route.obs", "route.nav", "precise.sp3", "clock.clk",
            "receiver.ubx", "sample.hdf", "sample.h5", "sample.hdf5",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assert_blocked(path)
        rinex_header = b"3.05 OBSERVATION DATA    " + b"RINEX VERSION" + b" / TYPE"
        self.assert_blocked("fixture.bin", rinex_header)
        self.assert_blocked("fixture.bin", b"\x89HDF\r\n\x1a\nrest")

    def test_high_precision_aliases_and_position_solution(self) -> None:
        paths = [
            "rtk_results/route.txt", "RTKResults/route.txt", "groundtruth/route.txt",
            "groundTruthPath/route.txt", "high_precision/route.txt",
            "highPrecisionTrajectory/route.txt", "postprocessed-truth/route.txt",
            "referenceTrajectory/route.txt", "route.pos",
            "rtk_data/route.txt", "ppkPath/route.txt", "rtkfree_rtk_data/route.txt",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assert_blocked(path)
        self.assertEqual(inspect_candidate("rtkfree-equivariant-gnss-ins/README.md", b"safe"), [])
        self.assertEqual(inspect_candidate("rtkfree/README.md", b"safe"), [])

    def test_weights_serialization_logs_backups_archives_and_private(self) -> None:
        paths = [
            "model.pt", "model.pth", "model.ckpt", "model.safetensors", "model.onnx",
            "object.pkl", "object.pickle", "object.joblib", "run.log", "copy.bak",
            "bundle.zip", "archive/data.tar.gz", "secrets/token.txt",
            "chat-exports/session.txt", "private_notes/chat.md", "unreviewed-notes/draft.md",
            "module.pyc",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assert_blocked(path)

    def test_content_secret_and_git_modes(self) -> None:
        fake_header = b"-----BEGIN " + b"PRIVATE KEY-----\nsynthetic"
        self.assert_blocked("notes.txt", fake_header)
        self.assert_blocked("safe.txt", b"target.txt", mode="120000")
        self.assert_blocked("vendor", b"0" * 20, mode="160000")
        self.assert_blocked("safe.txt", b"safe", mode="100600")

    def test_safe_candidates(self) -> None:
        self.assertEqual(inspect_candidate("src/package/module.py", b"value = 1\n"), [])
        self.assertEqual(inspect_candidate("docs/governance/policy.md", b"reviewed\n"), [])
        self.assertEqual(inspect_candidate("config/phase0.json", b"{}\n"), [])


if __name__ == "__main__":
    unittest.main()
