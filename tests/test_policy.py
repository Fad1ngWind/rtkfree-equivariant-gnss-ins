from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path

from rtkfree_equivariant_gnss_ins.policy import (
    InformationPolicyError,
    validate_deployable_config,
    validate_deployable_path,
)


ROOT = Path(__file__).resolve().parents[1]
TMP_PARENT = ROOT / ".phase0_tmp"


class InformationPolicyTests(unittest.TestCase):
    def test_canonical_rtkfree_repository_name_is_allowed(self) -> None:
        path = "/mnt/e/rtkfree-equivariant-gnss-ins/config/phase0.json"
        self.assertEqual(validate_deployable_path(path).as_posix(), path)

    def test_snake_kebab_camel_and_compact_markers_are_blocked(self) -> None:
        unsafe_values = [
            {"groundTruthPath": "/vault/route"},
            {"RTKResults": "disabled"},
            {"highPrecision": "none"},
            {"labels": "/vault/postprocessed-truth/route"},
            {"labels": "/vault/referenceTrajectory/route"},
            {"useRTK": False},
            {"rtkEnabled": False},
            {"rtkDataRoot": "disabled"},
            {"rtkLoss": 0.0},
            {"ppkPath": "disabled"},
            {"ppkTrajectory": "disabled"},
            {"ppkDataRoot": "disabled"},
            {"ppkLoss": 0.0},
            {"source": "useRTK"},
            {"source": "ppkTrajectory"},
        ]
        for value in unsafe_values:
            with self.subTest(value=value), self.assertRaises(InformationPolicyError):
                validate_deployable_config(value)
        explicit_markers = (
            "useRTK", "rtkEnabled", "rtkDataRoot", "rtkLoss",
            "ppkPath", "ppkTrajectory", "ppkDataRoot", "ppkLoss",
        )
        for marker in explicit_markers:
            for config in ({marker: False}, {"source": marker}):
                with self.subTest(config=config), self.assertRaises(InformationPolicyError):
                    validate_deployable_config(config)
            with self.subTest(path=marker), self.assertRaises(InformationPolicyError):
                validate_deployable_path(f"{marker}/route.txt")
        for path in ("rtk_data/route.txt", "ppkPath/route.txt", "rtkfree_rtk_data/route.txt"):
            with self.subTest(path=path), self.assertRaises(InformationPolicyError):
                validate_deployable_path(path)
        self.assertEqual(validate_deployable_path("rtkfree").as_posix(), "rtkfree")

    def test_nested_supported_containers_and_pathlike_are_checked(self) -> None:
        safe = {
            "sources": ("wls_spp_pvt", "low_cost_imu"),
            "seeds": {1, 2},
            "path": Path("deployable/session.json"),
        }
        self.assertIs(validate_deployable_config(safe), safe)
        unsafe = {"nested": (Path("groundTruth/route.pos"),)}
        with self.assertRaises(InformationPolicyError):
            validate_deployable_config(unsafe)

    def test_unsupported_or_unsafe_leaf_fails_closed(self) -> None:
        for value in (object(), b"bytes", float("nan"), float("inf")):
            with self.subTest(kind=type(value).__name__), self.assertRaises(InformationPolicyError):
                validate_deployable_config({"value": value})
        with self.assertRaises(InformationPolicyError):
            validate_deployable_config({1: "non-text key"})

    def test_cycle_fails_closed(self) -> None:
        cyclic: list[object] = []
        cyclic.append(cyclic)
        with self.assertRaises(InformationPolicyError):
            validate_deployable_config({"cycle": cyclic})

    def test_symbolic_link_chain_is_blocked(self) -> None:
        TMP_PARENT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="policy-", dir=TMP_PARENT) as raw:
            root = Path(raw)
            target = root / "target"
            target.mkdir()
            link = root / "link"
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink unsupported: {type(exc).__name__}")
            with self.assertRaises(InformationPolicyError):
                validate_deployable_path(link / "safe.json")
        try:
            TMP_PARENT.rmdir()
        except OSError:
            pass

    def test_non_pathlike_value_is_rejected(self) -> None:
        with self.assertRaises(InformationPolicyError):
            validate_deployable_path(123)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
