from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from rtkfree_equivariant_gnss_ins.phase4_validate import (
    compare_phase4_development_runs,
    file_sha256,
    state_dict_tensor_sha256,
)


class Phase4ValidationTests(unittest.TestCase):
    def test_tensor_hash_is_order_independent_and_detects_change(self) -> None:
        first = {
            "b": torch.tensor([2.0], dtype=torch.float32),
            "a": torch.tensor([1.0], dtype=torch.float32),
        }
        reordered = {"a": first["a"].clone(), "b": first["b"].clone()}
        changed = {"a": first["a"].clone(), "b": torch.tensor([3.0])}
        self.assertEqual(state_dict_tensor_sha256(first), state_dict_tensor_sha256(reordered))
        self.assertNotEqual(state_dict_tensor_sha256(first), state_dict_tensor_sha256(changed))

    def test_file_hash_detects_one_byte_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "evidence.json"
            path.write_bytes(b"{}\n")
            before = file_sha256(path)
            path.write_bytes(b"{ }\n")
            self.assertNotEqual(before, file_sha256(path))

    def test_comparison_rejects_artifact_mismatch(self) -> None:
        first = {
            "profile_id": "ordinary_causal_student_v1",
            "artifact_hashes": {"summary.json": "a"},
            "initial_model_tensor_sha256": "b",
        }
        second = {
            **first,
            "artifact_hashes": {"summary.json": "c"},
        }
        with self.assertRaises(ValueError):
            compare_phase4_development_runs(first, second)


if __name__ == "__main__":
    unittest.main()
