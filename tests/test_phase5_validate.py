from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from rtkfree_equivariant_gnss_ins.phase4_validate import state_dict_tensor_sha256
from rtkfree_equivariant_gnss_ins.phase5_augmentation import augmentation_yaw_rad
from rtkfree_equivariant_gnss_ins.phase5_validate import (
    compare_phase5_development_runs,
    compare_phase5_smoke_runs,
    validate_phase5_smoke_run,
)


class Phase5ValidationTests(unittest.TestCase):
    def _build_smoke_fixture(self, directory: Path) -> tuple[object, object]:
        phase4 = SimpleNamespace(
            smoke_optimizer_steps=4,
            training_mask_cycle_s=(0, 20, 30, 0),
        )
        variants = (
            SimpleNamespace(
                variant_id="ordinary",
                architecture="phase4_ordinary_causal_student",
                rotation_augmentation=False,
                strict_so2=False,
                trainable_parameter_count=6994,
            ),
            SimpleNamespace(
                variant_id="rotation_augmented",
                architecture="phase4_ordinary_causal_student",
                rotation_augmentation=True,
                strict_so2=False,
                trainable_parameter_count=6994,
            ),
            SimpleNamespace(
                variant_id="strict_so2",
                architecture="gravity_aware_typed_scalar_vector_student",
                rotation_augmentation=False,
                strict_so2=True,
                trainable_parameter_count=6780,
            ),
        )
        phase5 = SimpleNamespace(
            profile_id="gravity_aware_so2_comparison_v1",
            base_phase4_sha256=(
                "7d6607edfe38725942ec0a4df6cac77af47d1b41d79a7b376ae30b0c5f1772cd"
            ),
            augmentation_seed=3407,
            variants=variants,
        )
        normalizer = {
            "fit_split": "train",
            "imu_center": [0.0] * 7,
            "imu_scale": [1.0] * 7,
            "quality_center": [0.0] * 5,
            "quality_scale": [1.0] * 5,
            "teacher_covariance_used": False,
        }
        (directory / "normalizer.json").write_text(
            json.dumps(normalizer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summaries: dict[str, object] = {}
        for variant_index, variant in enumerate(variants):
            yaw = (
                augmentation_yaw_rad(3407, 0) if variant.rotation_augmentation else 0.0
            )
            records = []
            for index, duration in enumerate(phase4.training_mask_cycle_s):
                records.append(
                    {
                        "optimizer_step": index + 1,
                        "training_pass_index": 0,
                        "training_yaw_rad": yaw,
                        "mask_duration_s": duration,
                        "total_loss": 3.0 - index * 0.1,
                        "weak_label_loss": 3.0 - index * 0.1,
                        "physics_loss": None,
                        "gradient_norm_before_clip": 1.0,
                        "finite": True,
                    }
                )
            (directory / f"{variant.variant_id}_training.jsonl").write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )
            state_dict = {"weight": torch.tensor([float(variant_index + 1)])}
            torch.save(state_dict, directory / f"{variant.variant_id}_final.pt")
            final_hash = state_dict_tensor_sha256(state_dict)
            initial_hash = "ordinary-initial" if variant_index < 2 else "strict-initial"
            summaries[variant.variant_id] = {
                "architecture": variant.architecture,
                "strict_so2": variant.strict_so2,
                "rotation_augmentation": variant.rotation_augmentation,
                "physics_weight": 0.0,
                "trainable_parameter_count": variant.trainable_parameter_count,
                "initial_model_tensor_sha256": initial_hash,
                "final_model_tensor_sha256": final_hash,
                "common_unrotated_probe_before": 3.0 if variant_index < 2 else 4.0,
                "common_unrotated_probe_after": 2.5,
                "common_unrotated_probe_change": -0.5 if variant_index < 2 else -1.5,
                "all_training_values_finite": True,
                "optimizer_steps": 4,
                "training_mask_cycle_s": [0, 20, 30, 0],
                "training_pass_yaw_rad": yaw,
            }
        summary = {
            "schema_version": 1,
            "profile_id": phase5.profile_id,
            "base_phase4_config_sha256": phase5.base_phase4_sha256,
            "mode": "smoke",
            "torch_version": "2.13.0+cpu",
            "torch_cuda": None,
            "seed": 3407,
            "record_count": 764,
            "ordinary_augmented_initial_match": True,
            "strict_initialization_uses_same_seed": True,
            "variants": summaries,
            "teacher_is_weak_pseudo_label": True,
            "teacher_covariance_used": False,
            "teacher_input_to_student": False,
            "high_precision_reference_used": False,
            "physics_residual_improvement_claimed": False,
            "accuracy_or_superiority_claimed": False,
            "smoke_passed": True,
        }
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return phase4, phase5

    def test_smoke_validator_rejects_tampered_augmentation_yaw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            phase4, phase5 = self._build_smoke_fixture(directory)
            validate_phase5_smoke_run(phase4, phase5, directory)
            path = directory / "rotation_augmented_training.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            records[2]["training_yaw_rad"] = 0.0
            path.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_phase5_smoke_run(phase4, phase5, directory)

    def test_smoke_validator_rejects_tampered_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            phase4, phase5 = self._build_smoke_fixture(directory)
            torch.save(
                {"weight": torch.tensor([99.0])}, directory / "strict_so2_final.pt"
            )
            with self.assertRaises(ValueError):
                validate_phase5_smoke_run(phase4, phase5, directory)

    def test_smoke_comparison_rejects_one_artifact_change(self) -> None:
        first = {
            "profile_id": "gravity_aware_so2_comparison_v1",
            "training_artifact_hashes": {"summary.json": "a"},
            "initial_model_tensor_sha256": {"ordinary": "b"},
        }
        second = {
            **first,
            "training_artifact_hashes": {"summary.json": "changed"},
        }
        with self.assertRaises(ValueError):
            compare_phase5_smoke_runs(first, second)

    def test_development_comparison_rejects_checkpoint_selection_change(self) -> None:
        first = {
            "profile_id": "gravity_aware_so2_comparison_v1",
            "training_artifact_hashes": {"strict_so2_best.pt:tensor": "a"},
            "initial_model_tensor_sha256": {"strict_so2": "b"},
            "initial_validation": {"strict_so2": {"composite": 3.0}},
            "best_validation_composite": {"strict_so2": 2.0},
            "best_optimizer_step": {"strict_so2": 30},
        }
        second = {**first, "best_optimizer_step": {"strict_so2": 40}}
        with self.assertRaises(ValueError):
            compare_phase5_development_runs(first, second)

    def test_development_comparison_accepts_identical_training_evidence(self) -> None:
        report = {
            "profile_id": "gravity_aware_so2_comparison_v1",
            "training_artifact_hashes": {"summary.json": "a", "ordinary_final.pt:tensor": "b"},
            "initial_model_tensor_sha256": {"ordinary": "c"},
            "initial_validation": {"ordinary": {"composite": 3.0}},
            "best_validation_composite": {"ordinary": 2.0},
            "best_optimizer_step": {"ordinary": 40},
        }
        result = compare_phase5_development_runs(report, dict(report))
        self.assertTrue(result["byte_and_tensor_identical"])
        self.assertEqual(result["compared_artifact_count"], 2)


if __name__ == "__main__":
    unittest.main()
