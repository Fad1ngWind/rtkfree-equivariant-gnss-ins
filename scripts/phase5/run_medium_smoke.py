#!/usr/bin/env python3
"""Run the matched four-step Phase 5 smoke outside the repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from rtkfree_equivariant_gnss_ins.phase3_config import load_fixed_eskf_config
from rtkfree_equivariant_gnss_ins.phase3_forward import GnssOutage
from rtkfree_equivariant_gnss_ins.phase4_config import load_phase4_config
from rtkfree_equivariant_gnss_ins.phase4_data import assemble_phase4_sequence
from rtkfree_equivariant_gnss_ins.phase4_learning import (
    fit_deployable_normalizer,
    tensor_state,
)
from rtkfree_equivariant_gnss_ins.phase5_config import load_phase5_config
from rtkfree_equivariant_gnss_ins.phase5_learning import (
    new_phase5_student,
    training_sequence_for_variant,
    unroll_phase5_no_physics,
)


def _model_hash(student: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(student.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _outage_for_duration(sequence: object, config: object, duration_s: int) -> tuple[GnssOutage, ...]:
    if duration_s == 0:
        return ()
    start_index = config.training_mask_start_step
    start_ns = sequence.steps[start_index].timestamp_ns_utc
    return (GnssOutage(start_ns, duration_s * 1_000_000_000),)


def _probe(
    student: nn.Module,
    sequence: object,
    phase4_config: object,
    normalizer: object,
    variant: object,
) -> float:
    initial = tensor_state(sequence.initial_student_state, torch.float32)
    with torch.no_grad():
        result = unroll_phase5_no_physics(
            student,
            sequence,
            phase4_config,
            normalizer,
            0,
            phase4_config.unroll_steps,
            initial,
            student.initial_hidden(initial),
            variant,
        )
    return float(result.weak_label_loss)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run matched Phase 5 real-data smoke training.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if "RTKFREE_SEALED_REFERENCE_ROOT" in os.environ:
        raise ValueError("sealed-reference environment must be absent during Phase 5 development")
    output_directory = args.output_dir.resolve()
    if not output_directory.is_absolute() or output_directory.exists():
        raise ValueError("smoke output directory must be a new absolute path")
    if output_directory.is_relative_to(ROOT.resolve()):
        raise ValueError("smoke outputs must remain outside the repository")
    if torch.version.cuda is not None or "+cpu" not in torch.__version__:
        raise ValueError("Phase 5 smoke requires the locked CPU-only PyTorch environment")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    phase4_config = load_phase4_config(
        ROOT / "config" / "phase4" / "ordinary_student_v1.json"
    )
    phase5_config = load_phase5_config(
        ROOT / "config" / "phase5" / "gravity_aware_so2_v1.json",
        ROOT / "config" / "phase4" / "ordinary_student_v1.json",
    )
    phase3_config = load_fixed_eskf_config(ROOT / "config" / "phase3" / "fixed_eskf_v1.json")
    commit = "075f96b6a6d9252b37486ecb175b4ae690c56f54"
    session = "UrbanNav-HK-Medium-Urban-1"
    sequence = assemble_phase4_sequence(
        phase4_config,
        phase3_config,
        args.data_root / "standardized" / "UrbanNav" / commit / session / "gps_l1ca_broadcast_spp_v1" / "pvt.jsonl",
        args.data_root / "standardized" / "UrbanNav" / commit / session / "imu_contract_v1" / "imu.csv",
        args.data_root / "deployable" / "UrbanNav" / commit / session / "calibration" / "gnss_imu_extrinsic.json",
        args.teacher_run_dir / "fixed_eskf.jsonl",
    )
    normalizer = fit_deployable_normalizer(sequence, phase4_config)
    output_directory.mkdir(parents=True, mode=0o700)
    (output_directory / "normalizer.json").write_text(
        json.dumps(
            {
                "fit_split": "train",
                "imu_center": normalizer.imu_center.tolist(),
                "imu_scale": normalizer.imu_scale.tolist(),
                "quality_center": normalizer.quality_center.tolist(),
                "quality_scale": normalizer.quality_scale.tolist(),
                "teacher_covariance_used": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summaries: dict[str, object] = {}
    initial_hashes: dict[str, str] = {}
    for variant in phase5_config.variants:
        student = new_phase5_student(variant, phase4_config, phase5_config)
        initial_hashes[variant.variant_id] = _model_hash(student)
        parameter_count = sum(parameter.numel() for parameter in student.parameters())
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=phase4_config.learning_rate,
            weight_decay=phase4_config.weight_decay,
        )
        before = _probe(student, sequence, phase4_config, normalizer, variant)
        training_sequence, training_yaw_rad = training_sequence_for_variant(
            sequence,
            variant,
            phase5_config,
            training_pass_index=0,
        )
        records: list[dict[str, object]] = []
        for optimizer_step in range(phase4_config.smoke_optimizer_steps):
            initial = tensor_state(training_sequence.initial_student_state, torch.float32)
            duration_s = phase4_config.training_mask_cycle_s[
                optimizer_step % len(phase4_config.training_mask_cycle_s)
            ]
            result = unroll_phase5_no_physics(
                student,
                training_sequence,
                phase4_config,
                normalizer,
                0,
                phase4_config.unroll_steps,
                initial,
                student.initial_hidden(initial),
                variant,
                _outage_for_duration(training_sequence, phase4_config, duration_s),
            )
            optimizer.zero_grad(set_to_none=True)
            result.total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                phase4_config.gradient_norm_clip,
            )
            optimizer.step()
            values = (
                float(result.total_loss.detach()),
                float(result.weak_label_loss.detach()),
                float(gradient_norm),
            )
            records.append(
                {
                    "optimizer_step": optimizer_step + 1,
                    "training_pass_index": 0,
                    "training_yaw_rad": training_yaw_rad,
                    "mask_duration_s": duration_s,
                    "total_loss": values[0],
                    "weak_label_loss": values[1],
                    "physics_loss": None,
                    "gradient_norm_before_clip": values[2],
                    "finite": all(math.isfinite(value) for value in values),
                }
            )
        after = _probe(student, sequence, phase4_config, normalizer, variant)
        (output_directory / f"{variant.variant_id}_training.jsonl").write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in records),
            encoding="utf-8",
        )
        torch.save(student.state_dict(), output_directory / f"{variant.variant_id}_final.pt")
        summaries[variant.variant_id] = {
            "architecture": variant.architecture,
            "strict_so2": variant.strict_so2,
            "rotation_augmentation": variant.rotation_augmentation,
            "physics_weight": variant.physics_weight,
            "trainable_parameter_count": parameter_count,
            "initial_model_tensor_sha256": initial_hashes[variant.variant_id],
            "final_model_tensor_sha256": _model_hash(student),
            "common_unrotated_probe_before": before,
            "common_unrotated_probe_after": after,
            "common_unrotated_probe_change": after - before,
            "all_training_values_finite": all(value["finite"] for value in records),
            "optimizer_steps": phase4_config.smoke_optimizer_steps,
            "training_mask_cycle_s": list(phase4_config.training_mask_cycle_s),
            "training_pass_yaw_rad": training_yaw_rad,
        }

    ordinary_augmented_initial_match = (
        initial_hashes["ordinary"] == initial_hashes["rotation_augmented"]
    )
    smoke_passed = ordinary_augmented_initial_match and all(
        value["physics_weight"] == 0.0
        and value["all_training_values_finite"]
        and value["trainable_parameter_count"]
        == next(
            variant.trainable_parameter_count
            for variant in phase5_config.variants
            if variant.variant_id == variant_id
        )
        for variant_id, value in summaries.items()
    )
    summary = {
        "schema_version": 1,
        "profile_id": phase5_config.profile_id,
        "base_phase4_config_sha256": phase5_config.base_phase4_sha256,
        "mode": "smoke",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "seed": phase4_config.seed,
        "record_count": len(sequence.steps),
        "ordinary_augmented_initial_match": ordinary_augmented_initial_match,
        "strict_initialization_uses_same_seed": True,
        "variants": summaries,
        "teacher_is_weak_pseudo_label": True,
        "teacher_covariance_used": False,
        "teacher_input_to_student": False,
        "high_precision_reference_used": False,
        "physics_residual_improvement_claimed": False,
        "accuracy_or_superiority_claimed": False,
        "smoke_passed": smoke_passed,
    }
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not smoke_passed:
        raise RuntimeError(f"Phase 5 smoke gate failed; inspect {summary_path}")
    print("PASS Phase 5 matched real-data smoke")
    print(f"summary={summary_path}")
    for variant_id, value in summaries.items():
        print(
            f"{variant_id} weak_before={value['common_unrotated_probe_before']:.9f} "
            f"weak_after={value['common_unrotated_probe_after']:.9f} "
            f"finite={str(value['all_training_values_finite']).lower()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
