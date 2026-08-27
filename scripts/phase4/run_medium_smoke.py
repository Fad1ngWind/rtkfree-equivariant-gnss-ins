#!/usr/bin/env python3
"""Run the matched four-step Phase 4 smoke outside the repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch


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
    unroll_loss,
)
from rtkfree_equivariant_gnss_ins.phase4_student import OrdinaryCausalStudent


def _model_hash(student: OrdinaryCausalStudent) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(student.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _new_student(config: object) -> OrdinaryCausalStudent:
    torch.manual_seed(config.seed)
    return OrdinaryCausalStudent(
        quality_dim=len(config.quality_fields),
        hidden_dim=config.hidden_dim,
        previous_velocity_scale_n_mps=config.previous_velocity_scale_n_mps,
        pre_position_scale_n_m=config.pre_position_scale_n_m,
        pre_velocity_scale_n_mps=config.pre_velocity_scale_n_mps,
        pre_attitude_scale_rad=config.pre_attitude_scale_rad,
        post_correction_scale=config.post_correction_scale,
    )


def _outage_for_duration(sequence: object, config: object, duration_s: int) -> tuple[GnssOutage, ...]:
    if duration_s == 0:
        return ()
    start_index = config.training_mask_start_step
    start_ns = sequence.steps[start_index].timestamp_ns_utc
    return (GnssOutage(start_ns, duration_s * 1_000_000_000),)


def _probe(student: OrdinaryCausalStudent, sequence: object, config: object, normalizer: object) -> dict[str, float]:
    initial = tensor_state(sequence.initial_student_state, torch.float32)
    with torch.no_grad():
        result = unroll_loss(
            student,
            sequence,
            config,
            normalizer,
            0,
            config.unroll_steps,
            initial,
            student.initial_hidden(initial),
            0.0,
        )
    return {
        "weak_label_loss": float(result.weak_label_loss),
        "physics_loss": float(result.physics_loss),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run matched Phase 4 real-data smoke training.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if "RTKFREE_SEALED_REFERENCE_ROOT" in os.environ:
        raise ValueError("sealed-reference environment must be absent during Phase 4 development")
    output_directory = args.output_dir.resolve()
    if not output_directory.is_absolute() or output_directory.exists():
        raise ValueError("smoke output directory must be a new absolute path")
    if output_directory.is_relative_to(ROOT.resolve()):
        raise ValueError("smoke outputs must remain outside the repository")
    if torch.version.cuda is not None or "+cpu" not in torch.__version__:
        raise ValueError("Phase 4 smoke requires the locked CPU-only PyTorch environment")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = load_phase4_config(ROOT / "config" / "phase4" / "ordinary_student_v1.json")
    phase3_config = load_fixed_eskf_config(ROOT / "config" / "phase3" / "fixed_eskf_v1.json")
    commit = "075f96b6a6d9252b37486ecb175b4ae690c56f54"
    session = "UrbanNav-HK-Medium-Urban-1"
    sequence = assemble_phase4_sequence(
        config,
        phase3_config,
        args.data_root / "standardized" / "UrbanNav" / commit / session / "gps_l1ca_broadcast_spp_v1" / "pvt.jsonl",
        args.data_root / "standardized" / "UrbanNav" / commit / session / "imu_contract_v1" / "imu.csv",
        args.data_root / "deployable" / "UrbanNav" / commit / session / "calibration" / "gnss_imu_extrinsic.json",
        args.teacher_run_dir / "fixed_eskf.jsonl",
    )
    normalizer = fit_deployable_normalizer(sequence, config)
    output_directory.mkdir(parents=True, mode=0o700)
    normalizer_path = output_directory / "normalizer.json"
    normalizer_path.write_text(
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

    variant_summaries: dict[str, object] = {}
    initial_hashes: list[str] = []
    for variant in config.variants:
        student = _new_student(config)
        initial_hash = _model_hash(student)
        initial_hashes.append(initial_hash)
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        before = _probe(student, sequence, config, normalizer)
        training_records: list[dict[str, object]] = []
        for optimizer_step in range(config.smoke_optimizer_steps):
            initial = tensor_state(sequence.initial_student_state, torch.float32)
            duration_s = config.training_mask_cycle_s[optimizer_step]
            result = unroll_loss(
                student,
                sequence,
                config,
                normalizer,
                0,
                config.unroll_steps,
                initial,
                student.initial_hidden(initial),
                variant.physics_weight,
                _outage_for_duration(sequence, config, duration_s),
            )
            optimizer.zero_grad(set_to_none=True)
            result.total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                config.gradient_norm_clip,
            )
            optimizer.step()
            record = {
                "optimizer_step": optimizer_step + 1,
                "mask_duration_s": duration_s,
                "total_loss": float(result.total_loss.detach()),
                "weak_label_loss": float(result.weak_label_loss.detach()),
                "physics_loss": float(result.physics_loss.detach()),
                "gradient_norm_before_clip": float(gradient_norm),
                "finite": all(
                    math.isfinite(value)
                    for value in (
                        float(result.total_loss.detach()),
                        float(result.weak_label_loss.detach()),
                        float(result.physics_loss.detach()),
                        float(gradient_norm),
                    )
                ),
            }
            training_records.append(record)
        after = _probe(student, sequence, config, normalizer)
        log_path = output_directory / f"{variant.variant_id}_training.jsonl"
        log_path.write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in training_records),
            encoding="utf-8",
        )
        torch.save(student.state_dict(), output_directory / f"{variant.variant_id}_final.pt")
        variant_summaries[variant.variant_id] = {
            "physics_weight": variant.physics_weight,
            "initial_model_tensor_sha256": initial_hash,
            "final_model_tensor_sha256": _model_hash(student),
            "probe_before": before,
            "probe_after": after,
            "probe_weak_label_loss_decreased": (
                after["weak_label_loss"] < before["weak_label_loss"]
            ),
            "all_training_values_finite": all(value["finite"] for value in training_records),
            "optimizer_steps": config.smoke_optimizer_steps,
            "training_mask_cycle_s": list(config.training_mask_cycle_s),
        }

    fair_initialization = len(set(initial_hashes)) == 1
    smoke_passed = fair_initialization and all(
        value["probe_weak_label_loss_decreased"] and value["all_training_values_finite"]
        for value in variant_summaries.values()
    )
    summary = {
        "schema_version": 1,
        "profile_id": config.profile_id,
        "mode": "smoke",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "seed": config.seed,
        "record_count": len(sequence.steps),
        "fair_initialization": fair_initialization,
        "variants": variant_summaries,
        "teacher_is_weak_pseudo_label": True,
        "teacher_covariance_used": False,
        "teacher_input_to_student": False,
        "high_precision_reference_used": False,
        "smoke_passed": smoke_passed,
    }
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not smoke_passed:
        raise RuntimeError(f"Phase 4 smoke gate failed; inspect {summary_path}")
    print("PASS Phase 4 matched real-data smoke")
    print(f"summary={summary_path}")
    print(f"fair_initialization={str(fair_initialization).lower()}")
    for variant_id, value in variant_summaries.items():
        print(
            f"{variant_id} weak_before={value['probe_before']['weak_label_loss']:.9f} "
            f"weak_after={value['probe_after']['weak_label_loss']:.9f} "
            f"finite={str(value['all_training_values_finite']).lower()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
