#!/usr/bin/env python3
"""Run the one bounded matched Phase 5 development comparison."""

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
    detach_state,
    fit_deployable_normalizer,
    tensor_state,
)
from rtkfree_equivariant_gnss_ins.phase5_config import load_phase5_config
from rtkfree_equivariant_gnss_ins.phase5_learning import (
    detach_phase5_hidden,
    new_phase5_student,
    training_sequence_for_variant,
    unroll_phase5_no_physics,
)


def state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validation_score(
    student: nn.Module,
    sequence: object,
    phase4_config: object,
    normalizer: object,
    variant: object,
) -> dict[str, object]:
    """Score every variant on the identical unrotated weak-label validation view."""

    validation_start, validation_stop = phase4_config.splits["validation"]
    mask_start = validation_start + phase4_config.training_mask_start_step
    scenario_losses: dict[str, float] = {}
    with torch.no_grad():
        for duration_s in (0,) + phase4_config.outage_durations_s:
            outages = (
                ()
                if duration_s == 0
                else (
                    GnssOutage(
                        sequence.steps[mask_start].timestamp_ns_utc,
                        duration_s * 1_000_000_000,
                    ),
                )
            )
            initial = tensor_state(sequence.initial_student_state, torch.float32)
            result = unroll_phase5_no_physics(
                student,
                sequence,
                phase4_config,
                normalizer,
                0,
                validation_stop,
                initial,
                student.initial_hidden(initial),
                variant,
                outages,
                score_start=validation_start,
            )
            scenario_losses[f"mask_{duration_s}s"] = float(result.weak_label_loss)
    return {
        "composite": sum(scenario_losses.values()) / len(scenario_losses),
        "scenario_weak_label_losses": scenario_losses,
    }


def training_outage(
    sequence: object,
    config: object,
    start: int,
    stop: int,
    duration_s: int,
) -> tuple[GnssOutage, ...]:
    if duration_s == 0 or stop - start <= config.training_mask_start_step + duration_s:
        return ()
    mask_start = start + config.training_mask_start_step
    return (
        GnssOutage(
            sequence.steps[mask_start].timestamp_ns_utc,
            duration_s * 1_000_000_000,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded matched Phase 5 development training.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if "RTKFREE_SEALED_REFERENCE_ROOT" in os.environ:
        raise ValueError("sealed-reference environment must be absent during Phase 5 development")
    output_directory = args.output_dir.resolve()
    if not output_directory.is_absolute() or output_directory.exists():
        raise ValueError("development output directory must be a new absolute path")
    if output_directory.is_relative_to(ROOT.resolve()):
        raise ValueError("development outputs must remain outside the repository")
    if torch.version.cuda is not None or "+cpu" not in torch.__version__:
        raise ValueError("Phase 5 development requires the locked CPU-only PyTorch environment")

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

    train_start, train_stop = phase4_config.splits["train"]
    initial_hashes: dict[str, str] = {}
    summaries: dict[str, object] = {}
    for variant in phase5_config.variants:
        student = new_phase5_student(variant, phase4_config, phase5_config)
        initial_hash = state_dict_hash(student.state_dict())
        initial_hashes[variant.variant_id] = initial_hash
        parameter_count = sum(parameter.numel() for parameter in student.parameters())
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=phase4_config.learning_rate,
            weight_decay=phase4_config.weight_decay,
        )
        initial_validation = validation_score(
            student,
            sequence,
            phase4_config,
            normalizer,
            variant,
        )
        best_score = math.inf
        best_step: int | None = None
        best_state: dict[str, torch.Tensor] | None = None
        training_pass_index = 0
        training_sequence, training_yaw_rad = training_sequence_for_variant(
            sequence,
            variant,
            phase5_config,
            training_pass_index,
        )
        state = tensor_state(training_sequence.initial_student_state, torch.float32)
        hidden = student.initial_hidden(state)
        cursor = train_start
        records: list[dict[str, object]] = []
        for optimizer_step in range(phase4_config.development_optimizer_steps):
            if cursor >= train_stop:
                training_pass_index += 1
                training_sequence, training_yaw_rad = training_sequence_for_variant(
                    sequence,
                    variant,
                    phase5_config,
                    training_pass_index,
                )
                cursor = train_start
                state = tensor_state(training_sequence.initial_student_state, torch.float32)
                hidden = student.initial_hidden(state)
            stop = min(cursor + phase4_config.unroll_steps, train_stop)
            duration_s = phase4_config.training_mask_cycle_s[
                optimizer_step % len(phase4_config.training_mask_cycle_s)
            ]
            result = unroll_phase5_no_physics(
                student,
                training_sequence,
                phase4_config,
                normalizer,
                cursor,
                stop,
                state,
                hidden,
                variant,
                training_outage(
                    training_sequence,
                    phase4_config,
                    cursor,
                    stop,
                    duration_s,
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            result.total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                phase4_config.gradient_norm_clip,
            )
            optimizer.step()
            finite_values = (
                float(result.total_loss.detach()),
                float(result.weak_label_loss.detach()),
                float(gradient_norm),
            )
            finite = all(math.isfinite(value) for value in finite_values)
            record: dict[str, object] = {
                "optimizer_step": optimizer_step + 1,
                "training_pass_index": training_pass_index,
                "training_yaw_rad": training_yaw_rad,
                "train_start_index": cursor,
                "train_stop_index": stop,
                "mask_duration_s": duration_s,
                "total_loss": finite_values[0],
                "weak_label_loss": finite_values[1],
                "physics_loss": None,
                "gradient_norm_before_clip": finite_values[2],
                "finite": finite,
            }
            state = detach_state(result.final_state)
            hidden = detach_phase5_hidden(result.final_hidden)
            cursor = stop
            if (optimizer_step + 1) % phase5_config.checkpoint_every_steps == 0:
                validation = validation_score(
                    student,
                    sequence,
                    phase4_config,
                    normalizer,
                    variant,
                )
                record["validation"] = validation
                score = float(validation["composite"])
                if math.isfinite(score) and score < best_score:
                    best_score = score
                    best_step = optimizer_step + 1
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in student.state_dict().items()
                    }
            records.append(record)
            if not finite:
                raise RuntimeError(
                    f"nonfinite {variant.variant_id} training value at step {optimizer_step + 1}"
                )
        if best_state is None or best_step is None:
            raise RuntimeError(f"no finite validation checkpoint for {variant.variant_id}")
        final_validation = validation_score(
            student,
            sequence,
            phase4_config,
            normalizer,
            variant,
        )
        final_state_dict = {
            name: value.detach().cpu().clone()
            for name, value in student.state_dict().items()
        }
        torch.save(best_state, output_directory / f"{variant.variant_id}_best.pt")
        torch.save(final_state_dict, output_directory / f"{variant.variant_id}_final.pt")
        (output_directory / f"{variant.variant_id}_training.jsonl").write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in records),
            encoding="utf-8",
        )
        pass_yaws = {
            str(record["training_pass_index"]): record["training_yaw_rad"]
            for record in records
        }
        summaries[variant.variant_id] = {
            "architecture": variant.architecture,
            "strict_so2": variant.strict_so2,
            "rotation_augmentation": variant.rotation_augmentation,
            "physics_weight": variant.physics_weight,
            "trainable_parameter_count": parameter_count,
            "initial_model_tensor_sha256": initial_hash,
            "best_model_tensor_sha256": state_dict_hash(best_state),
            "final_model_tensor_sha256": state_dict_hash(final_state_dict),
            "initial_validation": initial_validation,
            "best_validation_composite": best_score,
            "best_optimizer_step": best_step,
            "final_validation": final_validation,
            "optimizer_steps": phase4_config.development_optimizer_steps,
            "checkpoint_every_steps": phase5_config.checkpoint_every_steps,
            "training_pass_yaws_rad": pass_yaws,
            "all_training_values_finite": all(record["finite"] for record in records),
        }

    ordinary_augmented_initial_match = (
        initial_hashes["ordinary"] == initial_hashes["rotation_augmented"]
    )
    development_passed = ordinary_augmented_initial_match and all(
        value["physics_weight"] == 0.0
        and value["all_training_values_finite"]
        and value["optimizer_steps"] == phase4_config.development_optimizer_steps
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
        "mode": "bounded_development",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "seed": phase4_config.seed,
        "record_count": len(sequence.steps),
        "optimizer_steps_per_variant": phase4_config.development_optimizer_steps,
        "ordinary_augmented_initial_match": ordinary_augmented_initial_match,
        "strict_initialization_uses_same_seed": True,
        "variants": summaries,
        "selection_uses_only_common_unrotated_validation_weak_label_loss": True,
        "teacher_is_weak_pseudo_label": True,
        "teacher_covariance_used": False,
        "teacher_input_to_student": False,
        "high_precision_reference_used": False,
        "physics_residual_improvement_claimed": False,
        "accuracy_or_superiority_claimed": False,
        "development_passed": development_passed,
    }
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not development_passed:
        raise RuntimeError(f"Phase 5 bounded development gate failed; inspect {summary_path}")
    print("PASS Phase 5 bounded matched development training")
    print(f"summary={summary_path}")
    for variant_id, value in summaries.items():
        print(
            f"{variant_id} best_step={value['best_optimizer_step']} "
            f"initial_validation={value['initial_validation']['composite']:.9f} "
            f"best_validation={value['best_validation_composite']:.9f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
