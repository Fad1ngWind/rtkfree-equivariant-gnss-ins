#!/usr/bin/env python3
"""Run the single prospective Phase 6 Medium freeze workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.phase4.diagnose_medium_models import diagnose_variant, state_dict_hash
from scripts.phase5.diagnose_medium_models import so2_rollout_diagnostic
from scripts.phase5.run_medium_development import training_outage, validation_score
from rtkfree_equivariant_gnss_ins.phase3_config import load_fixed_eskf_config
from rtkfree_equivariant_gnss_ins.phase4_config import load_phase4_config
from rtkfree_equivariant_gnss_ins.phase4_data import assemble_phase4_sequence
from rtkfree_equivariant_gnss_ins.phase4_learning import (
    detach_state,
    fit_deployable_normalizer,
    tensor_state,
    unroll_loss,
)
from rtkfree_equivariant_gnss_ins.phase5_config import load_phase5_config
from rtkfree_equivariant_gnss_ins.phase5_learning import (
    detach_phase5_hidden,
    new_phase5_student,
    training_sequence_for_variant,
)
from rtkfree_equivariant_gnss_ins.phase6_config import file_sha256, load_phase6_config
from rtkfree_equivariant_gnss_ins.phase6_diagnostics import low_quality_stratum_report
from rtkfree_equivariant_gnss_ins.phase6_selection import (
    interaction_direction,
    select_physics,
    select_structure,
)


PHASE4_PATH = ROOT / "config" / "phase4" / "ordinary_student_v1.json"
PHASE5_PATH = ROOT / "config" / "phase5" / "gravity_aware_so2_v1.json"
PHASE6_PATH = ROOT / "config" / "phase6" / "final_freeze_v1.json"


@dataclass(frozen=True)
class Arm:
    structure: str
    physics_weight: float

    @property
    def arm_id(self) -> str:
        suffix = "p0" if self.physics_weight == 0.0 else "p01"
        return f"{self.structure}_{suffix}"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_normalizer(path: Path, normalizer: object) -> None:
    _write_json(
        path,
        {
            "fit_split": "train",
            "imu_center": normalizer.imu_center.tolist(),
            "imu_scale": normalizer.imu_scale.tolist(),
            "quality_center": normalizer.quality_center.tolist(),
            "quality_scale": normalizer.quality_scale.tolist(),
            "teacher_covariance_used": False,
        },
    )


def _variant_map(phase5_config: object) -> dict[str, object]:
    return {variant.variant_id: variant for variant in phase5_config.variants}


def _train_one(
    arm: Arm,
    seed: int,
    output_directory: Path,
    sequence: object,
    normalizer: object,
    phase4_config: object,
    phase5_config: object,
    phase6_config: object,
    diagnostics: bool,
) -> dict[str, Any]:
    if output_directory.exists():
        raise ValueError(f"run output already exists: {output_directory}")
    output_directory.mkdir(parents=True, mode=0o700)
    variant = _variant_map(phase5_config)[arm.structure]
    student = new_phase5_student(variant, phase4_config, phase5_config, seed=seed)
    initial_hash = state_dict_hash(student.state_dict())
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=phase4_config.learning_rate,
        weight_decay=phase4_config.weight_decay,
    )
    initial_validation = validation_score(
        student, sequence, phase4_config, normalizer, variant
    )
    train_start, train_stop = phase4_config.splits["train"]
    training_pass_index = 0
    training_sequence, training_yaw_rad = training_sequence_for_variant(
        sequence, variant, phase5_config, training_pass_index
    )
    state = tensor_state(training_sequence.initial_student_state, torch.float32)
    hidden = student.initial_hidden(state)
    cursor = train_start
    records: list[dict[str, object]] = []
    best_score = math.inf
    best_step: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for optimizer_step in range(phase6_config.optimizer_steps):
        if cursor >= train_stop:
            training_pass_index += 1
            training_sequence, training_yaw_rad = training_sequence_for_variant(
                sequence, variant, phase5_config, training_pass_index
            )
            cursor = train_start
            state = tensor_state(training_sequence.initial_student_state, torch.float32)
            hidden = student.initial_hidden(state)
        stop = min(cursor + phase4_config.unroll_steps, train_stop)
        duration_s = phase4_config.training_mask_cycle_s[
            optimizer_step % len(phase4_config.training_mask_cycle_s)
        ]
        result = unroll_loss(
            student,
            training_sequence,
            phase4_config,
            normalizer,
            cursor,
            stop,
            state,
            hidden,
            arm.physics_weight,
            training_outage(
                training_sequence,
                phase4_config,
                cursor,
                stop,
                duration_s,
            ),
            compute_physics=arm.physics_weight > 0.0,
        )
        optimizer.zero_grad(set_to_none=True)
        result.total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(), phase4_config.gradient_norm_clip
        )
        optimizer.step()
        finite_values = (
            float(result.total_loss.detach()),
            float(result.weak_label_loss.detach()),
            float(result.physics_loss.detach()) if arm.physics_weight > 0.0 else None,
            float(gradient_norm),
        )
        finite = all(
            value is None or math.isfinite(value) for value in finite_values
        )
        record: dict[str, object] = {
            "optimizer_step": optimizer_step + 1,
            "training_pass_index": training_pass_index,
            "training_yaw_rad": training_yaw_rad,
            "train_start_index": cursor,
            "train_stop_index": stop,
            "mask_duration_s": duration_s,
            "total_loss": finite_values[0],
            "weak_label_loss": finite_values[1],
            "physics_loss": finite_values[2],
            "gradient_norm_before_clip": finite_values[3],
            "finite": finite,
        }
        state = detach_state(result.final_state)
        hidden = detach_phase5_hidden(result.final_hidden)
        cursor = stop
        if (optimizer_step + 1) % phase6_config.checkpoint_every_steps == 0:
            validation = validation_score(
                student, sequence, phase4_config, normalizer, variant
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
            _write_json(
                output_directory / "failure.json",
                {"arm_id": arm.arm_id, "seed": seed, "failed_step": optimizer_step + 1},
            )
            raise RuntimeError(f"nonfinite training value for {arm.arm_id} seed {seed}")
    if best_state is None or best_step is None:
        raise RuntimeError(f"no finite checkpoint for {arm.arm_id} seed {seed}")
    final_state = {
        name: value.detach().cpu().clone() for name, value in student.state_dict().items()
    }
    log_path = output_directory / "training.jsonl"
    log_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    best_path = output_directory / "best.pt"
    final_path = output_directory / "final.pt"
    torch.save(best_state, best_path)
    torch.save(final_state, final_path)
    final_validation = validation_score(
        student, sequence, phase4_config, normalizer, variant
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "arm_id": arm.arm_id,
        "structure": arm.structure,
        "physics_weight": arm.physics_weight,
        "seed": seed,
        "optimizer_steps": phase6_config.optimizer_steps,
        "checkpoint_every_steps": phase6_config.checkpoint_every_steps,
        "initial_model_tensor_sha256": initial_hash,
        "best_model_tensor_sha256": state_dict_hash(best_state),
        "final_model_tensor_sha256": state_dict_hash(final_state),
        "training_log_sha256": file_sha256(log_path),
        "initial_validation": initial_validation,
        "best_validation_composite": best_score,
        "best_optimizer_step": best_step,
        "final_validation": final_validation,
        "all_training_values_finite": True,
        "teacher_covariance_used": False,
        "high_precision_reference_used": False,
    }
    if diagnostics:
        student.load_state_dict(best_state, strict=True)
        student.eval()
        reference_free = diagnose_variant(
            student, sequence, phase4_config, normalizer
        )
        strict_mapping: dict[str, object] | None = None
        strict_float64_pass = True
        if arm.structure == "strict_so2":
            strict_float32 = so2_rollout_diagnostic(
                student,
                sequence,
                phase4_config,
                phase5_config,
                normalizer,
                strict_expected=True,
            )
            strict_model64 = new_phase5_student(
                variant, phase4_config, phase5_config, seed=seed
            ).to(dtype=torch.float64)
            strict_model64.load_state_dict(best_state, strict=True)
            strict_model64.eval()
            strict_float64 = so2_rollout_diagnostic(
                strict_model64,
                sequence,
                phase4_config,
                phase5_config,
                normalizer,
                strict_expected=True,
            )
            strict_mapping = {"float32": strict_float32, "float64": strict_float64}
            strict_float64_pass = bool(strict_float64["classification_passed"])
        low_quality = low_quality_stratum_report(
            student,
            sequence,
            phase4_config,
            phase6_config,
            normalizer,
        )
        qualification_passed = bool(
            reference_free["degeneracy_checks_passed"]
            and reference_free["causality_and_memory_checks_passed"]
            and reference_free["numerical_checks_passed"]
            and low_quality["robustness_gate_passed"]
            and strict_float64_pass
        )
        diagnostic_report = {
            "schema_version": 1,
            "arm_id": arm.arm_id,
            "seed": seed,
            "reference_free": reference_free,
            "low_quality_stratum": low_quality,
            "strict_so2_complete_mapping": strict_mapping,
            "qualification_passed": qualification_passed,
            "high_precision_reference_used": False,
        }
        diagnostic_path = output_directory / "diagnostics.json"
        _write_json(diagnostic_path, diagnostic_report)
        summary.update(
            {
                "diagnostics_sha256": file_sha256(diagnostic_path),
                "qualification_passed": qualification_passed,
                "independent_physics_huber": reference_free[
                    "independent_physics_residual_on_diagnostic_split"
                ]["normalized_component_mean_huber"],
                "low_quality_robustness_gate_passed": low_quality[
                    "robustness_gate_passed"
                ],
                "strict_float64_structural_passed": strict_float64_pass,
                "strict_float32_deployed_rollout_passed": (
                    None
                    if strict_mapping is None
                    else strict_mapping["float32"]["strict_property_passed"]
                ),
            }
        )
    _write_json(output_directory / "summary.json", summary)
    return summary


def _metric_matrix(
    runs: dict[str, dict[str, dict[str, Any]]],
    arm_ids: dict[str, str],
    seeds: tuple[int, ...],
    metric: str,
) -> dict[str, dict[int, float]]:
    return {
        label: {
            seed: float(runs[arm_id][str(seed)][metric])
            for seed in seeds
        }
        for label, arm_id in arm_ids.items()
    }


def _run_directory(root: Path, arm: Arm, seed: int) -> Path:
    return root / "runs" / arm.arm_id / f"seed_{seed}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the single frozen Phase 6 workflow.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if "RTKFREE_SEALED_REFERENCE_ROOT" in os.environ:
        raise ValueError("sealed-reference environment must be absent during Phase 6")
    output_root = args.output_dir.resolve()
    if not output_root.is_absolute() or output_root.exists():
        raise ValueError("Phase 6 output must be a new absolute directory")
    if output_root.is_relative_to(ROOT.resolve()):
        raise ValueError("Phase 6 data, weights, logs, and results must remain outside Git")
    if torch.version.cuda is not None or "+cpu" not in torch.__version__:
        raise ValueError("Phase 6 requires the existing locked CPU-only PyTorch environment")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    phase4_config = load_phase4_config(PHASE4_PATH)
    phase5_config = load_phase5_config(PHASE5_PATH, PHASE4_PATH)
    phase6_config = load_phase6_config(
        PHASE6_PATH, PHASE4_PATH, PHASE5_PATH
    )
    phase3_config = load_fixed_eskf_config(
        ROOT / "config" / "phase3" / "fixed_eskf_v1.json"
    )
    commit = "075f96b6a6d9252b37486ecb175b4ae690c56f54"
    session = "UrbanNav-HK-Medium-Urban-1"
    sequence = assemble_phase4_sequence(
        phase4_config,
        phase3_config,
        args.data_root
        / "standardized"
        / "UrbanNav"
        / commit
        / session
        / "gps_l1ca_broadcast_spp_v1"
        / "pvt.jsonl",
        args.data_root
        / "standardized"
        / "UrbanNav"
        / commit
        / session
        / "imu_contract_v1"
        / "imu.csv",
        args.data_root
        / "deployable"
        / "UrbanNav"
        / commit
        / session
        / "calibration"
        / "gnss_imu_extrinsic.json",
        args.teacher_run_dir / "fixed_eskf.jsonl",
    )
    normalizer = fit_deployable_normalizer(sequence, phase4_config)
    output_root.mkdir(parents=True, mode=0o700)
    _write_normalizer(output_root / "normalizer.json", normalizer)
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        base_arms = (
            Arm("ordinary", 0.0),
            Arm("rotation_augmented", 0.0),
            Arm("strict_so2", 0.0),
            Arm("ordinary", 0.1),
            Arm("strict_so2", 0.1),
        )
        for arm in base_arms:
            runs[arm.arm_id] = {}
            for seed in phase6_config.seeds:
                print(f"START arm={arm.arm_id} seed={seed}", flush=True)
                summary = _train_one(
                    arm,
                    seed,
                    _run_directory(output_root, arm, seed),
                    sequence,
                    normalizer,
                    phase4_config,
                    phase5_config,
                    phase6_config,
                    diagnostics=True,
                )
                runs[arm.arm_id][str(seed)] = summary
                print(
                    f"COMPLETE arm={arm.arm_id} seed={seed} "
                    f"validation={summary['best_validation_composite']:.9f}",
                    flush=True,
                )
                if summary["qualification_passed"] is not True:
                    raise RuntimeError(f"qualification failed: {arm.arm_id} seed {seed}")

        structure_arm_ids = {
            structure: Arm(structure, 0.0).arm_id
            for structure in phase6_config.structure_variants
        }
        structure_validation = _metric_matrix(
            runs,
            structure_arm_ids,
            phase6_config.seeds,
            "best_validation_composite",
        )
        structure_decision = select_structure(
            structure_validation, phase6_config.seeds
        )
        selected_structure = str(structure_decision["selected_structure"])
        if selected_structure == "rotation_augmented":
            rotation_physics = Arm("rotation_augmented", 0.1)
            runs[rotation_physics.arm_id] = {}
            for seed in phase6_config.seeds:
                print(
                    f"START arm={rotation_physics.arm_id} seed={seed}", flush=True
                )
                summary = _train_one(
                    rotation_physics,
                    seed,
                    _run_directory(output_root, rotation_physics, seed),
                    sequence,
                    normalizer,
                    phase4_config,
                    phase5_config,
                    phase6_config,
                    diagnostics=True,
                )
                runs[rotation_physics.arm_id][str(seed)] = summary
                print(
                    f"COMPLETE arm={rotation_physics.arm_id} seed={seed} "
                    f"validation={summary['best_validation_composite']:.9f}",
                    flush=True,
                )
                if summary["qualification_passed"] is not True:
                    raise RuntimeError(
                        f"qualification failed: {rotation_physics.arm_id} seed {seed}"
                    )

        no_arm = Arm(selected_structure, 0.0).arm_id
        physics_arm = Arm(selected_structure, 0.1).arm_id
        physics_validation = _metric_matrix(
            runs,
            {"no_physics": no_arm, "physics": physics_arm},
            phase6_config.seeds,
            "best_validation_composite",
        )
        physics_huber = _metric_matrix(
            runs,
            {"no_physics": no_arm, "physics": physics_arm},
            phase6_config.seeds,
            "independent_physics_huber",
        )
        physics_decision = select_physics(
            physics_validation, physics_huber, phase6_config.seeds
        )

        interaction_arms = {
            "ordinary_no_physics": Arm("ordinary", 0.0).arm_id,
            "ordinary_physics": Arm("ordinary", 0.1).arm_id,
            "strict_no_physics": Arm("strict_so2", 0.0).arm_id,
            "strict_physics": Arm("strict_so2", 0.1).arm_id,
        }
        interaction = {
            "common_validation_composite": interaction_direction(
                _metric_matrix(
                    runs,
                    interaction_arms,
                    phase6_config.seeds,
                    "best_validation_composite",
                ),
                phase6_config.seeds,
            ),
            "independent_physics_huber": interaction_direction(
                _metric_matrix(
                    runs,
                    interaction_arms,
                    phase6_config.seeds,
                    "independent_physics_huber",
                ),
                phase6_config.seeds,
            ),
        }
        final_arm = Arm(
            selected_structure,
            float(physics_decision["selected_physics_weight"]),
        )
        designated_seed = phase6_config.designated_checkpoint_seed
        expected = runs[final_arm.arm_id][str(designated_seed)]
        print(
            f"START final_reproduction arm={final_arm.arm_id} seed={designated_seed}",
            flush=True,
        )
        reproduction = _train_one(
            final_arm,
            designated_seed,
            output_root / "reproduction",
            sequence,
            normalizer,
            phase4_config,
            phase5_config,
            phase6_config,
            diagnostics=False,
        )
        reproduction_keys = (
            "initial_model_tensor_sha256",
            "best_model_tensor_sha256",
            "final_model_tensor_sha256",
            "training_log_sha256",
            "best_validation_composite",
            "best_optimizer_step",
            "final_validation",
        )
        reproduction_matches = all(
            reproduction[key] == expected[key] for key in reproduction_keys
        )
        if not reproduction_matches:
            raise RuntimeError("final designated checkpoint reproduction differs")
        print("COMPLETE final_reproduction exact_match=true", flush=True)

        root_summary = {
            "schema_version": 1,
            "profile_id": phase6_config.profile_id,
            "phase6_config_sha256": file_sha256(PHASE6_PATH),
            "phase4_config_sha256": phase6_config.phase4_sha256,
            "phase5_config_sha256": phase6_config.phase5_sha256,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "seeds": list(phase6_config.seeds),
            "optimizer_steps": phase6_config.optimizer_steps,
            "checkpoint_every_steps": phase6_config.checkpoint_every_steps,
            "runs": runs,
            "structure_decision": structure_decision,
            "physics_decision": physics_decision,
            "interaction_report_only": interaction,
            "final_primary": {
                "arm_id": final_arm.arm_id,
                "structure": selected_structure,
                "physics_weight": final_arm.physics_weight,
                "designated_seed": designated_seed,
                "checkpoint_relative_path": (
                    f"runs/{final_arm.arm_id}/seed_{designated_seed}/best.pt"
                ),
                "checkpoint_tensor_sha256": expected["best_model_tensor_sha256"],
            },
            "reproduction": {
                "count": 1,
                "matches_designated_training_artifacts": reproduction_matches,
                "summary": reproduction,
            },
            "all_required_runs_qualified": True,
            "selection_uses_reference_truth": False,
            "high_precision_reference_used": False,
            "teacher_covariance_used": False,
            "predictive_covariance_implemented": False,
            "cross_receiver_or_route_data_used": False,
            "accuracy_or_superiority_claimed": False,
            "workflow_completed": True,
        }
        _write_json(output_root / "summary.json", root_summary)
    except Exception as exc:
        _write_json(
            output_root / "failure.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "failed_seed_replacement_allowed": False,
                "high_precision_reference_used": False,
            },
        )
        raise

    print("COMPLETE Phase 6 prospective Medium workflow")
    print(f"summary={output_root / 'summary.json'}")
    print(f"selected_structure={selected_structure}")
    print(f"selected_physics_weight={final_arm.physics_weight}")
    print(f"reproduction_matches={str(reproduction_matches).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
