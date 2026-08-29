"""Read-only artifact validation for the prospective Phase 6 workflow."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .phase4_validate import state_dict_tensor_sha256
from .phase5_augmentation import augmentation_yaw_rad
from .phase6_config import Phase6Config, file_sha256
from .phase6_selection import interaction_direction, select_physics, select_structure


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path.name} must contain an object")
    return value


def _tensor_hash(path: Path) -> str:
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(state_dict, dict) and bool(state_dict), f"invalid {path.name}")
    return state_dict_tensor_sha256(state_dict)


def _finite_tree(value: object) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    return False


def _arm_id(structure: str, physics_weight: float) -> str:
    return f"{structure}_{'p0' if physics_weight == 0.0 else 'p01'}"


def _run_value(
    runs: dict[str, Any], arm_id: str, seed: int, metric: str
) -> float:
    return float(runs[arm_id][str(seed)][metric])


def validate_phase6_decisions(
    summary: dict[str, Any],
    config: Phase6Config,
) -> dict[str, object]:
    """Recompute every selection decision from the frozen run metrics."""

    runs = summary["runs"]
    for arm in runs.values():
        for value in arm.values():
            _require(value["qualification_passed"] is True, "an unqualified run entered selection")
    structure_validation = {
        structure: {
            seed: _run_value(
                runs, _arm_id(structure, 0.0), seed, "best_validation_composite"
            )
            for seed in config.seeds
        }
        for structure in config.structure_variants
    }
    expected_structure = select_structure(structure_validation, config.seeds)
    _require(
        summary["structure_decision"] == expected_structure,
        "structure decision differs from the prospective rule",
    )
    selected_structure = str(expected_structure["selected_structure"])
    selected_arm_ids = {
        "no_physics": _arm_id(selected_structure, 0.0),
        "physics": _arm_id(selected_structure, 0.1),
    }
    physics_validation = {
        label: {
            seed: _run_value(runs, arm_id, seed, "best_validation_composite")
            for seed in config.seeds
        }
        for label, arm_id in selected_arm_ids.items()
    }
    physics_huber = {
        label: {
            seed: _run_value(runs, arm_id, seed, "independent_physics_huber")
            for seed in config.seeds
        }
        for label, arm_id in selected_arm_ids.items()
    }
    expected_physics = select_physics(physics_validation, physics_huber, config.seeds)
    _require(
        summary["physics_decision"] == expected_physics,
        "physics decision differs from the prospective rule",
    )

    interaction_arms = {
        "ordinary_no_physics": _arm_id("ordinary", 0.0),
        "ordinary_physics": _arm_id("ordinary", 0.1),
        "strict_no_physics": _arm_id("strict_so2", 0.0),
        "strict_physics": _arm_id("strict_so2", 0.1),
    }
    expected_interaction: dict[str, object] = {}
    for name, metric in (
        ("common_validation_composite", "best_validation_composite"),
        ("independent_physics_huber", "independent_physics_huber"),
    ):
        matrix = {
            label: {
                seed: _run_value(runs, arm_id, seed, metric)
                for seed in config.seeds
            }
            for label, arm_id in interaction_arms.items()
        }
        expected_interaction[name] = interaction_direction(matrix, config.seeds)
    _require(
        summary["interaction_report_only"] == expected_interaction,
        "interaction report differs from the frozen difference-of-differences",
    )
    final = summary["final_primary"]
    expected_weight = float(expected_physics["selected_physics_weight"])
    expected_arm = _arm_id(selected_structure, expected_weight)
    _require(final["arm_id"] == expected_arm, "final arm differs from selection")
    _require(final["structure"] == selected_structure, "final structure differs")
    _require(final["physics_weight"] == expected_weight, "final physics weight differs")
    _require(
        final["designated_seed"] == config.designated_checkpoint_seed,
        "final seed differs from the freeze",
    )
    expected_checkpoint_hash = runs[expected_arm][str(config.designated_checkpoint_seed)][
        "best_model_tensor_sha256"
    ]
    _require(
        final["checkpoint_tensor_sha256"] == expected_checkpoint_hash,
        "final checkpoint hash differs from the designated run",
    )
    return {
        "selected_structure": selected_structure,
        "selected_physics_weight": expected_weight,
        "final_arm_id": expected_arm,
        "decisions_validated": True,
    }


def _validate_training_run(
    directory: Path,
    value: dict[str, Any],
    structure: str,
    physics_weight: float,
    seed: int,
    phase4_config: object,
    phase5_config: object,
    phase6_config: Phase6Config,
    diagnostics_required: bool,
) -> int:
    _require(directory.is_dir(), f"missing run directory: {directory}")
    disk_summary = _load_json(directory / "summary.json")
    _require(disk_summary == value, f"root summary differs from {directory}")
    _require(value["arm_id"] == _arm_id(structure, physics_weight), "arm ID mismatch")
    _require(value["structure"] == structure, "structure mismatch")
    _require(value["physics_weight"] == physics_weight, "physics weight mismatch")
    _require(value["seed"] == seed, "seed mismatch")
    _require(value["optimizer_steps"] == phase6_config.optimizer_steps, "step mismatch")
    _require(
        value["checkpoint_every_steps"] == phase6_config.checkpoint_every_steps,
        "checkpoint cadence mismatch",
    )
    _require(value["all_training_values_finite"] is True, "training was nonfinite")
    _require(value["teacher_covariance_used"] is False, "teacher covariance entered training")
    _require(
        value["high_precision_reference_used"] is False,
        "high-precision reference entered training",
    )
    log_path = directory / "training.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    _require(len(records) == phase6_config.optimizer_steps, "training log length mismatch")
    _require(file_sha256(log_path) == value["training_log_sha256"], "log hash mismatch")
    train_start, train_stop = phase4_config.splits["train"]
    steps_per_pass = math.ceil((train_stop - train_start) / phase4_config.unroll_steps)
    validation_scores: dict[int, float] = {}
    for index, record in enumerate(records):
        optimizer_step = index + 1
        pass_index = index // steps_per_pass
        offset = index % steps_per_pass
        expected_start = train_start + offset * phase4_config.unroll_steps
        expected_stop = min(expected_start + phase4_config.unroll_steps, train_stop)
        expected_yaw = (
            augmentation_yaw_rad(phase5_config.augmentation_seed, pass_index)
            if structure == "rotation_augmented"
            else 0.0
        )
        _require(record["optimizer_step"] == optimizer_step, "optimizer sequence mismatch")
        _require(record["training_pass_index"] == pass_index, "pass sequence mismatch")
        _require(record["training_yaw_rad"] == expected_yaw, "augmentation schedule mismatch")
        _require(
            record["train_start_index"] == expected_start
            and record["train_stop_index"] == expected_stop,
            "unroll cursor mismatch",
        )
        _require(
            record["mask_duration_s"]
            == phase4_config.training_mask_cycle_s[
                index % len(phase4_config.training_mask_cycle_s)
            ],
            "mask cycle mismatch",
        )
        _require(record["finite"] is True and _finite_tree(record), "nonfinite log record")
        if physics_weight == 0.0:
            _require(record["physics_loss"] is None, "no-physics arm logged physics")
        else:
            _require(isinstance(record["physics_loss"], float), "physics loss is missing")
        has_validation = optimizer_step % phase6_config.checkpoint_every_steps == 0
        _require(("validation" in record) == has_validation, "validation cadence mismatch")
        if has_validation:
            validation = record["validation"]
            scenario = validation["scenario_weak_label_losses"]
            composite = sum(float(item) for item in scenario.values()) / len(scenario)
            _require(composite == validation["composite"], "validation composite mismatch")
            validation_scores[optimizer_step] = float(composite)
    expected_step, expected_score = min(
        validation_scores.items(), key=lambda item: (item[1], item[0])
    )
    _require(value["best_optimizer_step"] == expected_step, "best step mismatch")
    _require(value["best_validation_composite"] == expected_score, "best score mismatch")
    best_hash = _tensor_hash(directory / "best.pt")
    final_hash = _tensor_hash(directory / "final.pt")
    _require(best_hash == value["best_model_tensor_sha256"], "best checkpoint mismatch")
    _require(final_hash == value["final_model_tensor_sha256"], "final checkpoint mismatch")
    artifact_count = 4
    if diagnostics_required:
        diagnostic_path = directory / "diagnostics.json"
        diagnostic = _load_json(diagnostic_path)
        _require(
            file_sha256(diagnostic_path) == value["diagnostics_sha256"],
            "diagnostic hash mismatch",
        )
        _require(diagnostic["qualification_passed"] is True, "qualification failed")
        _require(value["qualification_passed"] is True, "summary qualification failed")
        _require(
            diagnostic["high_precision_reference_used"] is False,
            "diagnostic used high-precision reference",
        )
        low_quality = diagnostic["low_quality_stratum"]
        _require(low_quality["union_epoch_count"] == 53, "low-quality count mismatch")
        _require(low_quality["enters_model_selection"] is False, "stratum entered selection")
        _require(low_quality["robustness_gate_passed"] is True, "stratum gate failed")
        mapping = diagnostic["strict_so2_complete_mapping"]
        if structure == "strict_so2":
            _require(mapping is not None, "strict mapping report missing")
            _require(
                mapping["float64"]["classification_passed"] is True,
                "strict float64 structural classification failed",
            )
        else:
            _require(mapping is None, "non-strict arm has a strict mapping report")
        artifact_count += 1
    return artifact_count


def validate_phase6_run(
    phase4_config: object,
    phase5_config: object,
    phase6_config: Phase6Config,
    phase6_config_path: Path,
    run_directory: Path,
) -> dict[str, object]:
    """Validate the complete workflow without reading deployable data or reference truth."""

    run_directory = run_directory.resolve()
    _require(run_directory.is_dir(), f"missing Phase 6 directory: {run_directory}")
    _require(not (run_directory / "failure.json").exists(), "Phase 6 failure artifact exists")
    summary = _load_json(run_directory / "summary.json")
    _require(summary["profile_id"] == phase6_config.profile_id, "profile mismatch")
    _require(
        summary["phase6_config_sha256"] == file_sha256(phase6_config_path),
        "Phase 6 config hash mismatch",
    )
    _require(summary["seeds"] == list(phase6_config.seeds), "seed declaration mismatch")
    _require(summary["optimizer_steps"] == 200, "optimizer budget mismatch")
    _require(summary["checkpoint_every_steps"] == 10, "checkpoint cadence mismatch")
    for key in (
        "all_required_runs_qualified",
        "workflow_completed",
    ):
        _require(summary[key] is True, f"summary gate failed: {key}")
    for key in (
        "selection_uses_reference_truth",
        "high_precision_reference_used",
        "teacher_covariance_used",
        "predictive_covariance_implemented",
        "cross_receiver_or_route_data_used",
        "accuracy_or_superiority_claimed",
    ):
        _require(summary[key] is False, f"information/claim boundary failed: {key}")

    runs = summary["runs"]
    selected_structure = summary["structure_decision"]["selected_structure"]
    expected_arms = [
        _arm_id("ordinary", 0.0),
        _arm_id("rotation_augmented", 0.0),
        _arm_id("strict_so2", 0.0),
        _arm_id("ordinary", 0.1),
        _arm_id("strict_so2", 0.1),
    ]
    if selected_structure == "rotation_augmented":
        expected_arms.append(_arm_id("rotation_augmented", 0.1))
    _require(set(runs) == set(expected_arms), "run arm set differs from the freeze")
    artifacts = 2
    for arm_id in expected_arms:
        structure, suffix = arm_id.rsplit("_", 1)
        physics_weight = 0.0 if suffix == "p0" else 0.1
        _require(
            set(runs[arm_id]) == {str(seed) for seed in phase6_config.seeds},
            "run seed set differs from the freeze",
        )
        for seed in phase6_config.seeds:
            artifacts += _validate_training_run(
                run_directory / "runs" / arm_id / f"seed_{seed}",
                runs[arm_id][str(seed)],
                structure,
                physics_weight,
                seed,
                phase4_config,
                phase5_config,
                phase6_config,
                diagnostics_required=True,
            )
    decisions = validate_phase6_decisions(summary, phase6_config)
    final = summary["final_primary"]
    checkpoint_path = run_directory / final["checkpoint_relative_path"]
    _require(_tensor_hash(checkpoint_path) == final["checkpoint_tensor_sha256"], "final checkpoint mismatch")
    reproduction = summary["reproduction"]
    _require(reproduction["count"] == 1, "reproduction count differs from freeze")
    _require(
        reproduction["matches_designated_training_artifacts"] is True,
        "final reproduction differs",
    )
    artifacts += _validate_training_run(
        run_directory / "reproduction",
        reproduction["summary"],
        final["structure"],
        final["physics_weight"],
        phase6_config.designated_checkpoint_seed,
        phase4_config,
        phase5_config,
        phase6_config,
        diagnostics_required=False,
    )
    return {
        **decisions,
        "run_directory": str(run_directory),
        "validated_artifact_count": artifacts,
        "high_precision_reference_used": False,
        "validated": True,
    }
