"""Reference-free artifact validation for the frozen Phase 5 comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .phase4_validate import file_sha256, state_dict_tensor_sha256
from .phase5_augmentation import augmentation_yaw_rad


_VARIANT_IDS = ("ordinary", "rotation_augmented", "strict_so2")
_SUMMARY_COMMON = {
    "schema_version": 1,
    "profile_id": "gravity_aware_so2_comparison_v1",
    "base_phase4_config_sha256": (
        "7d6607edfe38725942ec0a4df6cac77af47d1b41d79a7b376ae30b0c5f1772cd"
    ),
    "torch_version": "2.13.0+cpu",
    "torch_cuda": None,
    "seed": 3407,
    "record_count": 764,
    "ordinary_augmented_initial_match": True,
    "strict_initialization_uses_same_seed": True,
    "teacher_is_weak_pseudo_label": True,
    "teacher_covariance_used": False,
    "teacher_input_to_student": False,
    "high_precision_reference_used": False,
    "physics_residual_improvement_claimed": False,
    "accuracy_or_superiority_claimed": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_exact_keys(value: object, expected: set[str], name: str) -> None:
    _require(isinstance(value, dict), f"{name} must be an object")
    _require(set(value) == expected, f"{name} keys differ from the frozen schema")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path.name} must contain a JSON object")
    return value


def _load_training_log(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _require(all(isinstance(record, dict) for record in records), f"invalid {path.name}")
    return records


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(state_dict, dict) and bool(state_dict), f"invalid {path.name}")
    return state_dict


def _finite_tree(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    return False


def _validate_normalizer(normalizer: dict[str, Any]) -> None:
    _require_exact_keys(
        normalizer,
        {
            "fit_split",
            "imu_center",
            "imu_scale",
            "quality_center",
            "quality_scale",
            "teacher_covariance_used",
        },
        "normalizer",
    )
    _require(normalizer["fit_split"] == "train", "normalizer was not fit on train only")
    _require(normalizer["teacher_covariance_used"] is False, "normalizer uses covariance")
    _require(_finite_tree(normalizer), "normalizer contains a nonfinite value")
    for key, length in (
        ("imu_center", 7),
        ("imu_scale", 7),
        ("quality_center", 5),
        ("quality_scale", 5),
    ):
        _require(len(normalizer[key]) == length, f"normalizer field {key} has wrong size")
    _require(
        all(float(value) > 0.0 for key in ("imu_scale", "quality_scale") for value in normalizer[key]),
        "normalizer contains a nonpositive scale",
    )


def _validate_summary_common(summary: dict[str, Any], phase5_config: object) -> None:
    for key, expected in _SUMMARY_COMMON.items():
        _require(summary.get(key) == expected, f"summary boundary mismatch: {key}")
    _require(
        summary["profile_id"] == phase5_config.profile_id,
        "summary profile differs from loaded Phase 5 config",
    )
    _require(
        summary["base_phase4_config_sha256"] == phase5_config.base_phase4_sha256,
        "summary Phase 4 base differs from loaded Phase 5 config",
    )


def _expected_variants(phase5_config: object) -> dict[str, object]:
    variants = {variant.variant_id: variant for variant in phase5_config.variants}
    _require(tuple(variants) == _VARIANT_IDS, "loaded Phase 5 variant order differs from freeze")
    return variants


def _validate_variant_declaration(value: dict[str, Any], variant: object) -> None:
    for key, expected in (
        ("architecture", variant.architecture),
        ("strict_so2", variant.strict_so2),
        ("rotation_augmentation", variant.rotation_augmentation),
        ("physics_weight", 0.0),
        ("trainable_parameter_count", variant.trainable_parameter_count),
    ):
        _require(value.get(key) == expected, f"{variant.variant_id} declaration mismatch: {key}")


def _validate_training_record_common(
    record: dict[str, Any],
    variant: object,
    phase5_config: object,
    optimizer_step: int,
    training_pass_index: int,
    expected_mask_duration_s: int,
) -> None:
    _require(record["optimizer_step"] == optimizer_step, "optimizer-step sequence mismatch")
    _require(record["training_pass_index"] == training_pass_index, "training-pass mismatch")
    expected_yaw = (
        augmentation_yaw_rad(phase5_config.augmentation_seed, training_pass_index)
        if variant.rotation_augmentation
        else 0.0
    )
    _require(record["training_yaw_rad"] == expected_yaw, "training yaw schedule mismatch")
    _require(record["mask_duration_s"] == expected_mask_duration_s, "mask cycle mismatch")
    _require(record["physics_loss"] is None, "no-physics run logged a physics loss")
    _require(record["finite"] is True and _finite_tree(record), "nonfinite training record")


def _validate_validation(value: object, outage_durations_s: tuple[int, ...]) -> float:
    _require_exact_keys(value, {"composite", "scenario_weak_label_losses"}, "validation")
    scenario = value["scenario_weak_label_losses"]
    expected_keys = {"mask_0s"} | {f"mask_{duration}s" for duration in outage_durations_s}
    _require_exact_keys(scenario, expected_keys, "validation scenarios")
    _require(_finite_tree(value), "validation contains a nonfinite value")
    expected_composite = sum(float(scenario[key]) for key in scenario) / len(scenario)
    composite = float(value["composite"])
    _require(composite == expected_composite, "validation composite is inconsistent")
    return composite


def validate_phase5_smoke_run(
    phase4_config: object,
    phase5_config: object,
    run_directory: Path,
) -> dict[str, Any]:
    """Validate one four-step smoke without loading deployable data or reference truth."""

    run_directory = run_directory.resolve()
    _require(run_directory.is_dir(), f"missing smoke directory: {run_directory}")
    summary_path = run_directory / "summary.json"
    normalizer_path = run_directory / "normalizer.json"
    summary = _load_json(summary_path)
    normalizer = _load_json(normalizer_path)
    _require_exact_keys(
        summary,
        set(_SUMMARY_COMMON)
        | {
            "mode",
            "variants",
            "smoke_passed",
        },
        "smoke summary",
    )
    _validate_summary_common(summary, phase5_config)
    _require(summary["mode"] == "smoke", "smoke mode mismatch")
    _require(summary["smoke_passed"] is True, "smoke did not pass its generation gate")
    _validate_normalizer(normalizer)

    expected_variants = _expected_variants(phase5_config)
    variants = summary["variants"]
    _require_exact_keys(variants, set(expected_variants), "smoke variants")
    artifact_hashes = {
        "summary.json": file_sha256(summary_path),
        "normalizer.json": file_sha256(normalizer_path),
    }
    initial_hashes: dict[str, str] = {}
    probes_before: dict[str, float] = {}
    for variant_id, variant in expected_variants.items():
        value = variants[variant_id]
        _require_exact_keys(
            value,
            {
                "architecture",
                "strict_so2",
                "rotation_augmentation",
                "physics_weight",
                "trainable_parameter_count",
                "initial_model_tensor_sha256",
                "final_model_tensor_sha256",
                "common_unrotated_probe_before",
                "common_unrotated_probe_after",
                "common_unrotated_probe_change",
                "all_training_values_finite",
                "optimizer_steps",
                "training_mask_cycle_s",
                "training_pass_yaw_rad",
            },
            f"smoke {variant_id}",
        )
        _validate_variant_declaration(value, variant)
        _require(value["all_training_values_finite"] is True, f"nonfinite {variant_id} smoke")
        _require(value["optimizer_steps"] == phase4_config.smoke_optimizer_steps, "smoke steps mismatch")
        _require(
            value["training_mask_cycle_s"] == list(phase4_config.training_mask_cycle_s),
            "smoke mask declaration mismatch",
        )
        expected_yaw = (
            augmentation_yaw_rad(phase5_config.augmentation_seed, 0)
            if variant.rotation_augmentation
            else 0.0
        )
        _require(value["training_pass_yaw_rad"] == expected_yaw, "smoke yaw mismatch")
        before = float(value["common_unrotated_probe_before"])
        after = float(value["common_unrotated_probe_after"])
        _require(
            math.isfinite(before)
            and math.isfinite(after)
            and value["common_unrotated_probe_change"] == after - before,
            f"invalid {variant_id} smoke probe",
        )
        probes_before[variant_id] = before
        initial_hashes[variant_id] = value["initial_model_tensor_sha256"]

        log_path = run_directory / f"{variant_id}_training.jsonl"
        records = _load_training_log(log_path)
        _require(len(records) == phase4_config.smoke_optimizer_steps, "smoke log length mismatch")
        record_keys = {
            "optimizer_step",
            "training_pass_index",
            "training_yaw_rad",
            "mask_duration_s",
            "total_loss",
            "weak_label_loss",
            "physics_loss",
            "gradient_norm_before_clip",
            "finite",
        }
        for index, record in enumerate(records):
            _require_exact_keys(record, record_keys, f"smoke {variant_id} record")
            _validate_training_record_common(
                record,
                variant,
                phase5_config,
                optimizer_step=index + 1,
                training_pass_index=0,
                expected_mask_duration_s=phase4_config.training_mask_cycle_s[
                    index % len(phase4_config.training_mask_cycle_s)
                ],
            )
        artifact_hashes[log_path.name] = file_sha256(log_path)

        model_path = run_directory / f"{variant_id}_final.pt"
        tensor_hash = state_dict_tensor_sha256(_load_state_dict(model_path))
        _require(
            tensor_hash == value["final_model_tensor_sha256"],
            f"{variant_id} smoke checkpoint hash mismatch",
        )
        artifact_hashes[f"{model_path.name}:tensor"] = tensor_hash

    _require(
        initial_hashes["ordinary"] == initial_hashes["rotation_augmented"],
        "ordinary and augmented smoke initializations differ",
    )
    _require(
        probes_before["ordinary"] == probes_before["rotation_augmented"],
        "ordinary and augmented unrotated initial probes differ",
    )
    return {
        "run_directory": str(run_directory),
        "profile_id": phase5_config.profile_id,
        "training_artifact_hashes": artifact_hashes,
        "initial_model_tensor_sha256": initial_hashes,
        "weak_label_probe_after": {
            variant_id: float(variants[variant_id]["common_unrotated_probe_after"])
            for variant_id in expected_variants
        },
        "accuracy_or_superiority_claimed": False,
        "validated": True,
    }


def _steps_per_training_pass(phase4_config: object) -> int:
    train_start, train_stop = phase4_config.splits["train"]
    return math.ceil((train_stop - train_start) / phase4_config.unroll_steps)


def validate_phase5_development_run(
    phase4_config: object,
    phase5_config: object,
    run_directory: Path,
    diagnostics_name: str | None = None,
    failed_diagnostics_name: str | None = None,
) -> dict[str, Any]:
    """Validate one bounded development run and optional diagnostic evidence."""

    run_directory = run_directory.resolve()
    _require(run_directory.is_dir(), f"missing development directory: {run_directory}")
    for name in (diagnostics_name, failed_diagnostics_name):
        if name is not None:
            _require(Path(name).name == name and Path(name).suffix == ".json", "bad diagnostic name")

    summary_path = run_directory / "summary.json"
    normalizer_path = run_directory / "normalizer.json"
    summary = _load_json(summary_path)
    normalizer = _load_json(normalizer_path)
    _require_exact_keys(
        summary,
        set(_SUMMARY_COMMON)
        | {
            "mode",
            "optimizer_steps_per_variant",
            "variants",
            "selection_uses_only_common_unrotated_validation_weak_label_loss",
            "development_passed",
        },
        "development summary",
    )
    _validate_summary_common(summary, phase5_config)
    _require(summary["mode"] == "bounded_development", "development mode mismatch")
    _require(
        summary["optimizer_steps_per_variant"] == phase4_config.development_optimizer_steps,
        "development step declaration mismatch",
    )
    _require(
        summary["selection_uses_only_common_unrotated_validation_weak_label_loss"] is True,
        "selection boundary mismatch",
    )
    _require(summary["development_passed"] is True, "development generation gate failed")
    _validate_normalizer(normalizer)

    expected_variants = _expected_variants(phase5_config)
    variants = summary["variants"]
    _require_exact_keys(variants, set(expected_variants), "development variants")
    artifact_hashes = {
        "summary.json": file_sha256(summary_path),
        "normalizer.json": file_sha256(normalizer_path),
    }
    initial_hashes: dict[str, str] = {}
    initial_validations: dict[str, object] = {}
    training_passes = _steps_per_training_pass(phase4_config)
    train_start, train_stop = phase4_config.splits["train"]
    for variant_id, variant in expected_variants.items():
        value = variants[variant_id]
        _require_exact_keys(
            value,
            {
                "architecture",
                "strict_so2",
                "rotation_augmentation",
                "physics_weight",
                "trainable_parameter_count",
                "initial_model_tensor_sha256",
                "best_model_tensor_sha256",
                "final_model_tensor_sha256",
                "initial_validation",
                "best_validation_composite",
                "best_optimizer_step",
                "final_validation",
                "optimizer_steps",
                "checkpoint_every_steps",
                "training_pass_yaws_rad",
                "all_training_values_finite",
            },
            f"development {variant_id}",
        )
        _validate_variant_declaration(value, variant)
        _require(value["all_training_values_finite"] is True, f"nonfinite {variant_id} development")
        _require(value["optimizer_steps"] == phase4_config.development_optimizer_steps, "step mismatch")
        _require(
            value["checkpoint_every_steps"] == phase5_config.checkpoint_every_steps,
            "checkpoint schedule mismatch",
        )
        initial_validations[variant_id] = value["initial_validation"]
        _validate_validation(value["initial_validation"], phase4_config.outage_durations_s)
        initial_hashes[variant_id] = value["initial_model_tensor_sha256"]

        log_path = run_directory / f"{variant_id}_training.jsonl"
        records = _load_training_log(log_path)
        _require(len(records) == phase4_config.development_optimizer_steps, "log length mismatch")
        validation_scores: dict[int, float] = {}
        observed_pass_yaws: dict[str, float] = {}
        base_record_keys = {
            "optimizer_step",
            "training_pass_index",
            "training_yaw_rad",
            "train_start_index",
            "train_stop_index",
            "mask_duration_s",
            "total_loss",
            "weak_label_loss",
            "physics_loss",
            "gradient_norm_before_clip",
            "finite",
        }
        for index, record in enumerate(records):
            optimizer_step = index + 1
            pass_index = index // training_passes
            offset = index % training_passes
            expected_start = train_start + offset * phase4_config.unroll_steps
            expected_stop = min(expected_start + phase4_config.unroll_steps, train_stop)
            has_validation = optimizer_step % phase5_config.checkpoint_every_steps == 0
            expected_keys = base_record_keys | ({"validation"} if has_validation else set())
            _require_exact_keys(record, expected_keys, f"development {variant_id} record")
            _validate_training_record_common(
                record,
                variant,
                phase5_config,
                optimizer_step=optimizer_step,
                training_pass_index=pass_index,
                expected_mask_duration_s=phase4_config.training_mask_cycle_s[
                    index % len(phase4_config.training_mask_cycle_s)
                ],
            )
            _require(
                record["train_start_index"] == expected_start
                and record["train_stop_index"] == expected_stop,
                "truncated-unroll cursor mismatch",
            )
            observed_pass_yaws[str(pass_index)] = float(record["training_yaw_rad"])
            if has_validation:
                validation_scores[optimizer_step] = _validate_validation(
                    record["validation"], phase4_config.outage_durations_s
                )
        _require(
            value["training_pass_yaws_rad"] == observed_pass_yaws,
            f"{variant_id} pass-yaw summary mismatch",
        )
        expected_best_step, expected_best_score = min(
            validation_scores.items(), key=lambda item: (item[1], item[0])
        )
        _require(value["best_optimizer_step"] == expected_best_step, "best step mismatch")
        _require(value["best_validation_composite"] == expected_best_score, "best score mismatch")
        final_score = _validate_validation(value["final_validation"], phase4_config.outage_durations_s)
        _require(
            value["final_validation"] == records[-1]["validation"]
            and final_score == validation_scores[phase4_config.development_optimizer_steps],
            "final validation differs from the final checkpoint",
        )
        artifact_hashes[log_path.name] = file_sha256(log_path)

        for checkpoint in ("best", "final"):
            model_path = run_directory / f"{variant_id}_{checkpoint}.pt"
            tensor_hash = state_dict_tensor_sha256(_load_state_dict(model_path))
            _require(
                tensor_hash == value[f"{checkpoint}_model_tensor_sha256"],
                f"{variant_id} {checkpoint} checkpoint hash mismatch",
            )
            artifact_hashes[f"{model_path.name}:tensor"] = tensor_hash

    _require(
        initial_hashes["ordinary"] == initial_hashes["rotation_augmented"],
        "ordinary and augmented initializations differ",
    )
    _require(
        initial_validations["ordinary"] == initial_validations["rotation_augmented"],
        "ordinary and augmented initial validations differ",
    )

    diagnostic_report: dict[str, Any] | None = None
    diagnostic_hashes: dict[str, str] = {}
    if diagnostics_name is not None:
        diagnostics_path = run_directory / diagnostics_name
        diagnostics = _load_json(diagnostics_path)
        diagnostic_report = _validate_legacy_diagnostics_audit(
            diagnostics, phase4_config, phase5_config
        )
        diagnostic_hashes[diagnostics_name] = file_sha256(diagnostics_path)
    initial_float32_audit: dict[str, Any] | None = None
    if failed_diagnostics_name is not None:
        failed_path = run_directory / failed_diagnostics_name
        failed = _load_json(failed_path)
        initial_float32_audit = _validate_initial_float32_audit_provenance(
            failed, phase5_config
        )
        diagnostic_hashes[failed_diagnostics_name] = file_sha256(failed_path)

    return {
        "run_directory": str(run_directory),
        "profile_id": phase5_config.profile_id,
        "training_artifact_hashes": artifact_hashes,
        "diagnostic_artifact_hashes": diagnostic_hashes,
        "initial_model_tensor_sha256": initial_hashes,
        "initial_validation": initial_validations,
        "best_validation_composite": {
            variant_id: float(variants[variant_id]["best_validation_composite"])
            for variant_id in expected_variants
        },
        "best_optimizer_step": {
            variant_id: int(variants[variant_id]["best_optimizer_step"])
            for variant_id in expected_variants
        },
        "diagnostics": diagnostic_report,
        "initial_float32_audit_provenance": initial_float32_audit,
        "accuracy_or_superiority_claimed": False,
        "validated": True,
    }


def _validate_reference_free_variant(value: dict[str, Any], variant_id: str) -> float:
    for key in (
        "degeneracy_checks_passed",
        "causality_and_memory_checks_passed",
        "numerical_checks_passed",
    ):
        _require(value.get(key) is True, f"{variant_id} diagnostic gate failed: {key}")
    _require(
        value["teacher_copying_tendency"]["all_outputs_exact_teacher_copy"] is False,
        f"{variant_id} copies all teacher outputs",
    )
    _require(
        value["spp_copying"]["all_outputs_exact_spp_copy"] is False,
        f"{variant_id} copies all SPP positions",
    )
    _require(
        value["gnss_correction"]["all_corrections_zero_at_1e-8"] is False,
        f"{variant_id} produces no GNSS correction",
    )
    _require(
        value["reject_all_gnss"]["normal_and_reject_all_outputs_identical_at_1e-7"]
        is False,
        f"{variant_id} rejects all GNSS",
    )
    _require(
        value["future_information"]["full_and_truncated_prefix_bit_identical"] is True,
        f"{variant_id} future-prefix audit failed",
    )
    _require(
        value["absolute_position_memory"]["translation_consistency_pass"] is True
        and value["absolute_position_memory"]["route_or_session_generalization_claimed"] is False,
        f"{variant_id} translation or route-claim boundary failed",
    )
    numerical = value["numerical_stability"]
    _require(numerical["all_checked_outputs_finite"] is True, f"{variant_id} is nonfinite")
    _require(
        float(numerical["maximum_speed_mps"])
        <= float(numerical["maximum_allowed_speed_mps"]),
        f"{variant_id} exceeds the fixed speed diagnostic",
    )
    masks = value["controlled_outages"]
    _require_exact_keys(masks, {"20s", "30s"}, f"{variant_id} controlled outages")
    for duration in (20, 30):
        mask = masks[f"{duration}s"]
        for key, expected in (
            ("masked_valid_pvt_count", duration),
            ("prefix_before_mask_bit_identical_to_unmasked", True),
            ("recovery_gnss_available", True),
            ("all_outputs_finite", True),
        ):
            _require(mask.get(key) == expected, f"{variant_id} {duration}s outage mismatch: {key}")
        _require(_finite_tree(mask), f"{variant_id} {duration}s outage contains nonfinite data")
    physics = value["independent_physics_residual_on_diagnostic_split"]
    huber = float(physics["normalized_component_mean_huber"])
    _require(math.isfinite(huber) and int(physics["valid_step_count"]) == 104, "bad physics diagnostic")
    return huber


def _validate_so2_dtype(
    value: dict[str, Any],
    phase4_config: object,
    phase5_config: object,
    dtype: str,
    strict_expected: bool,
) -> dict[str, bool]:
    if dtype == "float64":
        atol, rtol = phase5_config.float64_atol, phase5_config.float64_rtol
    else:
        atol, rtol = phase5_config.float32_atol, phase5_config.float32_rtol
    for key, expected in (
        ("dtype", dtype),
        ("atol", atol),
        ("rtol", rtol),
        ("angles_rad", list(phase5_config.property_angles_rad)),
        ("complete_real_sequence_step_count", phase4_config.splits["diagnostic"][1]),
        ("strict_equivariance_expected", strict_expected),
        ("sensor_remounting_tested", False),
    ):
        _require(value.get(key) == expected, f"{dtype} SO(2) field mismatch: {key}")
    angle_results = value["angle_commutation"]
    expected_angle_keys = {f"{angle:+.2f}" for angle in phase5_config.property_angles_rad}
    _require_exact_keys(angle_results, expected_angle_keys, f"{dtype} angle commutation")
    angle_passes = [case["passed"] is True for case in angle_results.values()]
    strict_property_passed = (
        value["identity"]["passed"] is True
        and value["composition"]["passed"] is True
        and all(angle_passes)
    )
    counterexample_present = any(not passed for passed in angle_passes)
    classification_passed = (
        strict_property_passed if strict_expected else counterexample_present
    )
    _require(
        value["strict_property_passed"] is strict_property_passed,
        f"{dtype} strict-property result is internally inconsistent",
    )
    _require(
        value["non_strict_counterexample_present"] is counterexample_present,
        f"{dtype} counterexample result is internally inconsistent",
    )
    _require(
        value["classification_passed"] is classification_passed,
        f"{dtype} classification result is internally inconsistent",
    )
    _require(_finite_tree(value), f"{dtype} SO(2) diagnostic contains nonfinite data")
    return {
        "strict_property_passed": strict_property_passed,
        "counterexample_present": counterexample_present,
        "classification_passed": classification_passed,
    }


def _validate_legacy_diagnostics_audit(
    diagnostics: dict[str, Any],
    phase4_config: object,
    phase5_config: object,
) -> dict[str, Any]:
    _require_exact_keys(
        diagnostics,
        {
            "schema_version",
            "profile_id",
            "variants",
            "diagnostics_passed",
            "float32_complete_mapping_gate_passed",
            "teacher_is_weak_pseudo_label",
            "teacher_covariance_used",
            "high_precision_reference_used",
            "physics_residual_improvement_claimed",
            "accuracy_or_superiority_claim_supported",
            "sensor_remounting_generalization_claim_supported",
        },
        "legacy diagnostics audit",
    )
    for key, expected in (
        ("schema_version", 1),
        ("profile_id", phase5_config.profile_id),
        ("teacher_is_weak_pseudo_label", True),
        ("teacher_covariance_used", False),
        ("high_precision_reference_used", False),
        ("physics_residual_improvement_claimed", False),
        ("accuracy_or_superiority_claim_supported", False),
        ("sensor_remounting_generalization_claim_supported", False),
    ):
        _require(diagnostics.get(key) == expected, f"diagnostic boundary mismatch: {key}")
    _require(
        isinstance(diagnostics.get("diagnostics_passed"), bool)
        and isinstance(diagnostics.get("float32_complete_mapping_gate_passed"), bool),
        "legacy diagnostic outcome fields must be booleans",
    )
    variants = diagnostics["variants"]
    expected_variants = _expected_variants(phase5_config)
    _require_exact_keys(variants, set(expected_variants), "diagnostic variants")
    physics_huber: dict[str, float] = {}
    float64_classification: dict[str, bool] = {}
    float32_classification: dict[str, bool] = {}
    float64_strict_property: dict[str, bool] = {}
    float32_strict_property: dict[str, bool] = {}
    for variant_id, variant in expected_variants.items():
        value = variants[variant_id]
        physics_huber[variant_id] = _validate_reference_free_variant(value, variant_id)
        mapping = value["so2_complete_mapping"]
        _require_exact_keys(
            mapping,
            {
                "float32",
                "float64",
                "structural_classification_passed",
                "float32_runtime_classification_passed",
            },
            f"{variant_id} complete mapping",
        )
        float32_result = _validate_so2_dtype(
            mapping["float32"], phase4_config, phase5_config, "float32", variant.strict_so2
        )
        float64_result = _validate_so2_dtype(
            mapping["float64"], phase4_config, phase5_config, "float64", variant.strict_so2
        )
        float32_classification[variant_id] = float32_result["classification_passed"]
        float64_classification[variant_id] = float64_result["classification_passed"]
        float32_strict_property[variant_id] = float32_result["strict_property_passed"]
        float64_strict_property[variant_id] = float64_result["strict_property_passed"]
        _require(
            mapping["structural_classification_passed"]
            is float64_result["classification_passed"],
            f"{variant_id} legacy float64 summary mismatch",
        )
        _require(
            mapping["float32_runtime_classification_passed"]
            is float32_result["classification_passed"],
            f"{variant_id} float32 classification mismatch",
        )
        _require(
            value["strict_so2_structural_claim_allowed"]
            is (variant.strict_so2 and float64_result["strict_property_passed"]),
            f"{variant_id} strict structural claim mismatch",
        )
        _require(
            value["strict_so2_float32_full_rollout_claim_allowed"]
            is (variant.strict_so2 and float32_result["strict_property_passed"]),
            f"{variant_id} float32 claim summary mismatch",
        )
        _require(
            value["rotation_augmentation_is_empirical_only"] is variant.rotation_augmentation,
            f"{variant_id} augmentation classification mismatch",
        )
    _require(
        math.isclose(physics_huber["ordinary"], 0.374354878, rel_tol=0.0, abs_tol=5e-10),
        "ordinary Phase 4 no-physics residual was not preserved",
    )
    reference_free_safety_checks_passed = True
    float64_structural_so2_checks_passed = all(float64_classification.values())
    float32_deployed_rollout_checks_passed = all(float32_classification.values())
    _require(
        diagnostics["diagnostics_passed"]
        is (reference_free_safety_checks_passed and float64_structural_so2_checks_passed),
        "legacy diagnostics_passed field is inconsistent with its original computation",
    )
    _require(
        diagnostics["float32_complete_mapping_gate_passed"]
        is float32_deployed_rollout_checks_passed,
        "legacy float32 outcome field is inconsistent",
    )
    return {
        "legacy_schema_version": 1,
        "legacy_diagnostics_passed_field": diagnostics["diagnostics_passed"],
        "legacy_field_semantics": "reference_free_safety_and_float64_structural_only",
        "reference_free_safety_checks_passed": reference_free_safety_checks_passed,
        "float64_structural_so2_checks_passed": float64_structural_so2_checks_passed,
        "float32_deployed_rollout_checks_passed": float32_deployed_rollout_checks_passed,
        "strict_so2_float64_property_passed": float64_strict_property["strict_so2"],
        "strict_so2_float32_property_passed": float32_strict_property["strict_so2"],
        "ordinary_and_augmented_float64_counterexamples_present": all(
            not float64_strict_property[variant_id]
            for variant_id in ("ordinary", "rotation_augmented")
        ),
        "dtype_outcomes_are_audit_results_not_success_conditions": True,
        "diagnostic_physics_huber": physics_huber,
        "physics_residual_improvement_claimed": False,
        "accuracy_or_superiority_claim_supported": False,
        "sensor_remounting_generalization_claim_supported": False,
    }


def _validate_initial_float32_audit_provenance(
    diagnostics: dict[str, Any], phase5_config: object
) -> dict[str, Any]:
    _require(diagnostics.get("schema_version") == 1, "failed diagnostics schema mismatch")
    _require(diagnostics.get("profile_id") == phase5_config.profile_id, "failed profile mismatch")
    _require(diagnostics.get("diagnostics_passed") is False, "failed diagnostic was overwritten")
    variants = diagnostics.get("variants")
    _require(isinstance(variants, dict) and set(variants) == set(_VARIANT_IDS), "failed variants mismatch")
    strict = variants["strict_so2"]
    mapping = strict.get("so2_complete_mapping")
    _require(isinstance(mapping, dict), "missing failed strict SO(2) evidence")
    _require(mapping.get("dtype") == "float32", "failed audit was not the frozen float32 audit")
    _require(mapping.get("strict_property_passed") is False, "failed strict property was hidden")
    _require(mapping.get("classification_passed") is False, "failed classification was hidden")
    _require(strict.get("strict_so2_claim_allowed") is False, "failed report allowed strict claim")
    for key in (
        "teacher_covariance_used",
        "high_precision_reference_used",
        "physics_residual_improvement_claimed",
        "accuracy_or_superiority_claim_supported",
        "sensor_remounting_generalization_claim_supported",
    ):
        _require(diagnostics.get(key) is False, f"failed-report boundary mismatch: {key}")
    return {
        "provenance_retained": True,
        "audit_dtype": "float32",
        "strict_so2_complete_rollout_passed": False,
        "outcome": "exceeded_predeclared_tolerance",
        "outcome_is_not_a_success_gate": True,
    }


def compare_phase5_smoke_runs(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Require byte-identical logs/JSON and tensor-identical smoke checkpoints."""

    _require(first["profile_id"] == second["profile_id"], "smoke profile mismatch")
    _require(
        first["training_artifact_hashes"] == second["training_artifact_hashes"],
        "Phase 5 smoke training artifacts are not deterministic",
    )
    _require(
        first["initial_model_tensor_sha256"] == second["initial_model_tensor_sha256"],
        "smoke initialization mismatch",
    )
    return {
        "byte_and_tensor_identical": True,
        "compared_artifact_count": len(first["training_artifact_hashes"]),
    }


def compare_phase5_development_runs(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    """Require exact independent reproduction of all development training artifacts."""

    _require(first["profile_id"] == second["profile_id"], "development profile mismatch")
    _require(
        first["training_artifact_hashes"] == second["training_artifact_hashes"],
        "Phase 5 development training artifacts are not deterministic",
    )
    for key in (
        "initial_model_tensor_sha256",
        "initial_validation",
        "best_validation_composite",
        "best_optimizer_step",
    ):
        _require(first[key] == second[key], f"development comparison mismatch: {key}")
    return {
        "byte_and_tensor_identical": True,
        "compared_artifact_count": len(first["training_artifact_hashes"]),
        "diagnostics_compared": False,
    }
