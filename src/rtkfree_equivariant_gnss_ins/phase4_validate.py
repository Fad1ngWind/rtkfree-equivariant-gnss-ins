"""Reference-free structural validation for Phase 4 development artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_tensor_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"model entry {name!r} is not a tensor")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path.name} must contain a JSON object")
    return value


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


def _load_training_log(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _require(all(isinstance(record, dict) for record in records), f"invalid {path.name}")
    return records


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(state_dict, dict) and bool(state_dict), f"invalid model file {path.name}")
    return state_dict


def _validate_diagnostics(
    diagnostics: dict[str, Any],
    config: object,
) -> dict[str, float]:
    _require(diagnostics.get("schema_version") == 1, "unexpected diagnostics schema")
    _require(diagnostics.get("profile_id") == config.profile_id, "diagnostics profile mismatch")
    for key, expected in (
        ("diagnostics_passed", True),
        ("teacher_is_weak_pseudo_label", True),
        ("teacher_covariance_used", False),
        ("high_precision_reference_used", False),
        ("accuracy_or_superiority_claim_supported", False),
    ):
        _require(diagnostics.get(key) is expected, f"diagnostics boundary mismatch: {key}")

    expected_variants = {variant.variant_id for variant in config.variants}
    variants = diagnostics.get("variants")
    _require(isinstance(variants, dict), "missing diagnostic variants")
    _require(set(variants) == expected_variants, "diagnostic variant set mismatch")
    physics_huber: dict[str, float] = {}
    for variant_id, value in variants.items():
        _require(value.get("degeneracy_checks_passed") is True, f"{variant_id} degeneracy gate failed")
        _require(
            value.get("causality_and_memory_checks_passed") is True,
            f"{variant_id} causality gate failed",
        )
        _require(value.get("numerical_checks_passed") is True, f"{variant_id} numerical gate failed")
        _require(
            value["teacher_copying_tendency"]["all_outputs_exact_teacher_copy"] is False,
            f"{variant_id} copies every teacher state",
        )
        _require(
            value["spp_copying"]["all_outputs_exact_spp_copy"] is False,
            f"{variant_id} copies every SPP position",
        )
        _require(
            value["gnss_correction"]["all_corrections_zero_at_1e-8"] is False,
            f"{variant_id} emits zero GNSS corrections",
        )
        _require(
            value["reject_all_gnss"]["normal_and_reject_all_outputs_identical_at_1e-7"] is False,
            f"{variant_id} rejects all GNSS",
        )
        _require(
            value["future_information"]["full_and_truncated_prefix_bit_identical"] is True,
            f"{variant_id} future-prefix audit failed",
        )
        _require(
            value["absolute_position_memory"]["translation_consistency_pass"] is True,
            f"{variant_id} translation audit failed",
        )
        numerical = value["numerical_stability"]
        _require(numerical["all_checked_outputs_finite"] is True, f"{variant_id} is nonfinite")
        _require(
            float(numerical["maximum_speed_mps"])
            <= float(numerical["maximum_allowed_speed_mps"]),
            f"{variant_id} exceeds the fixed speed diagnostic",
        )
        physics = value["independent_physics_residual_on_diagnostic_split"]
        huber = float(physics["normalized_component_mean_huber"])
        _require(math.isfinite(huber) and int(physics["valid_step_count"]) > 0, "invalid physics diagnostic")
        physics_huber[variant_id] = huber

        masks = value.get("controlled_outages")
        _require(isinstance(masks, dict), f"missing {variant_id} outage diagnostics")
        _require(set(masks) == {f"{duration}s" for duration in config.outage_durations_s}, "mask set mismatch")
        for duration_s in config.outage_durations_s:
            mask = masks[f"{duration_s}s"]
            for key, expected in (
                ("masked_valid_pvt_count", duration_s),
                ("prefix_before_mask_bit_identical_to_unmasked", True),
                ("recovery_gnss_available", True),
                ("all_outputs_finite", True),
            ):
                _require(mask.get(key) == expected, f"{variant_id} {duration_s}s mask mismatch: {key}")
            displacement = mask.get("same_start_endpoint_displacements")
            _require(
                isinstance(displacement, dict)
                and set(displacement)
                == {
                    "student_masked_pre_gnss",
                    "same_start_pure_imu",
                    "student_unmasked_pre_gnss",
                    "teacher_weak_label",
                },
                f"missing detailed {variant_id} {duration_s}s displacement audit",
            )
            _require(_finite_tree(displacement), f"nonfinite {variant_id} outage displacement")
            for speed_key in (
                "student_masked_endpoint_speed_mps",
                "pure_imu_endpoint_speed_mps",
                "pure_imu_maximum_bridge_speed_mps",
            ):
                _require(math.isfinite(float(mask[speed_key])), f"nonfinite {variant_id} {speed_key}")
    return physics_huber


def validate_phase4_development_run(
    config: object,
    run_directory: Path,
    diagnostics_name: str,
) -> dict[str, Any]:
    run_directory = run_directory.resolve()
    _require(run_directory.is_dir(), f"missing run directory: {run_directory}")
    _require(Path(diagnostics_name).name == diagnostics_name, "diagnostics name must be a basename")
    _require(Path(diagnostics_name).suffix == ".json", "diagnostics name must end in .json")

    summary_path = run_directory / "summary.json"
    normalizer_path = run_directory / "normalizer.json"
    diagnostics_path = run_directory / diagnostics_name
    summary = _load_json(summary_path)
    normalizer = _load_json(normalizer_path)
    diagnostics = _load_json(diagnostics_path)

    for key, expected in (
        ("schema_version", 1),
        ("profile_id", config.profile_id),
        ("mode", "bounded_development"),
        ("torch_version", "2.13.0+cpu"),
        ("torch_cuda", None),
        ("seed", config.seed),
        ("record_count", 764),
        ("optimizer_steps_per_variant", config.development_optimizer_steps),
        ("fair_initialization", True),
        ("selection_uses_only_validation_weak_label_loss", True),
        ("teacher_is_weak_pseudo_label", True),
        ("teacher_covariance_used", False),
        ("teacher_input_to_student", False),
        ("high_precision_reference_used", False),
        ("development_passed", True),
    ):
        _require(summary.get(key) == expected, f"development summary mismatch: {key}")
    _require(normalizer.get("fit_split") == "train", "normalizer was not fit on train only")
    _require(normalizer.get("teacher_covariance_used") is False, "normalizer uses teacher covariance")
    _require(_finite_tree(normalizer), "normalizer contains a nonfinite value")

    expected_variants = {variant.variant_id: variant for variant in config.variants}
    variants = summary.get("variants")
    _require(isinstance(variants, dict) and set(variants) == set(expected_variants), "variant mismatch")
    initial_hashes: set[str] = set()
    artifact_hashes: dict[str, str] = {
        "summary.json": file_sha256(summary_path),
        "normalizer.json": file_sha256(normalizer_path),
        diagnostics_name: file_sha256(diagnostics_path),
    }
    for variant_id, variant in expected_variants.items():
        value = variants[variant_id]
        _require(float(value["physics_weight"]) == variant.physics_weight, "physics weight mismatch")
        _require(value["all_training_values_finite"] is True, f"nonfinite {variant_id} training")
        _require(int(value["optimizer_steps"]) == config.development_optimizer_steps, "step mismatch")
        best_step = int(value["best_optimizer_step"])
        _require(1 <= best_step <= config.development_optimizer_steps, "invalid best step")
        initial_score = float(value["initial_validation"]["composite"])
        best_score = float(value["best_validation_composite"])
        _require(math.isfinite(initial_score) and math.isfinite(best_score), "nonfinite validation")
        _require(best_score < initial_score, f"{variant_id} did not improve its validation objective")
        initial_hashes.add(value["initial_model_tensor_sha256"])

        log_path = run_directory / f"{variant_id}_training.jsonl"
        records = _load_training_log(log_path)
        _require(len(records) == config.development_optimizer_steps, f"{variant_id} log length mismatch")
        _require(
            [record["optimizer_step"] for record in records]
            == list(range(1, config.development_optimizer_steps + 1)),
            f"{variant_id} optimizer-step sequence mismatch",
        )
        _require(all(record.get("finite") is True and _finite_tree(record) for record in records), "bad log")
        if variant.physics_weight == 0.0:
            _require(all(record["physics_loss"] is None for record in records), "no-physics logged physics")
        else:
            _require(all(record["physics_loss"] is not None for record in records), "missing physics loss")
        artifact_hashes[log_path.name] = file_sha256(log_path)

        for checkpoint in ("best", "final"):
            model_path = run_directory / f"{variant_id}_{checkpoint}.pt"
            tensor_hash = state_dict_tensor_sha256(_load_state_dict(model_path))
            expected_hash = value[f"{checkpoint}_model_tensor_sha256"]
            _require(tensor_hash == expected_hash, f"{variant_id} {checkpoint} model hash mismatch")
            artifact_hashes[f"{model_path.name}:tensor"] = tensor_hash
    _require(len(initial_hashes) == 1, "matched variants do not share initialization")

    physics_huber = _validate_diagnostics(diagnostics, config)
    return {
        "run_directory": str(run_directory),
        "profile_id": config.profile_id,
        "artifact_hashes": artifact_hashes,
        "initial_model_tensor_sha256": next(iter(initial_hashes)),
        "best_validation_composite": {
            variant_id: float(variants[variant_id]["best_validation_composite"])
            for variant_id in expected_variants
        },
        "diagnostic_physics_huber": physics_huber,
        "physics_residual_improvement_claimed": False,
        "high_precision_reference_used": False,
        "validated": True,
    }


def compare_phase4_development_runs(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    _require(first["profile_id"] == second["profile_id"], "comparison profile mismatch")
    _require(
        first["artifact_hashes"] == second["artifact_hashes"],
        "Phase 4 development artifacts are not deterministic",
    )
    _require(
        first["initial_model_tensor_sha256"] == second["initial_model_tensor_sha256"],
        "comparison initialization mismatch",
    )
    return {
        "byte_and_tensor_identical": True,
        "compared_artifact_count": len(first["artifact_hashes"]),
    }
