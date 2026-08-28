#!/usr/bin/env python3
"""Run frozen reference-free and full-mapping SO(2) diagnostics for Phase 5."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.phase4.diagnose_medium_models import (
    diagnose_variant,
    load_normalizer,
    state_dict_hash,
)
from rtkfree_equivariant_gnss_ins.phase3_config import load_fixed_eskf_config
from rtkfree_equivariant_gnss_ins.phase4_config import load_phase4_config
from rtkfree_equivariant_gnss_ins.phase4_data import assemble_phase4_sequence
from rtkfree_equivariant_gnss_ins.phase4_learning import inference_rollout
from rtkfree_equivariant_gnss_ins.phase5_augmentation import rotate_phase4_sequence
from rtkfree_equivariant_gnss_ins.phase5_config import load_phase5_config
from rtkfree_equivariant_gnss_ins.phase5_group import (
    transform_mean_state,
    transform_state_increment,
)
from rtkfree_equivariant_gnss_ins.phase5_learning import new_phase5_student


def load_student(
    path: Path,
    variant: object,
    phase4_config: object,
    phase5_config: object,
    expected_hash: str,
) -> nn.Module:
    student = new_phase5_student(variant, phase4_config, phase5_config)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if state_dict_hash(state_dict) != expected_hash:
        raise ValueError(f"model tensor hash differs from development summary: {path.name}")
    student.load_state_dict(state_dict, strict=True)
    student.eval()
    return student


def _comparison(values: tuple[tuple[Tensor, Tensor], ...], atol: float, rtol: float) -> dict[str, object]:
    maximum_absolute_error = 0.0
    maximum_tolerance_ratio = 0.0
    passed = True
    for actual, expected in values:
        error = torch.abs(actual - expected)
        allowance = atol + rtol * torch.abs(expected)
        maximum_absolute_error = max(maximum_absolute_error, float(torch.max(error)))
        maximum_tolerance_ratio = max(
            maximum_tolerance_ratio,
            float(torch.max(error / allowance)),
        )
        passed = passed and bool(torch.all(error <= allowance))
    return {
        "passed": passed,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_tolerance_ratio": maximum_tolerance_ratio,
    }


def _compare_rollout_to_transformed_baseline(
    actual: tuple,
    baseline: tuple,
    yaw_rad: float,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    yaw = torch.tensor(yaw_rad, dtype=torch.float64)
    pairs: list[tuple[Tensor, Tensor]] = []
    for actual_step, baseline_step in zip(actual, baseline, strict=True):
        expected_pre = transform_mean_state(baseline_step.pre_gnss, yaw)
        expected_posterior = transform_mean_state(baseline_step.posterior, yaw)
        expected_correction = transform_state_increment(baseline_step.gnss_correction, yaw)
        pairs.extend(
            (
                (actual_step.pre_gnss.position_n_m, expected_pre.position_n_m),
                (actual_step.pre_gnss.velocity_n_mps, expected_pre.velocity_n_mps),
                (actual_step.pre_gnss.rotation_n_from_b, expected_pre.rotation_n_from_b),
                (actual_step.posterior.position_n_m, expected_posterior.position_n_m),
                (actual_step.posterior.velocity_n_mps, expected_posterior.velocity_n_mps),
                (actual_step.posterior.rotation_n_from_b, expected_posterior.rotation_n_from_b),
                (actual_step.gnss_correction, expected_correction),
            )
        )
    return _comparison(tuple(pairs), atol, rtol)


def _compare_rollouts(
    actual: tuple,
    expected: tuple,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    pairs: list[tuple[Tensor, Tensor]] = []
    for actual_step, expected_step in zip(actual, expected, strict=True):
        pairs.extend(
            (
                (actual_step.pre_gnss.position_n_m, expected_step.pre_gnss.position_n_m),
                (actual_step.pre_gnss.velocity_n_mps, expected_step.pre_gnss.velocity_n_mps),
                (actual_step.pre_gnss.rotation_n_from_b, expected_step.pre_gnss.rotation_n_from_b),
                (actual_step.posterior.position_n_m, expected_step.posterior.position_n_m),
                (actual_step.posterior.velocity_n_mps, expected_step.posterior.velocity_n_mps),
                (actual_step.posterior.rotation_n_from_b, expected_step.posterior.rotation_n_from_b),
                (actual_step.gnss_correction, expected_step.gnss_correction),
            )
        )
    return _comparison(tuple(pairs), atol, rtol)


def so2_rollout_diagnostic(
    student: nn.Module,
    sequence: object,
    phase4_config: object,
    phase5_config: object,
    normalizer: object,
    strict_expected: bool,
) -> dict[str, object]:
    """Test the deployed recursive map, not an isolated layer."""

    dtype = next(student.parameters()).dtype
    if dtype == torch.float64:
        dtype_name = "float64"
        atol = phase5_config.float64_atol
        rtol = phase5_config.float64_rtol
    elif dtype == torch.float32:
        dtype_name = "float32"
        atol = phase5_config.float32_atol
        rtol = phase5_config.float32_rtol
    else:
        raise ValueError("SO(2) diagnostic supports only frozen float32/float64 dtypes")
    baseline = inference_rollout(student, sequence, phase4_config, normalizer)
    angle_results: dict[str, object] = {}
    for angle in phase5_config.property_angles_rad:
        transformed = inference_rollout(
            student,
            rotate_phase4_sequence(sequence, angle),
            phase4_config,
            normalizer,
        )
        angle_results[f"{angle:+.2f}"] = _compare_rollout_to_transformed_baseline(
            transformed,
            baseline,
            angle,
            atol,
            rtol,
        )
    identity_outputs = inference_rollout(
        student,
        rotate_phase4_sequence(sequence, 0.0),
        phase4_config,
        normalizer,
    )
    identity = _compare_rollouts(identity_outputs, baseline, atol, rtol)

    first = phase5_config.property_angles_rad[0]
    second = phase5_config.property_angles_rad[1]
    sequential_outputs = inference_rollout(
        student,
        rotate_phase4_sequence(rotate_phase4_sequence(sequence, first), second),
        phase4_config,
        normalizer,
    )
    composed_outputs = inference_rollout(
        student,
        rotate_phase4_sequence(sequence, first + second),
        phase4_config,
        normalizer,
    )
    composition = _compare_rollouts(sequential_outputs, composed_outputs, atol, rtol)
    all_angle_cases_pass = all(value["passed"] for value in angle_results.values())
    strict_property_passed = identity["passed"] and composition["passed"] and all_angle_cases_pass
    counterexample_present = any(not value["passed"] for value in angle_results.values())
    classification_passed = strict_property_passed if strict_expected else counterexample_present
    return {
        "dtype": dtype_name,
        "atol": atol,
        "rtol": rtol,
        "angles_rad": list(phase5_config.property_angles_rad),
        "complete_real_sequence_step_count": len(baseline),
        "angle_commutation": angle_results,
        "identity": identity,
        "composition": composition,
        "strict_equivariance_expected": strict_expected,
        "strict_property_passed": strict_property_passed,
        "non_strict_counterexample_present": counterexample_present,
        "classification_passed": classification_passed,
        "sensor_remounting_tested": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 5 reference-free model diagnostics.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-run-dir", type=Path, required=True)
    parser.add_argument("--development-run-dir", type=Path, required=True)
    parser.add_argument("--report-name", default="diagnostics.json")
    args = parser.parse_args()
    if "RTKFREE_SEALED_REFERENCE_ROOT" in os.environ:
        raise ValueError("sealed-reference environment must be absent during Phase 5 diagnostics")
    development_directory = args.development_run_dir.resolve()
    if not development_directory.is_absolute() or not development_directory.is_dir():
        raise ValueError("development run directory must be an existing absolute directory")
    if development_directory.is_relative_to(ROOT.resolve()):
        raise ValueError("diagnostic inputs and outputs must remain outside the repository")
    report_name = Path(args.report_name)
    if report_name.name != args.report_name or report_name.suffix != ".json":
        raise ValueError("report name must be a JSON basename inside the development run directory")
    report_path = development_directory / report_name
    if report_path.exists():
        raise ValueError("diagnostics output already exists")

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
    run_summary = json.loads((development_directory / "summary.json").read_text(encoding="utf-8"))
    if (
        run_summary.get("profile_id") != phase5_config.profile_id
        or run_summary.get("development_passed") is not True
        or run_summary.get("high_precision_reference_used") is not False
    ):
        raise ValueError("development summary differs from the Phase 5 diagnostic contract")
    normalizer = load_normalizer(development_directory / "normalizer.json")
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

    variants: dict[str, object] = {}
    for variant in phase5_config.variants:
        expected_hash = run_summary["variants"][variant.variant_id][
            "best_model_tensor_sha256"
        ]
        student = load_student(
            development_directory / f"{variant.variant_id}_best.pt",
            variant,
            phase4_config,
            phase5_config,
            expected_hash,
        )
        reference_free = diagnose_variant(student, sequence, phase4_config, normalizer)
        symmetry_float32 = so2_rollout_diagnostic(
            student,
            sequence,
            phase4_config,
            phase5_config,
            normalizer,
            strict_expected=variant.strict_so2,
        )
        symmetry_float64 = so2_rollout_diagnostic(
            copy.deepcopy(student).to(dtype=torch.float64),
            sequence,
            phase4_config,
            phase5_config,
            normalizer,
            strict_expected=variant.strict_so2,
        )
        variants[variant.variant_id] = {
            **reference_free,
            "so2_complete_mapping": {
                "float32": symmetry_float32,
                "float64": symmetry_float64,
                "float64_structural_classification_passed": symmetry_float64[
                    "classification_passed"
                ],
                "float32_deployed_rollout_classification_passed": symmetry_float32[
                    "classification_passed"
                ],
            },
            "strict_so2_structural_claim_allowed": (
                variant.strict_so2 and symmetry_float64["strict_property_passed"]
            ),
            "strict_so2_float32_full_rollout_claim_allowed": (
                variant.strict_so2 and symmetry_float32["strict_property_passed"]
            ),
            "rotation_augmentation_is_empirical_only": variant.rotation_augmentation,
        }
    reference_free_safety_checks_passed = all(
        value["degeneracy_checks_passed"]
        and value["causality_and_memory_checks_passed"]
        and value["numerical_checks_passed"]
        for value in variants.values()
    )
    float64_structural_so2_checks_passed = all(
        value["so2_complete_mapping"]["float64_structural_classification_passed"]
        for value in variants.values()
    )
    float32_deployed_rollout_checks_passed = all(
        value["so2_complete_mapping"]["float32_deployed_rollout_classification_passed"]
        for value in variants.values()
    )
    report = {
        "schema_version": 2,
        "profile_id": phase5_config.profile_id,
        "variants": variants,
        "audit_completed": True,
        "reference_free_safety_checks_passed": reference_free_safety_checks_passed,
        "float64_structural_so2_checks_passed": float64_structural_so2_checks_passed,
        "float32_deployed_rollout_checks_passed": float32_deployed_rollout_checks_passed,
        "teacher_is_weak_pseudo_label": True,
        "teacher_covariance_used": False,
        "high_precision_reference_used": False,
        "physics_residual_improvement_claimed": False,
        "accuracy_or_superiority_claim_supported": False,
        "sensor_remounting_generalization_claim_supported": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not reference_free_safety_checks_passed:
        raise RuntimeError(f"Phase 5 reference-free safety checks failed; inspect {report_path}")
    print("COMPLETE Phase 5 reference-free and SO(2) diagnostic audit")
    print(f"report={report_path}")
    print(
        "reference_free_safety_checks_passed="
        f"{str(reference_free_safety_checks_passed).lower()}"
    )
    print(
        "float64_structural_so2_checks_passed="
        f"{str(float64_structural_so2_checks_passed).lower()}"
    )
    print(
        "float32_deployed_rollout_checks_passed="
        f"{str(float32_deployed_rollout_checks_passed).lower()}"
    )
    for variant_id, value in variants.items():
        symmetry = value["so2_complete_mapping"]
        print(
            f"{variant_id} float32_strict={str(symmetry['float32']['strict_property_passed']).lower()} "
            f"float64_strict={str(symmetry['float64']['strict_property_passed']).lower()} "
            f"float64_counterexample={str(symmetry['float64']['non_strict_counterexample_present']).lower()} "
            f"finite={str(value['numerical_checks_passed']).lower()}"
        )
        print(
            f"{variant_id} physics_huber="
            f"{value['independent_physics_residual_on_diagnostic_split']['normalized_component_mean_huber']:.9f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
