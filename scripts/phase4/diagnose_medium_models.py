#!/usr/bin/env python3
"""Run direct reference-free degeneracy and causal diagnostics on Phase 4 models."""

from __future__ import annotations

import argparse
import copy
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
    InferenceStep,
    RobustNormalizer,
    inference_rollout,
    tensor_state,
)
from rtkfree_equivariant_gnss_ins.phase4_state import (
    MeanState,
    propagate_imu_sequence,
    state_boxminus,
)
from rtkfree_equivariant_gnss_ins.phase4_student import OrdinaryCausalStudent


def state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_normalizer(path: Path) -> RobustNormalizer:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw["fit_split"] != "train" or raw["teacher_covariance_used"] is not False:
        raise ValueError("diagnostics require train-only deployable normalization")
    return RobustNormalizer(
        imu_center=np.asarray(raw["imu_center"], dtype=np.float64),
        imu_scale=np.asarray(raw["imu_scale"], dtype=np.float64),
        quality_center=np.asarray(raw["quality_center"], dtype=np.float64),
        quality_scale=np.asarray(raw["quality_scale"], dtype=np.float64),
    )


def load_student(path: Path, config: object, expected_hash: str) -> OrdinaryCausalStudent:
    student = OrdinaryCausalStudent(
        quality_dim=len(config.quality_fields),
        hidden_dim=config.hidden_dim,
        previous_velocity_scale_n_mps=config.previous_velocity_scale_n_mps,
        pre_position_scale_n_m=config.pre_position_scale_n_m,
        pre_velocity_scale_n_mps=config.pre_velocity_scale_n_mps,
        pre_attitude_scale_rad=config.pre_attitude_scale_rad,
        post_correction_scale=config.post_correction_scale,
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if state_dict_hash(state_dict) != expected_hash:
        raise ValueError(f"model tensor hash differs from development summary: {path.name}")
    student.load_state_dict(state_dict, strict=True)
    student.eval()
    return student


def error_vector(prediction: MeanState, reference: MeanState) -> np.ndarray:
    return state_boxminus(prediction, reference).detach().cpu().numpy()


def error_summary(errors: np.ndarray) -> dict[str, float]:
    return {
        "position_norm_median_m": float(np.median(np.linalg.norm(errors[:, 0:3], axis=1))),
        "position_vector_rmse_m": float(np.sqrt(np.mean(np.sum(errors[:, 0:3] ** 2, axis=1)))),
        "velocity_norm_median_mps": float(np.median(np.linalg.norm(errors[:, 3:6], axis=1))),
        "velocity_vector_rmse_mps": float(np.sqrt(np.mean(np.sum(errors[:, 3:6] ** 2, axis=1)))),
        "attitude_norm_median_rad": float(np.median(np.linalg.norm(errors[:, 6:9], axis=1))),
        "attitude_vector_rmse_rad": float(np.sqrt(np.mean(np.sum(errors[:, 6:9] ** 2, axis=1)))),
    }


def all_finite(outputs: tuple[InferenceStep, ...]) -> bool:
    return all(
        bool(torch.all(torch.isfinite(value)))
        for output in outputs
        for value in (
            output.pre_gnss.position_n_m,
            output.pre_gnss.velocity_n_mps,
            output.pre_gnss.rotation_n_from_b,
            output.posterior.position_n_m,
            output.posterior.velocity_n_mps,
            output.posterior.rotation_n_from_b,
            output.gnss_correction,
        )
    )


def prefix_equal(first: tuple[InferenceStep, ...], second: tuple[InferenceStep, ...]) -> bool:
    if len(first) != len(second):
        return False
    return all(
        torch.equal(left.posterior.position_n_m, right.posterior.position_n_m)
        and torch.equal(left.posterior.velocity_n_mps, right.posterior.velocity_n_mps)
        and torch.equal(left.posterior.rotation_n_from_b, right.posterior.rotation_n_from_b)
        for left, right in zip(first, second, strict=True)
    )


def diagnostic_physics(
    outputs: tuple[InferenceStep, ...],
    sequence: object,
    config: object,
) -> dict[str, object]:
    start, stop = config.splits["diagnostic"]
    gravity = torch.as_tensor(sequence.gravity_n_mps2, dtype=torch.float32)
    earth_rate = torch.as_tensor(sequence.earth_rotation_n_radps, dtype=torch.float32)
    residuals: list[np.ndarray] = []
    excluded = 0
    with torch.no_grad():
        for index in range(start, stop):
            data_step = sequence.steps[index]
            if data_step.imu_timing_gap:
                excluded += 1
                continue
            previous = (
                tensor_state(sequence.initial_student_state, torch.float32)
                if index == 0
                else outputs[index - 1].posterior
            )
            reference = propagate_imu_sequence(
                previous,
                torch.as_tensor(data_step.linear_acceleration_b_mps2, dtype=torch.float32),
                torch.as_tensor(data_step.angular_velocity_b_radps, dtype=torch.float32),
                torch.as_tensor(data_step.dt_s, dtype=torch.float32),
                gravity,
                earth_rate,
            )
            residuals.append(error_vector(outputs[index].pre_gnss, reference))
    array = np.vstack(residuals)
    normalized = array / np.asarray(config.state_loss_scale)
    absolute = np.abs(normalized)
    huber = np.where(absolute <= config.huber_delta, 0.5 * absolute**2, absolute - 0.5)
    return {
        **error_summary(array),
        "normalized_component_mean_huber": float(np.mean(huber)),
        "valid_step_count": int(array.shape[0]),
        "excluded_timing_gap_step_count": excluded,
    }


def mask_diagnostic(
    student: OrdinaryCausalStudent,
    normal_outputs: tuple[InferenceStep, ...],
    sequence: object,
    config: object,
    normalizer: RobustNormalizer,
    duration_s: int,
) -> dict[str, object]:
    start_ns = (
        sequence.initialization_timestamp_ns_utc
        + config.benchmark_start_after_initialization_s * 1_000_000_000
    )
    outage = GnssOutage(start_ns, duration_s * 1_000_000_000)
    masked_outputs = inference_rollout(student, sequence, config, normalizer, (outage,))
    start_index = next(
        index for index, step in enumerate(sequence.steps) if step.timestamp_ns_utc == start_ns
    )
    recovery_index = next(
        index
        for index, step in enumerate(sequence.steps)
        if step.timestamp_ns_utc == outage.end_ns_utc
    )
    masked_valid_count = sum(
        step.pvt_solution_valid and outage.contains(step.timestamp_ns_utc)
        for step in sequence.steps
    )
    prefix_is_identical = prefix_equal(
        normal_outputs[:start_index],
        masked_outputs[:start_index],
    )
    endpoint_vs_unmasked = error_vector(
        masked_outputs[recovery_index].pre_gnss,
        normal_outputs[recovery_index].pre_gnss,
    )
    teacher = tensor_state(sequence.steps[recovery_index].weak_pseudo_label, torch.float32)
    endpoint_vs_teacher = error_vector(masked_outputs[recovery_index].pre_gnss, teacher)

    common_start = normal_outputs[start_index - 1].posterior
    pure_imu = common_start
    pure_imu_max_speed_mps = float(torch.linalg.vector_norm(pure_imu.velocity_n_mps))
    gravity = torch.as_tensor(sequence.gravity_n_mps2, dtype=torch.float32)
    earth_rate = torch.as_tensor(sequence.earth_rotation_n_radps, dtype=torch.float32)
    with torch.no_grad():
        for index in range(start_index, recovery_index + 1):
            data_step = sequence.steps[index]
            pure_imu = propagate_imu_sequence(
                pure_imu,
                torch.as_tensor(data_step.linear_acceleration_b_mps2, dtype=torch.float32),
                torch.as_tensor(data_step.angular_velocity_b_radps, dtype=torch.float32),
                torch.as_tensor(data_step.dt_s, dtype=torch.float32),
                gravity,
                earth_rate,
            )
            pure_imu_max_speed_mps = max(
                pure_imu_max_speed_mps,
                float(torch.linalg.vector_norm(pure_imu.velocity_n_mps)),
            )
    endpoint_vs_pure_imu = error_vector(masked_outputs[recovery_index].pre_gnss, pure_imu)
    endpoint_displacements = {
        "student_masked_pre_gnss": error_summary(
            error_vector(masked_outputs[recovery_index].pre_gnss, common_start)[None, :]
        ),
        "same_start_pure_imu": error_summary(error_vector(pure_imu, common_start)[None, :]),
        "student_unmasked_pre_gnss": error_summary(
            error_vector(normal_outputs[recovery_index].pre_gnss, common_start)[None, :]
        ),
        "teacher_weak_label": error_summary(error_vector(teacher, common_start)[None, :]),
    }
    recovery_correction = masked_outputs[recovery_index].gnss_correction.detach().cpu().numpy()
    return {
        "start_timestamp_ns_utc": start_ns,
        "recovery_timestamp_ns_utc": outage.end_ns_utc,
        "masked_valid_pvt_count": masked_valid_count,
        "prefix_before_mask_bit_identical_to_unmasked": prefix_is_identical,
        "recovery_gnss_available": masked_outputs[recovery_index].gnss_available,
        "endpoint_pre_gnss_vs_unmasked": error_summary(endpoint_vs_unmasked[None, :]),
        "endpoint_pre_gnss_vs_teacher_weak_label": error_summary(endpoint_vs_teacher[None, :]),
        "endpoint_pre_gnss_vs_same_start_pure_imu": error_summary(endpoint_vs_pure_imu[None, :]),
        "same_start_endpoint_displacements": endpoint_displacements,
        "student_masked_endpoint_speed_mps": float(
            torch.linalg.vector_norm(masked_outputs[recovery_index].pre_gnss.velocity_n_mps)
        ),
        "pure_imu_endpoint_speed_mps": float(torch.linalg.vector_norm(pure_imu.velocity_n_mps)),
        "pure_imu_maximum_bridge_speed_mps": pure_imu_max_speed_mps,
        "recovery_correction_position_norm_m": float(np.linalg.norm(recovery_correction[0:3])),
        "recovery_correction_velocity_norm_mps": float(np.linalg.norm(recovery_correction[3:6])),
        "recovery_correction_attitude_norm_rad": float(np.linalg.norm(recovery_correction[6:9])),
        "all_outputs_finite": all_finite(masked_outputs),
    }


def diagnose_variant(
    student: OrdinaryCausalStudent,
    sequence: object,
    config: object,
    normalizer: RobustNormalizer,
) -> dict[str, object]:
    normal = inference_rollout(student, sequence, config, normalizer)
    reject = inference_rollout(student, sequence, config, normalizer, reject_all_gnss=True)
    translation_audit_student = copy.deepcopy(student).to(dtype=torch.float64)
    translation_normal = inference_rollout(
        translation_audit_student,
        sequence,
        config,
        normalizer,
    )
    shifted = inference_rollout(
        translation_audit_student,
        sequence,
        config,
        normalizer,
        coordinate_shift_n_m=(1000.0, -700.0, 50.0),
    )
    diagnostic_start, diagnostic_stop = config.splits["diagnostic"]

    teacher_errors = np.vstack(
        [
            error_vector(
                normal[index].posterior,
                tensor_state(sequence.steps[index].weak_pseudo_label, torch.float32),
            )
            for index in range(diagnostic_start, diagnostic_stop)
        ]
    )
    teacher_exact = np.all(np.abs(teacher_errors) < 1e-6, axis=1)

    spp_distances: list[float] = []
    corrections: list[np.ndarray] = []
    lever = torch.as_tensor(sequence.antenna_lever_imu_m, dtype=torch.float32)
    for index in range(diagnostic_start, diagnostic_stop):
        data_step = sequence.steps[index]
        output = normal[index]
        if data_step.pvt_solution_valid and data_step.pvt_antenna_position_n_m is not None:
            predicted_antenna = (
                output.posterior.position_n_m
                + output.posterior.rotation_n_from_b @ lever
            ).detach().cpu().numpy()
            spp_distances.append(
                float(np.linalg.norm(predicted_antenna - data_step.pvt_antenna_position_n_m))
            )
        if output.gnss_available:
            corrections.append(output.gnss_correction.detach().cpu().numpy())
    spp_array = np.asarray(spp_distances)
    correction_array = np.vstack(corrections)

    rejection_errors = np.vstack(
        [
            error_vector(normal[index].posterior, reject[index].posterior)
            for index in range(diagnostic_start, diagnostic_stop)
        ]
    )
    shift = np.array((1000.0, -700.0, 50.0), dtype=np.float64)
    shift_position_error = np.vstack(
        [
            (
                shifted[index].posterior.position_n_m
                - translation_normal[index].posterior.position_n_m
            ).detach().cpu().numpy()
            - shift
            for index in range(len(normal))
        ]
    )
    shift_velocity_error = np.vstack(
        [
            (
                shifted[index].posterior.velocity_n_mps
                - translation_normal[index].posterior.velocity_n_mps
            ).detach().cpu().numpy()
            for index in range(len(normal))
        ]
    )
    shift_rotation_error = np.vstack(
        [
            error_vector(
                shifted[index].posterior,
                translation_normal[index].posterior,
            )[6:9]
            for index in range(len(translation_normal))
        ]
    )
    translation_max_error = max(
        float(np.max(np.abs(shift_position_error))),
        float(np.max(np.abs(shift_velocity_error))),
        float(np.max(np.abs(shift_rotation_error))),
    )
    future_cut = diagnostic_start + (diagnostic_stop - diagnostic_start) // 2
    truncated = inference_rollout(
        student,
        sequence,
        config,
        normalizer,
        stop=future_cut,
    )
    future_prefix_pass = prefix_equal(normal[:future_cut], truncated)

    rotations = torch.stack([output.posterior.rotation_n_from_b for output in normal])
    orthogonality = rotations.transpose(-1, -2) @ rotations
    max_rotation_error = float(
        torch.max(torch.abs(orthogonality - torch.eye(3, dtype=rotations.dtype)))
    )
    minimum_determinant = float(torch.min(torch.linalg.det(rotations)))
    maximum_speed = max(float(torch.linalg.vector_norm(output.posterior.velocity_n_mps)) for output in normal)
    translation_tolerance = 1e-9
    route_shift_pass = translation_max_error <= translation_tolerance
    correction_all_zero = bool(np.all(np.abs(correction_array) < 1e-8))
    rejects_all_gnss = bool(np.all(np.abs(rejection_errors) < 1e-7))
    numerical_pass = (
        all_finite(normal)
        and all_finite(reject)
        and all_finite(shifted)
        and max_rotation_error < 1e-4
        and minimum_determinant > 0.9999
        and maximum_speed <= config.maximum_diagnostic_speed_mps
    )
    masks = {
        f"{duration_s}s": mask_diagnostic(
            student,
            normal,
            sequence,
            config,
            normalizer,
            duration_s,
        )
        for duration_s in config.outage_durations_s
    }
    mask_pass = all(
        value["masked_valid_pvt_count"] == int(key[:-1])
        and value["prefix_before_mask_bit_identical_to_unmasked"]
        and value["recovery_gnss_available"]
        and value["all_outputs_finite"]
        for key, value in masks.items()
    )
    degeneracy_pass = (
        not bool(np.all(teacher_exact))
        and not bool(np.all(spp_array < 1e-6))
        and not correction_all_zero
        and not rejects_all_gnss
    )
    return {
        "teacher_copying_tendency": {
            **error_summary(teacher_errors),
            "exact_copy_count_at_1e-6": int(np.sum(teacher_exact)),
            "record_count": int(teacher_errors.shape[0]),
            "all_outputs_exact_teacher_copy": bool(np.all(teacher_exact)),
        },
        "spp_copying": {
            "valid_record_count": int(spp_array.size),
            "antenna_position_distance_median_m": float(np.median(spp_array)),
            "antenna_position_distance_rms_m": float(np.sqrt(np.mean(spp_array**2))),
            "exact_copy_count_at_1e-6_m": int(np.sum(spp_array < 1e-6)),
            "all_outputs_exact_spp_copy": bool(np.all(spp_array < 1e-6)),
        },
        "gnss_correction": {
            "available_record_count": int(correction_array.shape[0]),
            "position_norm_median_m": float(np.median(np.linalg.norm(correction_array[:, 0:3], axis=1))),
            "velocity_norm_median_mps": float(np.median(np.linalg.norm(correction_array[:, 3:6], axis=1))),
            "attitude_norm_median_rad": float(np.median(np.linalg.norm(correction_array[:, 6:9], axis=1))),
            "all_corrections_zero_at_1e-8": correction_all_zero,
        },
        "reject_all_gnss": {
            **error_summary(rejection_errors),
            "normal_and_reject_all_outputs_identical_at_1e-7": rejects_all_gnss,
        },
        "future_information": {
            "prefix_record_count": future_cut,
            "full_and_truncated_prefix_bit_identical": future_prefix_pass,
        },
        "absolute_position_memory": {
            "coordinate_shift_n_m": shift.tolist(),
            "maximum_translation_consistency_error": translation_max_error,
            "float64_structural_audit_tolerance": translation_tolerance,
            "translation_consistency_pass": route_shift_pass,
            "route_or_session_generalization_claimed": False,
        },
        "numerical_stability": {
            "all_checked_outputs_finite": numerical_pass,
            "maximum_rotation_orthogonality_error": max_rotation_error,
            "minimum_rotation_determinant": minimum_determinant,
            "maximum_speed_mps": maximum_speed,
            "maximum_allowed_speed_mps": config.maximum_diagnostic_speed_mps,
        },
        "independent_physics_residual_on_diagnostic_split": diagnostic_physics(
            normal,
            sequence,
            config,
        ),
        "controlled_outages": masks,
        "degeneracy_checks_passed": degeneracy_pass,
        "causality_and_memory_checks_passed": future_prefix_pass and route_shift_pass and mask_pass,
        "numerical_checks_passed": numerical_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 4 reference-free model diagnostics.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-run-dir", type=Path, required=True)
    parser.add_argument("--development-run-dir", type=Path, required=True)
    parser.add_argument("--report-name", default="diagnostics.json")
    args = parser.parse_args()
    if "RTKFREE_SEALED_REFERENCE_ROOT" in os.environ:
        raise ValueError("sealed-reference environment must be absent during Phase 4 diagnostics")
    report_name = Path(args.report_name)
    if report_name.name != args.report_name or report_name.suffix != ".json":
        raise ValueError("report name must be a JSON basename inside the development run directory")
    report_path = args.development_run_dir / report_name
    if report_path.exists():
        raise ValueError("diagnostics output already exists")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = load_phase4_config(ROOT / "config" / "phase4" / "ordinary_student_v1.json")
    phase3_config = load_fixed_eskf_config(ROOT / "config" / "phase3" / "fixed_eskf_v1.json")
    run_summary = json.loads((args.development_run_dir / "summary.json").read_text(encoding="utf-8"))
    normalizer = load_normalizer(args.development_run_dir / "normalizer.json")
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
    variants: dict[str, object] = {}
    for variant in config.variants:
        expected_hash = run_summary["variants"][variant.variant_id]["best_model_tensor_sha256"]
        student = load_student(
            args.development_run_dir / f"{variant.variant_id}_best.pt",
            config,
            expected_hash,
        )
        variants[variant.variant_id] = diagnose_variant(student, sequence, config, normalizer)
    diagnostics_passed = all(
        value["degeneracy_checks_passed"]
        and value["causality_and_memory_checks_passed"]
        and value["numerical_checks_passed"]
        for value in variants.values()
    )
    report = {
        "schema_version": 1,
        "profile_id": config.profile_id,
        "variants": variants,
        "diagnostics_passed": diagnostics_passed,
        "teacher_is_weak_pseudo_label": True,
        "teacher_covariance_used": False,
        "high_precision_reference_used": False,
        "accuracy_or_superiority_claim_supported": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not diagnostics_passed:
        raise RuntimeError(f"Phase 4 direct diagnostic gate failed; inspect {report_path}")
    print("PASS Phase 4 direct reference-free diagnostics")
    print(f"report={report_path}")
    for variant_id, value in variants.items():
        print(
            f"{variant_id} teacher_copy={str(value['teacher_copying_tendency']['all_outputs_exact_teacher_copy']).lower()} "
            f"spp_copy={str(value['spp_copying']['all_outputs_exact_spp_copy']).lower()} "
            f"zero_correction={str(value['gnss_correction']['all_corrections_zero_at_1e-8']).lower()} "
            f"rejects_all_gnss={str(value['reject_all_gnss']['normal_and_reject_all_outputs_identical_at_1e-7']).lower()}"
        )
        print(
            f"{variant_id} physics_huber="
            f"{value['independent_physics_residual_on_diagnostic_split']['normalized_component_mean_huber']:.9f} "
            f"finite={str(value['numerical_checks_passed']).lower()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
