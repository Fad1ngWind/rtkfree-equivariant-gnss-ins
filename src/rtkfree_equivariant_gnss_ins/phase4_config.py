"""Validation for the single frozen Phase 4 ordinary-student profile."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Phase4Variant:
    variant_id: str
    physics_weight: float


@dataclass(frozen=True)
class Phase4Config:
    profile_id: str
    pvt_sha256: str
    imu_sha256: str
    teacher_sha256: str
    quality_fields: tuple[str, ...]
    hidden_dim: int
    pre_position_scale_n_m: tuple[float, float, float]
    pre_velocity_scale_n_mps: tuple[float, float, float]
    pre_attitude_scale_rad: tuple[float, float, float]
    post_correction_scale: tuple[float, ...]
    record_count: int
    splits: dict[str, tuple[int, int]]
    initialization_timestamp_ns_utc: int
    benchmark_start_after_initialization_s: int
    outage_durations_s: tuple[int, ...]
    training_mask_cycle_s: tuple[int, ...]
    training_mask_start_step: int
    normalization_minimum_scale: float
    gnss_innovation_scale_n_m: tuple[float, float, float]
    previous_velocity_scale_n_mps: tuple[float, float, float]
    state_loss_scale: tuple[float, ...]
    huber_delta: float
    seed: int
    learning_rate: float
    weight_decay: float
    gradient_norm_clip: float
    unroll_steps: int
    smoke_optimizer_steps: int
    development_optimizer_steps: int
    variants: tuple[Phase4Variant, ...]
    maximum_diagnostic_speed_mps: float


def load_phase4_config(path: Path) -> Phase4Config:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        inputs = raw["inputs"]
        sequence = raw["development_sequence"]
        model = raw["model"]
        normalization = raw["normalization"]
        loss = raw["loss"]
        masks = raw["controlled_masks"]
        optimization = raw["optimization"]
        diagnostics = raw["diagnostics"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("cannot read fixed Phase 4 configuration") from exc

    if raw.get("schema_version") != 1 or raw.get("profile_id") != "ordinary_causal_student_v1":
        raise ValueError("unsupported Phase 4 profile")
    if inputs.get("teacher_role") != "training_weak_pseudo_label_only":
        raise ValueError("teacher must remain a training-only weak pseudo-label source")
    if inputs.get("teacher_covariance_used") is not False:
        raise ValueError("teacher covariance weighting is prohibited")
    if model != {
        "architecture": "single_ordinary_gru",
        "hidden_dim": 32,
        "imu_feature_dim": 7,
        "quality_dim": 5,
        "position_is_recursive_increment": True,
        "pre_position_increment_scale_n_m": [30.0, 30.0, 10.0],
        "pre_velocity_output_scale_n_mps": [20.0, 20.0, 10.0],
        "pre_attitude_increment_scale_rad": [0.1, 0.1, 0.1],
        "post_correction_scale": [20.0, 20.0, 10.0, 5.0, 5.0, 2.0, 0.15, 0.15, 0.15],
        "current_gnss_enters_pre_branch": False,
        "predictive_covariance_head": False,
        "learned_bias_state": False,
    }:
        raise ValueError("Phase 4 must use the one frozen ordinary mean-state student")
    quality_fields = tuple(str(value) for value in inputs["quality_fields"])
    if quality_fields != (
        "used_satellite_count",
        "postfit_residual_rms_m",
        "dop.pdop",
        "dop.hdop",
        "dop.vdop",
    ):
        raise ValueError("Phase 4 deployable quality fields differ from the frozen set")
    prohibited_tokens = ("time", "route", "file", "device", "teacher", "covariance")
    if any(any(token in field.lower() for token in prohibited_tokens) for field in quality_fields):
        raise ValueError("student quality features contain a prohibited shortcut")
    if normalization.get("fit_source") != "train_split_deployable_fields_only":
        raise ValueError("normalization must be fit on deployable training fields only")
    if loss.get("teacher_covariance_weighting") is not False:
        raise ValueError("teacher covariance cannot weight the Phase 4 loss")
    if loss.get("physics_reference_gradient") != "stop_at_inertial_reference":
        raise ValueError("unsupported independent-physics gradient convention")
    if loss.get("physics_validity") != "exclude_imu_timing_gap_steps":
        raise ValueError("physics residual must exclude frozen IMU timing-gap steps")
    if masks.get("interval") != "left_closed_right_open":
        raise ValueError("Phase 4 masks must retain the Phase 3 interval definition")
    if masks.get("planned_duration_visible_to_student") is not False:
        raise ValueError("planned outage duration cannot be a student input")
    durations = tuple(int(value) for value in masks["durations_s"])
    if durations != (20, 30):
        raise ValueError("Phase 4 requires exactly 20 and 30 second masks")
    if tuple(int(value) for value in masks["training_cycle_s"]) != (0, 20, 30, 0):
        raise ValueError("unsupported fixed training-mask cycle")

    record_count = int(sequence["record_count"])
    splits = {
        str(name): (int(bounds[0]), int(bounds[1]))
        for name, bounds in sequence["splits"].items()
    }
    expected_splits = {
        "train": (0, 458),
        "guard_1": (458, 488),
        "validation": (488, 626),
        "guard_2": (626, 656),
        "diagnostic": (656, 764),
    }
    if record_count != 764 or splits != expected_splits:
        raise ValueError("Phase 4 chronological split differs from the frozen boundary")
    if sequence.get("route_generalization_claim_allowed") is not False:
        raise ValueError("one development session cannot support route generalization")

    variants = tuple(
        Phase4Variant(str(value["id"]), float(value["physics_weight"]))
        for value in raw["variants"]
    )
    if variants != (
        Phase4Variant("no_physics", 0.0),
        Phase4Variant("physics", 0.1),
    ):
        raise ValueError("only the matched no-physics and physics variants are allowed")
    if optimization.get("optimizer") != "AdamW":
        raise ValueError("unsupported Phase 4 optimizer")
    if optimization.get("selection") != "minimum_finite_validation_weak_label_state_loss":
        raise ValueError("Phase 4 selection must use the common weak-label score")
    if diagnostics != {
        "maximum_speed_mps": 60.0,
        "translation_consistency_tolerance": "float64_structural_audit_1e-9",
    }:
        raise ValueError("Phase 4 direct diagnostic thresholds differ from the freeze")

    return Phase4Config(
        profile_id=raw["profile_id"],
        pvt_sha256=str(inputs["pvt_sha256"]),
        imu_sha256=str(inputs["imu_sha256"]),
        teacher_sha256=str(inputs["teacher_sha256"]),
        quality_fields=quality_fields,
        hidden_dim=int(model["hidden_dim"]),
        pre_position_scale_n_m=tuple(
            float(value) for value in model["pre_position_increment_scale_n_m"]
        ),
        pre_velocity_scale_n_mps=tuple(
            float(value) for value in model["pre_velocity_output_scale_n_mps"]
        ),
        pre_attitude_scale_rad=tuple(
            float(value) for value in model["pre_attitude_increment_scale_rad"]
        ),
        post_correction_scale=tuple(
            float(value) for value in model["post_correction_scale"]
        ),
        record_count=record_count,
        splits=splits,
        initialization_timestamp_ns_utc=int(sequence["initialization_timestamp_ns_utc"]),
        benchmark_start_after_initialization_s=int(
            masks["benchmark_start_after_initialization_s"]
        ),
        outage_durations_s=durations,
        training_mask_cycle_s=tuple(int(value) for value in masks["training_cycle_s"]),
        training_mask_start_step=int(masks["training_start_step_within_unroll"]),
        normalization_minimum_scale=float(normalization["minimum_scale"]),
        gnss_innovation_scale_n_m=tuple(
            float(value) for value in normalization["gnss_innovation_scale_n_m"]
        ),
        previous_velocity_scale_n_mps=tuple(
            float(value) for value in normalization["previous_velocity_scale_n_mps"]
        ),
        state_loss_scale=tuple(
            float(value)
            for key in ("position_n_m", "velocity_n_mps", "attitude_rad")
            for value in loss["state_scales"][key]
        ),
        huber_delta=float(loss["huber_delta"]),
        seed=int(optimization["seed"]),
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        gradient_norm_clip=float(optimization["gradient_norm_clip"]),
        unroll_steps=int(optimization["unroll_steps"]),
        smoke_optimizer_steps=int(optimization["smoke_optimizer_steps"]),
        development_optimizer_steps=int(optimization["development_optimizer_steps"]),
        variants=variants,
        maximum_diagnostic_speed_mps=float(diagnostics["maximum_speed_mps"]),
    )
