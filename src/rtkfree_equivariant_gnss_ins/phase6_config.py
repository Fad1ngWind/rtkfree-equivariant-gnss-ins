"""Strict loader for the prospective Phase 6 final-freeze contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .phase4_config import load_phase4_config
from .phase5_config import load_phase5_config


@dataclass(frozen=True)
class Phase6Config:
    profile_id: str
    phase4_sha256: str
    phase5_sha256: str
    seeds: tuple[int, ...]
    optimizer_steps: int
    checkpoint_every_steps: int
    structure_variants: tuple[str, ...]
    interaction_structures: tuple[str, ...]
    physics_weights: tuple[float, ...]
    designated_checkpoint_seed: int
    final_reproduction_count: int
    low_quality_hdop_threshold: float
    low_quality_residual_threshold_m: float
    low_quality_valid_epoch_count: int
    low_quality_union_epoch_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_keys(value: object, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} keys differ from the Phase 6 freeze")
    return value


def _derived_seed(experiment_id: str, index: int) -> int:
    payload = f"{experiment_id}:seed:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFF_FFFF


def load_phase6_config(
    path: Path,
    phase4_path: Path,
    phase5_path: Path,
) -> Phase6Config:
    """Reject any config that expands or changes the accepted prospective contract."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read Phase 6 config") from exc
    root = _require_keys(
        raw,
        {
            "schema_version",
            "profile_id",
            "base_phase4",
            "base_phase5",
            "development_data",
            "optimization",
            "arms",
            "selection",
            "low_quality_stratum",
            "scope",
        },
        "Phase 6 config",
    )
    if root["schema_version"] != 1 or root["profile_id"] != "phase6_final_freeze_v1":
        raise ValueError("Phase 6 profile identity differs from the freeze")

    base4 = _require_keys(root["base_phase4"], {"path", "sha256"}, "base Phase 4")
    base5 = _require_keys(root["base_phase5"], {"path", "sha256"}, "base Phase 5")
    expected4 = "7d6607edfe38725942ec0a4df6cac77af47d1b41d79a7b376ae30b0c5f1772cd"
    expected5 = "7556a1585ea456b61331c120b26f548437eefa7e342ed0691dec83abefaa6f59"
    if base4 != {"path": "config/phase4/ordinary_student_v1.json", "sha256": expected4}:
        raise ValueError("Phase 4 base declaration differs from the freeze")
    if base5 != {"path": "config/phase5/gravity_aware_so2_v1.json", "sha256": expected5}:
        raise ValueError("Phase 5 base declaration differs from the freeze")
    if file_sha256(phase4_path) != expected4 or file_sha256(phase5_path) != expected5:
        raise ValueError("a frozen base config hash differs")
    phase4 = load_phase4_config(phase4_path)
    phase5 = load_phase5_config(phase5_path, phase4_path)

    data = _require_keys(
        root["development_data"],
        {
            "session",
            "pvt_sha256",
            "imu_sha256",
            "teacher_sha256",
            "extrinsic_sha256",
            "no_new_data",
            "sealed_reference_prohibited",
        },
        "development data",
    )
    if data != {
        "session": "UrbanNav-HK-Medium-Urban-1",
        "pvt_sha256": phase4.pvt_sha256,
        "imu_sha256": phase4.imu_sha256,
        "teacher_sha256": phase4.teacher_sha256,
        "extrinsic_sha256": "873d9da5c7957ba0fd61e903275ea5f281dde3baade9517c0555ad1b5ff3f693",
        "no_new_data": True,
        "sealed_reference_prohibited": True,
    }:
        raise ValueError("Phase 6 data boundary differs from the freeze")

    optimization = _require_keys(
        root["optimization"],
        {
            "seed_experiment_id",
            "seed_derivation",
            "seeds",
            "optimizer_steps",
            "checkpoint_every_steps",
            "inherit_phase5_optimizer_and_masks",
            "one_run_per_arm_and_seed",
        },
        "optimization",
    )
    seeds = tuple(int(value) for value in optimization["seeds"])
    expected_seeds = tuple(_derived_seed("P6-STRUCTURE-001", index) for index in range(3))
    if optimization != {
        "seed_experiment_id": "P6-STRUCTURE-001",
        "seed_derivation": "sha256_utf8_first_u32_big_endian_mask_31bit",
        "seeds": list(expected_seeds),
        "optimizer_steps": 200,
        "checkpoint_every_steps": 10,
        "inherit_phase5_optimizer_and_masks": True,
        "one_run_per_arm_and_seed": True,
    } or seeds != expected_seeds:
        raise ValueError("Phase 6 optimization differs from the prospective freeze")
    if phase5.checkpoint_every_steps != optimization["checkpoint_every_steps"]:
        raise ValueError("Phase 6 checkpoint cadence differs from Phase 5")

    arms = _require_keys(
        root["arms"],
        {
            "structure_no_physics",
            "physics_weight_values",
            "interaction_structures",
            "reuse_no_physics_for_interaction",
            "conditional_rotation_physics_only_if_selected",
        },
        "arms",
    )
    if arms != {
        "structure_no_physics": ["ordinary", "rotation_augmented", "strict_so2"],
        "physics_weight_values": [0.0, 0.1],
        "interaction_structures": ["ordinary", "strict_so2"],
        "reuse_no_physics_for_interaction": True,
        "conditional_rotation_physics_only_if_selected": True,
    }:
        raise ValueError("Phase 6 arms differ from the approved minimum")
    if tuple(variant.variant_id for variant in phase5.variants) != tuple(
        arms["structure_no_physics"]
    ):
        raise ValueError("Phase 6 structure order differs from Phase 5")

    selection = _require_keys(
        root["selection"],
        {
            "qualification_gate_first",
            "structure_primary_metric",
            "structure_rule",
            "physics_secondary_metric",
            "physics_rule",
            "physics_weight_search_prohibited",
            "designated_checkpoint_seed",
            "final_reproduction_count",
        },
        "selection",
    )
    expected_selection = {
        "qualification_gate_first": True,
        "structure_primary_metric": "common_unrotated_validation_weak_label_composite",
        "structure_rule": (
            "unique_lowest_three_seed_median_and_at_least_two_of_three_seed_wins_else_ordinary"
        ),
        "physics_secondary_metric": "independent_diagnostic_physics_huber",
        "physics_rule": (
            "lower_paired_median_and_at_least_two_of_three_improvement_directions_and_"
            "validation_median_not_worse"
        ),
        "physics_weight_search_prohibited": True,
        "designated_checkpoint_seed": expected_seeds[0],
        "final_reproduction_count": 1,
    }
    if selection != expected_selection:
        raise ValueError("Phase 6 selection rule differs from the prospective freeze")

    low_quality = _require_keys(
        root["low_quality_stratum"],
        {
            "role",
            "split",
            "valid_pvt_only",
            "union_fields",
            "empirical_quantile",
            "rank_rule",
            "hdop_threshold",
            "postfit_residual_rms_m_threshold",
            "valid_epoch_count",
            "union_epoch_count",
            "enters_model_selection",
        },
        "low-quality stratum",
    )
    expected_low_quality = {
        "role": "deployable_metadata_robustness_report_only",
        "split": "diagnostic",
        "valid_pvt_only": True,
        "union_fields": ["dop.hdop", "postfit_residual_rms_m"],
        "empirical_quantile": 0.75,
        "rank_rule": "ceil_q_times_n_one_based_inclusive",
        "hdop_threshold": 1.827707154949175,
        "postfit_residual_rms_m_threshold": 32.6901790767606,
        "valid_epoch_count": 108,
        "union_epoch_count": 53,
        "enters_model_selection": False,
    }
    if low_quality != expected_low_quality:
        raise ValueError("Phase 6 low-quality stratum differs from the freeze")

    scope = _require_keys(
        root["scope"],
        {
            "predictive_covariance",
            "cross_receiver_claim",
            "cross_route_claim",
            "sensor_remounting_claim",
            "rotation_augmented_in_interaction_factorial",
            "truth_accuracy_claim_in_phase6",
        },
        "scope",
    )
    if scope != {
        "predictive_covariance": False,
        "cross_receiver_claim": False,
        "cross_route_claim": False,
        "sensor_remounting_claim": False,
        "rotation_augmented_in_interaction_factorial": False,
        "truth_accuracy_claim_in_phase6": False,
    }:
        raise ValueError("Phase 6 claim boundary differs from the freeze")

    return Phase6Config(
        profile_id=root["profile_id"],
        phase4_sha256=expected4,
        phase5_sha256=expected5,
        seeds=seeds,
        optimizer_steps=optimization["optimizer_steps"],
        checkpoint_every_steps=optimization["checkpoint_every_steps"],
        structure_variants=tuple(arms["structure_no_physics"]),
        interaction_structures=tuple(arms["interaction_structures"]),
        physics_weights=tuple(float(value) for value in arms["physics_weight_values"]),
        designated_checkpoint_seed=selection["designated_checkpoint_seed"],
        final_reproduction_count=selection["final_reproduction_count"],
        low_quality_hdop_threshold=low_quality["hdop_threshold"],
        low_quality_residual_threshold_m=low_quality["postfit_residual_rms_m_threshold"],
        low_quality_valid_epoch_count=low_quality["valid_epoch_count"],
        low_quality_union_epoch_count=low_quality["union_epoch_count"],
    )
