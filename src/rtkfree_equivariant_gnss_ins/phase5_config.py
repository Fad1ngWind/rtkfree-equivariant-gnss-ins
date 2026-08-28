"""Strict loader for the predeclared Phase 5 fair-comparison profile."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .phase4_config import load_phase4_config


@dataclass(frozen=True)
class Phase5Variant:
    variant_id: str
    architecture: str
    rotation_augmentation: bool
    strict_so2: bool
    physics_weight: float
    trainable_parameter_count: int


@dataclass(frozen=True)
class Phase5Config:
    profile_id: str
    base_phase4_sha256: str
    variants: tuple[Phase5Variant, ...]
    scalar_hidden_dim: int
    vector_hidden_channels: int
    capacity_relative_tolerance: float
    capacity_minimum: int
    capacity_maximum: int
    augmentation_seed: int
    property_angles_rad: tuple[float, ...]
    float64_atol: float
    float64_rtol: float
    float32_atol: float
    float32_rtol: float
    checkpoint_every_steps: int


def _require_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys differ from the Phase 5 freeze")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_phase5_config(path: Path, phase4_path: Path) -> Phase5Config:
    """Load Phase 5 only if its exact comparison and Phase 4 base remain frozen."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read Phase 5 config") from exc
    if not isinstance(raw, dict):
        raise ValueError("Phase 5 config root must be an object")
    _require_keys(
        raw,
        {
            "schema_version",
            "profile_id",
            "base_phase4",
            "variants",
            "strict_model",
            "capacity_control",
            "augmentation",
            "property_test",
            "selection",
            "matched_phase4_fields",
            "scope",
        },
        "Phase 5 config",
    )
    if raw["schema_version"] != 1 or raw["profile_id"] != "gravity_aware_so2_comparison_v1":
        raise ValueError("Phase 5 profile identity differs from the freeze")

    base = raw["base_phase4"]
    _require_keys(base, {"profile_id", "config_path", "config_sha256"}, "base Phase 4")
    expected_base_hash = "7d6607edfe38725942ec0a4df6cac77af47d1b41d79a7b376ae30b0c5f1772cd"
    if base != {
        "profile_id": "ordinary_causal_student_v1",
        "config_path": "config/phase4/ordinary_student_v1.json",
        "config_sha256": expected_base_hash,
    }:
        raise ValueError("base Phase 4 declaration differs from the freeze")
    if _sha256(phase4_path) != expected_base_hash:
        raise ValueError("base Phase 4 config hash differs from the Phase 5 freeze")
    phase4 = load_phase4_config(phase4_path)
    if phase4.profile_id != base["profile_id"] or phase4.seed != 3407:
        raise ValueError("base Phase 4 profile or seed differs from the Phase 5 freeze")

    expected_variants = (
        ("ordinary", "phase4_ordinary_causal_student", False, False, 6994),
        ("rotation_augmented", "phase4_ordinary_causal_student", True, False, 6994),
        ("strict_so2", "gravity_aware_typed_scalar_vector_student", False, True, 6780),
    )
    variants_raw = raw["variants"]
    if not isinstance(variants_raw, list) or len(variants_raw) != len(expected_variants):
        raise ValueError("Phase 5 must contain exactly three variants")
    variants: list[Phase5Variant] = []
    for value, expected in zip(variants_raw, expected_variants, strict=True):
        _require_keys(
            value,
            {
                "id",
                "architecture",
                "rotation_augmentation",
                "strict_so2",
                "physics_weight",
                "trainable_parameter_count",
            },
            "Phase 5 variant",
        )
        actual = (
            value["id"],
            value["architecture"],
            value["rotation_augmentation"],
            value["strict_so2"],
            value["trainable_parameter_count"],
        )
        if actual != expected or value["physics_weight"] != 0.0:
            raise ValueError("Phase 5 variant differs from the frozen comparison")
        variants.append(
            Phase5Variant(
                variant_id=value["id"],
                architecture=value["architecture"],
                rotation_augmentation=value["rotation_augmentation"],
                strict_so2=value["strict_so2"],
                physics_weight=float(value["physics_weight"]),
                trainable_parameter_count=value["trainable_parameter_count"],
            )
        )

    strict_model = raw["strict_model"]
    if strict_model != {"scalar_hidden_dim": 30, "vector_hidden_channels": 5}:
        raise ValueError("strict model dimensions differ from the freeze")
    capacity = raw["capacity_control"]
    if capacity != {
        "ordinary_parameter_count": 6994,
        "relative_tolerance": 0.15,
        "minimum_parameter_count": 5945,
        "maximum_parameter_count": 8043,
    }:
        raise ValueError("capacity control differs from the freeze")
    if not all(
        capacity["minimum_parameter_count"]
        <= variant.trainable_parameter_count
        <= capacity["maximum_parameter_count"]
        for variant in variants
    ):
        raise ValueError("a Phase 5 variant violates the capacity tolerance")

    augmentation = raw["augmentation"]
    if augmentation != {
        "schedule": "sha256_phase5_yaw_v1_seed_and_zero_based_training_pass",
        "seed": 3407,
        "range": "open_minus_pi_to_pi",
        "one_coordinate_view_per_training_pass": True,
        "extra_optimizer_steps": 0,
    }:
        raise ValueError("augmentation protocol differs from the freeze")
    property_test = raw["property_test"]
    if property_test != {
        "angles_rad": [0.37, -1.11, 2.23],
        "float64": {"atol": 1e-09, "rtol": 1e-08},
        "float32": {"atol": 5e-06, "rtol": 5e-05},
        "identity_required": True,
        "composition_required": True,
        "complete_rollout_required": True,
    }:
        raise ValueError("property-test protocol differs from the freeze")
    selection = raw["selection"]
    if selection != {
        "checkpoint_every_steps": 10,
        "rule": "minimum_finite_common_unrotated_validation_weak_label_state_loss",
    }:
        raise ValueError("selection protocol differs from the freeze")
    if raw["matched_phase4_fields"] != [
        "inputs",
        "development_sequence",
        "normalization",
        "controlled_masks",
        "optimization",
        "loss.kind",
        "loss.huber_delta",
        "loss.state_scales",
        "loss.teacher_covariance_weighting",
    ]:
        raise ValueError("matched Phase 4 fields differ from the freeze")
    if raw["scope"] != {
        "primary_group": "SO(2)",
        "coordinate_equivariance_only": True,
        "sensor_remounting_claim": False,
        "sealed_reference_used": False,
        "phase4_negative_physics_huber": {
            "no_physics": 0.374354878,
            "physics": 0.418493815,
        },
    }:
        raise ValueError("Phase 5 scope differs from the freeze")

    return Phase5Config(
        profile_id=raw["profile_id"],
        base_phase4_sha256=expected_base_hash,
        variants=tuple(variants),
        scalar_hidden_dim=strict_model["scalar_hidden_dim"],
        vector_hidden_channels=strict_model["vector_hidden_channels"],
        capacity_relative_tolerance=capacity["relative_tolerance"],
        capacity_minimum=capacity["minimum_parameter_count"],
        capacity_maximum=capacity["maximum_parameter_count"],
        augmentation_seed=augmentation["seed"],
        property_angles_rad=tuple(property_test["angles_rad"]),
        float64_atol=property_test["float64"]["atol"],
        float64_rtol=property_test["float64"]["rtol"],
        float32_atol=property_test["float32"]["atol"],
        float32_rtol=property_test["float32"]["rtol"],
        checkpoint_every_steps=selection["checkpoint_every_steps"],
    )
