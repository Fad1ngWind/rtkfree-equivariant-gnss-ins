from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from rtkfree_equivariant_gnss_ins.phase6_config import load_phase6_config
from rtkfree_equivariant_gnss_ins.phase6_selection import (
    interaction_direction,
    select_physics,
    select_structure,
)
from rtkfree_equivariant_gnss_ins.phase6_validate import validate_phase6_decisions


ROOT = Path(__file__).resolve().parents[1]


class Phase6DecisionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_phase6_config(
            ROOT / "config" / "phase6" / "final_freeze_v1.json",
            ROOT / "config" / "phase4" / "ordinary_student_v1.json",
            ROOT / "config" / "phase5" / "gravity_aware_so2_v1.json",
        )
        seeds = self.config.seeds
        metrics = {
            "ordinary_p0": ([1.0, 1.1, 1.2], [0.4, 0.5, 0.6]),
            "rotation_augmented_p0": ([2.0, 2.1, 2.2], [0.3, 0.4, 0.5]),
            "strict_so2_p0": ([3.0, 3.1, 3.2], [0.2, 0.3, 0.4]),
            "ordinary_p01": ([1.3, 1.4, 1.5], [0.3, 0.4, 0.5]),
            "strict_so2_p01": ([2.7, 2.8, 2.9], [0.1, 0.2, 0.3]),
        }
        runs = {
            arm: {
                str(seed): {
                    "qualification_passed": True,
                    "best_validation_composite": values[0][index],
                    "independent_physics_huber": values[1][index],
                    "best_model_tensor_sha256": f"{arm}-{seed}",
                }
                for index, seed in enumerate(seeds)
            }
            for arm, values in metrics.items()
        }
        structure_matrix = {
            structure: {
                seed: runs[f"{structure}_p0"][str(seed)]["best_validation_composite"]
                for seed in seeds
            }
            for structure in self.config.structure_variants
        }
        structure = select_structure(structure_matrix, seeds)
        physics_validation = {
            "no_physics": {
                seed: runs["ordinary_p0"][str(seed)]["best_validation_composite"]
                for seed in seeds
            },
            "physics": {
                seed: runs["ordinary_p01"][str(seed)]["best_validation_composite"]
                for seed in seeds
            },
        }
        physics_huber = {
            "no_physics": {
                seed: runs["ordinary_p0"][str(seed)]["independent_physics_huber"]
                for seed in seeds
            },
            "physics": {
                seed: runs["ordinary_p01"][str(seed)]["independent_physics_huber"]
                for seed in seeds
            },
        }
        physics = select_physics(physics_validation, physics_huber, seeds)
        interaction: dict[str, object] = {}
        for name, metric in (
            ("common_validation_composite", "best_validation_composite"),
            ("independent_physics_huber", "independent_physics_huber"),
        ):
            interaction[name] = interaction_direction(
                {
                    "ordinary_no_physics": {
                        seed: runs["ordinary_p0"][str(seed)][metric] for seed in seeds
                    },
                    "ordinary_physics": {
                        seed: runs["ordinary_p01"][str(seed)][metric] for seed in seeds
                    },
                    "strict_no_physics": {
                        seed: runs["strict_so2_p0"][str(seed)][metric] for seed in seeds
                    },
                    "strict_physics": {
                        seed: runs["strict_so2_p01"][str(seed)][metric] for seed in seeds
                    },
                },
                seeds,
            )
        self.summary = {
            "runs": runs,
            "structure_decision": structure,
            "physics_decision": physics,
            "interaction_report_only": interaction,
            "final_primary": {
                "arm_id": "ordinary_p0",
                "structure": "ordinary",
                "physics_weight": 0.0,
                "designated_seed": seeds[0],
                "checkpoint_tensor_sha256": f"ordinary_p0-{seeds[0]}",
            },
        }

    def test_recomputes_frozen_decisions(self) -> None:
        report = validate_phase6_decisions(self.summary, self.config)
        self.assertTrue(report["decisions_validated"])
        self.assertEqual(report["final_arm_id"], "ordinary_p0")

    def test_rejects_tampered_structure_selection(self) -> None:
        tampered = deepcopy(self.summary)
        tampered["structure_decision"]["selected_structure"] = "strict_so2"
        with self.assertRaises(ValueError):
            validate_phase6_decisions(tampered, self.config)

    def test_rejects_unqualified_run(self) -> None:
        tampered = deepcopy(self.summary)
        first_seed = str(self.config.seeds[0])
        tampered["runs"]["strict_so2_p01"][first_seed]["qualification_passed"] = False
        with self.assertRaises(ValueError):
            validate_phase6_decisions(tampered, self.config)


if __name__ == "__main__":
    unittest.main()
