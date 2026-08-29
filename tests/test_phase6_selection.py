from __future__ import annotations

import unittest

from rtkfree_equivariant_gnss_ins.phase6_selection import (
    interaction_direction,
    select_physics,
    select_structure,
)


SEEDS = (11, 22, 33)


class Phase6SelectionTests(unittest.TestCase):
    def test_structure_requires_unique_median_and_two_seed_wins(self) -> None:
        report = select_structure(
            {
                "ordinary": {11: 3.0, 22: 3.0, 33: 3.0},
                "rotation_augmented": {11: 2.0, 22: 2.5, 33: 4.0},
                "strict_so2": {11: 4.0, 22: 4.0, 33: 2.0},
            },
            SEEDS,
        )
        self.assertEqual(report["selected_structure"], "rotation_augmented")
        self.assertEqual(report["seed_win_count"]["rotation_augmented"], 2)

    def test_structure_falls_back_to_ordinary_when_support_is_unstable(self) -> None:
        report = select_structure(
            {
                "ordinary": {11: 1.0, 22: 4.0, 33: 4.0},
                "rotation_augmented": {11: 4.0, 22: 1.0, 33: 5.0},
                "strict_so2": {11: 3.0, 22: 3.0, 33: 3.0},
            },
            SEEDS,
        )
        self.assertEqual(report["unique_lowest_median_candidate"], "strict_so2")
        self.assertEqual(report["selected_structure"], "ordinary")

    def test_physics_requires_all_three_frozen_conditions(self) -> None:
        report = select_physics(
            {
                "no_physics": {11: 3.0, 22: 3.2, 33: 3.1},
                "physics": {11: 2.9, 22: 3.1, 33: 3.1},
            },
            {
                "no_physics": {11: 0.4, 22: 0.5, 33: 0.6},
                "physics": {11: 0.3, 22: 0.4, 33: 0.7},
            },
            SEEDS,
        )
        self.assertTrue(report["physics_accepted"])
        self.assertEqual(report["selected_physics_weight"], 0.1)

    def test_physics_rejected_when_validation_median_is_worse(self) -> None:
        report = select_physics(
            {
                "no_physics": {11: 3.0, 22: 3.0, 33: 3.0},
                "physics": {11: 3.2, 22: 3.2, 33: 3.2},
            },
            {
                "no_physics": {11: 0.4, 22: 0.5, 33: 0.6},
                "physics": {11: 0.3, 22: 0.4, 33: 0.5},
            },
            SEEDS,
        )
        self.assertFalse(report["physics_accepted"])
        self.assertEqual(report["selected_physics_weight"], 0.0)

    def test_interaction_is_report_only(self) -> None:
        report = interaction_direction(
            {
                "ordinary_no_physics": {11: 10.0, 22: 10.0, 33: 10.0},
                "ordinary_physics": {11: 9.0, 22: 9.0, 33: 9.0},
                "strict_no_physics": {11: 8.0, 22: 8.0, 33: 8.0},
                "strict_physics": {11: 6.0, 22: 6.0, 33: 6.0},
            },
            SEEDS,
        )
        self.assertEqual(report["three_seed_median"], -1.0)
        self.assertEqual(report["negative_count"], 3)
        self.assertFalse(report["enters_model_selection"])


if __name__ == "__main__":
    unittest.main()
