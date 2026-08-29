"""Pure prospective Phase 6 selection and interaction rules."""

from __future__ import annotations

import math
from statistics import median
from typing import Mapping, Sequence


STRUCTURES = ("ordinary", "rotation_augmented", "strict_so2")


def _validated_matrix(
    values: Mapping[str, Mapping[int, float]],
    rows: Sequence[str],
    seeds: Sequence[int],
    name: str,
) -> dict[str, dict[int, float]]:
    if tuple(values) != tuple(rows):
        raise ValueError(f"{name} row order differs from the freeze")
    result: dict[str, dict[int, float]] = {}
    for row in rows:
        if tuple(values[row]) != tuple(seeds):
            raise ValueError(f"{name} seed order differs from the freeze")
        result[row] = {seed: float(values[row][seed]) for seed in seeds}
        if not all(math.isfinite(value) for value in result[row].values()):
            raise ValueError(f"{name} contains a nonfinite value")
    return result


def select_structure(
    validation: Mapping[str, Mapping[int, float]],
    seeds: Sequence[int],
) -> dict[str, object]:
    """Select a no-physics structure or fall back to ordinary by simplicity."""

    matrix = _validated_matrix(validation, STRUCTURES, seeds, "structure validation")
    medians = {name: float(median(matrix[name].values())) for name in STRUCTURES}
    minimum_median = min(medians.values())
    median_winners = [name for name in STRUCTURES if medians[name] == minimum_median]
    unique_candidate = median_winners[0] if len(median_winners) == 1 else None
    per_seed_winners: dict[str, str | None] = {}
    win_counts = {name: 0 for name in STRUCTURES}
    for seed in seeds:
        minimum = min(matrix[name][seed] for name in STRUCTURES)
        winners = [name for name in STRUCTURES if matrix[name][seed] == minimum]
        winner = winners[0] if len(winners) == 1 else None
        per_seed_winners[str(seed)] = winner
        if winner is not None:
            win_counts[winner] += 1
    supported = unique_candidate is not None and win_counts[unique_candidate] >= 2
    selected = unique_candidate if supported else "ordinary"
    return {
        "selected_structure": selected,
        "unique_lowest_median_candidate": unique_candidate,
        "candidate_supported": supported,
        "three_seed_median_validation": medians,
        "per_seed_winner": per_seed_winners,
        "seed_win_count": win_counts,
        "fallback_to_ordinary": not supported,
    }


def select_physics(
    validation: Mapping[str, Mapping[int, float]],
    physics_huber: Mapping[str, Mapping[int, float]],
    seeds: Sequence[int],
) -> dict[str, object]:
    """Apply the matched no-physics versus 0.1 physics decision exactly."""

    rows = ("no_physics", "physics")
    validation_matrix = _validated_matrix(validation, rows, seeds, "physics validation")
    physics_matrix = _validated_matrix(physics_huber, rows, seeds, "physics Huber")
    validation_medians = {
        name: float(median(validation_matrix[name].values())) for name in rows
    }
    huber_medians = {name: float(median(physics_matrix[name].values())) for name in rows}
    improved_seeds = [
        seed
        for seed in seeds
        if physics_matrix["physics"][seed] < physics_matrix["no_physics"][seed]
    ]
    accepted = (
        huber_medians["physics"] < huber_medians["no_physics"]
        and len(improved_seeds) >= 2
        and validation_medians["physics"] <= validation_medians["no_physics"]
    )
    return {
        "selected_physics_weight": 0.1 if accepted else 0.0,
        "physics_accepted": accepted,
        "three_seed_median_validation": validation_medians,
        "three_seed_median_physics_huber": huber_medians,
        "physics_huber_improved_seed_count": len(improved_seeds),
        "physics_huber_improved_seeds": improved_seeds,
    }


def interaction_direction(
    values: Mapping[str, Mapping[int, float]],
    seeds: Sequence[int],
) -> dict[str, object]:
    """Report the frozen structure-by-physics difference-of-differences direction."""

    rows = ("ordinary_no_physics", "ordinary_physics", "strict_no_physics", "strict_physics")
    matrix = _validated_matrix(values, rows, seeds, "interaction")
    by_seed = {
        str(seed): (
            matrix["strict_physics"][seed]
            - matrix["strict_no_physics"][seed]
            - matrix["ordinary_physics"][seed]
            + matrix["ordinary_no_physics"][seed]
        )
        for seed in seeds
    }
    directions = tuple(by_seed.values())
    return {
        "difference_of_differences_by_seed": by_seed,
        "three_seed_median": float(median(directions)),
        "negative_count": sum(value < 0.0 for value in directions),
        "zero_count": sum(value == 0.0 for value in directions),
        "positive_count": sum(value > 0.0 for value in directions),
        "enters_model_selection": False,
    }
