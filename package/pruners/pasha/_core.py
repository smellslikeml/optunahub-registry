"""Framework-free ranking-stability core for PASHA.

PASHA (Progressive ASHA, arXiv:2207.06940) caps the maximum resource of a
successive-halving schedule and grows the cap one rung at a time, but only
while the ranking of configurations keeps changing between the top two active
rungs. This module implements the two decision pieces that do not depend on
Optuna:

- :func:`evaluate_soft_ranking` -- the soft-ranking consistency check deciding
  whether the ordering observed at the top rung agrees with the ordering at the
  rung below it, up to a tolerance ``epsilon``.
- :func:`estimate_epsilon` -- the automatic choice of ``epsilon`` from pairs of
  configurations whose relative order flips back and forth across steps.

This is a clean-room reimplementation of the algorithm published in the PASHA
paper (the reference implementation ships in Apache-2.0 licensed syne-tune).
"""

from __future__ import annotations

import itertools
import math

import numpy as np


__all__ = ["decide_resource_increase", "estimate_epsilon", "evaluate_soft_ranking"]

# A rung ranking: ``(trial identifier, metric value)`` pairs, one per
# configuration that already reported at the rung's resource level.
RungRanking = list[tuple[int, float]]


def sort_rung(rung: RungRanking, maximize: bool) -> RungRanking:
    """Order the entries of one rung from the best value to the worst one."""
    return sorted(rung, key=lambda entry: entry[1], reverse=maximize)


def _is_worse_by_more_than(other: float, reference: float, epsilon: float, maximize: bool) -> bool:
    gap = other - reference if not maximize else reference - other
    return gap > epsilon


def evaluate_soft_ranking(
    sorted_top_rung: RungRanking,
    sorted_previous_rung: RungRanking,
    epsilon: float,
    maximize: bool,
) -> bool:
    """Check whether two rungs agree on the ranking, up to ``epsilon``.

    Both rungs must be sorted best-first (see :func:`sort_rung`), and the
    previous rung must already be restricted to the configurations present in
    the top rung. Every configuration of the previous rung owns a group of
    configurations whose values are within ``epsilon`` of its own; the budget
    is kept (the rankings count as consistent) only if each configuration of
    the top rung stays inside the group of its same-position counterpart of
    the previous rung.

    Args:
        sorted_top_rung:
            Ranking of the top active rung, sorted best-first.
        sorted_previous_rung:
            Ranking of the rung below it, sorted best-first and restricted to
            the configurations of the top rung.
        epsilon:
            Soft tolerance; configurations whose values differ by no more than
            ``epsilon`` share a group. With fewer than two configurations in
            the previous rung the tolerance degenerates to ``0.0`` (an exact
            order comparison).
        maximize:
            Whether the objective is maximized.

    Returns:
        ``True`` when the current resource budget should be kept.
    """
    if len(sorted_previous_rung) < 2:
        epsilon = 0.0

    previous_rung_groups: list[set[int]] = []
    for idx, (_, value) in enumerate(sorted_previous_rung):
        group = {sorted_previous_rung[idx][0]}
        # Configurations ranked after the current one; the scan stops at the
        # first one that is more than ``epsilon`` worse.
        for after in range(idx + 1, len(sorted_previous_rung)):
            other = sorted_previous_rung[after][1]
            if _is_worse_by_more_than(other, value, epsilon, maximize):
                break
            group.add(sorted_previous_rung[after][0])
        # Configurations ranked before the current one; the scan stops at the
        # first one that is more than ``epsilon`` better.
        for before in range(idx - 1, -1, -1):
            other = sorted_previous_rung[before][1]
            if _is_worse_by_more_than(value, other, epsilon, maximize):
                break
            group.add(sorted_previous_rung[before][0])
        previous_rung_groups.append(group)

    for idx, (trial_id, _) in enumerate(sorted_top_rung):
        if idx >= len(previous_rung_groups) or trial_id not in previous_rung_groups[idx]:
            return False
    return True


def decide_resource_increase(
    top_rung: RungRanking,
    previous_rung: RungRanking,
    epsilon: float,
    maximize: bool,
) -> bool:
    """Decide whether PASHA should grow the resource cap by one more rung.

    Args:
        top_rung:
            ``(trial id, value)`` pairs recorded at the top active rung.
        previous_rung:
            ``(trial id, value)`` pairs recorded at the rung below it.
        epsilon:
            Soft tolerance used by :func:`evaluate_soft_ranking`; ``0.0``
            yields a hard (exact order) comparison.
        maximize:
            Whether the objective is maximized.

    Returns:
        ``True`` only when both rungs hold at least two configurations and the
        soft ranking of the top rung disagrees with the rung below it. With
        fewer observations there is nothing to compare and the budget is kept.
    """
    if len(top_rung) < 2 or len(previous_rung) < 2:
        return False

    top_ids = {trial_id for trial_id, _ in top_rung}
    previous_restricted = [entry for entry in previous_rung if entry[0] in top_ids]
    sorted_top_rung = sort_rung(top_rung, maximize)
    sorted_previous_rung = sort_rung(previous_restricted, maximize)
    return not evaluate_soft_ranking(sorted_top_rung, sorted_previous_rung, epsilon, maximize)


def _order_flip_then_back(first: dict[int, float], second: dict[int, float], step: int) -> bool:
    """Whether the order of two curves at ``step`` flipped earlier and returned."""
    current = first[step] > second[step]
    flipped = False
    earlier_steps = sorted(
        (other for other in first if other < step and other in second), reverse=True
    )
    for earlier in earlier_steps:
        if (first[earlier] > second[earlier]) != current:
            flipped = True
        elif flipped:
            return True
    return False


def estimate_epsilon(
    per_epoch_results: dict[int, dict[int, float]],
    upper_step: int,
    lower_step: int,
    previous_epsilon: float = 0.0,
) -> float:
    """Estimate the soft-ranking tolerance from noisy learning curves.

    ``per_epoch_results`` maps a trial identifier to its metric value at each
    reported step. Only steps in the band ``(lower_step, upper_step]`` (the one
    covered by the top two active rungs) are scanned, from the highest step
    down. A pair of configurations is noisy when their relative order at some
    earlier step flipped and later flipped back to the order they show at the
    step where the pair is first seen in the band; the estimate is the 90th
    percentile of the metric distance of such pairs. When no noisy pair
    exists, ``previous_epsilon`` is returned unchanged.

    Args:
        per_epoch_results:
            Per-trial learning curves, ``{trial id: {step: value}}``.
        upper_step:
            Resource level of the top active rung (inclusive band bound).
        lower_step:
            Resource level of the rung below it (exclusive band bound).
        previous_epsilon:
            Tolerance to keep when no noisy pair is found.

    Returns:
        The estimated tolerance.
    """
    band_steps = sorted(
        {
            step
            for curves in per_epoch_results.values()
            for step in curves
            if lower_step < step <= upper_step
        },
        reverse=True,
    )

    seen_pairs: set[tuple[int, int]] = set()
    noisy_distances: list[float] = []
    for step in band_steps:
        reporters = sorted(
            trial_id
            for trial_id, curves in per_epoch_results.items()
            if step in curves and math.isfinite(curves[step])
        )
        for pair in itertools.combinations(reporters, 2):
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            first, second = pair
            if _order_flip_then_back(per_epoch_results[first], per_epoch_results[second], step):
                distance = abs(per_epoch_results[first][step] - per_epoch_results[second][step])
                if math.isfinite(distance):
                    noisy_distances.append(distance)

    if not noisy_distances:
        return previous_epsilon
    return float(np.percentile(noisy_distances, 90))
