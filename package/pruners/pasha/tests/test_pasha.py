"""Unit tests for the PASHA ranking-stability core and pruner."""

from __future__ import annotations

from collections.abc import Callable

import optuna
import pytest

from pasha import PASHAPruner
from pasha import decide_resource_increase
from pasha import estimate_epsilon


# ── core: soft-ranking decisions ──────────────────────────────────────────────


def test_single_rung_never_grows() -> None:
    # With only one populated rung there is nothing to compare.
    only_top = [(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)]
    assert not decide_resource_increase(only_top, [], epsilon=0.5, maximize=False)


def test_stable_order_keeps_budget() -> None:
    top = [(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)]
    previous = [(0, 1.1), (1, 2.2), (2, 3.3), (3, 4.4)]
    assert not decide_resource_increase(top, previous, epsilon=0.5, maximize=False)


def test_scrambled_top_rung_grows() -> None:
    top = [(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)]
    previous = [(3, 1.1), (2, 2.2), (1, 3.3), (0, 4.4)]  # exact reversal
    assert decide_resource_increase(top, previous, epsilon=0.5, maximize=False)


def test_soft_mode_tolerates_swaps_inside_epsilon() -> None:
    # Trials 1 and 2 swap between the rungs, but by less than the tolerance.
    top = [(0, 1.0), (1, 2.1), (2, 3.0), (3, 4.0)]
    previous = [(0, 1.05), (2, 2.0), (1, 3.05), (3, 4.1)]
    assert not decide_resource_increase(top, previous, epsilon=1.5, maximize=False)
    # A tight (hard) tolerance sees the same data as a ranking change.
    assert decide_resource_increase(top, previous, epsilon=0.05, maximize=False)


def test_maximize_direction_is_supported() -> None:
    top = [(0, 40.0), (1, 30.0), (2, 20.0), (3, 10.0)]
    previous = [(0, 41.0), (1, 31.0), (2, 21.0), (3, 11.0)]
    assert not decide_resource_increase(top, previous, epsilon=0.5, maximize=True)
    previous_reversed = [(3, 41.0), (2, 31.0), (1, 21.0), (0, 11.0)]
    assert decide_resource_increase(top, previous_reversed, epsilon=0.5, maximize=True)


# ── core: automatic epsilon ───────────────────────────────────────────────────


def test_estimate_epsilon_uses_flip_then_back_pairs() -> None:
    curves = {
        0: {1: 1.0, 2: 3.0, 3: 1.0},
        1: {1: 2.0, 2: 2.0, 3: 2.0},
    }
    # The order of trials 0 and 1 flips at step 2 and flips back at step 3,
    # where they are 1.0 apart.
    assert estimate_epsilon(curves, upper_step=3, lower_step=1) == pytest.approx(1.0)


def test_estimate_epsilon_keeps_previous_without_flips() -> None:
    curves = {
        0: {1: 1.0, 2: 1.5, 3: 2.0},
        1: {1: 3.0, 2: 3.5, 3: 4.0},
    }
    assert estimate_epsilon(
        curves, upper_step=3, lower_step=1, previous_epsilon=0.25
    ) == pytest.approx(0.25)


# ── pruner: studies over a rung ladder with steps 1, 3, 9, 27 ────────────────


def _constant_objective(trial: optuna.trial.Trial) -> float:
    # The ranking never changes across rungs, so PASHA must keep the cap.
    x = trial.suggest_float("x", 0.0, 1.0)
    for step in (1, 3, 9, 27):
        trial.report(x, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return x


def _reversal_objective(trial: optuna.trial.Trial) -> float:
    # Below step 3 a smaller x looks better; from step 3 on the ranking of the
    # surviving trials reverses, which makes the top-two-rung ranking unstable.
    x = trial.suggest_float("x", 0.0, 1.0)
    value = x
    for step in (1, 3, 9, 27):
        value = x if step < 3 else 1.0 - x
        trial.report(value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return value


def _make_study(
    pruner: optuna.pruners.BasePruner, objective: Callable[[optuna.trial.Trial], float]
) -> optuna.study.Study:
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        sampler=optuna.samplers.RandomSampler(seed=0),
    )
    study.optimize(objective, n_trials=12)
    return study


def _last_steps(study: optuna.study.Study) -> list[int]:
    return [t.last_step for t in study.trials if t.last_step is not None]


def test_stable_ranking_keeps_the_cap() -> None:
    pruner = PASHAPruner(min_resource=1, reduction_factor=3)
    study = _make_study(pruner, _constant_objective)
    # Successive halving still prunes the bottom of each rung, but nothing may
    # pass the cap at rung 1 (step 3).
    assert pruner.current_max_rung == 1
    assert max(_last_steps(study)) == 3
    states = [t.state for t in study.trials]
    assert states.count(optuna.trial.TrialState.PRUNED) > 0


def test_unstable_ranking_grows_the_cap() -> None:
    pruner = PASHAPruner(min_resource=1, reduction_factor=3)
    study = _make_study(pruner, _reversal_objective)
    # The reversal between rungs 0 and 1 grows the cap to rung 2 (step 9);
    # rungs 1 and 2 agree again, so growth stops there.
    assert pruner.current_max_rung == 2
    assert max(_last_steps(study)) == 9


def test_max_rung_bounds_growth() -> None:
    pruner = PASHAPruner(min_resource=1, reduction_factor=3, max_rung=1)
    study = _make_study(pruner, _reversal_objective)
    # The ranking is unstable, but the cap may not leave rung 1.
    assert pruner.current_max_rung == 1
    assert max(_last_steps(study)) == 3


def test_hard_mode_uses_zero_epsilon() -> None:
    pruner = PASHAPruner(min_resource=1, reduction_factor=3, ranking_mode="hard")
    assert pruner.epsilon == 0.0


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        PASHAPruner(ranking_mode="fuzzy")
    with pytest.raises(ValueError):
        PASHAPruner(epsilon=-1.0)
    with pytest.raises(ValueError):
        PASHAPruner(epsilon="nope")
    with pytest.raises(ValueError):
        PASHAPruner(max_rung=-1)
    with pytest.raises(ValueError):
        PASHAPruner(reduction_factor=1)
