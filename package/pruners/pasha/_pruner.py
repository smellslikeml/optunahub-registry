"""Optuna glue for PASHA: :class:`PASHAPruner`.

:class:`PASHAPruner` is a drop-in replacement for
:class:`optuna.pruners.SuccessiveHalvingPruner` that adds the progressive
resource allocation of PASHA (Progressive ASHA, arXiv:2207.06940): the maximum
resource is capped at the second rung and the cap grows by one rung only while
the ranking of configurations across the top two active rungs is still
unstable, so the study stops allocating long runs once the ranking settles.

The ranking machinery lives in :mod:`._core`; this module re-derives the
successive-halving rung bookkeeping from public Optuna APIs (the
``completed_rung_{k}`` system attributes written at each promotion step) and
delegates the promotion rule to the parent class semantics.
"""

from __future__ import annotations

import math

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.study import StudyDirection

from ._core import decide_resource_increase
from ._core import estimate_epsilon


def _completed_rung_key(rung: int) -> str:
    """System-attribute key under which rung values are stored."""
    return f"completed_rung_{rung}"


def _current_rung(trial: optuna.trial.FrozenTrial) -> int:
    """The lowest rung the trial has not recorded a value at yet."""
    rung = 0
    while _completed_rung_key(rung) in trial.system_attrs:
        rung += 1
    return rung


def _estimate_min_resource(trials: list[optuna.trial.FrozenTrial]) -> int | None:
    """The ``SuccessiveHalvingPruner`` heuristic: max completed step / 100."""
    last_steps = [
        t.last_step
        for t in trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.last_step is not None
    ]
    if not last_steps:
        return None
    return max(max(last_steps) // 100, 1)


def _is_promotable(
    value: float,
    competing_values: list[float],
    reduction_factor: int,
    study_direction: StudyDirection,
) -> bool:
    """The ``SuccessiveHalvingPruner`` rule: promote the top ``1 / eta`` fraction."""
    promotable_idx = (len(competing_values) // reduction_factor) - 1
    if promotable_idx == -1:
        # With fewer than ``eta`` competitors, promote only the best value.
        promotable_idx = 0
    competing_values.sort()
    if study_direction == StudyDirection.MAXIMIZE:
        return value >= competing_values[-(promotable_idx + 1)]
    return value <= competing_values[promotable_idx]


def _collect_rung(
    trials: list[optuna.trial.FrozenTrial],
    rung: int,
    extra: tuple[int, float] | None = None,
) -> list[tuple[int, float]]:
    """Collect ``(trial number, value)`` pairs recorded at ``rung``.

    ``extra`` folds in a value recorded during the current call, which the
    snapshot of ``trials`` may predate; the freshest value for a trial wins.
    """
    key = _completed_rung_key(rung)
    entries: dict[int, float] = {}
    for t in trials:
        if key in t.system_attrs:
            entries[t.number] = float(t.system_attrs[key])
    if extra is not None:
        entries[extra[0]] = extra[1]
    return list(entries.items())


class PASHAPruner(SuccessiveHalvingPruner):
    """Pruner using Progressive ASHA (PASHA).

    PASHA extends asynchronous successive halving with a maximum-resource cap
    that starts at the second rung (the top of the two lowest rungs) and grows
    by exactly one rung each time the ranking of configurations across the top
    two active rungs is unstable. Once the ranking settles, trials that reach
    the cap are pruned there and the remaining compute is saved. See
    `arXiv:2207.06940 <https://arxiv.org/abs/2207.06940>`__ for details.

    Example:

        .. testcode::

            import optuna
            import optunahub

            module = optunahub.load_module(package="pruners/pasha")

            pruner = module.PASHAPruner(min_resource=1, reduction_factor=3)
            study = optuna.create_study(direction="minimize", pruner=pruner)
            study.optimize(objective, n_trials=30)

    Args:
        min_resource:
            Inherited from :class:`~optuna.pruners.SuccessiveHalvingPruner`;
            the resource level of the first rung. With ``"auto"`` no pruning
            happens before the first trial completes.
        reduction_factor:
            Inherited from :class:`~optuna.pruners.SuccessiveHalvingPruner`;
            the rung promotion factor :math:`\\eta`.
        min_early_stopping_rate:
            Inherited from :class:`~optuna.pruners.SuccessiveHalvingPruner`.
        bootstrap_count:
            Inherited from :class:`~optuna.pruners.SuccessiveHalvingPruner`.
        ranking_mode:
            ``"soft"`` compares rung rankings up to the ``epsilon`` tolerance
            (noise that does not move a configuration out of its group counts
            as stability); ``"hard"`` compares exact orders (``epsilon = 0``).
        epsilon:
            The soft-ranking tolerance, or ``"auto"`` to estimate it on the
            fly from pairs of trials whose relative order flips back and forth
            across steps (the 90th percentile of their metric distances).
            Ignored (kept at ``0.0``) when ``ranking_mode="hard"``.
        max_rung:
            The highest rung index the cap may grow to (the absolute top
            rung). ``None`` means no fixed limit beyond the study's own step
            range.
    """

    def __init__(
        self,
        min_resource: str | int = "auto",
        reduction_factor: int = 4,
        min_early_stopping_rate: int = 0,
        bootstrap_count: int = 0,
        *,
        ranking_mode: str = "soft",
        epsilon: float | str = "auto",
        max_rung: int | None = None,
    ) -> None:
        super().__init__(
            min_resource=min_resource,
            reduction_factor=reduction_factor,
            min_early_stopping_rate=min_early_stopping_rate,
            bootstrap_count=bootstrap_count,
        )
        if ranking_mode not in ("soft", "hard"):
            raise ValueError(
                f"The value of `ranking_mode` is {ranking_mode}, "
                "but must be either 'soft' or 'hard'"
            )
        if isinstance(epsilon, str) and epsilon != "auto":
            raise ValueError(
                f"The value of `epsilon` is {epsilon}, but must be `epsilon >= 0` or 'auto'"
            )
        if not isinstance(epsilon, str) and epsilon < 0:
            raise ValueError(
                f"The value of `epsilon` is {epsilon}, but must be `epsilon >= 0` or 'auto'"
            )
        if max_rung is not None and max_rung < 0:
            raise ValueError(
                f"The value of `max_rung` is {max_rung}, but must be `max_rung >= 0` or None"
            )

        self._ranking_mode = ranking_mode
        if ranking_mode == "hard":
            self._epsilon: float = 0.0
            self._epsilon_is_auto = False
        elif isinstance(epsilon, str):  # "auto", validated above
            self._epsilon = 0.0
            self._epsilon_is_auto = True
        else:
            self._epsilon = float(epsilon)
            self._epsilon_is_auto = False

        self._max_rung = max_rung
        # The cap starts at the second rung (the top of the two lowest rungs);
        # with a single rung (``max_rung == 0``) there is nothing to grow.
        self._current_max_rung = min(max_rung, 1) if max_rung is not None else 1
        self._per_epoch_results: dict[int, dict[int, float]] = {}
        # Declare the inherited lazily-resolved attribute's type for mypy
        # (annotation only — assigning here would drop a user-supplied min_resource).
        self._min_resource: int | None

    @property
    def current_max_rung(self) -> int:
        """The highest rung index trials may currently be promoted to."""
        return self._current_max_rung

    @property
    def epsilon(self) -> float:
        """The soft-ranking tolerance currently in use."""
        return self._epsilon

    def _rung_level(self, rung: int) -> int:
        """The step at which ``rung`` is evaluated."""
        assert self._min_resource is not None
        return self._min_resource * self._reduction_factor ** (
            self._min_early_stopping_rate + rung
        )

    def _refresh_epsilon(self) -> None:
        """Re-estimate the tolerance from the curves inside the active band."""
        if not self._epsilon_is_auto or self._min_resource is None or self._current_max_rung < 1:
            return
        self._epsilon = estimate_epsilon(
            self._per_epoch_results,
            upper_step=self._rung_level(self._current_max_rung),
            lower_step=self._rung_level(self._current_max_rung - 1),
            previous_epsilon=self._epsilon,
        )

    def _grow_if_unstable(
        self,
        study: optuna.study.Study,
        trials: list[optuna.trial.FrozenTrial],
        trial: optuna.trial.FrozenTrial,
        value: float,
    ) -> bool:
        """Grow the cap by one rung if the top-two-rung ranking is unstable.

        Returns ``True`` when the cap was grown, in which case the reporting
        trial continues to the new rung instead of being pruned at the old cap.
        """
        if self._max_rung is not None and self._current_max_rung >= self._max_rung:
            return False
        top_rung = _collect_rung(trials, self._current_max_rung, extra=(trial.number, value))
        previous_rung = _collect_rung(trials, self._current_max_rung - 1)
        maximize = study.direction == StudyDirection.MAXIMIZE
        if decide_resource_increase(top_rung, previous_rung, self._epsilon, maximize):
            self._current_max_rung += 1
            return True
        return False

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        step = trial.last_step
        if step is None:
            return False

        trials = study.get_trials(deepcopy=False)

        # PASHA bookkeeping: the per-step curves feed the automatic tolerance.
        self._per_epoch_results[trial.number] = dict(trial.intermediate_values)
        self._refresh_epsilon()

        rung = _current_rung(trial)
        value = trial.intermediate_values[step]

        while True:
            if self._min_resource is None:
                self._min_resource = _estimate_min_resource(trials)
                if self._min_resource is None:
                    return False
            assert self._min_resource is not None

            if step < self._rung_level(rung):
                return False

            if math.isnan(value):
                return True

            rung_key = _completed_rung_key(rung)
            study._storage.set_trial_system_attr(trial._trial_id, rung_key, value)

            # PASHA re-evaluates the cap on *every* report that completes the
            # current top rung and grows it by one while the top-two-rung ranking
            # is unstable. This is a property of the rungs, not of whether this
            # particular trial is promotable, so it must run before the
            # successive-halving promotion gate (otherwise the first trial to
            # reach the cap decides growth alone, with a single observation, and
            # the cap never grows once later trials are pruned at it).
            if rung >= self._current_max_rung and self._grow_if_unstable(
                study, trials, trial, value
            ):
                # The report revealed the top two rungs are still unstable and
                # grew the cap: this trial continues to the new rung rather than
                # being halved at the old cap (optuna prunes terminally, so a
                # trial capped now could never be resumed once the cap grows).
                rung += 1
                continue

            competing_values = [
                t.system_attrs[rung_key] for t in trials if rung_key in t.system_attrs
            ]
            competing_values.append(value)

            if len(competing_values) <= self._bootstrap_count:
                return True

            if not _is_promotable(
                value, competing_values, self._reduction_factor, study.direction
            ):
                return True

            # At or above the (settled) cap: the trial is pruned here.
            if rung >= self._current_max_rung:
                return True

            rung += 1
