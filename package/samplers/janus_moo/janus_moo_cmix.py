from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import NSGAIISampler
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_EPS = 1e-12


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < _EPS, 1.0, sd)
    return (X - mu) / sd, mu, sd


def _fit_diag_quadratic_metric(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray | None:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    finite = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[finite]
    y = y[finite]
    if X.ndim != 2 or X.shape[0] < 4 or X.shape[1] <= 0:
        return None

    Z, _mu, sd = _standardize(X)
    design = np.c_[np.ones(Z.shape[0]), Z, 0.5 * Z * Z]
    penalty = float(ridge) * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    try:
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    except np.linalg.LinAlgError:
        return None

    hdiag = np.maximum(np.abs(coef[1 + Z.shape[1] :]), _EPS) / (sd * sd)
    return np.diag(hdiag)


def _fit_component_metric(
    X: np.ndarray, components: np.ndarray, ridge: float
) -> np.ndarray | None:
    X = np.asarray(X, dtype=float)
    S = np.asarray(components, dtype=float)
    if S.ndim == 1:
        S = S[:, None]
    finite = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(S), axis=1)
    X = X[finite]
    S = S[finite]
    if X.ndim != 2 or S.ndim != 2 or X.shape[0] != S.shape[0] or X.shape[0] < 4:
        return None
    if X.shape[1] <= 0 or S.shape[1] <= 0:
        return None

    Z, _mu, sd = _standardize(X)
    S = np.sqrt(np.maximum(S, 0.0))
    S = S - S.mean(axis=0)
    try:
        AinvZt = np.linalg.solve(Z.T @ Z + float(ridge) * np.eye(Z.shape[1]), Z.T)
    except np.linalg.LinAlgError:
        return None
    W = (AinvZt @ S).T
    J = W / sd[None, :]
    metric = J.T @ J
    if not np.all(np.isfinite(metric)):
        return None
    return 0.5 * (metric + metric.T)


def _nondominated_indices(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.all(np.isfinite(values), axis=1)
    idx = np.where(finite)[0]
    if idx.size == 0:
        return np.arange(values.shape[0])
    vals = values[idx]
    keep = np.ones(vals.shape[0], dtype=bool)
    for i, point in enumerate(vals):
        if not keep[i]:
            continue
        dominated = np.all(vals <= point, axis=1) & np.any(vals < point, axis=1)
        dominated[i] = False
        keep[dominated] = False
    return idx[keep]


def _crowding_like(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n, m = values.shape
    dist = np.zeros(n, dtype=float)
    if n <= 2:
        dist[:] = np.inf
        return dist
    for j in range(m):
        order = np.argsort(values[:, j], kind="mergesort")
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        span = values[order[-1], j] - values[order[0], j]
        if not np.isfinite(span) or span <= _EPS:
            continue
        dist[order[1:-1]] += (values[order[2:], j] - values[order[:-2], j]) / span
    return dist


class JanusMooCmixSampler(NSGAIISampler):
    """NSGA-II selection with JANUS-style Cmix candidate generation for MOO.

    NSGA-II still owns Pareto selection. This sampler occasionally replaces a
    child with inverse-curvature Gaussian samples estimated from either
    trial.user_attrs['component_losses'] or a diagonal quadratic scalar fallback.

    The defaults are intentionally conservative: short warm-up, low replacement
    probability, small normalized step size, and Pareto-front center selection.
    For Optuna-scale runs (often ~1000 trials), the warm-up is measured in a few
    populations rather than the long 60-generation delay used by 20k-evaluation
    pymoo experiments.
    """

    def __init__(
        self,
        *,
        population_size: int = 100,
        seed: int | None = None,
        cmix_prob: float = 0.02,
        cmix_window: int | str = "auto",
        cmix_ridge: float = 1.0,
        cmix_lambda: float = 1.0,
        cmix_sigma: float = 0.002,
        cmix_start_generation: int = 3,
        cmix_start_trial: int | None = 100,
        component_attr: str = "component_losses",
        scalar_fallback: str = "diag_quadratic",
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(population_size=population_size, seed=seed, **kwargs)
        self._verbose = bool(verbose)
        self._cmix_prob = float(cmix_prob)
        self._cmix_window = (
            3 * int(population_size) if str(cmix_window).lower() == "auto" else int(cmix_window)
        )
        self._cmix_ridge = float(cmix_ridge)
        self._cmix_lambda = float(cmix_lambda)
        self._cmix_sigma = float(cmix_sigma)
        if cmix_start_trial is None:
            self._cmix_start_trial = max(
                self._cmix_window, max(0, int(cmix_start_generation)) * int(population_size)
            )
        else:
            self._cmix_start_trial = max(0, int(cmix_start_trial))
        self._component_attr = component_attr
        self._scalar_fallback = str(scalar_fallback or "diag_quadratic").lower()
        self._search_space = IntersectionSearchSpace()
        self._cmix_rng = np.random.default_rng(seed)
        self._last_cmix_mode = "none"
        self._cmix_hits = 0
        self._cmix_fail = 0

    def infer_relative_search_space(
        self,
        study: optuna.study.Study,
        trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        return {
            name: dist
            for name, dist in self._search_space.calculate(study).items()
            if not dist.single()
        }

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        nsga_params = super().sample_relative(study, trial, search_space)
        if trial.number < self._cmix_start_trial:
            return nsga_params
        if len(search_space) > 1 and self._cmix_rng.random() < self._cmix_prob:
            params = self._sample_cmix(study, search_space)
            if params:
                self._cmix_hits += 1
                return params
            self._cmix_fail += 1
        return nsga_params

    def _sample_cmix(
        self,
        study: optuna.study.Study,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any] | None:
        param_names = sorted(search_space.keys())
        transforms = self._build_transforms(search_space, param_names)
        if transforms is None:
            return None

        trials = [
            trial
            for trial in study._get_trials(deepcopy=False, use_cache=True)
            if trial.state == TrialState.COMPLETE and trial.values is not None
        ]
        rows: list[list[float]] = []
        used_trials: list[FrozenTrial] = []
        components: list[list[float]] = []
        has_components = True
        for completed in trials[-max(4, self._cmix_window) :]:
            if not all(name in completed.params for name in param_names):
                continue
            try:
                rows.append(
                    [
                        self._to_norm(completed.params[name], transforms[name])
                        for name in param_names
                    ]
                )
            except (TypeError, ValueError):
                rows.pop()
                continue
            used_trials.append(completed)
            component_value = completed.user_attrs.get(self._component_attr)
            if component_value is None:
                has_components = False
            elif has_components:
                try:
                    components.append([float(v) for v in component_value])
                except (TypeError, ValueError):
                    has_components = False

        if len(rows) < 4:
            return None
        X = np.asarray(rows, dtype=float)
        metric = None
        if has_components and len(components) == len(rows):
            metric = _fit_component_metric(
                X, np.asarray(components, dtype=float), self._cmix_ridge
            )
            self._last_cmix_mode = "component_losses"
        if metric is None and self._scalar_fallback in ("diag_quadratic", "diagq", "quad"):
            values = np.asarray(
                [self._scalar_value(study, trial) for trial in used_trials], dtype=float
            )
            metric = _fit_diag_quadratic_metric(X, values, self._cmix_ridge)
            self._last_cmix_mode = "diag_quadratic"
        if metric is None:
            self._last_cmix_mode = "none"
            return None

        center = self._choose_center(study, used_trials, param_names, transforms)
        if center is None:
            center = X.mean(axis=0)
        cov = self._inverse_metric_cov(metric, X.shape[1])
        if cov is None:
            return None
        for _ in range(8):
            sample = self._cmix_rng.multivariate_normal(center, cov)
            sample = np.clip(sample, 0.0, 1.0)
            params = self._from_norm_vector(sample, search_space, param_names, transforms)
            if params is not None:
                return params
        return None

    def _build_transforms(
        self,
        search_space: dict[str, BaseDistribution],
        param_names: list[str],
    ) -> dict[str, dict[str, Any]] | None:
        transforms: dict[str, dict[str, Any]] = {}
        for name in param_names:
            dist = search_space[name]
            if not isinstance(dist, (FloatDistribution, IntDistribution)):
                return None
            low = float(dist.low)
            high = float(dist.high)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                return None
            transforms[name] = {
                "low": low,
                "high": high,
                "step": getattr(dist, "step", None),
                "is_int": isinstance(dist, IntDistribution),
            }
        return transforms

    def _to_norm(self, value: Any, transform: dict[str, Any]) -> float:
        return (float(value) - transform["low"]) / (transform["high"] - transform["low"])

    def _from_norm_vector(
        self,
        sample: np.ndarray,
        search_space: dict[str, BaseDistribution],
        param_names: list[str],
        transforms: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        params: dict[str, Any] = {}
        for idx, name in enumerate(param_names):
            transform = transforms[name]
            value = transform["low"] + float(sample[idx]) * (transform["high"] - transform["low"])
            dist = search_space[name]
            step = transform.get("step")
            if step is not None:
                value = transform["low"] + round((value - transform["low"]) / float(step)) * float(
                    step
                )
            if transform["is_int"]:
                value = int(round(value))
                value = int(min(max(value, transform["low"]), transform["high"]))
            else:
                value = float(min(max(value, transform["low"]), transform["high"]))
            if not dist._contains(dist.to_internal_repr(value)):
                return None
            params[name] = value
        return params

    def _scalar_value(self, study: optuna.study.Study, trial: FrozenTrial) -> float:
        values = np.asarray(trial.values, dtype=float)
        values = np.where(np.isfinite(values), values, 1.0)
        return float(np.mean(values))

    def _choose_center(
        self,
        study: optuna.study.Study,
        trials: list[FrozenTrial],
        param_names: list[str],
        transforms: dict[str, dict[str, Any]],
    ) -> np.ndarray | None:
        complete = [trial for trial in trials if all(name in trial.params for name in param_names)]
        if not complete:
            return None
        value_rows = [trial.values for trial in complete if trial.values is not None]
        if len(value_rows) == len(complete) and value_rows:
            values = np.asarray(value_rows, dtype=float)
            front_idx = _nondominated_indices(values)
            front = [complete[int(i)] for i in front_idx]
            if len(front) > 1:
                cd = _crowding_like(values[front_idx])
                order = np.argsort(-cd, kind="mergesort")
                top = [front[int(i)] for i in order[: max(1, min(5, len(front)))]]
            else:
                top = front
        else:
            top = []
        if not top:
            ranked = sorted(complete, key=lambda trial: self._scalar_value(study, trial))
            top = ranked[: max(1, min(5, len(ranked)))]
        rows = [
            [self._to_norm(trial.params[name], transforms[name]) for name in param_names]
            for trial in top
        ]
        return np.asarray(rows, dtype=float).mean(axis=0)

    def _inverse_metric_cov(self, metric: np.ndarray, dim: int) -> np.ndarray | None:
        try:
            cov = np.linalg.inv(metric + self._cmix_lambda * np.eye(dim))
        except np.linalg.LinAlgError:
            return None
        cov = 0.5 * (cov + cov.T)
        trace = float(np.trace(cov))
        if not np.isfinite(trace) or trace <= _EPS:
            return None
        cov *= (max(self._cmix_sigma, _EPS) ** 2 * dim) / trace
        cov += 1e-8 * np.eye(dim)
        return cov

    def after_trial(
        self,
        study: optuna.study.Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        population_size = self._population_size or 1
        period = max(1, int(population_size))
        if self._verbose and trial.number > 0 and trial.number % period == 0:
            print(
                f"[JANUS-MOO-CMIX] trial={trial.number}, hit={self._cmix_hits}, "
                f"fail={self._cmix_fail}, mode={self._last_cmix_mode}",
                flush=True,
            )
        return super().after_trial(study, trial, state, values)
