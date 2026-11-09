"""
JANUS sampler (evotorch backend).

Unified Gauss-Newton metric framework: ONE estimated Jacobian J drives BOTH
halves of the search.

  EXPLOIT (unchanged from mosaic_gn): a GN/LM elite
      Δ = −(JᵀJ + λI)⁻¹ Jᵀ s,  x_b + tΔ
  is assembled and BOUNDED-injected (whitened step ‖z‖ clipped to κ√d).

  EXPLORE (CMIX): the SAME GN curvature metric creates a
  tail of inverse-curvature candidates:
      T  = (JᵀJ + λ_mix I)⁻¹
      x_tail ~ N(mean, σ² · trace_normalize(T))
  This keeps CMA's native covariance adaptation intact while adding targeted
  low-curvature exploration.

IMPORTANT — w is an ABSOLUTE weight, not w_fac·c1. The report's "w ≈ 2·c1 sweet
spot" was tuned at low dim; at AWB's d=1135 evotorch's c_1 ≈ 1.55e-6, so w_fac·c1
would blend essentially nothing. cmix_w defaults to 0.02 (dimension-robust).

cmix reuses the JᵀJ computed during the GN assembly the SAME generation, so there
is no extra Jacobian estimation. Defaults use GN+CMIX with adaptive fuse_start
and gn_window, popsize supplied by cma_opts when needed, and 40% candidate
budgets. Requires per-trial component_losses (or rae) as the residual vector r.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging as _py_logging
import math
import random
from typing import Any

import numpy as np
import optuna
from optuna import logging
from optuna._imports import try_import
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


with try_import() as _imports:
    from evotorch import Problem
    from evotorch.algorithms import CMAES
    from evotorch.algorithms.cmaes import _h_sig
    import torch


for _n in ("evotorch", "evotorch.core"):
    _py_logging.getLogger(_n).setLevel(_py_logging.WARNING)

_logger = logging.get_logger(__name__)
_EPS = 1e-10


def dynamic_normalize_params(
    params: dict[str, Any],
    norm_min: float = 0,
    norm_max_range: list[float] = [1, 2],
) -> dict[str, Any]:
    low, high = params["low"], params["high"]
    step = params.get("step", None)
    original_range = high - low
    best: dict[str, Any] = {"nm": None, "sf": None, "st": None}
    min_dp = float("inf")
    for nm in [
        x * 0.01 for x in range(int(norm_max_range[0] * 100), int(norm_max_range[1] * 100) + 1)
    ]:
        sf = (nm - norm_min) / original_range
        dp = len(str(sf).split(".")[1]) if "." in str(sf) else 0
        if dp < min_dp:
            min_dp = dp
            best = {"nm": nm, "sf": sf, "st": round(step * sf, 10) if step else None}
    return {"low": norm_min, "high": best["nm"], "step": best["st"], "scale": best["sf"]}


def resolve_pca_k(pca_k: int | str, n: int, d: int, min_extra: int = 10) -> int:
    if isinstance(pca_k, str) and pca_k.lower() == "auto":
        budget = max(1, n - int(min_extra))
        q_sample = int((math.sqrt(1.0 + 8.0 * budget) - 3.0) // 2)
        q_dim = int(math.ceil(math.sqrt(float(d))))
        return int(max(1, min(d, q_sample, q_dim)))
    return int(max(1, min(int(pca_k), d)))


def _pseudo_target(f: np.ndarray, target: str = "raw") -> np.ndarray:
    f = np.asarray(f, dtype=float).reshape(-1)
    target = str(target or "raw").lower()
    if target in ("rank", "ranks", "rank_normalized"):
        order = np.argsort(f)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(f.size, dtype=float)
        if f.size > 1:
            ranks = ranks / float(f.size - 1)
        return ranks - float(np.mean(ranks))
    if target in ("log", "log_shift", "log1p"):
        shift = float(np.min(f))
        y = np.log1p(np.maximum(f - shift, 0.0))
        return y - float(np.mean(y))
    return f - float(np.mean(f))


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    return (X - mu) / sd, mu, sd


def _positive_hessian_metric(
    H: np.ndarray, mode: str, fit_r2: float = 0.0
) -> dict[str, Any] | None:
    H = 0.5 * (np.asarray(H, dtype=float) + np.asarray(H, dtype=float).T)
    if not np.all(np.isfinite(H)):
        return None
    try:
        evals, evecs = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        return None
    pos = evals > max(float(np.max(np.abs(evals))), 1.0) * 1e-10
    if not np.any(pos):
        return None
    lam = evals[pos]
    V = evecs[:, pos]
    JtJ = V @ np.diag(lam) @ V.T
    return {
        "JtJ": 0.5 * (JtJ + JtJ.T),
        "mode": mode,
        "rank": int(lam.size),
        "fit_r2": float(fit_r2),
    }


def _linear_pseudo(
    Z: np.ndarray, y: np.ndarray, sd: np.ndarray, ridge: float
) -> dict[str, Any] | None:
    n, d = Z.shape
    try:
        w = np.linalg.solve(Z.T @ Z + float(ridge) * np.eye(d), Z.T @ y)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(w)):
        return None
    g = w / sd
    JtJ = np.outer(g, g)
    return {"JtJ": JtJ, "mode": "linear", "rank": 1, "fit_r2": 0.0}


def _fit_full_quadratic(
    Z: np.ndarray, y: np.ndarray, sd: np.ndarray, ridge: float
) -> dict[str, Any] | None:
    n, d = Z.shape
    iu = np.triu_indices(d)
    Phi = np.concatenate([Z, Z[:, iu[0]] * Z[:, iu[1]]], axis=1)
    if n < Phi.shape[1] + 2:
        return None
    try:
        coef = np.linalg.solve(Phi.T @ Phi + float(ridge) * np.eye(Phi.shape[1]), Phi.T @ y)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coef)):
        return None
    yhat = Phi @ coef
    sst = float(np.sum(y**2))
    fit_r2 = 0.0 if sst <= _EPS else max(0.0, 1.0 - float(np.sum((y - yhat) ** 2)) / sst)
    hvec = coef[d:]
    H_std = np.zeros((d, d), dtype=float)
    H_std[iu] = hvec
    H_std = H_std + H_std.T
    inv_sd = 1.0 / sd
    H = H_std * inv_sd[:, None] * inv_sd[None, :]
    return _positive_hessian_metric(H, "quadratic", fit_r2=fit_r2)


def _fit_diag_quadratic(
    Z: np.ndarray, y: np.ndarray, sd: np.ndarray, ridge: float
) -> dict[str, Any] | None:
    n, d = Z.shape
    if n < 2 * d + 10:
        return None
    Phi = np.concatenate([Z, Z * Z], axis=1)
    try:
        coef = np.linalg.solve(Phi.T @ Phi + float(ridge) * np.eye(2 * d), Phi.T @ y)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coef)):
        return None
    yhat = Phi @ coef
    sst = float(np.sum(y**2))
    fit_r2 = 0.0 if sst <= _EPS else max(0.0, 1.0 - float(np.sum((y - yhat) ** 2)) / sst)
    hdiag = np.maximum(2.0 * coef[d:] / (sd * sd), 0.0)
    if not np.any(hdiag > 0):
        return None
    return {
        "JtJ": np.diag(hdiag),
        "mode": "diag_quadratic",
        "rank": int(np.count_nonzero(hdiag > 0)),
        "fit_r2": float(fit_r2),
    }


def _fit_pca_quadratic(
    Z: np.ndarray, y: np.ndarray, sd: np.ndarray, pca_k: int | str, ridge: float
) -> dict[str, Any] | None:
    n, d = Z.shape
    if isinstance(pca_k, str) and pca_k.lower() in ("none", "off", "false", "0"):
        return None
    if not isinstance(pca_k, str) and int(pca_k) <= 0:
        return None
    q = int(max(1, min(resolve_pca_k(pca_k, n, d, min_extra=10), n - 2)))
    try:
        _, _, vt = np.linalg.svd(Z, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if vt.shape[0] < q:
        return None
    basis = vt[:q].T
    U = Z @ basis
    iu = np.triu_indices(q)
    Phi = np.concatenate([U, U[:, iu[0]] * U[:, iu[1]]], axis=1)
    if n < Phi.shape[1] + 2:
        return None
    try:
        coef = np.linalg.solve(Phi.T @ Phi + float(ridge) * np.eye(Phi.shape[1]), Phi.T @ y)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coef)):
        return None
    yhat = Phi @ coef
    sst = float(np.sum(y**2))
    fit_r2 = 0.0 if sst <= _EPS else max(0.0, 1.0 - float(np.sum((y - yhat) ** 2)) / sst)
    hvec = coef[q:]
    H_sub = np.zeros((q, q), dtype=float)
    H_sub[iu] = hvec
    H_sub = H_sub + H_sub.T
    H_std = basis @ H_sub @ basis.T
    inv_sd = 1.0 / sd
    H = H_std * inv_sd[:, None] * inv_sd[None, :]
    out = _positive_hessian_metric(H, "pca_quadratic", fit_r2=fit_r2)
    if out is not None:
        out["q"] = int(q)
    return out


def pseudo_metric(
    X: np.ndarray,
    f: np.ndarray,
    mode: str = "linear",
    ridge: float = 1.0,
    pca_k: int | str = "auto",
    target: str = "raw",
) -> dict[str, Any] | None:
    X = np.asarray(X, dtype=float)
    f = np.asarray(f, dtype=float).reshape(-1)
    if X.ndim != 2 or f.ndim != 1 or X.shape[0] != f.shape[0]:
        return None
    finite = np.all(np.isfinite(X), axis=1) & np.isfinite(f)
    X = X[finite]
    f = f[finite]
    if X.shape[0] < 4 or X.shape[1] <= 0:
        return None
    Z, _mu, sd = _standardize(X)
    y = _pseudo_target(f, target=target)
    mode = str(mode or "linear").lower()

    def linear() -> dict[str, Any] | None:
        return _linear_pseudo(Z, y, sd, ridge)

    if mode in ("none", "off", "false"):
        return None
    if mode in ("linear", "lin"):
        return linear()
    if mode in ("quadratic", "quad"):
        return _fit_full_quadratic(Z, y, sd, ridge) or linear()
    if mode in ("diag_quadratic", "diagq", "diagonal_quadratic"):
        return _fit_diag_quadratic(Z, y, sd, ridge) or linear()
    if mode in ("pca_quadratic", "pcaq"):
        return _fit_pca_quadratic(Z, y, sd, pca_k, ridge) or linear()
    if mode in ("diag_pca_quadratic", "diag_pcaq", "diagpca"):
        diag = _fit_diag_quadratic(Z, y, sd, ridge)
        pca = _fit_pca_quadratic(Z, y, sd, pca_k, ridge)
        if diag is None and pca is None:
            return linear()
        if diag is None:
            return pca
        if pca is None or pca.get("mode") == "linear":
            return diag
        JtJ = diag["JtJ"] + pca["JtJ"]
        return {
            "JtJ": 0.5 * (JtJ + JtJ.T),
            "mode": "diag_pca_quadratic",
            "rank": int(diag.get("rank", 0) + pca.get("rank", 0)),
            "fit_r2": max(float(diag.get("fit_r2", 0.0)), float(pca.get("fit_r2", 0.0))),
            "q": pca.get("q"),
        }
    return linear()


class JanusSampler(BaseSampler):
    """JANUS: GN exploit injection + candidate inverse-curvature CMIX exploration."""

    def __init__(
        self,
        x0: dict[str, Any] | None = None,
        sigma0: float | None = None,
        seed: int | None = None,
        cma_opts: dict[str, Any] | None = None,
        n_startup_trials: int = 1,
        independent_sampler: BaseSampler | None = None,
        warn_independent_sampling: bool = True,
        n_jobs: int = 1,
        fuse_every: int = 4,
        fuse_start: int | str = "auto",
        kappa: float = 1.0,
        gn_window: int | str = "auto",
        gn_lambda: float = 1.0,
        gn_ridge: float = 1.0,
        cmix_w: float | str = "auto",
        cmix_lambda: float = 1.0,
        cmix_w_base: float = 0.5,
        cmix_w_min: float = 0.01,
        cmix_w_max: float = 0.20,
        pseudo_mode: str = "linear",
        gn_pseudo_mode: str | None = None,
        cmix_pseudo_mode: str | None = None,
        pseudo_target: str = "raw",
        pca_k: int | str = "auto",
        cmix_mode: str = "portfolio",
        cmix_candidate_frac: float = 0.4,
        cmix_candidate_center: str = "mean",
        cmix_candidate_cov: str = "metric",
        verbose: bool = False,
    ) -> None:
        _imports.check()
        self._verbose = bool(verbose)
        self._x0 = x0
        self._sigma0 = sigma0
        if seed is None:
            seed = random.randint(1, 2**31 - 1)
        self._seed = seed
        self._cma_opts = cma_opts or {}
        self._n_startup_trials = n_startup_trials
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._warn_independent_sampling = warn_independent_sampling
        self._search_space = IntersectionSearchSpace()
        self._n_jobs = n_jobs
        self._fuse_every = fuse_every
        self._fuse_start = fuse_start
        self._kappa = kappa
        self._gn_window = gn_window
        self._gn_lambda = gn_lambda
        self._gn_ridge = gn_ridge
        self._cmix_w = cmix_w
        self._cmix_lambda = cmix_lambda
        self._cmix_w_base = cmix_w_base
        self._cmix_w_min = cmix_w_min
        self._cmix_w_max = cmix_w_max
        self._pseudo_mode = pseudo_mode
        self._gn_pseudo_mode = gn_pseudo_mode or pseudo_mode
        self._cmix_pseudo_mode = cmix_pseudo_mode or pseudo_mode
        self._pseudo_target = pseudo_target
        self._pca_k = pca_k
        self._cmix_mode = cmix_mode
        self._cmix_candidate_frac = cmix_candidate_frac
        self._cmix_candidate_center = cmix_candidate_center
        self._cmix_candidate_cov = cmix_candidate_cov
        self.optimizer_dict: dict[int, _JanusOptimizer] = {}

    def reseed_rng(self) -> None:
        self._seed = random.randint(1, 2**31 - 1)
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: optuna.study.Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return {n: d for n, d in self._search_space.calculate(study).items() if not d.single()}

    def sample_independent(
        self,
        study: optuna.study.Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self._raise_error_if_multi_objective(study)
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        self._raise_error_if_multi_objective(study)
        if len(search_space) <= 1:
            return {}
        completed = self._get_trials(study)
        if len(completed) < self._n_startup_trials:
            return {}
        if self._x0 is None:
            x0: dict[str, Any] = {}
            for n, d in search_space.items():
                if isinstance(d, IntDistribution):
                    x0[n] = int(np.mean([d.high, d.low]))
                elif isinstance(d, FloatDistribution):
                    x0[n] = float(np.mean([d.high, d.low]))
                elif isinstance(d, CategoricalDistribution):
                    x0[n] = d.choices[(len(d.choices) - 1) // 2]
            self._x0 = x0
        if self._sigma0 is None:
            self._sigma0 = 0.2
        popsize = self._cma_opts.get("popsize") or (4 + int(3 * math.log(len(search_space))))
        thread_id = trial.number % self._n_jobs

        opt = self.optimizer_dict.get(thread_id)
        if opt is None:
            opt = _JanusOptimizer(
                search_space,
                completed,
                study.direction,
                self._x0,
                self._sigma0,
                self._cma_opts,
                self._seed + thread_id,
                self._fuse_every,
                self._fuse_start,
                self._kappa,
                self._gn_window,
                self._gn_lambda,
                self._gn_ridge,
                self._cmix_w,
                self._cmix_lambda,
                self._cmix_w_base,
                self._cmix_w_min,
                self._cmix_w_max,
                self._pseudo_mode,
                self._gn_pseudo_mode,
                self._cmix_pseudo_mode,
                self._pseudo_target,
                self._pca_k,
                self._cmix_mode,
                self._cmix_candidate_frac,
                self._cmix_candidate_center,
                self._cmix_candidate_cov,
                self._verbose,
            )
        else:
            opt._completed_trials = completed
        opt._update_es()
        self.optimizer_dict[thread_id] = opt
        return opt.ask(trial.number % popsize)

    def before_trial(self, study: optuna.study.Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: optuna.study.Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)

    def _get_trials(self, study: optuna.study.Study) -> list[FrozenTrial]:
        return [
            t
            for t in study._get_trials(deepcopy=False, use_cache=True)
            if t.state == TrialState.COMPLETE
        ]


class _JanusOptimizer:
    def __init__(
        self,
        search_space: dict[str, BaseDistribution],
        completed_trials: list[FrozenTrial],
        study_direction: StudyDirection,
        x0: dict[str, Any],
        sigma0: float,
        cma_opts: dict[str, Any],
        seed: int,
        fuse_every: int,
        fuse_start: int | str,
        kappa: float,
        gn_window: int | str,
        gn_lambda: float,
        gn_ridge: float,
        cmix_w: float | str,
        cmix_lambda: float,
        cmix_w_base: float = 0.5,
        cmix_w_min: float = 0.01,
        cmix_w_max: float = 0.20,
        pseudo_mode: str = "linear",
        gn_pseudo_mode: str | None = None,
        cmix_pseudo_mode: str | None = None,
        pseudo_target: str = "raw",
        pca_k: int | str = "auto",
        cmix_mode: str = "candidates",
        cmix_candidate_frac: float = 0.05,
        cmix_candidate_center: str = "mean",
        cmix_candidate_cov: str = "metric",
        verbose: bool = False,
    ) -> None:
        self._verbose = bool(verbose)
        self._search_space = search_space
        self._param_names = list(sorted(search_space.keys()))
        self._study_direction = study_direction
        self._completed_trials = completed_trials
        self._fuse_every = fuse_every
        self._fuse_start = fuse_start
        self._kappa = kappa
        self._gn_window = gn_window
        self._gn_lambda = gn_lambda
        self._gn_ridge = gn_ridge
        self._cmix_w = cmix_w  # may be "auto" until first GN assembly resolves it
        self._cmix_lambda = cmix_lambda
        self._cmix_w_base = cmix_w_base
        self._cmix_w_min = cmix_w_min
        self._cmix_w_max = cmix_w_max
        self._pseudo_mode = pseudo_mode
        self._gn_pseudo_mode = gn_pseudo_mode or pseudo_mode
        self._cmix_pseudo_mode = cmix_pseudo_mode or pseudo_mode
        self._pseudo_target = pseudo_target
        self._pca_k = pca_k
        self._cmix_mode = str(cmix_mode or "candidates").lower()
        self._cmix_candidate_frac = float(cmix_candidate_frac)
        self._cmix_candidate_center = str(cmix_candidate_center or "mean").lower()
        self._cmix_candidate_cov = str(cmix_candidate_cov or "metric").lower()
        self.trials_used: list[int] = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._rng = np.random.default_rng(int(seed) % (2**32 - 1))
        self._archive_cursor = 0

        self._param_transforms: dict[str, dict[str, Any]] = {}
        for name in self._param_names:
            dist = search_space[name]
            if isinstance(dist, (FloatDistribution, IntDistribution)):
                norm = dynamic_normalize_params(
                    {"low": dist.low, "high": dist.high, "step": dist.step}
                )
                self._param_transforms[name] = {
                    "bias": dist.low,
                    "scale": norm["scale"],
                    "low": norm["low"],
                    "high": norm["high"],
                }
        d = len(self._param_names)
        self._d = d
        self._sqrt_d = math.sqrt(d)
        self._popsize = cma_opts.get("popsize") or (4 + int(3 * math.log(d)))

        lo = np.array([self._param_transforms[n]["low"] for n in self._param_names], float)
        hi = np.array([self._param_transforms[n]["high"] for n in self._param_names], float)
        self._lo_t = torch.tensor(lo, dtype=torch.float, device=self._device)
        self._hi_t = torch.tensor(hi, dtype=torch.float, device=self._device)
        self._lo_np = lo
        self._hi_np = hi
        init_mean = (self._lo_t + self._hi_t) / 2.0
        stdev_init = max(float(((self._hi_t - self._lo_t) / 6.0).min().item()), _EPS)

        torch.manual_seed(int(seed) % (2**31 - 1))
        self._problem = Problem(
            "min",
            objective_func=None,
            initial_bounds=(self._lo_t, self._hi_t),
            solution_length=d,
            device=self._device,
        )
        self._es = CMAES(
            self._problem,
            stdev_init=stdev_init,
            center_init=init_mean,
            popsize=self._popsize,
            limit_C_decomposition=True,
        )

        self._generation = 0
        self._pending_xs: Any = None
        self._win_x: list[np.ndarray] = []
        self._win_r: list[np.ndarray] = []
        self._win_f: list[float] = []
        self._seen: set[int] = set()
        self._n_injected = 0
        self._last_clip = 0.0
        self._gn_fail = 0
        self._last_JtJ: np.ndarray | None = None  # numpy [d,d], set fresh each GN assembly
        self._last_pseudo_mode: str | None = None
        self._pca_hits = 0
        self._pca_fail = 0
        self._n_cmix = 0
        self._cmix_fail = 0
        self._cmix_w_eff: float | None = None  # resolved numeric weight (set when m is known)

    def _resolve_cmix_w(self, m: int) -> float:
        """Resolve cmix_w to a numeric weight. 'auto' scales with how overdetermined
        the Jacobian regression is: w = w_base · min(m, window) / d, clipped.
        At AWB (m=111, window=300, d=1135): 0.5·111/1135 ≈ 0.049."""
        if self._cmix_w_eff is not None:
            return self._cmix_w_eff
        if isinstance(self._cmix_w, str) and self._cmix_w.lower() == "auto":
            r = float(min(m, self._resolve_gn_window(m)))
            w = self._cmix_w_base * r / float(self._d)
            auto_min = self._cmix_w_base / float(self._d)
            w = float(min(max(w, auto_min), self._cmix_w_max))
        else:
            w = float(self._cmix_w)
        self._cmix_w_eff = w
        return w

    def _resolve_gn_window(self, m: int | None = None) -> int:
        setting = self._gn_window
        if not (isinstance(setting, str) and setting.lower() == "auto"):
            return int(max(self._popsize, setting))
        min_window = max(4 * self._popsize, 80)
        if m is None:
            target = max(min_window, 300)
        else:
            target = max(min_window, 3 * int(m))
        return int(min(max(target, min_window), 1200))

    def _resolve_fuse_start(self, window: int) -> int:
        setting = self._fuse_start
        if not (isinstance(setting, str) and setting.lower() == "auto"):
            return int(max(self._popsize, min(int(setting), window)))
        return int(max(2 * self._popsize, min(200, max(self._popsize, window // 2))))

    def _harvest(self) -> None:
        for t in self._completed_trials:
            if t.number in self._seen:
                continue
            rae = (
                (t.user_attrs.get("component_losses") or t.user_attrs.get("rae"))
                if t.user_attrs
                else None
            )
            if rae is None:
                continue
            try:
                x = np.array([self._to_cma(n, t.params[n]) for n in self._param_names], float)
            except Exception:
                continue
            fval = float(t.value) if t.value is not None else float(np.mean(rae))
            if t.value is not None and self._study_direction == StudyDirection.MAXIMIZE:
                fval = -fval
            self._win_x.append(x)
            self._win_r.append(np.asarray(rae, float))
            self._win_f.append(fval)
            self._seen.add(t.number)
        m = len(self._win_r[-1]) if self._win_r else None
        window = self._resolve_gn_window(m)
        if len(self._win_x) > window:
            self._win_x = self._win_x[-window:]
            self._win_r = self._win_r[-window:]
            self._win_f = self._win_f[-window:]

    # ---- GN/LM elite + cache JᵀJ for CMIX ----
    def _assemble_gn(self) -> np.ndarray | None:
        self._last_JtJ = None
        if self._win_r:
            window = self._resolve_gn_window(len(self._win_r[-1]))
        else:
            window = self._resolve_gn_window(None)
        fuse_start = self._resolve_fuse_start(window)
        if len(self._win_x) < max(fuse_start, self._popsize):
            return None
        X = np.array(self._win_x)
        R = np.array(self._win_r)
        if R.ndim != 2 or R.shape[0] != X.shape[0]:
            return None
        S = np.sqrt(np.maximum(R, 0.0))
        d = self._d
        self._resolve_cmix_w(S.shape[1])
        mu = X.mean(0)
        sd = X.std(0) + 1e-9
        Z = (X - mu) / sd
        try:
            AinvZt = np.linalg.solve(Z.T @ Z + self._gn_ridge * np.eye(d), Z.T)
        except np.linalg.LinAlgError:
            self._gn_fail += 1
            return None
        Sc = S - S.mean(0)
        W = (AinvZt @ Sc).T
        J = W / sd[None, :]
        gn_JtJ = J.T @ J
        pseudo = pseudo_metric(
            X,
            np.asarray(self._win_f, dtype=float),
            mode=self._gn_pseudo_mode,
            ridge=self._gn_ridge,
            pca_k=self._pca_k,
            target=self._pseudo_target,
        )
        use_pseudo = pseudo is not None and str(pseudo.get("mode", "linear")) != "linear"
        if use_pseudo:
            assert pseudo is not None
            self._last_JtJ = np.asarray(pseudo["JtJ"], dtype=float)
            self._last_pseudo_mode = str(pseudo.get("mode"))
            if "pca" in self._last_pseudo_mode:
                self._pca_hits += 1
        else:
            self._last_JtJ = gn_JtJ
            self._last_pseudo_mode = "gn_residual"
            if "pca" in str(self._gn_pseudo_mode).lower():
                self._pca_fail += 1
        ib = int(np.argmin(R.mean(1)))
        x_b = X[ib]
        s_b = S[ib]
        H = self._last_JtJ + self._gn_lambda * np.eye(d)
        try:
            if use_pseudo:
                y = _pseudo_target(
                    np.asarray(self._win_f, dtype=float), target=self._pseudo_target
                )
                g = np.linalg.solve(Z.T @ Z + self._gn_ridge * np.eye(d), Z.T @ y) / sd
                delta = -np.linalg.solve(H, g)
            else:
                delta = -np.linalg.solve(H, J.T @ s_b)
        except np.linalg.LinAlgError:
            self._gn_fail += 1
            return None
        if not np.all(np.isfinite(delta)):
            self._gn_fail += 1
            return None
        best_t, best_pred = 1.0, np.inf
        for step in (0.25, 0.5, 1.0):
            if use_pseudo:
                pred = float(
                    self._win_f[ib]
                    + step * np.dot(g, delta)
                    + 0.5 * step * step * np.dot(delta, self._last_JtJ @ delta)
                )
            else:
                pred = float(np.sum((s_b + step * (J @ delta)) ** 2))
            if pred < best_pred:
                best_pred, best_t = pred, step
        elite = x_b + best_t * delta
        return np.clip(elite, self._lo_np, self._hi_np)

    def _candidate_metric(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._last_JtJ is None:
            return None
        if self._cmix_pseudo_mode != self._gn_pseudo_mode and self._win_x:
            cmix_pseudo = pseudo_metric(
                np.asarray(self._win_x, dtype=float),
                np.asarray(self._win_f, dtype=float),
                mode=self._cmix_pseudo_mode,
                ridge=self._gn_ridge,
                pca_k=self._pca_k,
                target=self._pseudo_target,
            )
            if cmix_pseudo is not None and str(cmix_pseudo.get("mode", "linear")) != "linear":
                self._last_JtJ = np.asarray(cmix_pseudo["JtJ"], dtype=float)
                self._last_pseudo_mode = str(cmix_pseudo.get("mode"))
                if "pca" in self._last_pseudo_mode:
                    self._pca_hits += 1
        d = self._d
        try:
            metric = np.linalg.inv(self._last_JtJ + self._cmix_lambda * np.eye(d))
        except np.linalg.LinAlgError:
            self._cmix_fail += 1
            return None
        if not np.all(np.isfinite(metric)):
            self._cmix_fail += 1
            return None
        metric = 0.5 * (metric + metric.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(metric)
        except np.linalg.LinAlgError:
            self._cmix_fail += 1
            return None
        if not np.all(np.isfinite(eigvals)):
            self._cmix_fail += 1
            return None
        eigvals = np.maximum(eigvals, 1e-12)
        C = self._es.C.detach().cpu().numpy().astype(np.float64)
        trC = float(np.trace(C))
        trM = float(np.sum(eigvals))
        if trC <= 0.0 or trM <= 0.0 or not np.isfinite(trC):
            self._cmix_fail += 1
            return None
        eigvals = eigvals * (trC / trM)
        return eigvecs, eigvals

    def _archive_centers(self, count: int) -> list[np.ndarray]:
        if count <= 0 or not self._win_x or not self._win_r:
            return []
        R = np.asarray(self._win_r, dtype=float)
        X = np.asarray(self._win_x, dtype=float)
        if R.ndim != 2 or X.ndim != 2 or R.shape[0] != X.shape[0]:
            return []
        finite = np.all(np.isfinite(R), axis=1) & np.all(np.isfinite(X), axis=1)
        if not np.any(finite):
            return []
        R = R[finite]
        X = X[finite]
        order = np.argsort(np.mean(R, axis=1), kind="mergesort")[
            : min(R.shape[0], max(200, 8 * count))
        ]
        R = R[order]
        X = X[order]
        keep = np.ones(R.shape[0], dtype=bool)
        for i in range(R.shape[0]):
            if not keep[i]:
                continue
            dominated = np.all(R <= R[i], axis=1) & np.any(R < R[i], axis=1)
            dominated[i] = False
            keep[dominated] = False
        archive_x = X[keep]
        archive_r = R[keep]
        if archive_x.shape[0] == 0:
            return []
        if archive_x.shape[0] > 1:
            lo = np.min(archive_r, axis=0)
            hi = np.max(archive_r, axis=0)
            norm_r = (archive_r - lo) / np.maximum(hi - lo, 1e-12)
            selected = [int(np.argmin(np.mean(norm_r, axis=1)))]
            while len(selected) < min(count, archive_x.shape[0]):
                dist = np.min(
                    np.linalg.norm(
                        norm_r[:, None, :] - norm_r[np.asarray(selected)][None, :, :], axis=2
                    ),
                    axis=1,
                )
                dist[np.asarray(selected)] = -1.0
                selected.append(int(np.argmax(dist)))
            archive_x = archive_x[np.asarray(selected)]
        centers = [
            archive_x[(self._archive_cursor + i) % archive_x.shape[0]] for i in range(count)
        ]
        self._archive_cursor = (self._archive_cursor + count) % max(1, archive_x.shape[0])
        return centers

    def _cmix_candidate_samples(self, count: int, elite_np: np.ndarray | None = None) -> Any:
        if count <= 0 or self._last_JtJ is None:
            return None
        metric = self._candidate_metric()
        if metric is None:
            return None
        eigvecs, eigvals = metric
        if self._cmix_candidate_center == "archive":
            centers = self._archive_centers(count)
            if centers:
                scale = float(max(self._es.sigma, _EPS))
                samples_list = []
                for center in centers:
                    noise = self._rng.normal(size=self._d) * np.sqrt(eigvals) * scale
                    samples_list.append(np.asarray(center, dtype=np.float64) + eigvecs @ noise)
                samples = np.clip(
                    np.asarray(samples_list, dtype=np.float64),
                    self._lo_np[None, :],
                    self._hi_np[None, :],
                )
                self._n_cmix += int(samples.shape[0])
                return torch.tensor(samples, dtype=torch.float, device=self._device)
        if self._cmix_candidate_center == "best" and self._win_x:
            center = np.array(self._win_x[int(np.argmin(self._win_f))], float)
        elif self._cmix_candidate_center == "elite" and elite_np is not None:
            center = np.asarray(elite_np, float)
        else:
            center = self._es.m.detach().cpu().numpy().astype(np.float64)
        scale = float(self._es.sigma)
        noise = np.random.randn(count, self._d) * np.sqrt(eigvals)[None, :]
        samples = center[None, :] + scale * (noise @ eigvecs.T)
        samples = np.clip(samples, self._lo_np[None, :], self._hi_np[None, :])
        self._n_cmix += int(samples.shape[0])
        return torch.tensor(samples, dtype=torch.float, device=self._device)

    def _cmix_shape_C(self) -> None:
        w = self._cmix_w_eff
        if self._last_JtJ is None or w is None or w <= 0.0:
            return
        d = self._d
        try:
            T = np.linalg.inv(self._last_JtJ + self._cmix_lambda * np.eye(d))
        except np.linalg.LinAlgError:
            self._cmix_fail += 1
            return
        if not np.all(np.isfinite(T)):
            self._cmix_fail += 1
            return
        T = 0.5 * (T + T.T)
        es = self._es
        C = es.C.detach().cpu().numpy().astype(np.float64)
        trC = float(np.trace(C))
        trT = float(np.trace(T))
        if trT <= 1e-30 or not np.isfinite(trC):
            self._cmix_fail += 1
            return
        Tn = T * (trC / trT)
        C_new = (1.0 - w) * C + w * Tn
        C_new = 0.5 * (C_new + C_new.T)
        es.C = torch.tensor(C_new, dtype=es.C.dtype, device=self._device)
        self._safe_decompose()
        self._n_cmix += 1

    def _bounded_elite(self, fused_np: np.ndarray) -> Any:
        es = self._es
        x_e = torch.tensor(fused_np, dtype=torch.float, device=self._device)
        y = (x_e - es.m) / es.sigma
        if es.separable:
            z = y / es.A
        else:
            z = torch.linalg.solve_triangular(es.A, y.unsqueeze(1), upper=False).squeeze(1)
        znorm = float(torch.linalg.norm(z))
        cap = self._kappa * self._sqrt_d
        if znorm > cap and znorm > 1e-12:
            z = z * (cap / znorm)
            self._last_clip = cap / znorm
        else:
            self._last_clip = 1.0
        if es.separable:
            x_new = es.m + es.sigma * (es.A * z)
        else:
            x_new = es.m + es.sigma * (es.A @ z)
        return torch.clamp(x_new, self._lo_t, self._hi_t)

    def _update_es(self) -> None:
        if not self._completed_trials:
            return
        self._harvest()
        used = set(self.trials_used)
        selected = [t for t in self._completed_trials if t.number not in used]
        selected.sort(key=lambda t: t.number)
        gen_before = self._generation
        for i in range(len(selected) // self._popsize):
            batch = selected[i * self._popsize : (i + 1) * self._popsize]
            xs_np = np.array(
                [[self._to_cma(n, t.params[n]) for n in self._param_names] for t in batch], float
            )
            ys_val = np.array([t.value for t in batch], float)
            if self._study_direction == StudyDirection.MAXIMIZE:
                ys_val = -ys_val
            for t in batch:
                self.trials_used.append(t.number)
            self._cma_tell(xs_np, ys_val)
            self._generation += 1
            if self._verbose and self._generation % 5 == 0:
                weff = self._cmix_w_eff if self._cmix_w_eff is not None else -1.0
                print(
                    f"[JANUS] gen={self._generation}, win={len(self._win_x)}, "
                    f"sigma={float(self._es.sigma):.6f}, injected={self._n_injected}, "
                    f"cmix={self._n_cmix}, w={weff:.4f}, "
                    f"last_clip={self._last_clip:.3f}, gn_fail={self._gn_fail}, "
                    f"cmix_fail={self._cmix_fail}, pseudo={self._last_pseudo_mode}, "
                    f"pca_hit={self._pca_hits}, pca_fail={self._pca_fail}",
                    flush=True,
                )
        if self._generation != gen_before:
            self._pending_xs = None

    def _cma_tell(self, xs_np: np.ndarray, fitness_np: np.ndarray) -> None:
        es = self._es
        xs = torch.tensor(xs_np, dtype=torch.float, device=self._device)
        ys = (xs - es.m.unsqueeze(0)) / es.sigma
        if es.separable:
            zs = ys / es.A.unsqueeze(0)
        else:
            zs = torch.linalg.solve_triangular(es.A, ys.T, upper=False).T
        fit = torch.tensor(fitness_np, dtype=torch.float, device=self._device)
        indices = torch.argsort(fit)
        ranks = torch.zeros_like(indices)
        ranks[indices] = torch.arange(self._popsize, dtype=indices.dtype, device=indices.device)
        assigned_weights = es.weights[ranks]
        local_m_disp, shaped_m_disp = es.update_m(zs, ys, assigned_weights)
        es.update_p_sigma(local_m_disp)
        es.update_sigma()
        h_sig = _h_sig(es.p_sigma, es.c_sigma, es._steps_count)
        es.update_p_c(shaped_m_disp, h_sig)
        es.update_C(zs, ys, assigned_weights, h_sig)
        es._steps_count += 1
        if es._steps_count % es.decompose_C_freq == 0:
            self._safe_decompose()

    def _safe_decompose(self) -> None:
        es = self._es
        try:
            es.decompose_C()
            return
        except Exception:
            pass
        for jit in (1e-9, 1e-7, 1e-5, 1e-3, 1e-1):
            try:
                idx = torch.arange(self._d, device=self._device)
                es.C[idx, idx] = es.C[idx, idx] + jit
                es.decompose_C()
                return
            except Exception:
                continue
        es.C = torch.eye(self._d, device=self._device, dtype=es.C.dtype)
        try:
            es.decompose_C()
        except Exception:
            pass

    def _generate(self) -> None:
        elite = None
        if self._generation % self._fuse_every == 0:
            elite = self._assemble_gn()
            if self._cmix_mode == "reshape":
                self._cmix_shape_C()
        _zs, _ys, xs = self._es.sample_distribution()
        xs = torch.clamp(xs, self._lo_t.unsqueeze(0), self._hi_t.unsqueeze(0))
        if self._cmix_mode in ("candidates", "candidate"):
            n_cand = int(round(self._popsize * self._cmix_candidate_frac))
            n_cand = max(0, min(self._popsize - 1, n_cand))
            cand = self._cmix_candidate_samples(n_cand, elite_np=elite)
            if cand is not None and cand.shape[0] > 0:
                xs[-cand.shape[0] :] = cand
        elif self._cmix_mode in ("portfolio", "hybrid"):
            tail_budget = self._popsize - 1
            n_cand = int(round(self._popsize * self._cmix_candidate_frac))
            n_cand = max(0, min(tail_budget, n_cand))
            cursor = self._popsize
            cand = self._cmix_candidate_samples(n_cand, elite_np=elite)
            if cand is not None and cand.shape[0] > 0:
                cursor -= cand.shape[0]
                xs[cursor : cursor + cand.shape[0]] = cand
        if elite is not None:
            xs[0] = self._bounded_elite(elite)
            self._n_injected += 1
        self._pending_xs = xs

    def ask(self, idx: int) -> dict[str, Any]:
        if self._pending_xs is None:
            self._generate()
        x = self._pending_xs[idx % self._pending_xs.shape[0]].detach().cpu().numpy()
        if not np.all(np.isfinite(x)):
            self._cmix_fail += 1
            x = self._lo_np + self._rng.random(self._d) * (self._hi_np - self._lo_np)
        x = np.clip(x, self._lo_np, self._hi_np)
        return {n: self._from_cma(n, v) for n, v in zip(self._param_names, x)}

    def _to_cma(self, name: str, val: Any) -> float:
        dist = self._search_space[name]
        if isinstance(dist, (IntDistribution, FloatDistribution)):
            t = self._param_transforms[name]
            return min(max((val - t["bias"]) * t["scale"], t["low"]), t["high"])
        elif isinstance(dist, CategoricalDistribution):
            return float(dist.choices.index(val))
        return float(val)

    def _from_cma(self, name: str, val: float) -> Any:
        dist = self._search_space[name]
        if isinstance(dist, (FloatDistribution, IntDistribution)):
            t = self._param_transforms[name]
            v = min(max(val / t["scale"] + t["bias"], dist.low), dist.high)
            if isinstance(dist, IntDistribution):
                if dist.step:
                    v = np.round((v - dist.low) / dist.step) * dist.step + dist.low
                return int(min(max(v, dist.low), dist.high))
            else:
                if dist.step:
                    v = np.round((v - dist.low) / dist.step) * dist.step + dist.low
                    return float(min(max(v, dist.low), dist.high))
                return round(float(v), 10)
        elif isinstance(dist, CategoricalDistribution):
            return dist.choices[int(np.round(val))]
        return val
