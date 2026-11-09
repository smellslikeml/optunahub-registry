---
author: Hongyuan Yu, Pufan Xu, Jiaojiao Yi, Yiding Tian, Mingrui Sun, Jiayuan Lu, Changyuan Wen
title: JANUS Sampler
description: A CMA-ES sampler augmented with JANUS Jacobian-aligned Gauss-Newton exploitation and inverse-curvature exploration for single-objective black-box optimization.
tags: [sampler, CMA-ES, Gauss-Newton, evolutionary algorithm, evotorch]
optuna_versions: [4.5.0]
license: MIT License
---

## Abstract

`JanusSampler` is a single-objective, plug-and-play sampler that augments a CMA-ES host (evotorch backend) with JANUS-style, geometry-guided candidate generation (arXiv:2608.22862). It is the single-objective counterpart of `JanusMooCmixSampler`.

Population optimizers such as CMA-ES drive search through rank-based selection: the signal says one candidate is better than another, but not the local *direction* responsible for the improvement. JANUS recovers this missing geometric signal from a single estimated Jacobian `J` and uses it for **both** halves of the search, without replacing the host optimizer:

- **Exploit** — a bounded Gauss-Newton / Levenberg–Marquardt step `Δ = −(JᵀJ + λI)⁻¹ Jᵀ s` is assembled from the recent trace and injected as one elite child (whitened step norm clipped to `κ√d`).
- **Explore (CMIX)** — the *same* curvature metric produces a tail of inverse-curvature candidates `x ~ N(mean, σ² · trace_normalize((JᵀJ + λ_mix I)⁻¹))`, adding targeted low-curvature exploration while CMA-ES keeps its native covariance adaptation.

The Jacobian is estimated directly from a per-trial residual/component vector when available (recommended, exposed via `trial.set_user_attr("component_losses", [...])`); otherwise JANUS falls back to a local linear/quadratic fit on the recent window of scalar objective values.

## APIs

- `JanusSampler(x0=None, sigma0=None, seed=None, cma_opts=None, n_startup_trials=1, independent_sampler=None, warn_independent_sampling=True, n_jobs=1, fuse_every=4, fuse_start="auto", kappa=1.0, gn_window="auto", gn_lambda=1.0, gn_ridge=1.0, cmix_w="auto", cmix_lambda=1.0, cmix_w_base=0.5, cmix_w_min=0.01, cmix_w_max=0.20, pseudo_mode="linear", gn_pseudo_mode=None, cmix_pseudo_mode=None, pseudo_target="raw", pca_k="auto", cmix_mode="portfolio", cmix_candidate_frac=0.4, cmix_candidate_center="mean", cmix_candidate_cov="metric", verbose=False)`
  - `x0`: Initial mean per parameter. If `None`, the search-space midpoint is used.
  - `sigma0`: Initial step size. Defaults to `0.2` when `None`.
  - `seed`: Seed for the random number generators. A random seed is drawn if `None`.
  - `cma_opts`: Extra CMA-ES options; `cma_opts["popsize"]` overrides the default population size.
  - `n_startup_trials`: Number of completed trials required before JANUS activates; earlier trials are drawn from `independent_sampler`.
  - `independent_sampler`: Sampler used for non-relative parameters and startup. Defaults to `RandomSampler`.
  - `warn_independent_sampling`: Whether to warn when parameters are sampled independently.
  - `n_jobs`: Number of parallel CMA-ES states keyed by `trial.number % n_jobs`.
  - `fuse_every`: Assemble a fresh Gauss-Newton elite/metric every this many generations.
  - `fuse_start`: Warm-up before the first fuse. `"auto"` derives it from population size and window.
  - `kappa`: Trust-region cap on the whitened elite step (`‖z‖ ≤ κ√d`).
  - `gn_window`: Sliding window of recent trials used to estimate `J`. `"auto"` adapts to the residual dimension.
  - `gn_lambda`: Levenberg–Marquardt damping for the Gauss-Newton step.
  - `gn_ridge`: Ridge regularization for the Jacobian regression.
  - `cmix_w`: Weight of the inverse-curvature blend into the covariance when `cmix_mode="reshape"`. `"auto"` scales with how overdetermined the regression is.
  - `cmix_lambda`: Damping added to the metric before inversion for CMIX.
  - `cmix_w_base`, `cmix_w_min`, `cmix_w_max`: Bounds and base used to resolve `cmix_w="auto"`.
  - `pseudo_mode`: Local surrogate used when no residual vector is available (`"linear"`, `"quadratic"`, `"diag_quadratic"`, `"pca_quadratic"`, `"diag_pca_quadratic"`).
  - `gn_pseudo_mode`, `cmix_pseudo_mode`: Per-half overrides of `pseudo_mode`.
  - `pseudo_target`: Target transform for the surrogate fit (`"raw"`, `"rank"`, `"log"`).
  - `pca_k`: Number of PCA components for PCA-quadratic surrogates. `"auto"` derives it from the sample count and dimension.
  - `cmix_mode`: How CMIX is applied (`"portfolio"`, `"candidates"`, `"reshape"`).
  - `cmix_candidate_frac`: Fraction of the population replaced by CMIX candidates.
  - `cmix_candidate_center`: Center of the CMIX distribution (`"mean"`, `"best"`, `"elite"`, `"archive"`).
  - `cmix_candidate_cov`: Covariance source for CMIX candidates.
  - `verbose`: If `True`, prints periodic diagnostics.

JANUS only samples non-conditional numerical (`FloatDistribution` / `IntDistribution`) parameters through the relative CMA-ES host; categorical and conditional parameters fall back to the independent sampler.

## Installation

This sampler requires the optional backend `torch` and `evotorch`:

```shell
pip install torch evotorch
```

## Example

```python
import numpy as np
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = np.array([trial.suggest_float(f"x{i}", -5.0, 5.0) for i in range(10)])
    residuals = (x - 0.5) ** 2
    # Optional: expose per-coordinate residuals so JANUS can estimate J directly.
    trial.set_user_attr("component_losses", residuals.tolist())
    return float(residuals.sum())


JanusSampler = optunahub.load_module(package="samplers/janus").JanusSampler

sampler = JanusSampler(seed=0, n_startup_trials=16)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=300)
print(f"Best value: {study.best_value:.6e}")
```

See `example.py` for a runnable version.

## Others

This sampler is the single-objective component of JANUS (Jacobian-Aligned Newton-Unified Search). See the `janus_moo` package for the multi-objective (NSGA-II) counterpart.

### Reference

Hongyuan Yu, Pufan Xu, Jiaojiao Yi, Yiding Tian, Mingrui Sun, Jiayuan Lu, and Changyuan Wen. 2026. JANUS: Online Jacobian-Aligned Infill for Black-Box Optimization. arXiv preprint arXiv:2608.22862.

### Bibtex

```
@article{yu2026janus,
    title   = {JANUS: Online Jacobian-Aligned Infill for Black-Box Optimization},
    author  = {Yu, Hongyuan and Xu, Pufan and Yi, Jiaojiao and Tian, Yiding and
               Sun, Mingrui and Lu, Jiayuan and Wen, Changyuan},
    journal = {arXiv preprint arXiv:2608.22862},
    year    = {2026}
}
```
