---
author: Hongyuan Yu, Pufan Xu, Jiaojiao Yi, Yiding Tian, Mingrui Sun, Jiayuan Lu, Changyuan Wen
title: JANUS Multi-objective Cmix Sampler
description: NSGA-II augmented with JANUS Jacobian-aligned inverse-curvature infill for multi-objective black-box optimization.
tags: [sampler, multi-objective, NSGA-II, Gauss-Newton, evolutionary algorithm]
optuna_versions: [4.5.0]
license: MIT License
---

## Abstract

`JanusMooCmixSampler` is a plug-and-play, training-free multi-objective sampler that augments Optuna's NSGA-II host with JANUS-style, geometry-guided candidate generation (arXiv:2608.22862).

Population optimizers such as NSGA-II drive search mainly through rank-based selection signals: a signal says one candidate dominates another, but not the local *direction* responsible for the improvement. JANUS recovers this missing local geometric signal **without replacing the host optimizer**. It estimates a local metric from the recent evaluation trace and occasionally replaces a child with inverse-curvature Gaussian samples drawn around the current Pareto front. NSGA-II keeps full control of Pareto selection, survival, and crowding — JANUS only contributes a few extra, metric-guided candidates.

The metric is estimated from either a per-trial residual/component vector (recommended, exposed via `trial.set_user_attr("component_losses", [...])`) or a diagonal-quadratic scalar fallback fit on recent objective values. Defaults are intentionally conservative: short warm-up, low replacement probability, and a small normalized step size.

## Class or Function Names

- `JanusMooCmixSampler(*, population_size=100, seed=None, cmix_prob=0.02, cmix_window="auto", cmix_ridge=1.0, cmix_lambda=1.0, cmix_sigma=0.002, cmix_start_generation=3, cmix_start_trial=100, component_attr="component_losses", scalar_fallback="diag_quadratic", verbose=False, **kwargs)`
  - `population_size`: Population size passed through to the NSGA-II host.
  - `seed`: Seed for the random number generator.
  - `cmix_prob`: Probability that a child is replaced by a JANUS infill candidate.
  - `cmix_window`: Sliding window of recent completed trials used to estimate the local metric. `"auto"` uses `3 * population_size`.
  - `cmix_ridge`: Ridge regularization for the metric regression.
  - `cmix_lambda`: Levenberg–Marquardt damping added to the metric before inversion.
  - `cmix_sigma`: Normalized step size (standard deviation) of the inverse-curvature Gaussian, in the unit hypercube.
  - `cmix_start_trial`: Number of warm-up trials before JANUS infill engages. If `None`, it is derived from `cmix_start_generation` and `population_size`.
  - `cmix_start_generation`: Warm-up measured in generations, used only when `cmix_start_trial=None`.
  - `component_attr`: Trial user-attribute key holding the per-trial residual/component vector.
  - `scalar_fallback`: Metric fallback when no component vector is available. Currently `"diag_quadratic"`.
  - `verbose`: If `True`, prints periodic infill statistics.
  - `**kwargs`: Additional keyword arguments forwarded to `optuna.samplers.NSGAIISampler`.

Note that JANUS infill only samples non-conditional numerical (`FloatDistribution` / `IntDistribution`) parameters; other parameter types and the selection operator are handled entirely by the NSGA-II host.

## Example

```python
import numpy as np
import optuna
import optunahub


def objective(trial: optuna.Trial) -> tuple[float, float]:
    x = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(5)])
    f1 = float(x[0])
    g = 1.0 + 9.0 * float(np.mean(x[1:]))
    f2 = g * (1.0 - np.sqrt(f1 / g))
    # Optional: expose a residual/component vector so JANUS can sharpen its metric.
    trial.set_user_attr("component_losses", [f1, float(f2)])
    return f1, float(f2)


JanusMooCmixSampler = optunahub.load_module(
    package="samplers/janus_moo"
).JanusMooCmixSampler

sampler = JanusMooCmixSampler(population_size=50, seed=0, cmix_prob=0.05)
study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
study.optimize(objective, n_trials=400)
print(f"Number of Pareto-optimal trials: {len(study.best_trials)}")
```

See `example.py` for a runnable version.

## Others

This sampler is the multi-objective component of JANUS (Jacobian-Aligned Newton-Unified Search).

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
