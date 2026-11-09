---
author: Remyx AI
title: PASHA Pruner
description: Progressive ASHA — grows the maximum resource only while the top-two-rung ranking is unstable, stopping early once it settles (multi-fidelity early stopping).
tags: [pruner, asha, successive halving, multi-fidelity, early stopping]
optuna_versions: [5.0.0]
license: MIT License
---

## Abstract

PASHA (Progressive ASHA) is a multi-fidelity pruner that avoids spending compute on long runs while the ranking of configurations is still churning. It wraps asynchronous successive halving with a resource cap that starts at the second rung and grows by a single rung only while the ranking across the top two active rungs is unstable. Once the ranking settles, trials that reach the cap are pruned there and the remaining budget is saved.

`PASHAPruner` implements the method of Bohdal et al. (ICLR 2023): successive-halving promotion with a progressively grown maximum resource, a soft-ranking consistency check (configurations within `epsilon` of each other share a group, so noise that does not move a configuration out of its group counts as stability), and an automatic `epsilon` estimated from pairs of trials whose relative order flips back and forth across steps.

## Class or Function Names

- `PASHAPruner(min_resource="auto", reduction_factor=4, min_early_stopping_rate=0, bootstrap_count=0, ranking_mode="soft", epsilon="auto", max_rung=None)`

  A subclass of `optuna.pruners.SuccessiveHalvingPruner`. The first four arguments keep their successive-halving meaning; the PASHA-specific ones are:

  - `ranking_mode`: `"soft"` compares rung rankings up to the `epsilon` tolerance, `"hard"` compares exact orders (equivalent to `epsilon=0`).
  - `epsilon`: the soft-ranking tolerance, or `"auto"` to estimate it on the fly from pairs of trials whose order flips back and forth across steps (the 90th percentile of their metric distances).
  - `max_rung`: the highest rung index the cap may grow to; `None` means no fixed limit beyond the study's own step range.

  The read-only properties `current_max_rung` and `epsilon` expose the pruner's current state.

- `decide_resource_increase(top_rung, previous_rung, epsilon, maximize)`: whether the resource cap should grow by one rung, given the `(trial id, value)` pairs recorded at the top two active rungs.

- `evaluate_soft_ranking(sorted_top_rung, sorted_previous_rung, epsilon, maximize)`: the soft-ranking consistency check itself (both rungs sorted best-first).

- `estimate_epsilon(per_epoch_results, upper_step, lower_step, previous_epsilon=0.0)`: the automatic tolerance estimate from per-trial learning curves.

## Installation

```shell
pip install numpy
```

## Example

```python
import optuna
import optunahub

module = optunahub.load_module(package="pruners/pasha")
pruner = module.PASHAPruner(min_resource=1, reduction_factor=3, max_rung=4)
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=30)
```

See [example.py](https://github.com/optuna/optunahub-registry/blob/main/package/pruners/pasha/example.py) for a full example whose objective reports intermediate values across a rung ladder (steps 1, 3, 9, 27, 81).

## Others

### Notes

- `PASHAPruner` keeps the resource cap as in-process instance state, so a single pruner instance must be shared by all trials of a study. In distributed optimization each process would grow its own cap; reconstructing the cap from shared storage is not implemented.
- The rung bookkeeping is re-derived from public APIs (the `completed_rung_{k}` system attributes written at each promotion step, read via `study.get_trials`), but the pruner still performs the same storage write as the parent class, hence the `optuna_versions` pin to the tested release.
- The objective must report intermediate values (`trial.report`) on a step ladder compatible with `min_resource * reduction_factor ** k`, exactly as for `SuccessiveHalvingPruner`.

### Reference

- Bohdal, Balles, Wistuba, Ermis, Archambeau, Zappella. PASHA: Efficient HPO and NAS with Progressive Resource Allocation, ICLR 2023 ([arXiv:2207.06940](https://arxiv.org/abs/2207.06940)).
- This package is a from-scratch reimplementation (MIT, this repository) of that algorithm; the original scheduler ships in [syne-tune](https://github.com/awslabs/syne-tune) (Apache-2.0) as `PASHARungSystem`.

### Bibtex

```bibtex
@inproceedings{bohdal2023pasha,
  title     = {PASHA: Efficient HPO and NAS with Progressive Resource Allocation},
  author    = {Bohdal, Okan and Balles, Luke and Wistuba, Martin and Ermis, Beyza and Archambeau, Cedric and Zappella, Luca},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}
```
