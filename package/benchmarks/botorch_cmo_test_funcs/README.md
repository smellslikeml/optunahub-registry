---
author: Shuhei Watanabe
title: BoTorch Constrained Multi-Objective Test Functions
description: A collection of 6 constrained multi-objective test problems (BNH, CONSTR, ConstrainedBraninCurrin, OSY, SRN, MW7) ported from BoTorch's test_functions.multi_objective module to OptunaHub's BaseProblem interface, with no dependency other than numpy.
tags: [benchmark, continuous optimization, constrained optimization, multi-objective, BoTorch]
optuna_versions: [5.0.0]
license: MIT License
---

## Abstract

This package ports 6 constrained multi-objective test problems from [BoTorch's `test_functions.multi_objective` module](https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py) to OptunaHub's `BaseProblem` interface: `BNH`, `CONSTR`, `ConstrainedBraninCurrin`, `OSY`, `SRN`, and `MW7`.
Each problem is reimplemented with plain Python and `numpy` only, so it can be evaluated without installing `torch` or `botorch`.

## Conventions

- Parameters are named `x0`, `x1`, ... following each problem's input vector order.
- `evaluate_constraints` keys are named `c0`, `c1`, ... following each problem's constraint order.
- BoTorch's constraint convention is that a slack `c_i(x) >= 0` is feasible; OptunaHub's `BaseProblem` convention is that a value `<= 0` is feasible. So every constraint returned here is the *negation* of the corresponding BoTorch slack.
- `OSY` uses BoTorch's raw `_evaluate_true` objectives verbatim (`f0` already has its minus sign baked into the formula) with `directions=[MINIMIZE, MINIMIZE]`, since BoTorch's own `negate=True` recommendation is only about fitting its maximize-oriented `ref_point`/hypervolume convention, not the sign of `f0` itself.

## APIs

All 6 classes below share the same interface, inherited from `optunahub.benchmarks.BaseProblem`:

- `search_space`: Return the search space.

  - Returns: `dict[str, optuna.distributions.BaseDistribution]`

- `directions`: Return the optimization directions. Always `[MINIMIZE, MINIMIZE]` for every problem in this package.

  - Returns: `list[optuna.study.StudyDirection]`

- `__call__(trial: optuna.Trial)`: Evaluate the objectives and constraints and return the objective values.

  - Args:
    - `trial`: Optuna trial object.
  - Returns: `list[float]`

- `evaluate(params: dict[str, float])`: Evaluate the objective functions.

  - Args:
    - `params`: Decision variable like `{"x0": x0_value, "x1": x1_value, ..., "xn": xn_value}`.
  - Returns: `list[float]` of length 2.

- `evaluate_constraints(params: dict[str, float])`: Evaluate the constraint functions.

  - Args:
    - `params`: Decision variable, same format as `evaluate`.
  - Returns: `dict[str, float]` keyed by `c0`, `c1`, .... A trial is feasible when every value is zero or less.

- `BNH()`

  - Dimension: 2. Constraints: 2. See (Garrido-Merchán & Hernández-Lobato, 2019).

- `CONSTR()`

  - Dimension: 2. Constraints: 2. See (Garrido-Merchán & Hernández-Lobato, 2019).

- `ConstrainedBraninCurrin()`

  - Dimension: 2. Constraints: 1. The Branin-Currin function with the disk constraint from (Gelbart et al., 2014).

- `OSY()`

  - Dimension: 6. Constraints: 6. See (Osyczka & Kundu, 1995).

- `SRN()`

  - Dimension: 2. Constraints: 2. See (Garrido-Merchán & Hernández-Lobato, 2019).

- `MW7(dim: int)`

  - `dim`: Number of decision variables. Must be at least 2.
  - Dimension: `dim`. Constraints: 2. Disconnected Pareto front. See (Ma & Wang, 2019).

## Example

```python
import optuna
import optunahub


cmo = optunahub.load_module("benchmarks/botorch_cmo_test_funcs")
problem = cmo.BNH()

study = optuna.create_study(
    sampler=optuna.samplers.NSGAIISampler(seed=42),
    directions=problem.directions,
)
study.optimize(problem, n_trials=100)
optuna.visualization.plot_pareto_front(study).show()
```

Tests can be performed by:

```bash
pytest package/benchmarks/botorch_cmo_test_funcs/tests
```

## Reference

Garrido-Merchán, E. C., & Hernández-Lobato, D. (2019). [Predictive Entropy Search for Multi-objective Bayesian Optimization with Constraints](https://doi.org/10.1016/j.neucom.2019.06.025). Neurocomputing, 361, 50-68.

Gelbart, M. A., Snoek, J., & Adams, R. P. (2014). [Bayesian Optimization with Unknown Constraints](https://arxiv.org/abs/1403.5607). UAI.

Osyczka, A., & Kundu, S. (1995). A New Method to Solve Generalized Multicriteria Optimization Problems Using the Simple Genetic Algorithm. Structural Optimization, 10, 94-99.

Ma, Z., & Wang, Y. (2019). Evolutionary Constrained Multiobjective Optimization: Test Suite Construction and Performance Comparisons. IEEE Transactions on Evolutionary Computation, 23(6), 972-986.
