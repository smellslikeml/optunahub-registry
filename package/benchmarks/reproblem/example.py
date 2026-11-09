from __future__ import annotations

import optuna
import optunahub


reproblem = optunahub.load_module("benchmarks/reproblem")
problem = reproblem.ConstrainedProblem("CRE21")
study = optuna.create_study(directions=problem.directions)
study.optimize(problem, n_trials=10)

if len(problem.directions) == 1:
    print(study.best_trial)
else:
    print(study.best_trials)
