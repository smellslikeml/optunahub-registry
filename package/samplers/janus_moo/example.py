"""Multi-objective JANUS example.

``JanusMooCmixSampler`` subclasses Optuna's NSGA-II. NSGA-II keeps full control
of Pareto selection and survival; JANUS only occasionally replaces a child with
inverse-curvature Gaussian samples estimated from the recent trace. This adds
targeted, metric-guided exploration without changing the selection operator.

Exposing a per-trial residual/component vector via
``trial.set_user_attr("component_losses", [...])`` sharpens the local metric;
otherwise a diagonal-quadratic scalar fallback is used.
"""

from __future__ import annotations

import numpy as np
import optuna
import optunahub


optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial) -> tuple[float, float]:
    # ZDT1-style 2-objective problem in 5 dimensions.
    d = 5
    x = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(d)])
    f1 = float(x[0])
    g = 1.0 + 9.0 * float(np.mean(x[1:]))
    f2 = g * (1.0 - np.sqrt(f1 / g))

    # Optional residual signal that sharpens JANUS's local curvature metric.
    trial.set_user_attr("component_losses", [f1, float(f2)])

    return f1, float(f2)


if __name__ == "__main__":
    JanusMooCmixSampler = optunahub.load_local_module(
        package="samplers/janus_moo",
        registry_root="./package",
    ).JanusMooCmixSampler

    sampler = JanusMooCmixSampler(
        population_size=50,
        seed=0,
        cmix_prob=0.05,  # fraction of children replaced by JANUS infill
        cmix_start_trial=100,  # warm-up before JANUS engages
    )
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    study.optimize(objective, n_trials=400)

    print(f"Number of Pareto-optimal trials: {len(study.best_trials)}")
    front = sorted((t.values for t in study.best_trials), key=lambda v: v[0])
    print("Pareto front (first 5):", [(round(a, 3), round(b, 3)) for a, b in front[:5]])
