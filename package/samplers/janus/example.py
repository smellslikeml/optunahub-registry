"""Single-objective JANUS example.

``JanusSampler`` is a drop-in relative sampler backed by a CMA-ES host
(evotorch backend). Every few generations it injects a Gauss-Newton exploitation
candidate plus inverse-curvature (CMIX) exploration candidates estimated from the
recent evaluation trace.

Optional but recommended: expose a per-trial residual/component vector via
``trial.set_user_attr("component_losses", [...])``. When present, JANUS estimates
the local Jacobian directly from these residuals; otherwise it falls back to a
local linear/quadratic fit on the recent window of scalar objective values.

This example requires the optional backend (see ``requirements.txt``):
``torch`` and ``evotorch``.
"""

from __future__ import annotations

import numpy as np
import optuna
import optunahub


optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial) -> float:
    # A 10-D shifted sphere with an explicit residual decomposition.
    d = 10
    shift = 0.5
    x = np.array([trial.suggest_float(f"x{i}", -5.0, 5.0) for i in range(d)])
    residuals = (x - shift) ** 2  # per-coordinate residuals r_i(x)

    # Exposing residuals lets JANUS estimate the local Jacobian directly.
    trial.set_user_attr("component_losses", residuals.tolist())

    return float(residuals.sum())


if __name__ == "__main__":
    JanusSampler = optunahub.load_local_module(
        package="samplers/janus",
        registry_root="./package",
    ).JanusSampler

    sampler = JanusSampler(seed=0, n_startup_trials=16)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=300)

    print(f"Best value: {study.best_value:.6e}")
    print(
        "Best params (first 3):",
        {k: round(v, 4) for k, v in list(study.best_params.items())[:3]},
    )
