from __future__ import annotations

import numpy as np
import optuna
import optunahub


JanusMooCmixSampler = optunahub.load_local_module(
    package="samplers/janus_moo", registry_root="package/"
).JanusMooCmixSampler


def objective(trial: optuna.Trial) -> tuple[float, float]:
    x = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(4)])
    f1 = float(x[0])
    g = 1.0 + 9.0 * float(np.mean(x[1:]))
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return f1, float(f2)


def test_sampler_runs_multi_objective() -> None:
    n_trials = 60
    sampler = JanusMooCmixSampler(
        population_size=10,
        seed=0,
        cmix_prob=0.2,
        cmix_start_trial=20,
    )
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    assert len(study.trials) == n_trials
    assert len(study.best_trials) >= 1
    for trial in study.best_trials:
        assert trial.values is not None
        assert len(trial.values) == 2


def test_sampler_uses_component_losses() -> None:
    n_trials = 60

    def objective_with_components(trial: optuna.Trial) -> tuple[float, float]:
        f1, f2 = objective(trial)
        trial.set_user_attr("component_losses", [f1, f2])
        return f1, f2

    sampler = JanusMooCmixSampler(
        population_size=10,
        seed=1,
        cmix_prob=0.2,
        cmix_start_trial=20,
    )
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    study.optimize(objective_with_components, n_trials=n_trials)

    assert len(study.trials) == n_trials
    assert len(study.best_trials) >= 1
