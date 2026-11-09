from __future__ import annotations

import numpy as np
import optuna
import optunahub
import pytest


pytest.importorskip("torch")
pytest.importorskip("evotorch")


JanusSampler = optunahub.load_local_module(
    package="samplers/janus", registry_root="package/"
).JanusSampler


def objective(trial: optuna.Trial) -> float:
    x = np.array([trial.suggest_float(f"x{i}", -5.0, 5.0) for i in range(4)])
    residuals = (x - 0.5) ** 2
    trial.set_user_attr("component_losses", residuals.tolist())
    return float(residuals.sum())


def test_sampler_runs_single_objective() -> None:
    n_trials = 60
    sampler = JanusSampler(seed=0, n_startup_trials=8)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    assert len(study.trials) == n_trials
    assert study.best_value is not None
    assert set(study.best_params.keys()) == {f"x{i}" for i in range(4)}


def test_sampler_falls_back_without_component_losses() -> None:
    n_trials = 60

    def objective_scalar(trial: optuna.Trial) -> float:
        x = np.array([trial.suggest_float(f"x{i}", -5.0, 5.0) for i in range(4)])
        return float(((x - 0.5) ** 2).sum())

    sampler = JanusSampler(seed=1, n_startup_trials=8)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective_scalar, n_trials=n_trials)

    assert len(study.trials) == n_trials
    assert study.best_value is not None
