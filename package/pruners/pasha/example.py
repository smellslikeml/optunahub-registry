import numpy as np
import optuna
import optunahub


def objective(trial: optuna.trial.Trial) -> float:
    # Later steps are more faithful measurements of the same configuration;
    # early steps are noisy enough to scramble the ranking, so PASHA grows the
    # resource cap for a while and stops once the ranking settles.
    x = trial.suggest_float("x", -1.0, 1.0)
    rng = np.random.RandomState(trial.number)
    value = float("nan")
    for step in (1, 3, 9, 27, 81):
        noise_scale = {1: 1.0, 3: 0.4, 9: 0.1, 27: 0.03, 81: 0.01}[step]
        value = float((x - 0.1) ** 2 + abs(rng.normal(scale=noise_scale)))
        trial.report(value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return value


module = optunahub.load_module(package="pruners/pasha")
pruner = module.PASHAPruner(
    min_resource=1,
    reduction_factor=3,
    max_rung=4,
    ranking_mode="soft",
    epsilon="auto",
)

study = optuna.create_study(
    direction="minimize",
    pruner=pruner,
    sampler=optuna.samplers.RandomSampler(seed=0),
)
study.optimize(objective, n_trials=30)

states = [t.state for t in study.trials]
n_pruned = states.count(optuna.trial.TrialState.PRUNED)
print(f"Pruned {n_pruned} of {len(states)} trials")
print(f"The resource cap settled at rung {pruner.current_max_rung}")
