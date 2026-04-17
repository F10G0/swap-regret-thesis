import numpy as np

from algorithms import BanditAlgorithm
from environments import BanditEnvironment
from .metrics import compute_instant_regret, cumulative_sum


def run_single_experiment(
    env: BanditEnvironment,
    algo: BanditAlgorithm,
    horizon: int,
) -> dict:
    """
    Run a single bandit experiment for a fixed horizon.

    Args:
        env: bandit environment
        algo: bandit algorithm
        horizon: number of rounds

    Returns:
        dict containing:
            - actions
            - rewards
            - instant_regret
            - cumulative_regret
            - cumulative_reward
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    env.reset()
    algo.reset()

    actions = np.zeros(horizon, dtype=int)
    rewards = np.zeros(horizon, dtype=float)
    instant_regrets = np.zeros(horizon, dtype=float)

    for t in range(horizon):
        action = algo.select_action()
        reward = env.pull(action)

        algo.update(action, reward)

        actions[t] = action
        rewards[t] = reward
        instant_regrets[t] = compute_instant_regret(env, action)

    cumulative_regret = cumulative_sum(instant_regrets)
    cumulative_reward = cumulative_sum(rewards)

    return {
        "actions": actions,
        "rewards": rewards,
        "instant_regret": instant_regrets,
        "cumulative_regret": cumulative_regret,
        "cumulative_reward": cumulative_reward,
        "final_regret": float(cumulative_regret[-1]),
        "final_reward": float(cumulative_reward[-1]),
    }


def run_multiple_experiments(
    env_factory,
    algo_factory,
    horizon: int,
    n_runs: int,
    base_seed: int = 0,
) -> list[dict]:
    """
    Run multiple independent experiments.

    Args:
        env_factory: callable returning a fresh environment
        algo_factory: callable returning a fresh algorithm
        horizon: number of rounds per run
        n_runs: number of independent runs
        base_seed: base random seed; run i uses seed = base_seed + i

    Returns:
        list of result dicts returned by run_single_experiment
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")

    results = []

    for i in range(n_runs):
        np.random.seed(base_seed + i)

        env = env_factory()
        algo = algo_factory()

        result = run_single_experiment(env, algo, horizon)
        results.append(result)

    return results
