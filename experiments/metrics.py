import numpy as np

from environments import BanditEnvironment


def compute_instant_regret(env: BanditEnvironment, action: int) -> float:
    """
    Compute the expected instantaneous regret of choosing the given action.

    Regret_t = mu* - mu(action)

    Args:
        env: bandit environment
        action: chosen arm

    Returns:
        float
    """
    return env.optimal_mean() - env.arm_mean(action)


def cumulative_sum(values: np.ndarray) -> np.ndarray:
    """
    Return cumulative sum of a 1D array.
    """
    return np.cumsum(values)


def average_cumulative_regret(results: list[dict]) -> np.ndarray:
    """
    Compute the average cumulative regret over multiple runs.

    Args:
        results: list of result dicts returned by run_single_experiment

    Returns:
        np.ndarray
    """
    if len(results) == 0:
        raise ValueError("results must not be empty")

    curves = [res["cumulative_regret"] for res in results]
    return np.mean(np.stack(curves, axis=0), axis=0)


def average_final_regret(results: list[dict]) -> float:
    """
    Compute the average final regret over multiple runs.
    """
    if len(results) == 0:
        raise ValueError("results must not be empty")

    finals = [res["final_regret"] for res in results]
    return float(np.mean(finals))
