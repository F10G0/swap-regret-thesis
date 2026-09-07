import numpy as np

from algorithms.external_regret.base import ExponentialWeightsAlgorithm


class Hedge(ExponentialWeightsAlgorithm):
    """Hedge with a fixed horizon or an explicit local-time schedule."""

    def __init__(self, n_actions: int, horizon: int | None = None, seed: int | None = None) -> None:
        if horizon is not None and (not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon <= 0):
            raise ValueError("horizon must be a positive integer or None")
        self.horizon = horizon
        super().__init__(n_actions, seed=seed)

    @property
    def learning_rate(self) -> float:
        """Use fixed T when provided, otherwise the next local round t + 1."""
        rate_time = self.horizon if self.horizon is not None else self.t + 1
        return np.sqrt(8.0 * np.log(self.n_actions) / rate_time)

    def _update_state(self, reward_vector: np.ndarray) -> None:
        self.cumulative_score += reward_vector
