from numbers import Real

import numpy as np

from algorithms.external_regret.base import ExponentialWeightsAlgorithm


class OptimisticHedge(ExponentialWeightsAlgorithm):
    """Reward-form optimistic Hedge from Chen and Peng (2020)."""

    def __init__(self, n_actions: int, horizon: int, learning_rate: float | None = None, seed: int | None = None) -> None:
        if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if learning_rate is not None and (not isinstance(learning_rate, Real) or isinstance(learning_rate, bool) or not np.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("learning_rate must be finite and strictly positive")
        self.horizon = horizon
        super().__init__(n_actions, seed=seed)
        self._learning_rate = (np.log(self.n_actions) / self.horizon) ** (1.0 / 6.0) if learning_rate is None else learning_rate

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    def _reset_state(self) -> None:
        super()._reset_state()
        self.previous_reward = np.zeros(self.n_actions, dtype=float)

    def _update_state(self, reward_vector: np.ndarray) -> None:
        self.cumulative_score += 2.0 * reward_vector - self.previous_reward
        self.previous_reward = np.copy(reward_vector)
