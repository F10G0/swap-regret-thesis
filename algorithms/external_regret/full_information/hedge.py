import numpy as np

from algorithms.external_regret.base import ExponentialWeightsAlgorithm


class Hedge(ExponentialWeightsAlgorithm):
    """
    Hedge algorithm.
    """

    def _reset_state(self) -> None:
        self.cumulative_reward = np.zeros(self.n_actions, dtype=float)

    def _update_state(self, reward_vector: np.ndarray) -> None:
        reward_vector = self._validate_reward_vector(reward_vector)
        self.cumulative_reward += reward_vector

    def _update_strategy(self) -> None:
        self.current_strategy = self._exponential_weights(self.cumulative_reward)
