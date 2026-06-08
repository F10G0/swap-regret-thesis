import numpy as np

from algorithms.external_regret.base import ExponentialWeightsAlgorithm


class Exp3(ExponentialWeightsAlgorithm):
    """
    Exp3 algorithm.
    """

    def _reset_state(self) -> None:
        self.estimated_cumulative_reward = np.zeros(self.n_actions, dtype=float)

    def _update_state(self, action: int, reward: float) -> None:
        self._validate_action(action)
        self._validate_reward(reward)

        probability = self.current_strategy[action]
        estimated_reward = reward / probability
        self.estimated_cumulative_reward[action] += estimated_reward

    def _update_strategy(self) -> None:
        self.current_strategy = self._exponential_weights(self.estimated_cumulative_reward)
