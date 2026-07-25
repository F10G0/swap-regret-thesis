import numpy as np

from algorithms.external_regret.base import ExponentialWeightsAlgorithm


class Exp3(ExponentialWeightsAlgorithm):
    """Reward-based Exp3 algorithm."""

    @property
    def learning_rate(self) -> float:
        """Return eta = sqrt(log(K) / (K T))."""
        return np.sqrt(np.log(self.n_actions) / (self.n_actions * self._rate_horizon))

    def _update_state(self, reward: float) -> None:
        probability = self.current_strategy[self.current_action]
        estimated_reward = reward / probability
        self.cumulative_score[self.current_action] += estimated_reward


class Exp3IX(ExponentialWeightsAlgorithm):
    """Loss-based Exp3 with implicit exploration."""

    @property
    def learning_rate(self) -> float:
        """Return eta = sqrt(log(K) / T)."""
        return np.sqrt(np.log(self.n_actions) / self._rate_horizon)

    @property
    def implicit_exploration(self) -> float:
        """Return gamma = eta / 2."""
        return self.learning_rate / 2.0

    def _update_state(self, reward: float) -> None:
        loss = 1.0 - reward
        probability = self.current_strategy[self.current_action]
        estimated_loss = loss / (probability + self.implicit_exploration)
        self.cumulative_score[self.current_action] -= estimated_loss
