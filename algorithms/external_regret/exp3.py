import numpy as np

from algorithms.external_regret.base import ExponentialWeightsAlgorithm


class Exp3IX(ExponentialWeightsAlgorithm):
    """Loss-based Exp3 with implicit exploration."""

    use_implicit_exploration = True

    @property
    def learning_rate(self) -> float:
        """Return eta = sqrt(c log(K) / (K H)) for the active rate horizon H."""
        coefficient = 2.0 if self.use_implicit_exploration and self.horizon else 1.0
        return np.sqrt(coefficient * np.log(self.n_actions) / (self.n_actions * self._rate_horizon))

    @property
    def implicit_exploration(self) -> float:
        """Return gamma = eta / 2 for Exp3-IX and zero for Exp3."""
        return self.learning_rate / 2.0 if self.use_implicit_exploration else 0.0

    def _update_state(self, reward: float) -> None:
        loss = 1.0 - reward
        probability = self.current_strategy[self.current_action]
        estimated_loss = loss / (probability + self.implicit_exploration)
        self.cumulative_score[self.current_action] -= estimated_loss


class Exp3(Exp3IX):
    """Loss-based Exp3 without implicit exploration."""

    use_implicit_exploration = False
