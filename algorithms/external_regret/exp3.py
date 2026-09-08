import numpy as np

from algorithms.external_regret.base import ExponentialWeightsAlgorithm


def _validate_horizon(horizon: int) -> int:
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    return horizon


class AuerExp3(ExponentialWeightsAlgorithm):
    """Auer et al. (2002) Exp3 with Blum-Mansour (2007) tuning."""

    def __init__(self, n_actions: int, horizon: int, seed: int | None = None) -> None:
        self.horizon = _validate_horizon(horizon)
        super().__init__(n_actions, seed=seed)
        self._explicit_exploration = min(1.0, np.sqrt(self.n_actions * np.log(self.n_actions) / self.horizon))
        self._learning_rate = self._explicit_exploration / self.n_actions

    @property
    def explicit_exploration(self) -> float:
        """Return gamma = min(1, sqrt(K log(K) / T))."""
        return self._explicit_exploration

    @property
    def learning_rate(self) -> float:
        """Return eta = gamma / K."""
        return self._learning_rate

    def _compute_strategy(self) -> np.ndarray:
        strategy = super()._compute_strategy()
        gamma = self.explicit_exploration
        return (1.0 - gamma) * strategy + gamma / self.n_actions

    def _update_state(self, reward: float) -> None:
        probability = self.current_strategy[self.current_action]
        self.cumulative_score[self.current_action] += reward / probability


class ImplicitExplorationAlgorithm(ExponentialWeightsAlgorithm):
    """Exponential weights with the implicit-exploration loss estimator."""

    @property
    def implicit_exploration(self) -> float:
        """Return gamma = eta / 2."""
        return self.learning_rate / 2.0

    def _update_state(self, reward: float) -> None:
        loss = 1.0 - reward
        probability = self.current_strategy[self.current_action]
        estimated_loss = loss / (probability + self.implicit_exploration)
        self.cumulative_score[self.current_action] -= estimated_loss


class Exp3IX(ImplicitExplorationAlgorithm):
    """Neu (2015) Exp3-IX."""

    def __init__(self, n_actions: int, horizon: int, seed: int | None = None) -> None:
        self.horizon = _validate_horizon(horizon)
        super().__init__(n_actions, seed=seed)
        self._learning_rate = np.sqrt(2.0 * np.log(self.n_actions) / (self.n_actions * self.horizon))

    @property
    def learning_rate(self) -> float:
        """Return eta = sqrt(2 log(K) / (K T))."""
        return self._learning_rate
