import numpy as np

from algorithms.external_regret.base import ExponentialWeightsAlgorithm


class Hedge(ExponentialWeightsAlgorithm):
    """Hedge algorithm."""

    @property
    def learning_rate(self) -> float:
        """Return eta = sqrt(8 log(K) / T)."""
        return np.sqrt(8.0 * np.log(self.n_actions) / self._rate_horizon)

    def _update_state(self, reward_vector: np.ndarray) -> None:
        self.cumulative_score += reward_vector
