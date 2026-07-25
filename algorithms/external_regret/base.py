from abc import abstractmethod

import numpy as np

from algorithms.base import Algorithm
from config import NUMERICAL_TOLERANCE


class ExponentialWeightsAlgorithm(Algorithm):
    """Base class for fixed-horizon and anytime exponential-weights algorithms."""

    @property
    def _rate_horizon(self) -> int:
        """Return the horizon used by the learning-rate schedule."""
        return max(self.horizon, self.t + 1)

    @property
    @abstractmethod
    def learning_rate(self) -> float:
        """Return the algorithm-specific learning rate."""
        pass

    def _reset_state(self) -> None:
        self.cumulative_score = np.zeros(self.n_actions, dtype=float)

    def _compute_strategy(self) -> np.ndarray:
        logits = self.learning_rate * self.cumulative_score
        logits -= np.max(logits)
        weights = np.exp(logits)
        strategy = np.maximum(weights / np.sum(weights), NUMERICAL_TOLERANCE)
        return strategy / np.sum(strategy)
