from abc import abstractmethod

import numpy as np

from algorithms.base import Algorithm
from config import NUMERICAL_TOLERANCE


class ExponentialWeightsAlgorithm(Algorithm):
    """Exponential weights with a learning rate owned by each concrete learner."""

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
