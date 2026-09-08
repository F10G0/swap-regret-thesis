import numpy as np

from algorithms.internal_regret.base import RegretMatchingBase
from algorithms.stationary import stationary_distribution


class RegretMatching(RegretMatchingBase):
    """Inertia-based Hart-Mas-Colell regret matching from equation (2.2)."""

    def _compute_strategy(self) -> np.ndarray:
        action = self.current_action
        strategy = np.maximum(self.cumulative_regret[action], 0.0) / (self.normalization * self.t)
        strategy[action] = 0.0
        strategy[action] = 1.0 - np.sum(strategy)
        return strategy


class StationaryRegretMatching(RegretMatchingBase):
    """Hart-Mas-Colell stationary-distribution procedure from equation (3.1)."""

    def _compute_strategy(self) -> np.ndarray:
        return stationary_distribution(self._regret_transition_matrix)
