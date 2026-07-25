import numpy as np

from algorithms.internal_regret.base import RegretMatchingBase
from algorithms.stationary import stationary_distribution


class RegretMatching(RegretMatchingBase):
    """Inertia-based Hart-Mas-Colell regret matching from equation (2.2)."""

    def _compute_strategy(self) -> np.ndarray:
        return self._regret_transition_matrix[self.current_action]


class StationaryRegretMatching(RegretMatchingBase):
    """Hart-Mas-Colell stationary-distribution procedure from equation (3.1)."""

    def _compute_strategy(self) -> np.ndarray:
        return stationary_distribution(self._regret_transition_matrix)
