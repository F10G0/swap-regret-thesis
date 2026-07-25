from collections.abc import Callable

import numpy as np

from algorithms.base import Algorithm
from algorithms.stationary import stationary_distribution


class StationaryReduction(Algorithm):
    """Base class for stationary-distribution-based swap-regret reductions."""

    def __init__(self, n_actions: int, inner_algorithm_factory: Callable[[int | None], Algorithm], horizon: int = 0, seed: int | None = None) -> None:
        self.inner_algorithm_factory = inner_algorithm_factory
        super().__init__(n_actions, horizon, seed)

    @property
    def _transition_matrix(self) -> np.ndarray:
        """Return the current inner-strategy transition matrix."""
        return np.vstack([learner.strategy() for learner in self.learners])

    def _reset_state(self) -> None:
        self.learners = [self.inner_algorithm_factory(int(self.rng.integers(0, 2**32 - 1))) for _ in range(self.n_actions)]

    def _compute_strategy(self) -> np.ndarray:
        return stationary_distribution(self._transition_matrix)
