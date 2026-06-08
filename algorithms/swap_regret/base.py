from collections.abc import Callable

import numpy as np

from algorithms.base import Algorithm
from algorithms.swap_regret.stationary import stationary_distribution


class StationaryReduction(Algorithm):
    """
    Base class for stationary-distribution-based swap-regret reductions.
    """

    def __init__(
        self, n_actions: int, learning_rate: float,
        inner_algorithm_factory: Callable[[int, float, int | None], Algorithm],
        seed: int | None = None,
    ) -> None:
        super().__init__(n_actions, seed)

        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        self.learning_rate = learning_rate
        self.inner_algorithm_factory = inner_algorithm_factory
        self.reset()

    def _reset_state(self) -> None:
        self.learners = [
            self.inner_algorithm_factory(self.n_actions, self.learning_rate, int(self.rng.integers(0, 2**32 - 1))) for _ in range(self.n_actions)
        ]
    
    def _update_strategy(self) -> None:
        """
        Compute the stationary distribution of the current transition matrix.
        """
        transition_matrix = self._transition_matrix()
        self.current_strategy = stationary_distribution(transition_matrix)
    
    def _transition_matrix(self) -> np.ndarray:
        """
        Return the transition matrix induced by the inner learners.
        """
        return np.vstack([learner.strategy() for learner in self.learners])
