import numpy as np

from algorithms.external_regret import Hedge
from algorithms.swap_regret.base import StationaryReduction


class BlumMansour(StationaryReduction):
    """
    Full-information Blum-Mansour reduction.
    """

    def __init__(self, n_actions: int, learning_rate: float, inner_algorithm_factory=Hedge, seed: int | None = None) -> None:
        super().__init__(n_actions, learning_rate, inner_algorithm_factory, seed)

    def _update_state(self, reward_vector: np.ndarray) -> None:
        reward_vector = self._validate_reward_vector(reward_vector)

        for i, learner in enumerate(self.learners):
            learner.update(self.current_strategy[i] * reward_vector)
