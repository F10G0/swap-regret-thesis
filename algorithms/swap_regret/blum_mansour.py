from functools import partial

import numpy as np

from algorithms.external_regret import Hedge, Exp3
from algorithms.swap_regret.base import StationaryReduction


class FullBM(StationaryReduction):
    """Full-information Blum-Mansour reduction with known-horizon inner learners."""

    def __init__(self, n_actions: int, horizon: int, inner_algorithm_factory=Hedge, seed: int | None = None) -> None:
        super().__init__(n_actions, partial(inner_algorithm_factory, n_actions, horizon), horizon, seed)

    def _update_state(self, reward_vector: np.ndarray) -> None:
        for i, learner in enumerate(self.learners):
            weighted_reward = self.current_strategy[i] * reward_vector
            learner.update(weighted_reward)


class BanditBM(StationaryReduction):
    """Bandit-feedback Blum-Mansour reduction with known-horizon inner learners."""

    def __init__(self, n_actions: int, horizon: int, inner_algorithm_factory=Exp3, seed: int | None = None) -> None:
        super().__init__(n_actions, partial(inner_algorithm_factory, n_actions, horizon), horizon, seed)

    def _update_state(self, reward: float) -> None:
        transition_matrix = self._transition_matrix
        probability = self.current_strategy[self.current_action]

        for i, learner in enumerate(self.learners):
            # Importance-weighted reward: r_i = p_i r_k q_i,k / p_k.
            observed_reward = self.current_strategy[i] * reward * transition_matrix[i, self.current_action] / probability
            learner.current_action = self.current_action
            learner.update(observed_reward)
