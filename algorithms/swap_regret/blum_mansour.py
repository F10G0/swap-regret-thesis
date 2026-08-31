from functools import partial

import numpy as np

from algorithms.external_regret import Exp3, Exp3IX, Hedge
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
    """Bandit Blum-Mansour reduction using weighted observed losses."""

    def __init__(self, n_actions: int, horizon: int, inner_algorithm_factory=Exp3, seed: int | None = None) -> None:
        super().__init__(n_actions, partial(inner_algorithm_factory, n_actions, horizon), horizon, seed)

    def _update_state(self, reward: float) -> None:
        transition_matrix = self._transition_matrix
        probability = self.current_strategy[self.current_action]

        for i, learner in enumerate(self.learners):
            # Pass a complementary reward so inner Exp3 importance-weights observed_loss.
            observed_loss = self.current_strategy[i] * (1.0 - reward) * transition_matrix[i, self.current_action] / probability
            learner.current_action = self.current_action
            learner.update(1.0 - observed_loss)


class LCEIX(BanditBM):
    """Bandit Blum-Mansour reduction with anytime Exp3-IX inner learners."""

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        super().__init__(n_actions, 0, Exp3IX, seed)
