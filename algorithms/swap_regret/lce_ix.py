from functools import partial

from algorithms.external_regret import Exp3IX
from algorithms.swap_regret.base import StationaryReduction


class LCEIX(StationaryReduction):
    """Learning for correlated equilibrium with implicit exploration."""

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        super().__init__(n_actions, partial(Exp3IX, n_actions, 0), seed=seed)

    def _update_state(self, reward: float) -> None:
        transition_matrix = self._transition_matrix
        probability = self.current_strategy[self.current_action]

        for i, learner in enumerate(self.learners):
            observed_loss = self.current_strategy[i] * (1.0 - reward) * transition_matrix[i, self.current_action] / probability
            learner.current_action = self.current_action
            learner.update(1.0 - observed_loss)
