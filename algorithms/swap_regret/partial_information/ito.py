from algorithms.external_regret import Exp3
from algorithms.swap_regret.base import StationaryReduction


class Ito(StationaryReduction):
    """
    Partial-information Ito reduction.
    """

    def __init__(self, n_actions: int, learning_rate: float, inner_algorithm_factory=Exp3, seed: int | None = None) -> None:
        super().__init__(n_actions, learning_rate, inner_algorithm_factory, seed)

    def _reset_state(self) -> None:
        super()._reset_state()
        self.selected_learner = None

    def _update_state(self, action: int, reward: float) -> None:
        if self.selected_learner is None:
            raise RuntimeError("sample_action() must be called before update.")

        self._validate_action(action)
        self._validate_reward(reward)

        # Feed bandit feedback only to A_j
        self.selected_learner.update(action, reward)
        self.selected_learner = None

    def sample_action(self) -> int:
        if self.selected_learner is not None:
            raise RuntimeError("sample_action() was called twice before update.")
        
        # First sample j_t ~ p
        self.selected_learner = self.learners[super().sample_action()]
        # Then sample i_t ~ q_j
        return self.selected_learner.sample_action()
