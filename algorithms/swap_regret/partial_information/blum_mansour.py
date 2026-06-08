from algorithms.external_regret import Exp3
from algorithms.swap_regret.base import StationaryReduction


class BlumMansour(StationaryReduction):
    """
    Partial-information Blum-Mansour reduction.
    """
    
    def __init__(self, n_actions: int, learning_rate: float, inner_algorithm_factory=Exp3, seed: int | None = None) -> None:
        super().__init__(n_actions, learning_rate, inner_algorithm_factory, seed)

    def _update_state(self, action: int, reward: float) -> None:
        self._validate_action(action)
        self._validate_reward(reward)

        transition_matrix = self._transition_matrix()
        probability = self.current_strategy[action]

        for i, learner in enumerate(self.learners):
            # g_i = p_i * r_k * q_i,k / p_k
            observed_reward = self.current_strategy[i] * reward * transition_matrix[i, action] / probability
            learner.update(action, observed_reward)
