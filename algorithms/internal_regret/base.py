import numpy as np

from algorithms.base import Algorithm


class RegretMatchingBase(Algorithm):
    """Base class for Hart-Mas-Colell regret-matching algorithms."""

    @property
    def normalization(self) -> int:
        """Return K as the fixed normalization."""
        return self.n_actions

    @property
    def _regret_transition_matrix(self) -> np.ndarray:
        """Return the transition matrix induced by positive regret."""
        transition_matrix = np.maximum(self.cumulative_regret, 0.0) / (self.normalization * self.t)
        np.fill_diagonal(transition_matrix, 1.0 - np.sum(transition_matrix, axis=1))
        return transition_matrix

    def _reset_state(self) -> None:
        self.cumulative_regret = np.zeros((self.n_actions, self.n_actions), dtype=float)

    def _update_state(self, reward_vector: np.ndarray) -> None:
        self.cumulative_regret[self.current_action] += reward_vector - reward_vector[self.current_action]
