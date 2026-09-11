import numpy as np


class RegretBundle:
    """Strategy-weighted gains G[i, j] = sum_t p_t[i](r_t[j] - r_t[i])."""

    def __init__(self, n_actions: int):
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        self.n_actions = n_actions
        self.cumulative_replacement_gains = np.zeros((n_actions, n_actions), dtype=float)

    def update(self, strategy: np.ndarray, payoff_vector: np.ndarray) -> None:
        replacement_gains = payoff_vector[None, :] - payoff_vector[:, None]
        weighted_replacement_gains = strategy[:, None] * replacement_gains
        self.cumulative_replacement_gains += weighted_replacement_gains

    @property
    def external_regret(self) -> float:
        """Return max_j sum_i G[i, j]."""
        fixed_action_gains = np.sum(self.cumulative_replacement_gains, axis=0)
        return float(np.max(fixed_action_gains))

    @property
    def internal_regret(self) -> float:
        """Return max_{i,j} G[i, j]."""
        return float(np.max(self.cumulative_replacement_gains))

    @property
    def swap_regret(self) -> float:
        """Return sum_i max_j G[i, j]."""
        best_replacement_gains = np.max(self.cumulative_replacement_gains, axis=1)
        return float(np.sum(best_replacement_gains))

    def summary(self, time: int) -> dict[str, float]:
        external_regret = self.external_regret
        internal_regret = self.internal_regret
        swap_regret = self.swap_regret

        return {
            "external_regret": external_regret,
            "average_external_regret": external_regret / time,
            "internal_regret": internal_regret,
            "average_internal_regret": internal_regret / time,
            "swap_regret": swap_regret,
            "average_swap_regret": swap_regret / time,
        }
