import numpy as np

from environments.base import PlayerOutcome


class RegretBundle:
    """
    Tracks external, internal, and swap regret from the same cumulative replacement-gain matrix
        G[i, j] = sum_t p_t[i] * (r_t[j] - r_t[i]).
    """

    def __init__(self, n_actions: int):
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")

        self.n_actions = n_actions
        self.cumulative_replacement_gains = np.zeros(
            (n_actions, n_actions),
            dtype=float,
        )

    def update(self, strategy: np.ndarray, outcome: PlayerOutcome) -> None:
        self._validate_strategy(strategy)
        self._validate_outcome(outcome)
        payoff_vector = outcome.payoff_vector

        replacement_gains = payoff_vector[None, :] - payoff_vector[:, None]
        weighted_replacement_gains = strategy[:, None] * replacement_gains
        self.cumulative_replacement_gains += weighted_replacement_gains

    @property
    def external_regret(self) -> float:
        """
        This corresponds to replacing every action by the same fixed action j.
            max_j sum_i G[i, j]
        """
        fixed_action_gains = np.sum(self.cumulative_replacement_gains, axis=0)
        return float(np.max(fixed_action_gains))

    @property
    def internal_regret(self) -> float:
        """
        This corresponds to replacing one departure action i by one action j.
            max_{i,j} G[i, j]
        """
        return float(np.max(self.cumulative_replacement_gains))

    @property
    def swap_regret(self) -> float:
        """
        This corresponds to choosing a separate replacement j for each departure action i.
            sum_i max_j G[i, j]
        """
        best_replacement_gains = np.max(self.cumulative_replacement_gains, axis=1)
        return float(np.sum(best_replacement_gains))

    def summary(self, time: int) -> dict[str, float]:
        if time <= 0:
            raise ValueError("time must be positive")

        return {
            "external_regret": self.external_regret,
            "average_external_regret": self.external_regret / time,
            "internal_regret": self.internal_regret,
            "average_internal_regret": self.internal_regret / time,
            "swap_regret": self.swap_regret,
            "average_swap_regret": self.swap_regret / time,
        }

    def _validate_strategy(self, strategy: np.ndarray) -> None:
        if strategy.shape != (self.n_actions,):
            raise ValueError(f"strategy must have shape ({self.n_actions},)")
        if np.any(strategy < 0.0):
            raise ValueError("strategy must not contain negative probabilities")
        if not np.isclose(np.sum(strategy), 1.0):
            raise ValueError("strategy must sum to 1")

    def _validate_outcome(self, outcome: PlayerOutcome) -> None:
        if outcome.payoff_vector.shape != (self.n_actions,):
            raise ValueError(f"payoff_vector must have shape ({self.n_actions},)")
