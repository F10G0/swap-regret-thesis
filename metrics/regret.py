import numpy as np


class BaseReplacementRegretBundle:
    """Base class using G[i, j] for the cumulative gain from replacing action i with action j."""

    def __init__(self, n_actions: int, regret_type: str):
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if not regret_type:
            raise ValueError("regret_type must not be empty")

        self.n_actions = n_actions
        self.regret_type = regret_type
        self.cumulative_replacement_gains = np.zeros((n_actions, n_actions), dtype=float)

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
        if time <= 0:
            raise ValueError("time must be positive")

        regret_type = self.regret_type
        external_regret = self.external_regret
        internal_regret = self.internal_regret
        swap_regret = self.swap_regret

        return {
            f"{regret_type}_external_regret": external_regret,
            f"average_{regret_type}_external_regret": external_regret / time,
            f"{regret_type}_internal_regret": internal_regret,
            f"average_{regret_type}_internal_regret": internal_regret / time,
            f"{regret_type}_swap_regret": swap_regret,
            f"average_{regret_type}_swap_regret": swap_regret / time,
        }

    def _validate_strategy(self, strategy: np.ndarray) -> None:
        if strategy.shape != (self.n_actions,):
            raise ValueError(f"strategy must have shape ({self.n_actions},)")
        if not np.all(np.isfinite(strategy)):
            raise ValueError("strategy must contain only finite probabilities")
        if np.any(strategy < 0.0):
            raise ValueError("strategy must not contain negative probabilities")
        if not np.isclose(np.sum(strategy), 1.0):
            raise ValueError("strategy must sum to 1")

    def _validate_payoff_vector(self, payoff_vector: np.ndarray) -> None:
        if payoff_vector.shape != (self.n_actions,):
            raise ValueError(f"payoff_vector must have shape ({self.n_actions},)")
        if not np.all(np.isfinite(payoff_vector)):
            raise ValueError("payoff_vector must contain only finite values")


class ExpectedRegretBundle(BaseReplacementRegretBundle):
    """Expected-regret tracker with G[i, j] = sum_t p_t[i](r_t[j] - r_t[i])."""

    def __init__(self, n_actions: int):
        super().__init__(n_actions, regret_type="expected")

    def update(self, strategy: np.ndarray, payoff_vector: np.ndarray) -> None:
        self._validate_strategy(strategy)
        self._validate_payoff_vector(payoff_vector)

        replacement_gains = payoff_vector[None, :] - payoff_vector[:, None]
        weighted_replacement_gains = strategy[:, None] * replacement_gains
        self.cumulative_replacement_gains += weighted_replacement_gains


class RealizedRegretBundle(BaseReplacementRegretBundle):
    """Realized-regret tracker with G[i, j] = sum_{t: a_t = i}(r_t[j] - r_t[i])."""

    def __init__(self, n_actions: int):
        super().__init__(n_actions, regret_type="realized")

    def update(self, action: int, payoff_vector: np.ndarray) -> None:
        self._validate_payoff_vector(payoff_vector)

        replacement_gains = payoff_vector - payoff_vector[action]
        self.cumulative_replacement_gains[action] += replacement_gains


class RegretBundles:
    """Expected and realized regret trackers for one player."""

    def __init__(self, n_actions: int):
        self.expected = ExpectedRegretBundle(n_actions)
        self.realized = RealizedRegretBundle(n_actions)

    def update(self, strategy: np.ndarray, action: int, payoff_vector: np.ndarray) -> None:
        self.expected.update(strategy, payoff_vector)
        self.realized.update(action, payoff_vector)
