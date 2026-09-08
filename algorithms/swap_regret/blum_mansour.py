from functools import partial

import numpy as np

from algorithms.external_regret import AuerExp3, Hedge
from algorithms.external_regret.exp3 import ImplicitExplorationAlgorithm
from algorithms.swap_regret.base import StationaryReduction


class FullBM(StationaryReduction):
    """Full-information Blum-Mansour reduction with known-horizon inner learners."""

    def __init__(self, n_actions: int, horizon: int, inner_algorithm_factory=Hedge, seed: int | None = None) -> None:
        if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon
        super().__init__(n_actions, partial(inner_algorithm_factory, n_actions, horizon=horizon), seed=seed)

    def _update_state(self, reward_vector: np.ndarray) -> None:
        for i, learner in enumerate(self.learners):
            weighted_reward = self.current_strategy[i] * reward_vector
            learner.update(weighted_reward)


class BanditBM(StationaryReduction):
    """Gain-based partial-information reduction from Blum and Mansour (2007)."""

    def __init__(self, n_actions: int, horizon: int, inner_algorithm_factory=AuerExp3, seed: int | None = None) -> None:
        if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon
        super().__init__(n_actions, partial(inner_algorithm_factory, n_actions, horizon=horizon), seed=seed)

    def _update_state(self, reward: float) -> None:
        probability = self.current_strategy[self.current_action]

        for i, learner in enumerate(self.learners):
            observed_gain = (self.current_strategy[i] * learner.current_strategy[self.current_action] * reward / probability)
            learner.current_action = self.current_action
            learner.update(observed_gain)


class LCEIXInner(ImplicitExplorationAlgorithm):
    """Huang-Pan LCE-IX inner learner with its paper-specific rate."""

    @property
    def learning_rate(self) -> float:
        """Return eta_t = sqrt(log(K) / t) at the active local time."""
        return np.sqrt(np.log(self.n_actions) / (self.t + 1))


class LCEIX(StationaryReduction):
    """Huang-Pan (2023) LCE-IX reduction."""

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        super().__init__(n_actions, partial(LCEIXInner, n_actions), seed=seed)

    def _update_state(self, reward: float) -> None:
        probability = self.current_strategy[self.current_action]

        for i, learner in enumerate(self.learners):
            observed_loss = (self.current_strategy[i] * (1.0 - reward) * learner.current_strategy[self.current_action] / probability)
            learner.current_action = self.current_action
            learner.update(1.0 - observed_loss)
