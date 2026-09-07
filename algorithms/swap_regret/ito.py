from functools import partial

import numpy as np

from algorithms.external_regret import Hedge, TsallisINF
from algorithms.swap_regret.base import StationaryReduction


class ItoBase(StationaryReduction):
    """Base class for Ito reductions with two-stage action sampling."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.selected_learner = None

    def sample_action(self) -> int:
        learner_index = int(self.rng.choice(self.n_actions, p=self.current_strategy))
        self.selected_learner = self.learners[learner_index]
        return self.selected_learner.sample_action()


class FullIto(ItoBase):
    """Full-information Ito reduction with anytime inner learners."""

    def __init__(self, n_actions: int, inner_algorithm_factory=Hedge, seed: int | None = None) -> None:
        super().__init__(n_actions, partial(inner_algorithm_factory, n_actions, horizon=None), seed=seed)

    def _update_state(self, reward_vector: np.ndarray) -> None:
        self.selected_learner.update(reward_vector)
        self.selected_learner = None


class BanditIto(ItoBase):
    """Ito (2020) bandit reduction with anytime Tsallis-INF learners."""

    def __init__(self, n_actions: int, inner_algorithm_factory=TsallisINF, seed: int | None = None) -> None:
        super().__init__(n_actions, partial(inner_algorithm_factory, n_actions), seed=seed)

    def _update_state(self, reward: float) -> None:
        self.selected_learner.update(reward)
        self.selected_learner = None
