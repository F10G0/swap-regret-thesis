import numpy as np

from .empirical_mean_base import EmpiricalMeanBanditAlgorithm


class ExploreThenCommit(EmpiricalMeanBanditAlgorithm):
    """
    Explore-Then-Commit (ETC) algorithm.

    The algorithm first explores each arm exactly m times in a round-robin
    fashion, then commits to the arm with the highest empirical mean.
    """

    def __init__(self, n_arms: int, m: int):
        if m <= 0:
            raise ValueError("m must be positive")
        self.m = m
        super().__init__(n_arms)

    def reset(self) -> None:
        super().reset()
        self.committed_arm = None

    def select_action(self) -> int:
        # Exploration phase: round-robin
        if self.in_exploration_phase():
            return self.t % self.n_arms
        # Commit phase: choose the empirically best arm once
        if self.committed_arm is None:
            self.committed_arm = int(np.argmax(self.empirical_means()))
        return self.committed_arm

    def in_exploration_phase(self) -> bool:
        return self.t < self.m * self.n_arms

    def __repr__(self) -> str:
        return (
            f"ExploreThenCommit(n_arms={self.n_arms}, m={self.m}, "
            f"t={self.t}, committed_arm={self.committed_arm})"
        )
    