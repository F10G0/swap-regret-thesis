import numpy as np
from .base import BanditAlgorithm


class ExploreThenCommit(BanditAlgorithm):
    """
    Explore-Then-Commit (ETC) algorithm.

    The algorithm first explores each arm exactly m times in a round-robin
    fashion, then commits to the arm with the highest empirical mean.
    """

    def __init__(self, n_arms: int, m: int):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        if m <= 0:
            raise ValueError("m must be positive")

        self.n_arms = n_arms
        self.m = m
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.reward_sums = np.zeros(self.n_arms, dtype=float)
        self.committed_arm = None

    def select_action(self) -> int:
        # Exploration phase: round-robin
        if self.in_exploration_phase():
            return self.t % self.n_arms

        # Commit phase: choose the empirically best arm once
        if self.committed_arm is None:
            empirical_means = self.empirical_means()
            self.committed_arm = int(np.argmax(empirical_means))

        return self.committed_arm

    def update(self, action: int, reward: float) -> None:
        if action < 0 or action >= self.n_arms:
            raise IndexError("invalid action index")

        self.counts[action] += 1
        self.reward_sums[action] += reward
        self.t += 1

    def in_exploration_phase(self) -> bool:
        return self.t < self.m * self.n_arms

    def empirical_means(self) -> np.ndarray:
        return self.reward_sums / np.maximum(self.counts, 1)

    def __repr__(self) -> str:
        return (
            f"ExploreThenCommit(n_arms={self.n_arms}, m={self.m}, "
            f"t={self.t}, committed_arm={self.committed_arm})"
        )
