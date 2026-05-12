import numpy as np

from .base import BanditAlgorithm


class EmpiricalMeanBanditAlgorithm(BanditAlgorithm):
    """
    Base class for bandit algorithms that maintain pull counts,
    reward sums, and empirical means.
    """

    def __init__(self, n_arms: int):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        self.n_arms = n_arms
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.reward_sums = np.zeros(self.n_arms, dtype=float)

    def update(self, action: int, reward: float) -> None:
        if action < 0 or action >= self.n_arms:
            raise IndexError("invalid action index")
        self.counts[action] += 1
        self.reward_sums[action] += reward
        self.t += 1

    def empirical_means(self) -> np.ndarray:
        return self.reward_sums / np.maximum(self.counts, 1)
    