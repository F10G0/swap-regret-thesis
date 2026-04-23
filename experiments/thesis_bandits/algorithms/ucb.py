import numpy as np
from .base import BanditAlgorithm


class UpperConfidenceBound(BanditAlgorithm):
    """
    Upper Confidence Bound (UCB) algorithm.

    The algorithm first plays each arm once, then repeatedly chooses the arm
    with the largest upper confidence bound.
    """

    def __init__(self, n_arms: int, exploration_factor: float = 2.0):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        if exploration_factor <= 0:
            raise ValueError("exploration_factor must be positive")

        self.n_arms = n_arms
        self.exploration_factor = exploration_factor
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.reward_sums = np.zeros(self.n_arms, dtype=float)

    def select_action(self) -> int:
        # Initialization phase: play each arm once
        if self.in_initialization_phase():
            return self.t
        
        empirical_means = self.empirical_means()
        bonuses = np.sqrt(
            self.exploration_factor * np.log(self.t) / self.counts
        )
        ucb_values = empirical_means + bonuses
        return int(np.argmax(ucb_values))

    def update(self, action: int, reward: float) -> None:
        if action < 0 or action >= self.n_arms:
            raise IndexError("invalid action index")

        self.counts[action] += 1
        self.reward_sums[action] += reward
        self.t += 1

    def in_initialization_phase(self) -> bool:
        return self.t < self.n_arms

    def empirical_means(self) -> np.ndarray:
        return self.reward_sums / np.maximum(self.counts, 1)

    def __repr__(self) -> str:
        return (
            f"UpperConfidenceBound(n_arms={self.n_arms}, "
            f"exploration_factor={self.exploration_factor}, t={self.t})"
        )
    