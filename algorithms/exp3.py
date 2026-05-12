import numpy as np

from .base import BanditAlgorithm


class Exp3(BanditAlgorithm):
    """
    Base class for Exp3-style adversarial bandit algorithms.

    This class implements exponential weighting with an importance-weighted
    loss estimator. Setting implicit_exploration = 0 gives vanilla Exp3.
    Setting implicit_exploration > 0 gives Exp3-IX style updates.
    """

    def __init__(self, n_arms: int, learning_rate: float, implicit_exploration: float = 0.0):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if implicit_exploration < 0.0:
            raise ValueError("implicit_exploration must be non-negative")
        self.n_arms = n_arms
        self.learning_rate = learning_rate
        self.implicit_exploration = implicit_exploration
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.estimated_cumulative_loss = np.zeros(self.n_arms, dtype=float)
        self.probabilities = np.full(self.n_arms, 1.0 / self.n_arms, dtype=float)

    def select_action(self) -> int:
        # Compute action probabilities via exponential weighting
        logits = -self.learning_rate * self.estimated_cumulative_loss
        logits -= np.max(logits) # numerical stability
        weights = np.exp(logits)
        self.probabilities = weights / np.sum(weights)
        # Sample an arm according to the current distribution
        return int(np.random.choice(self.n_arms, p=self.probabilities))

    def update(self, action: int, reward: float) -> None:
        if action < 0 or action >= self.n_arms:
            raise IndexError("invalid action index")
        if reward < 0.0 or reward > 1.0:
            raise ValueError("reward must be in [0, 1]")
        # Estimate loss using importance weighting (with optional smoothing)
        estimated_loss = (1.0 - reward) / (self.probabilities[action] + self.implicit_exploration)
        # Update cumulative loss for the selected arm
        self.estimated_cumulative_loss[action] += estimated_loss
        self.t += 1
        