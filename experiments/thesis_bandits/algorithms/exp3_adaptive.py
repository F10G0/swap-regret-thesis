import numpy as np

from .base import BanditAlgorithm


class Exp3AdaptiveLR(BanditAlgorithm):
    """
    Exp3-style adversarial bandit with time-varying learning rate.

    The learning rate is provided by a user-defined schedule eta_t = f(n_arms, t).
    Supports both Exp3 (ix_ratio = 0) and Exp3-IX (ix_ratio > 0).
    """

    def __init__(self, n_arms: int, learning_rate_fn: float, ix_ratio: float = 0.0):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        if learning_rate_fn is None:
            raise ValueError("learning_rate must be positive")
        if ix_ratio < 0.0:
            raise ValueError("implicit_exploration must be non-negative")
        self.n_arms = n_arms
        self.learning_rate_fn = learning_rate_fn
        self.ix_ratio = ix_ratio
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.estimated_cumulative_loss = np.zeros(self.n_arms, dtype=float)
        self.probabilities = np.full(self.n_arms, 1.0 / self.n_arms, dtype=float)

    def select_action(self) -> int:
        # Update learning rate and implicit exploration for current round
        self.learning_rate = self.compute_learning_rate()
        self.implicit_exploration = self.ix_ratio * self.learning_rate
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
        self.implicit_exploration = self.learning_rate * self.ix_ratio
        estimated_loss = (1.0 - reward) / (self.probabilities[action] + self.implicit_exploration)
        # Update cumulative loss for the selected arm
        self.estimated_cumulative_loss[action] += estimated_loss
        self.t += 1

    def compute_learning_rate(self) -> float:
        learning_rate = self.learning_rate_fn(self.n_arms, self.t + 1)
        if learning_rate <= 0:
            raise ValueError("learning_rate_fn must return a positive value")
        return learning_rate
        