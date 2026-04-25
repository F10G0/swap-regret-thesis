import numpy as np

from .empirical_mean_base import EmpiricalMeanBanditAlgorithm


class UpperConfidenceBound(EmpiricalMeanBanditAlgorithm):
    """
    Upper Confidence Bound (UCB) algorithm family.

    The algorithm first plays each arm once, then repeatedly chooses the arm
    with the largest upper confidence bound.
    The confidence bonus is parameterized by beta_fn(t).
    """

    def __init__(self, n_arms: int, beta_fn):
        if beta_fn is None:
            raise ValueError("beta_fn must not be None")
        self.beta_fn = beta_fn
        super().__init__(n_arms)

    def select_action(self) -> int:
        # Initialization phase: play each arm once
        if self.in_initialization_phase():
            return self.t
        # Select arm with highest upper confidence bound
        return self.best_ucb_arm()

    def in_initialization_phase(self) -> bool:
        return self.t < self.n_arms
    
    def best_ucb_arm(self) -> int:
        # Compute exploration parameter
        beta = self.beta_fn(self.t)
        if beta < 0.0:
            raise ValueError("beta_fn(t) must be nonnegative")
        # Compute UCB index for each arm
        empirical_means = self.empirical_means()
        bonuses = np.sqrt(beta / self.counts)
        ucb_values = empirical_means + bonuses
        return int(np.argmax(ucb_values))
    
    def __repr__(self) -> str:
        return (
            f"UpperConfidenceBound(n_arms={self.n_arms}, "
            f"t={self.t}, beta_fn={self.beta_fn})"
        )
    