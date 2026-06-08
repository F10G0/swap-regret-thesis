import numpy as np

from .empirical_mean_base import EmpiricalMeanBanditAlgorithm


class PhasedUCB(EmpiricalMeanBanditAlgorithm):
    """
    Phased Upper Confidence Bound (Phased UCB) algorithm.

    The algorithm first plays each arm once. Then it proceeds in phases:
    at the start of each phase, it selects the arm with the largest UCB
    value and plays it for a phase-dependent number of rounds.

    The confidence bonus is parameterized by beta_fn(t), and the phase
    length is controlled by phase_length_fn.
    """

    def __init__(self, n_arms: int, beta_fn, phase_length_fn):
        if beta_fn is None:
            raise ValueError("beta_fn must not be None")
        if phase_length_fn is None:
            raise ValueError("phase_length_fn must not be None")
        self.beta_fn = beta_fn
        self.phase_length_fn = phase_length_fn
        super().__init__(n_arms)

    def reset(self) -> None:
        super().reset()
        self.phase = 1
        self.phase_arm = None
        self.phase_remaining_pulls = 0

    def select_action(self) -> int:
        # Initialization phase: play each arm once
        if self.in_initialization_phase():
            return self.t
        # Continue current phase if it is not finished
        if self.phase_remaining_pulls > 0:
            return self.phase_arm
        # Start a new phase by choosing the arm with highest UCB value
        self.phase_arm = self.best_ucb_arm()
        self.phase_remaining_pulls = self.compute_phase_length(self.phase_arm)
        return self.phase_arm

    def update(self, action: int, reward: float) -> None:
        # During phase execution, decrease remaining pulls of current phase
        if not self.in_initialization_phase():
            self.phase_remaining_pulls -= 1
            # If the current phase is finished, move to the next phase
            if self.phase_remaining_pulls == 0:
                self.phase += 1
                self.phase_arm = None
        # Update empirical statistics (counts, reward_sums, t)
        super().update(action, reward)

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

    def compute_phase_length(self, arm: int) -> int:
        length = int(np.ceil(self.phase_length_fn(
            phase=self.phase,
            arm=arm,
            counts=self.counts.copy(),
            t=self.t,
        )))
        if length <= 0:
            raise ValueError("phase_length_fn must return a positive value")
        return length
    
    def __repr__(self) -> str:
        return (
            f"PhasedUCB("
            f"n_arms={self.n_arms}, "
            f"t={self.t}, "
            f"phase={self.phase}, "
            f"phase_arm={self.phase_arm}, "
            f"phase_remaining_pulls={self.phase_remaining_pulls}, "
            f"counts={self.counts.tolist()}, "
            f"reward_sums={self.reward_sums.tolist()}"
            f")"
        )
    