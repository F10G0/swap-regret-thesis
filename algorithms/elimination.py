import math
import numpy as np

from .empirical_mean_base import EmpiricalMeanBanditAlgorithm


class EliminationAlgorithm(EmpiricalMeanBanditAlgorithm):
    """
    Phased elimination algorithm for finite-armed stochastic bandits.

    In each phase, every active arm is pulled for a fixed number of rounds.
    Arms with sufficiently low phase-wise empirical mean are eliminated.
    """

    def __init__(self, n_arms: int, phase_length_fn):
        if phase_length_fn is None:
            raise ValueError("phase_length_fn must not be None")
        self.phase_length_fn = phase_length_fn
        super().__init__(n_arms)

    def reset(self) -> None:
        super().reset()
        # Phase-local statistics
        self.phase_counts = np.zeros(self.n_arms, dtype=int)
        self.phase_reward_sums = np.zeros(self.n_arms, dtype=float)
        # Phase state
        self.phase = 1
        self.active_arms = list(range(self.n_arms))
        self.active_index = 0
        self.phase_remaining_pulls = self.compute_phase_remaining_pulls()

    def select_action(self) -> int:
        if len(self.active_arms) == 0:
            raise RuntimeError("no active arms remain")
        # If only one arm remains, always play it
        if self.is_commitment_phase():
            return self.active_arms[0]
        # Round-robin over active arms within the current phase
        return self.active_arms[self.active_index]

    def update(self, action: int, reward: float) -> None:
        assert action in self.active_arms # Debug-only check; O(|active_arms|)
        # Update global statistics
        super().update(action, reward)
        # Update phase-local statistics
        self.phase_counts[action] += 1
        self.phase_reward_sums[action] += reward

        if not self.is_commitment_phase():
            # One scheduled pull of the current phase has been completed
            self.phase_remaining_pulls -= 1
            # If the current phase is complete, eliminate arms and start next phase
            if self.phase_remaining_pulls == 0:
                self._advance_phase()
            # Continue round-robin over active arms
            else:
                self.active_index = (self.active_index + 1) % len(self.active_arms)

    def is_commitment_phase(self) -> bool:
        return len(self.active_arms) == 1
    
    def compute_phase_remaining_pulls(self) -> int:
        m_l = math.ceil(self.phase_length_fn(self.phase))
        if m_l <= 0:
            raise ValueError("phase_length_fn must return a positive value")
        return m_l * len(self.active_arms)
    
    def phase_empirical_means(self) -> np.ndarray:
        means = np.zeros(self.n_arms, dtype=float)
        played = self.phase_counts > 0
        means[played] = self.phase_reward_sums[played] / self.phase_counts[played]
        return means

    def _advance_phase(self) -> None:
        # Compute phase-wise empirical means and best active arm
        means = self.phase_empirical_means()
        best_mean = max(means[arm] for arm in self.active_arms)
        threshold = 2.0 ** (-self.phase)
        # Eliminate arms whose mean is too far below the best arm
        new_active_arms = [arm for arm in self.active_arms if means[arm] + threshold >= best_mean]
        if not new_active_arms:
            raise RuntimeError("elimination removed all active arms")
        self.active_arms = new_active_arms
        # Reset phase-local statistics for the next phase
        self.phase_counts = np.zeros(self.n_arms, dtype=int)
        self.phase_reward_sums = np.zeros(self.n_arms, dtype=float)
        # Advance to next phase and reinitialize scheduling state
        self.phase += 1
        self.active_index = 0
        self.phase_remaining_pulls = self.compute_phase_remaining_pulls()

    def __repr__(self) -> str:
        return (
            f"EliminationAlgorithm("
            f"n_arms={self.n_arms}, "
            f"t={self.t}, "
            f"phase={self.phase}, "
            f"phase_remaining_pulls={self.phase_remaining_pulls}, "
            f"active_arms={self.active_arms}, "
            f"active_index={self.active_index}, "
            f"counts={self.counts.tolist()}, "
            f"reward_sums={self.reward_sums.tolist()}, "
            f"phase_counts={self.phase_counts.tolist()}, "
            f"phase_reward_sums={self.phase_reward_sums.tolist()}"
            f")"
        )
    