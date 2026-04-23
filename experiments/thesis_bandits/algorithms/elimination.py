import math
import numpy as np
from .base import BanditAlgorithm


class EliminationAlgorithm(BanditAlgorithm):
    """
    Phased elimination algorithm for finite-armed stochastic bandits.

    The algorithm proceeds in phases. In phase l, each active arm is pulled
    exactly m_l times, where m_l is determined by phase_length_fn(l).
    After that, arms whose phase-wise empirical mean is too far below the
    best active arm are eliminated.

    The elimination rule is:
        keep arm i if mean_i + 2^{-l} >= max_j mean_j
    """

    def __init__(self, n_arms: int, phase_length_fn):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        if phase_length_fn is None:
            raise ValueError("phase_length_fn must not be None")

        self.n_arms = n_arms
        self.phase_length_fn = phase_length_fn
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.reward_sums = np.zeros(self.n_arms, dtype=float)

        # Phase-local statistics
        self.phase_t = 0
        self.phase_counts = np.zeros(self.n_arms, dtype=int)
        self.phase_reward_sums = np.zeros(self.n_arms, dtype=float)

        # Phase state
        self.phase = 1
        self.phase_length = self.compute_phase_length()
        self.active_arms = list(range(self.n_arms))
        self.active_index = 0

    def select_action(self) -> int:
        if len(self.active_arms) == 0:
            raise RuntimeError("no active arms remain")

        # If only one arm remains, always play it
        if self.is_commitment_phase():
            return self.active_arms[0]

        return self.active_arms[self.active_index]

    def update(self, action: int, reward: float) -> None:
        if action < 0 or action >= self.n_arms:
            raise IndexError("invalid action index")
        assert action in self.active_arms # Debug-only check; O(|active_arms|)

        # Update global statistics
        self.counts[action] += 1
        self.reward_sums[action] += reward
        self.t += 1

        # Update phase-local statistics
        self.phase_counts[action] += 1
        self.phase_reward_sums[action] += reward
        self.phase_t += 1

        if not self.is_commitment_phase():
            # If the current phase is complete, eliminate arms and move to next phase
            if self.phase_t >= self.phase_length * len(self.active_arms):
                self._advance_phase()
            else:
                # Advance round-robin pointer
                self.active_index = (self.active_index + 1) % len(self.active_arms)

    def compute_phase_length(self) -> int:
        m_l = math.ceil(self.phase_length_fn(self.phase))
        if m_l <= 0:
            raise ValueError("phase_length_fn must return a positive value")
        return m_l
    
    def is_commitment_phase(self) -> bool:
        return len(self.active_arms) == 1

    def phase_empirical_means(self) -> np.ndarray:
        means = np.zeros(self.n_arms, dtype=float)
        played = self.phase_counts > 0
        means[played] = self.phase_reward_sums[played] / self.phase_counts[played]
        return means

    def empirical_means(self) -> np.ndarray:
        return self.reward_sums / np.maximum(self.counts, 1)

    def _advance_phase(self) -> None:
        means = self.phase_empirical_means()
        best_mean = max(means[arm] for arm in self.active_arms)
        threshold = 2.0 ** (-self.phase)

        new_active_arms = [arm for arm in self.active_arms if means[arm] + threshold >= best_mean]
        if not new_active_arms:
            raise RuntimeError("elimination removed all active arms")
        self.active_arms = new_active_arms

        # Reset phase-local statistics
        self.phase_counts = np.zeros(self.n_arms, dtype=int)
        self.phase_reward_sums = np.zeros(self.n_arms, dtype=float)
        self.phase_t = 0

        # Update phase state
        self.phase += 1
        self.phase_length = self.compute_phase_length()
        self.active_index = 0

    def __repr__(self) -> str:
        return (
            f"EliminationAlgorithm("
            f"n_arms={self.n_arms}, "
            f"t={self.t}, "
            f"phase={self.phase}, "
            f"phase_t={self.phase_t}, "
            f"phase_length={self.phase_length}, "
            f"active_arms={self.active_arms}, "
            f"active_index={self.active_index}, "
            f"counts={self.counts.tolist()}, "
            f"reward_sums={self.reward_sums.tolist()}, "
            f"phase_counts={self.phase_counts.tolist()}, "
            f"phase_reward_sums={self.phase_reward_sums.tolist()}"
            f")"
        )
    