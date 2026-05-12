import numpy as np

from .base import AdversarialBanditEnvironment


class AdversarialBandit(AdversarialBanditEnvironment):
    """
    Adversarial bandit environment defined by a fixed reward matrix.

    rewards[t, i] is the reward of arm i at round t.
    """

    def __init__(self, rewards: np.ndarray):
        rewards = np.asarray(rewards, dtype=float)

        if rewards.ndim != 2:
            raise ValueError("rewards must be a 2D array of shape (horizon, n_arms)")
        if rewards.shape[0] == 0 or rewards.shape[1] == 0:
            raise ValueError("rewards must have non-zero horizon and number of arms")
        if np.any(rewards < 0.0) or np.any(rewards > 1.0):
            raise ValueError("rewards must be in [0, 1]")

        self._rewards = rewards
        self._t = 0

    @property
    def n_arms(self) -> int:
        return self._rewards.shape[1]

    @property
    def horizon(self) -> int:
        return self._rewards.shape[0]

    def pull(self, action: int) -> float:
        if action < 0 or action >= self.n_arms:
            raise IndexError("invalid action index")
        if self._t >= self.horizon:
            raise RuntimeError("horizon exhausted")

        reward = self._rewards[self._t, action]
        self._t += 1
        return float(reward)

    def best_fixed_arm_reward(self) -> float:
        return float(np.max(np.sum(self._rewards, axis=0)))

    def best_fixed_arm(self) -> int:
        return int(np.argmax(np.sum(self._rewards, axis=0)))

    def reset(self) -> None:
        self._t = 0

    def __repr__(self) -> str:
        return f"AdversarialBandit(horizon={self.horizon}, n_arms={self.n_arms})"
    