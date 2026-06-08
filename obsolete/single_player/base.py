from abc import ABC, abstractmethod

import numpy as np


class SinglePlayerEnvironment(ABC):
    """
    Base class for single-player environments.
    """

    @property
    @abstractmethod
    def n_actions(self) -> int:
        pass

    @property
    @abstractmethod
    def optimal_action(self) -> int:
        pass

    @abstractmethod
    def step(self, action: int):
        pass

    def _validate_action(self, action: int) -> None:
        if action < 0 or action >= self.n_actions:
            raise IndexError("invalid action index")
        

class FixedRewardEnvironment(SinglePlayerEnvironment):
    """
    Single-player environment defined by a fixed reward matrix.

    rewards[t, a] is the reward of action a at round t.
    """

    def __init__(self, rewards: np.ndarray):
        rewards = np.asarray(rewards, dtype=float)

        if rewards.ndim != 2:
            raise ValueError("rewards must be a 2D array of shape (horizon, n_actions)")
        if rewards.shape[0] == 0 or rewards.shape[1] == 0:
            raise ValueError("rewards must have non-zero horizon and number of actions")
        if np.any((rewards < 0.0) | (rewards > 1.0)):
            raise ValueError("rewards must be in [0, 1]")

        self._rewards = rewards
        self._optimal_action = int(np.argmax(np.sum(rewards, axis=0)))
        self._t = 0

    @property
    def n_actions(self) -> int:
        return self._rewards.shape[1]

    @property
    def optimal_action(self) -> int:
        return self._optimal_action
    
    @property
    def horizon(self) -> int:
        return self._rewards.shape[0]

    @property
    def current_round(self) -> int:
        return self._t
    
    def next_rewards(self) -> np.ndarray:
        if self._t >= self.horizon:
            raise RuntimeError("horizon exhausted")

        rewards = self._rewards[self._t]
        self._t += 1
        return rewards.copy()

    def reset(self) -> None:
        self._t = 0
