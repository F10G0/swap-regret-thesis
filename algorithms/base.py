from abc import ABC, abstractmethod

import numpy as np


class Algorithm(ABC):
    """
    Base class for learning algorithms with a current action distribution.
    """

    current_strategy: np.ndarray | None

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        if n_actions <= 0:
            raise ValueError("n_actions must be positive.")

        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.current_strategy = None

    def reset(self) -> None:
        """Reset the internal state and recompute the strategy."""
        self._reset_state()
        self._update_strategy()

    def update(self, *args, **kwargs) -> None:
        """Update the internal state and recompute the strategy."""
        self._update_state(*args, **kwargs)
        self._update_strategy()

    @abstractmethod
    def _reset_state(self) -> None:
        """Reset algorithm-specific internal state."""
        pass

    @abstractmethod
    def _update_state(self, *args, **kwargs) -> None:
        """Update algorithm-specific internal state."""
        pass

    @abstractmethod
    def _update_strategy(self) -> None:
        """Recompute the current strategy from the internal state."""
        pass

    def strategy(self) -> np.ndarray:
        """Return a copy of the current action distribution."""
        self._validate_strategy()
        return self.current_strategy.copy()

    def sample_action(self) -> int:
        """Sample an action from the current action distribution."""
        self._validate_strategy()
        return int(self.rng.choice(self.n_actions, p=self.current_strategy))

    def _validate_strategy(self) -> None:
        if self.current_strategy is None:
            raise RuntimeError("strategy is not initialized")
        if np.any(self.current_strategy < 0.0):
            raise ValueError("strategy must not contain negative probabilities.")
        if not np.isclose(np.sum(self.current_strategy), 1.0):
            raise ValueError("strategy must sum to 1.")
    
    def _validate_action(self, action: int) -> None:
        if action < 0 or action >= self.n_actions:
            raise IndexError("invalid action index")
        
    def _validate_reward(self, reward: float) -> None:
        if reward < 0.0 or reward > 1.0:
            raise ValueError("reward must be in [0, 1]")

    def _validate_reward_vector(self, reward_vector: np.ndarray) -> np.ndarray:
        reward_vector = np.asarray(reward_vector, dtype=float)

        if reward_vector.shape != (self.n_actions,):
            raise ValueError(f"reward_vector must have shape ({self.n_actions},)")
        if np.any((reward_vector < 0.0) | (reward_vector > 1.0)):
            raise ValueError("reward_vector must contain values in [0, 1]")

        return reward_vector
