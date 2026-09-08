from abc import ABC, abstractmethod

import numpy as np


class Algorithm(ABC):
    """Base class for learning algorithms with a current action distribution."""

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        if n_actions <= 0:
            raise ValueError("n_actions must be positive.")

        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """Reset the internal state and strategy."""
        self._reset_state()
        self.t = 0
        self.current_action = None
        self.current_strategy = np.full(self.n_actions, 1.0 / self.n_actions)

    def update(self, feedback: float | np.ndarray) -> None:
        """Update from trusted environment feedback and recompute the strategy."""
        self._update_state(feedback)
        self.t += 1
        self.current_strategy = self._compute_strategy()

    @abstractmethod
    def _reset_state(self) -> None:
        """Reset algorithm-specific internal state."""
        pass

    @abstractmethod
    def _update_state(self, feedback: float | np.ndarray) -> None:
        """Update algorithm-specific internal state."""
        pass

    @abstractmethod
    def _compute_strategy(self) -> np.ndarray:
        """Compute the current strategy from the internal state."""
        pass

    def strategy(self) -> np.ndarray:
        """Return the current distribution; internal callers must not mutate it."""
        return self.current_strategy

    def sample_action(self) -> int:
        """Sample an action from the current action distribution."""
        self.current_action = int(self.rng.choice(self.n_actions, p=self.current_strategy))
        return self.current_action
