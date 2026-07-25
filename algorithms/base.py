from abc import ABC, abstractmethod

import numpy as np


class Algorithm(ABC):
    """Base class for learning algorithms with a current action distribution."""

    def __init__(self, n_actions: int, horizon: int = 0, seed: int | None = None) -> None:
        if n_actions <= 0:
            raise ValueError("n_actions must be positive.")
        if horizon < 0:
            raise ValueError("horizon must be non-negative")

        self.n_actions = n_actions
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """Reset the internal state and strategy."""
        self._reset_state()
        self.t = 0
        self.current_action = None
        self.current_strategy = np.full(self.n_actions, 1.0 / self.n_actions)

    def update(self, feedback: float | np.ndarray) -> None:
        """Update the internal state and recompute the strategy."""
        self._validate_feedback(feedback)
        self._update_state(feedback)
        self.t += 1
        self.current_strategy = self._compute_strategy()
        self._validate_strategy()

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
        """Return a copy of the current action distribution."""
        return self.current_strategy.copy()

    def sample_action(self) -> int:
        """Sample an action from the current action distribution."""
        self.current_action = int(self.rng.choice(self.n_actions, p=self.current_strategy))
        return self.current_action

    def _validate_strategy(self) -> None:
        if self.current_strategy is None:
            raise RuntimeError("strategy is not initialized")
        if self.current_strategy.shape != (self.n_actions,):
            raise ValueError(f"strategy must have shape ({self.n_actions},)")
        if not np.all(np.isfinite(self.current_strategy)):
            raise ValueError("strategy must contain only finite probabilities")
        if np.any(self.current_strategy < 0.0):
            raise ValueError("strategy must not contain negative probabilities.")
        if not np.isclose(np.sum(self.current_strategy), 1.0):
            raise ValueError("strategy must sum to 1.")

    def _validate_feedback(self, feedback: float | np.ndarray) -> None:
        if np.ndim(feedback) > 0 and feedback.shape != (self.n_actions,):
            raise ValueError(f"feedback must have shape ({self.n_actions},)")
        if not np.all(np.isfinite(feedback)):
            raise ValueError("feedback must contain only finite values")
        if np.any((feedback < 0.0) | (feedback > 1.0)):
            raise ValueError("feedback must contain values in [0, 1]")
