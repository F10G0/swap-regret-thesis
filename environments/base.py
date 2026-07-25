from abc import ABC, abstractmethod

import numpy as np


class FixedGameEnvironment(ABC):
    """Fixed game where payoff_tensor[i, a_1, ..., a_n] gives player i's payoff."""

    def __init__(self, payoff_tensor: np.ndarray) -> None:
        self.payoff_tensor = self._validate_payoff_tensor(payoff_tensor)
        self.actions: tuple[int, ...]

    @property
    def n_players(self) -> int:
        return self.payoff_tensor.shape[0]

    @property
    def n_actions(self) -> tuple[int, ...]:
        return self.payoff_tensor.shape[1:]

    def step(self, actions: tuple[int, ...]) -> None:
        self._validate_actions(actions)
        self.actions = actions

    # In bandit experiments, use only for evaluation—never as learner feedback.
    def deviation_payoffs(self, player: int) -> np.ndarray:
        self._validate_player(player)
        indices = list(self.actions)
        indices[player] = slice(None)
        return self.payoff_tensor[(player, *indices)].copy()

    @abstractmethod
    def feedback(self, player: int) -> float | np.ndarray:
        """Return one player's feedback for the current round."""
        pass

    @staticmethod
    def _validate_payoff_tensor(payoff_tensor: np.ndarray) -> np.ndarray:
        if payoff_tensor.ndim < 2:
            raise ValueError("payoff tensor must have shape (n_players, action_1, ..., action_n)")
        if payoff_tensor.shape[0] == 0:
            raise ValueError("number of players must be non-zero")
        if payoff_tensor.shape[0] != payoff_tensor.ndim - 1:
            raise ValueError("number of players must match number of action dimensions")
        if any(size == 0 for size in payoff_tensor.shape[1:]):
            raise ValueError("each player must have at least one action")
        if not np.all(np.isfinite(payoff_tensor)):
            raise ValueError("payoffs must contain only finite values")
        if np.any((payoff_tensor < 0.0) | (payoff_tensor > 1.0)):
            raise ValueError("payoffs must contain values in [0, 1]")
        return payoff_tensor.copy()

    def _validate_player(self, player: int) -> None:
        if not 0 <= player < self.n_players:
            raise IndexError("invalid player index")

    def _validate_actions(self, actions: tuple[int, ...]) -> None:
        if len(actions) != self.n_players:
            raise ValueError("number of actions must match number of players")
        for player, action in enumerate(actions):
            if not 0 <= action < self.n_actions[player]:
                raise IndexError(f"invalid action index {action} for player {player}")
