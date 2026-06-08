from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlayerOutcome:
    """
    Complete outcome for one player under a realized joint action.
    """

    player: int
    actions: tuple[int, ...]
    payoff: float
    payoff_vector: np.ndarray


class MultiPlayerEnvironment(ABC):
    """
    Base class for multi-player environments.
    """

    @property
    @abstractmethod
    def n_players(self) -> int:
        pass

    @property
    @abstractmethod
    def n_actions(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def evaluate(self, player: int, actions: tuple[int, ...]) -> PlayerOutcome:
        """
        Return the full outcome for evaluation and regret computation.

        This method should not be used by learning algorithms.
        """
        pass

    @abstractmethod
    def observe(self, player: int, actions: tuple[int, ...]) -> float | np.ndarray:
        """
        Return the feedback available to the learning algorithm.
        """
        pass

    def _check_player(self, player: int) -> None:
        if player < 0 or player >= self.n_players:
            raise IndexError("invalid player index")

    def _check_actions(self, actions: tuple[int, ...]) -> None:
        if len(actions) != self.n_players:
            raise ValueError("number of actions must match number of players")
        for player, action in enumerate(actions):
            if action < 0 or action >= self.n_actions[player]:
                raise IndexError(f"invalid action index {action} for player {player}")


class FixedGameEnvironment(MultiPlayerEnvironment):
    """
    Multi-player environment defined by a fixed payoff tensor.

    payoffs[i][a_1, ..., a_n] is the payoff of player i
    under the joint action (a_1, ..., a_n).
    """

    def __init__(self, payoffs: np.ndarray):
        payoffs = np.asarray(payoffs, dtype=float)

        if payoffs.ndim < 2:
            raise ValueError("payoffs must have shape (n_players, action_1, ..., action_n)")
        if payoffs.shape[0] == 0:
            raise ValueError("number of players must be non-zero")
        if any(size == 0 for size in payoffs.shape[1:]):
            raise ValueError("each player must have at least one action")
        if payoffs.shape[0] != len(payoffs.shape[1:]):
            raise ValueError("number of players must match number of action dimensions")

        self._payoffs = payoffs

    @property
    def n_players(self) -> int:
        return self._payoffs.shape[0]

    @property
    def n_actions(self) -> tuple[int, ...]:
        return self._payoffs.shape[1:]
    
    def evaluate(self, player: int, actions: tuple[int, ...]) -> PlayerOutcome:
        self._check_player(player)
        self._check_actions(actions)

        indices = list(actions)
        indices[player] = slice(None)
        payoff_vector = self._payoffs[(player, *indices)].copy()
        payoff = float(payoff_vector[actions[player]])

        return PlayerOutcome(
            player=player,
            actions=actions,
            payoff=payoff,
            payoff_vector=payoff_vector,
        )
