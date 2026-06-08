import numpy as np

from environments.base import FixedGameEnvironment


class RepeatedGame(FixedGameEnvironment):
    """
    Full-information repeated game.

    Each player observes the payoff of all its actions against
    the realized actions of the other players.
    """

    def observe(self, player: int, actions: tuple[int, ...]) -> np.ndarray:
        return self.evaluate(player, actions).payoff_vector

    def __repr__(self) -> str:
        return (
            f"RepeatedGame("
            f"n_players={self.n_players}, "
            f"n_actions={self.n_actions})"
        )
