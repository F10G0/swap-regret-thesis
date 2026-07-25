import numpy as np

from environments.base import FixedGameEnvironment


class RepeatedGame(FixedGameEnvironment):
    """Repeated game with full-information feedback."""

    def feedback(self, player: int) -> np.ndarray:
        return self.deviation_payoffs(player)


class BanditRepeatedGame(FixedGameEnvironment):
    """Repeated game with bandit feedback."""

    def feedback(self, player: int) -> float:
        self._validate_player(player)
        return float(self.payoff_tensor[(player, *self.actions)])
