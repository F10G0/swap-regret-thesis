from environments.base import FixedGameEnvironment


class BanditRepeatedGame(FixedGameEnvironment):
    """
    Bandit-feedback repeated game.

    Each player observes only the payoff of its realized action.
    """

    def observe(self, player: int, actions: tuple[int, ...]) -> float:
        return self.evaluate(player, actions).payoff

    def __repr__(self) -> str:
        return (
            f"BanditRepeatedGame("
            f"n_players={self.n_players}, "
            f"n_actions={self.n_actions})"
        )
