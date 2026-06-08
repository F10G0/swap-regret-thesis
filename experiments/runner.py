import numpy as np

from algorithms.base import Algorithm
from environments.base import FixedGameEnvironment
from experiments.recorder import CsvRecorder
from metrics import RegretBundle


def run_game(game_name: str, game: FixedGameEnvironment, algorithm_name: str, players: list[Algorithm], recorder: CsvRecorder, horizon: int) -> None:
    n_players = game.n_players
    if len(players) != n_players:
        raise ValueError("number of players must match game.n_players")

    regrets = [
        RegretBundle(game.n_actions[player_id])
        for player_id in range(n_players)
    ]

    for t in range(1, horizon + 1):
        actions = tuple(player.sample_action() for player in players)

        for player_id, player in enumerate(players):
            outcome = game.evaluate(player=player_id, actions=actions)
            regrets[player_id].update(player.strategy(), outcome)

            recorder.record(
                {
                    "game": game_name,
                    "algorithm": algorithm_name,
                    "t": t,
                    
                    "player": player_id,
                    "action": actions[player_id],
                    "payoff": outcome.payoff,

                    **regrets[player_id].summary(t),
                }
            )
            
            feedback = game.observe(player=player_id, actions=actions)
            if isinstance(feedback, np.ndarray):
                player.update(feedback)
            else:
                player.update(actions[player_id], feedback)
