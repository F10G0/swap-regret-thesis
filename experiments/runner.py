from collections.abc import Callable

from algorithms.base import Algorithm
from environments.base import FixedGameEnvironment
from experiments.recorder import CsvRecorder
from experiments.result_schema import regret_sources, resolve_regret_evaluation
from metrics.regret import RegretBundles


class ExperimentCancelled(RuntimeError):
    pass


def run_game(game_name: str, feedback_mode: str, game: FixedGameEnvironment, algorithm_name: str, players: list[Algorithm], recorder: CsvRecorder, horizon: int,
             metadata: dict | None = None, should_cancel: Callable[[], bool] | None = None, regret_evaluation: str = "feedback_aligned") -> None:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if feedback_mode not in {"full_information", "bandit"}:
        raise ValueError(f"unknown feedback mode: {feedback_mode}")
    if len(players) != game.n_players:
        raise ValueError("number of players must match game.n_players")
    for player_id, player in enumerate(players):
        if player.n_actions != game.n_actions[player_id]:
            raise ValueError(f"player {player_id} action count does not match the environment")

    metadata = metadata or {}
    regret_evaluation = resolve_regret_evaluation(feedback_mode, regret_evaluation)
    regrets = [RegretBundles(n_actions) for n_actions in game.n_actions]

    for t in range(1, horizon + 1):
        if should_cancel is not None and should_cancel():
            raise ExperimentCancelled("experiment cancelled")
        actions = tuple(player.sample_action() for player in players)
        game.step(actions)

        for player_id, (player, action) in enumerate(zip(players, actions)):
            strategy = player.strategy()
            feedback = game.feedback(player_id)
            regret = regrets[player_id]

            if feedback_mode == "full_information":
                payoff = float(feedback[action])
                deviation_payoffs = feedback
            else:
                payoff = feedback
                deviation_payoffs = game.deviation_payoffs(player_id)

            regret.update(strategy, action, deviation_payoffs)
            player.update(feedback)
            regret_summary = {}
            for source in regret_sources(regret_evaluation):
                regret_summary.update(getattr(regret, source).summary(t))

            recorder.record({
                "game": game_name,
                "algorithm": algorithm_name,
                **metadata,
                "t": t,
                "player": player_id,
                "action": action,
                "payoff": payoff,
                **regret_summary,
            })
