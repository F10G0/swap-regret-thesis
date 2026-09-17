from collections.abc import Callable

import numpy as np

from algorithms.base import Algorithm
from environments.base import FixedGameEnvironment
from experiments.recorder import CsvRecorder
from experiments.recording import (
    MAX_RECORDED_POINTS,
    encode_joint_action_histograms,
    joint_action_histogram_checkpoints,
    recording_checkpoints,
)
from experiments.result_schema import JOINT_ACTION_HISTOGRAM_FIELD
from metrics.regret import RegretBundle


class ExperimentCancelled(RuntimeError):
    pass


def round_progress_interval(horizon: int) -> int:
    return max(10_000, horizon // 100)


def run_game(game_name: str, feedback_mode: str, game: FixedGameEnvironment, algorithm_name: str, players: list[Algorithm], recorder: CsvRecorder, horizon: int,
             metadata: dict | None = None, should_cancel: Callable[[], bool] | None = None,
             max_recorded_points: int = MAX_RECORDED_POINTS, report_rounds: Callable[[int], None] | None = None) -> None:
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
    regrets = [RegretBundle(n_actions) for n_actions in game.n_actions]
    checkpoints = set(recording_checkpoints(horizon, max_recorded_points))
    histogram_checkpoints = set(joint_action_histogram_checkpoints(horizon))
    joint_action_counts = np.zeros(game.n_actions, dtype=np.int64)
    histogram_horizons = []
    histogram_counts = []
    progress_interval = round_progress_interval(horizon)
    reported_rounds = 0

    for t in range(1, horizon + 1):
        if should_cancel is not None and should_cancel():
            raise ExperimentCancelled("experiment cancelled")
        actions = tuple(player.sample_action() for player in players)
        game.step(actions)
        joint_action_counts[actions] += 1
        if t in histogram_checkpoints:
            histogram_horizons.append(t)
            histogram_counts.append(joint_action_counts.copy())

        for player_id, (player, action) in enumerate(zip(players, actions)):
            feedback = game.feedback(player_id)
            regret = regrets[player_id]

            if feedback_mode == "full_information":
                payoff = float(feedback[action])
                deviation_payoffs = feedback
            else:
                payoff = feedback
                deviation_payoffs = game.deviation_payoffs(player_id)

            regret.update(action, deviation_payoffs)
            player.update(feedback)
            if t not in checkpoints:
                continue
            regret_summary = regret.summary(t)
            histograms = {}
            if player_id == 0 and t == horizon:
                histograms[JOINT_ACTION_HISTOGRAM_FIELD] = encode_joint_action_histograms(
                    histogram_horizons, histogram_counts
                )

            recorder.record({
                "game": game_name,
                "algorithm": algorithm_name,
                **metadata,
                "t": t,
                "player": player_id,
                "action": action,
                "payoff": payoff,
                **histograms,
                **regret_summary,
            })

        if report_rounds is not None and (t - reported_rounds >= progress_interval or t == horizon):
            report_rounds(t - reported_rounds)
            reported_rounds = t
