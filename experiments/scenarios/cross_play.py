from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from algorithms.base import Algorithm
from config import CUSTOM_GAME_DIR
from environments.base import FixedGameEnvironment
from experiments.game_catalog import load_game_payoffs, payoff_tensor_digest
from experiments.recorder import CsvRecorder
from experiments.recording import MAX_RECORDED_POINTS, recording_checkpoints
from experiments.runner import run_game
from experiments.result_schema import regret_fieldnames
from experiments.spec import ExperimentSpec


@dataclass(frozen=True)
class AlgorithmFactory:
    """Declare whether a learner's parameters use the experiment horizon."""

    algorithm_class: type[Algorithm]
    uses_horizon: bool

    def create(self, n_actions: int, horizon: int, seed: int) -> Algorithm:
        if self.uses_horizon:
            return self.algorithm_class(n_actions=n_actions, horizon=horizon, seed=seed)
        return self.algorithm_class(n_actions=n_actions, seed=seed)


def player_seed(spec: ExperimentSpec, player_id: int) -> int:
    """Return a reproducible seed for one player and replicate."""
    return replicate_player_seeds(
        spec.seed,
        spec.replicate,
        len(spec.algorithm_names),
    )[player_id]


def replicate_player_seeds(
    base_seed: int,
    replicate: int,
    n_players: int,
) -> tuple[int, ...]:
    """Return the complete deterministic player-seed schedule for a replicate."""
    if base_seed < 0 or replicate < 0 or n_players <= 0:
        raise ValueError("seed inputs must be non-negative and include players")
    first_seed = base_seed + replicate * n_players
    return tuple(first_seed + player for player in range(n_players))


def run_cross_play_experiment(game_name: str, feedback_mode: str, algorithm_names: list[str], horizon: int, seed: int, replicate: int,
                              environment_factory: Callable[[np.ndarray], FixedGameEnvironment], algorithm_registry: dict[str, AlgorithmFactory], output_dir: str | Path,
                              should_cancel: Callable[[], bool] | None = None, custom_game_dir: str | Path = CUSTOM_GAME_DIR,
                              max_recorded_points: int = MAX_RECORDED_POINTS) -> Path:
    for name in algorithm_names:
        if name not in algorithm_registry:
            raise ValueError(f"unknown algorithm: {name}")

    payoff_tensor = load_game_payoffs(game_name, custom_game_dir)
    if len(algorithm_names) != payoff_tensor.shape[0]:
        raise ValueError(f"game {game_name} requires {payoff_tensor.shape[0]} player algorithms")
    spec = ExperimentSpec(
        game_name,
        feedback_mode,
        tuple(algorithm_names),
        horizon,
        seed,
        replicate,
        game_payoff_digest=payoff_tensor_digest(payoff_tensor),
    )
    game = environment_factory(payoff_tensor)
    players = [
        algorithm_registry[name].create(n_actions, horizon, player_seed(spec, player_id))
        for player_id, (name, n_actions) in enumerate(zip(spec.algorithm_names, game.n_actions))
    ]
    output_path = Path(output_dir) / f"{spec.run_id}.csv"

    if output_path.exists():
        raise FileExistsError(f"experiment {spec.run_id} already exists at {output_path}")

    checkpoints = recording_checkpoints(horizon, max_recorded_points)
    fieldnames = regret_fieldnames()
    if len(checkpoints) < horizon:
        fieldnames = fieldnames + ["action_history"]
    with CsvRecorder(fieldnames, output_path) as recorder:
        run_game(
            game_name=spec.game_name, feedback_mode=spec.feedback_mode, algorithm_name=spec.algorithm_profile_name, game=game, players=players, recorder=recorder, horizon=spec.horizon,
            metadata=spec.metadata(), should_cancel=should_cancel,
            max_recorded_points=max_recorded_points,
        )

    return output_path
