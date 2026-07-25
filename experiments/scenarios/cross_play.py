from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from algorithms.base import Algorithm
from environments.base import FixedGameEnvironment
from experiments.games import PAYOFF_FACTORIES
from experiments.recorder import CsvRecorder
from experiments.runner import run_game
from experiments.scenarios.fieldnames import regret_fieldnames
from experiments.spec import ExperimentSpec


@dataclass(frozen=True)
class AlgorithmFactory:
    algorithm_class: type[Algorithm]
    uses_horizon: bool = True

    def create(self, n_actions: int, horizon: int, seed: int) -> Algorithm:
        if self.uses_horizon:
            return self.algorithm_class(n_actions=n_actions, horizon=horizon, seed=seed)
        return self.algorithm_class(n_actions=n_actions, seed=seed)


def player_seed(spec: ExperimentSpec, player_id: int) -> int:
    """Return a reproducible seed for one player and replicate."""
    return spec.seed + spec.replicate * len(spec.algorithm_names) + player_id


def run_cross_play_experiment(game_name: str, feedback_mode: str, algorithm_names: list[str], horizon: int, seed: int, replicate: int,
                              environment_factory: Callable[[np.ndarray], FixedGameEnvironment], algorithm_registry: dict[str, AlgorithmFactory], output_dir: str | Path,
                              should_cancel: Callable[[], bool] | None = None) -> Path:
    if game_name not in PAYOFF_FACTORIES:
        raise ValueError(f"unknown game: {game_name}")
    for name in algorithm_names:
        if name not in algorithm_registry:
            raise ValueError(f"unknown algorithm: {name}")

    spec = ExperimentSpec(game_name, feedback_mode, tuple(algorithm_names), horizon, seed, replicate)
    game = environment_factory(PAYOFF_FACTORIES[game_name]())
    players = [
        algorithm_registry[name].create(n_actions, horizon, player_seed(spec, player_id))
        for player_id, (name, n_actions) in enumerate(zip(spec.algorithm_names, game.n_actions))
    ]
    output_path = Path(output_dir) / f"{spec.run_id}.csv"

    if output_path.exists():
        raise FileExistsError(f"experiment {spec.run_id} already exists at {output_path}")

    with CsvRecorder(regret_fieldnames(spec.feedback_mode), output_path) as recorder:
        run_game(
            game_name=spec.game_name, feedback_mode=spec.feedback_mode, algorithm_name=spec.algorithm_profile_name, game=game, players=players, recorder=recorder, horizon=spec.horizon,
            metadata=spec.metadata(), should_cancel=should_cancel,
        )

    return output_path


def main() -> None:
    from experiments.scenarios.bandit_cross_play import main as run_bandit_cross_play
    from experiments.scenarios.full_information_cross_play import main as run_full_information_cross_play

    run_full_information_cross_play()
    run_bandit_cross_play()


if __name__ == "__main__":
    main()
