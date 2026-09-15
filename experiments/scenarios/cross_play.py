from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from algorithms.base import Algorithm
from algorithms.external_regret import AuerExp3, Exp3IX, Hedge
from algorithms.internal_regret import RegretMatching, StationaryRegretMatching
from algorithms.swap_regret import BanditBM, BanditIto, FullBM, FullIto, LCEIX
from config import CUSTOM_GAME_DIR, HORIZON, RAW_DIR, SEED
from environments import BanditRepeatedGame, RepeatedGame
from experiments.game_catalog import load_game_payoffs, payoff_tensor_digest
from experiments.recorder import CsvRecorder
from experiments.recording import MAX_RECORDED_POINTS
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


ALGORITHMS_BY_FEEDBACK_MODE = {
    "full_information": {
        "hedge": AlgorithmFactory(Hedge, uses_horizon=True),
        "bm": AlgorithmFactory(FullBM, uses_horizon=True),
        "ito": AlgorithmFactory(FullIto, uses_horizon=False),
        "regret_matching": AlgorithmFactory(RegretMatching, uses_horizon=False),
        "stationary_regret_matching": AlgorithmFactory(StationaryRegretMatching, uses_horizon=False),
    },
    "bandit": {
        "auer_exp3": AlgorithmFactory(AuerExp3, uses_horizon=True),
        "exp3_ix": AlgorithmFactory(Exp3IX, uses_horizon=True),
        "bm": AlgorithmFactory(BanditBM, uses_horizon=True),
        "ito": AlgorithmFactory(BanditIto, uses_horizon=False),
        "lce_ix": AlgorithmFactory(LCEIX, uses_horizon=False),
    },
}
FEEDBACK_MODE_LABELS = {
    "full_information": "Full information",
    "bandit": "Bandit feedback",
}


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


def run_cross_play_experiment(game_name: str, algorithm_names: list[str], horizon: int = HORIZON, seed: int = SEED, replicate: int = 0,
                              output_dir: str | Path | None = None, should_cancel: Callable[[], bool] | None = None,
                              custom_game_dir: str | Path = CUSTOM_GAME_DIR, max_recorded_points: int = MAX_RECORDED_POINTS,
                              *, feedback_mode: str) -> Path:
    if feedback_mode not in ALGORITHMS_BY_FEEDBACK_MODE:
        raise ValueError(f"unknown feedback mode: {feedback_mode}")
    algorithm_registry = ALGORITHMS_BY_FEEDBACK_MODE[feedback_mode]
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
    game = (RepeatedGame if feedback_mode == "full_information" else BanditRepeatedGame)(payoff_tensor)
    players = [
        algorithm_registry[name].create(n_actions, horizon, player_seed(spec, player_id))
        for player_id, (name, n_actions) in enumerate(zip(spec.algorithm_names, game.n_actions))
    ]
    output_path = Path(RAW_DIR if output_dir is None else output_dir) / f"{spec.run_id}.csv"

    if output_path.exists():
        raise FileExistsError(f"experiment {spec.run_id} already exists at {output_path}")

    fieldnames = regret_fieldnames()
    with CsvRecorder(fieldnames, output_path) as recorder:
        run_game(
            game_name=spec.game_name, feedback_mode=spec.feedback_mode, algorithm_name=spec.algorithm_profile_name, game=game, players=players, recorder=recorder, horizon=spec.horizon,
            metadata=spec.metadata(), should_cancel=should_cancel,
            max_recorded_points=max_recorded_points,
        )

    return output_path
