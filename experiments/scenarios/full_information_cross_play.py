from collections.abc import Callable
from pathlib import Path

from algorithms.external_regret import Hedge
from algorithms.internal_regret import RegretMatching, StationaryRegretMatching
from algorithms.swap_regret import FullBM, FullIto
from config import CUSTOM_GAME_DIR, HORIZON, RAW_DIR, SEED
from environments import RepeatedGame
from experiments.scenarios.cross_play import AlgorithmFactory, run_cross_play_experiment


ALGORITHMS = {
    "hedge": AlgorithmFactory(Hedge),
    "bm": AlgorithmFactory(FullBM),
    "ito": AlgorithmFactory(FullIto, uses_horizon=False),
    "regret_matching": AlgorithmFactory(RegretMatching, uses_horizon=False),
    "stationary_regret_matching": AlgorithmFactory(StationaryRegretMatching, uses_horizon=False),
}


def run_full_information_cross_play_experiment(game_name: str, algorithm_names: list[str], horizon: int = HORIZON, seed: int = SEED, replicate: int = 0,
                                               output_dir: str | Path | None = None, should_cancel: Callable[[], bool] | None = None,
                                               custom_game_dir: str | Path = CUSTOM_GAME_DIR, regret_evaluation: str = "feedback_aligned") -> Path:
    return run_cross_play_experiment(game_name=game_name, feedback_mode="full_information", algorithm_names=algorithm_names, horizon=horizon, seed=seed, replicate=replicate,
                                     environment_factory=RepeatedGame, algorithm_registry=ALGORITHMS, output_dir=RAW_DIR if output_dir is None else output_dir,
                                     should_cancel=should_cancel, custom_game_dir=custom_game_dir, regret_evaluation=regret_evaluation)
