from collections.abc import Callable
from pathlib import Path

from algorithms.external_regret import AuerExp3, Exp3IX
from algorithms.swap_regret import BanditBM, BanditIto, LCEIX
from config import CUSTOM_GAME_DIR, HORIZON, RAW_DIR, SEED
from environments import BanditRepeatedGame
from experiments.recording import MAX_RECORDED_POINTS
from experiments.scenarios.cross_play import AlgorithmFactory, run_cross_play_experiment


ALGORITHMS = {
    "auer_exp3": AlgorithmFactory(AuerExp3, uses_horizon=True),
    "exp3_ix": AlgorithmFactory(Exp3IX, uses_horizon=True),
    "bm": AlgorithmFactory(BanditBM, uses_horizon=True),
    "ito": AlgorithmFactory(BanditIto, uses_horizon=False),
    "lce_ix": AlgorithmFactory(LCEIX, uses_horizon=False),
}


def run_bandit_cross_play_experiment(game_name: str, algorithm_names: list[str], horizon: int = HORIZON, seed: int = SEED, replicate: int = 0,
                                     output_dir: str | Path | None = None, should_cancel: Callable[[], bool] | None = None,
                                     custom_game_dir: str | Path = CUSTOM_GAME_DIR, regret_evaluation: str = "feedback_aligned", max_recorded_points: int = MAX_RECORDED_POINTS) -> Path:
    return run_cross_play_experiment(game_name=game_name, feedback_mode="bandit", algorithm_names=algorithm_names, horizon=horizon, seed=seed, replicate=replicate,
                                     environment_factory=BanditRepeatedGame, algorithm_registry=ALGORITHMS, output_dir=RAW_DIR if output_dir is None else output_dir,
                                     should_cancel=should_cancel, custom_game_dir=custom_game_dir, regret_evaluation=regret_evaluation, max_recorded_points=max_recorded_points)
