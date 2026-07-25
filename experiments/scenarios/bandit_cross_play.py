from collections.abc import Callable
from pathlib import Path

from algorithms.external_regret import Exp3, Exp3IX
from algorithms.swap_regret import BanditBM, BanditIto, LCEIX
from config import BANDIT_REPLICATES, HORIZON, RAW_DIR, SEED
from environments import BanditRepeatedGame
from experiments.games import PAYOFF_FACTORIES
from experiments.scenarios.cross_play import AlgorithmFactory, run_cross_play_experiment


ALGORITHMS = {
    "exp3": AlgorithmFactory(Exp3),
    "exp3_ix": AlgorithmFactory(Exp3IX),
    "bm": AlgorithmFactory(BanditBM),
    "ito": AlgorithmFactory(BanditIto, uses_horizon=False),
    "lce_ix": AlgorithmFactory(LCEIX, uses_horizon=False),
}


def run_bandit_cross_play_experiment(game_name: str, algorithm_names: list[str], horizon: int = HORIZON, seed: int = SEED, replicate: int = 0,
                                     output_dir: str | Path | None = None, should_cancel: Callable[[], bool] | None = None) -> Path:
    return run_cross_play_experiment(game_name=game_name, feedback_mode="bandit", algorithm_names=algorithm_names, horizon=horizon, seed=seed, replicate=replicate,
                                     environment_factory=BanditRepeatedGame, algorithm_registry=ALGORITHMS, output_dir=RAW_DIR if output_dir is None else output_dir,
                                     should_cancel=should_cancel)


def run_bandit_cross_play_replicates(game_name: str, algorithm_names: list[str], n_replicates: int = BANDIT_REPLICATES, horizon: int = HORIZON, seed: int = SEED,
                                     output_dir: str | Path | None = None) -> list[Path]:
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")

    return [
        run_bandit_cross_play_experiment(game_name=game_name, algorithm_names=algorithm_names, horizon=horizon, seed=seed, replicate=replicate, output_dir=output_dir)
        for replicate in range(n_replicates)
    ]


def main() -> None:
    for game_name in PAYOFF_FACTORIES:
        for algorithm_name_0 in ALGORITHMS:
            for algorithm_name_1 in ALGORITHMS:
                output_paths = run_bandit_cross_play_replicates(game_name=game_name, algorithm_names=[algorithm_name_0, algorithm_name_1])
                for output_path in output_paths:
                    print(f"[done] {output_path.stem}")


if __name__ == "__main__":
    main()
