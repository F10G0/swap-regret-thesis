import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from operator import index
from pathlib import Path
import tempfile

from experiments.games import PAYOFF_FACTORIES
from experiments.plots.plot_equilibrium_weights import plot_equilibrium_profile_weights
from web.equilibrium_figures import PRECOMPUTED_EQUILIBRIUM_DIR, equilibrium_figure_filename
from web.presentations import GAME_PRESENTATIONS


def _precompute_equilibrium_figure(game_name: str, equilibrium: str, output_path: Path) -> Path:
    payoff_tensor = PAYOFF_FACTORIES[game_name]()
    with tempfile.TemporaryDirectory(prefix=".precompute-equilibrium-", dir=output_path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory) / output_path.name
        plot_equilibrium_profile_weights(
            payoff_tensor,
            equilibrium,
            temporary_path,
            game_name=GAME_PRESENTATIONS.get(game_name, {}).get("label", game_name),
        )
        os.replace(temporary_path, output_path)
    return output_path


def precompute_equilibrium_figures(game_names=None, equilibria=("ce", "cce"), output_dir: str | Path = PRECOMPUTED_EQUILIBRIUM_DIR,
                                   overwrite: bool = False, workers: int = 1) -> list[Path]:
    selected_games = list(PAYOFF_FACTORIES) if game_names is None else list(game_names)
    unknown_games = set(selected_games) - PAYOFF_FACTORIES.keys()
    if unknown_games:
        raise ValueError(f"unknown games: {', '.join(sorted(unknown_games))}")
    if not set(equilibria) <= {"ce", "cce"}:
        raise ValueError("equilibria must contain only ce and cce")
    try:
        workers = index(workers)
    except TypeError as error:
        raise ValueError("workers must be a positive integer") from error
    if workers <= 0:
        raise ValueError("workers must be a positive integer")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    tasks = []
    for game_name in selected_games:
        if PAYOFF_FACTORIES[game_name]().ndim != 3:
            continue
        for equilibrium in equilibria:
            output_path = output_dir / equilibrium_figure_filename(game_name, equilibrium)
            if output_path.is_file() and not overwrite:
                generated.append(output_path)
                continue
            tasks.append((game_name, equilibrium, output_path))

    if workers == 1:
        generated.extend(_precompute_equilibrium_figure(*task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            generated.extend(executor.map(_precompute_equilibrium_figure_task, tasks))
    return generated


def _precompute_equilibrium_figure_task(task) -> Path:
    return _precompute_equilibrium_figure(*task)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute static CE/CCE profile-weight heatmaps.")
    parser.add_argument("--game", action="append", choices=PAYOFF_FACTORIES)
    parser.add_argument("--equilibrium", action="append", choices=("ce", "cce"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    for path in precompute_equilibrium_figures(
        arguments.game,
        tuple(arguments.equilibrium or ("ce", "cce")),
        overwrite=arguments.force,
        workers=arguments.workers,
    ):
        print(path)


if __name__ == "__main__":
    main()
