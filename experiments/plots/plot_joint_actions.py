from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import CUSTOM_GAME_DIR
from experiments.game_catalog import load_game_payoffs
from experiments.plots import save_figure_pair
from experiments.plots.style import publication_plot, HEATMAP_FIGURE_SIZE, heatmap
from experiments.results import iter_result_rows
from experiments.result_trajectories import load_result_empirical_distribution_trajectory


def joint_action_distribution(input_path: str | Path, custom_game_dir: str | Path = CUSTOM_GAME_DIR) -> tuple[str, np.ndarray]:
    input_path = Path(input_path)
    rows = iter_result_rows(input_path)
    first_row = next(rows, None)
    if first_row is None:
        raise ValueError("result file has no rows")

    game_name = first_row["game"]
    action_counts = load_game_payoffs(game_name, custom_game_dir).shape[1:]
    if len(action_counts) != 2:
        raise ValueError("joint-action heatmaps require exactly two players")
    empirical = load_result_empirical_distribution_trajectory(input_path, action_counts)
    return game_name, empirical.distributions[-1]


def mean_joint_action_distribution(input_paths: Iterable[str | Path], custom_game_dir: str | Path = CUSTOM_GAME_DIR) -> tuple[str, np.ndarray, int]:
    distributions = [joint_action_distribution(path, custom_game_dir) for path in input_paths]
    if not distributions:
        raise ValueError("at least one result file is required")
    game_name = distributions[0][0]
    if any(game != game_name for game, _ in distributions):
        raise ValueError("joint-action results must use the same game")
    return game_name, np.mean([distribution for _, distribution in distributions], axis=0), len(distributions)


@publication_plot
def plot_joint_actions(input_paths: str | Path | Iterable[str | Path], output_path: str | Path,
                       custom_game_dir: str | Path = CUSTOM_GAME_DIR,
                       information_rows: list[tuple[str, str]] | None = None) -> None:
    paths = [input_paths] if isinstance(input_paths, (str, Path)) else list(input_paths)
    _, frequencies, _ = mean_joint_action_distribution(paths, custom_game_dir)
    output_path = Path(output_path)
    figure, axes = plt.subplots(figsize=HEATMAP_FIGURE_SIZE)
    image = heatmap(axes, frequencies, vmax=max(float(np.max(frequencies)), 1.0 / frequencies.size),
                    label_format=lambda value: f"{100.0 * value:.1f}%")
    axes.set_xlabel("Player 1 action")
    axes.set_ylabel("Player 0 action")
    colorbar = figure.colorbar(image, ax=axes, label="Empirical frequency")
    colorbar.solids.set_rasterized(False)
    figure.tight_layout()
    if information_rows is None:
        save_figure_pair(figure, output_path)
    else:
        save_figure_pair(figure, output_path, information_rows=information_rows)
    plt.close(figure)
