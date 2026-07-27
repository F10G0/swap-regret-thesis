from collections.abc import Iterable
from itertools import chain
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.games import PAYOFF_FACTORIES
from experiments.plots import HEATMAP_COLORMAP
from experiments.results import iter_result_rows


def joint_action_distribution(input_path: str | Path) -> tuple[str, np.ndarray]:
    input_path = Path(input_path)
    rows = iter_result_rows(input_path)
    first_row = next(rows, None)
    if first_row is None:
        raise ValueError("result file has no rows")

    game_name = first_row["game"]
    action_counts = PAYOFF_FACTORIES[game_name]().shape[1:]
    n_players = len(action_counts)
    counts = np.zeros(action_counts, dtype=int)
    current_time = None
    actions = {}

    for row in chain((first_row,), rows):
        time = int(row["t"])
        if current_time is not None and time != current_time:
            if len(actions) != n_players:
                raise ValueError(f"round {current_time} has incomplete actions")
            counts[actions[0], actions[1]] += 1
            actions = {}
        current_time = time
        actions[int(row["player"])] = int(row["action"])

    if len(actions) != n_players:
        raise ValueError(f"round {current_time} has incomplete actions")
    counts[actions[0], actions[1]] += 1
    return game_name, counts / np.sum(counts)


def mean_joint_action_distribution(input_paths: Iterable[str | Path]) -> tuple[str, np.ndarray, int]:
    distributions = [joint_action_distribution(path) for path in input_paths]
    if not distributions:
        raise ValueError("at least one result file is required")
    game_name = distributions[0][0]
    if any(game != game_name for game, _ in distributions):
        raise ValueError("joint-action results must use the same game")
    return game_name, np.mean([distribution for _, distribution in distributions], axis=0), len(distributions)


def plot_joint_actions(input_paths: str | Path | Iterable[str | Path], output_path: str | Path) -> None:
    paths = [input_paths] if isinstance(input_paths, (str, Path)) else list(input_paths)
    game_name, frequencies, n_replicates = mean_joint_action_distribution(paths)
    action_counts = frequencies.shape
    output_path = Path(output_path)
    figure, axes = plt.subplots(figsize=(6.5, 5.5))
    image = axes.imshow(frequencies, cmap=HEATMAP_COLORMAP, origin="lower", vmin=0.0, vmax=max(float(np.max(frequencies)), 1.0 / frequencies.size))
    axes.set_xlabel("Player 1 action")
    axes.set_ylabel("Player 0 action")
    title = "empirical joint-action distribution" if n_replicates == 1 else f"mean empirical joint-action distribution ({n_replicates} replicates)"
    axes.set_title(f"{game_name}: {title}")
    axes.set_xticks(range(action_counts[1]))
    axes.set_yticks(range(action_counts[0]))

    if frequencies.size <= 100:
        for action_0 in range(action_counts[0]):
            for action_1 in range(action_counts[1]):
                value = frequencies[action_0, action_1]
                axes.text(action_1, action_0, f"{100.0 * value:.1f}%", ha="center", va="center", fontsize=7)

    figure.colorbar(image, ax=axes, label="Empirical frequency")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
