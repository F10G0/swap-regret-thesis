from itertools import chain
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.games import PAYOFF_FACTORIES
from experiments.results import EXPERIMENT_PLAYERS, iter_result_rows


def plot_joint_actions(input_path: str | Path, output_path: str | Path) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    rows = iter_result_rows(input_path)
    first_row = next(rows, None)
    if first_row is None:
        raise ValueError("result file has no rows")

    game_name = first_row["game"]
    action_counts = PAYOFF_FACTORIES[game_name]().shape[1:]
    counts = np.zeros(action_counts, dtype=int)
    current_time = None
    actions = {}

    for row in chain((first_row,), rows):
        time = int(row["t"])
        if current_time is not None and time != current_time:
            if len(actions) != EXPERIMENT_PLAYERS:
                raise ValueError(f"round {current_time} has incomplete actions")
            counts[actions[0], actions[1]] += 1
            actions = {}
        current_time = time
        actions[int(row["player"])] = int(row["action"])

    if len(actions) != EXPERIMENT_PLAYERS:
        raise ValueError(f"round {current_time} has incomplete actions")
    counts[actions[0], actions[1]] += 1

    frequencies = counts / np.sum(counts)
    figure, axes = plt.subplots(figsize=(6.5, 5.5))
    image = axes.imshow(frequencies, cmap="YlGn", vmin=0.0, vmax=max(float(np.max(frequencies)), 1.0 / frequencies.size))
    axes.set_xlabel("Player 1 action")
    axes.set_ylabel("Player 0 action")
    axes.set_title(f"{game_name}: empirical joint-action distribution")
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
