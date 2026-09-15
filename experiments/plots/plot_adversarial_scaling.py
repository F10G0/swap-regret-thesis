import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import ADVERSARIAL_SCALING_FIGURE_DIR, ADVERSARIAL_SCALING_RAW_DIR
from experiments.algorithm_labels import algorithm_label
from experiments.plots import remove_stale_figure_pairs, save_figure_pair
from experiments.plots.style import publication_plot, algorithm_style, regret_axis_label, finish_line_figure
from experiments.scenarios.adversarial_scaling import (
    load_adversarial_scaling_rows,
)


logger = logging.getLogger(__name__)


def aggregate_scaling_regret(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    action_counts = sorted({int(row["n_actions"]) for row in rows})
    samples = np.asarray(
        [
            [
                float(row[f"{row['target_regret']}_regret"])
                for row in rows
                if int(row["n_actions"]) == n_actions
            ]
            for n_actions in action_counts
        ]
    )
    return (
        np.asarray(action_counts),
        np.mean(samples, axis=1),
    )


@publication_plot
def _plot_scaling(rows: list[dict[str, str]], output_path: Path) -> None:
    first = rows[0]
    action_counts, means = aggregate_scaling_regret(rows)
    figure, axes = plt.subplots()
    axes.plot(
        action_counts,
        means,
        **algorithm_style(first["algorithm"]),
        label=algorithm_label(first["algorithm"]),
    )
    axes.set_xticks(action_counts)
    axes.set_xlabel(r"Actions $K$")
    axes.set_ylabel(regret_axis_label(first["target_regret"], "final"))
    finish_line_figure(figure, axes)
    save_figure_pair(figure, output_path)
    plt.close(figure)


def plot_adversarial_scaling_results(
    input_dir: str | Path = ADVERSARIAL_SCALING_RAW_DIR,
    output_dir: str | Path = ADVERSARIAL_SCALING_FIGURE_DIR,
    skip_invalid: bool = False,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for input_path in sorted(Path(input_dir).glob("*.csv")):
        try:
            rows = load_adversarial_scaling_rows(input_path)
        except (OSError, TypeError, ValueError, csv.Error) as error:
            if not skip_invalid:
                raise
            logger.warning(
                "Skipping invalid action-space scaling result %s: %s",
                input_path,
                error,
            )
            continue
        output_path = output_dir / f"{input_path.stem}_regret_by_actions.png"
        _plot_scaling(rows, output_path)
        generated.append(output_path)

    remove_stale_figure_pairs(output_dir, generated)
    return generated


def main() -> None:
    plot_adversarial_scaling_results()


if __name__ == "__main__":
    main()
