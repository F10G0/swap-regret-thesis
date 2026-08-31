import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import ADVERSARIAL_SCALING_FIGURE_DIR, ADVERSARIAL_SCALING_RAW_DIR
from experiments.algorithm_labels import algorithm_label
from experiments.plots import confidence_free_figure_path, remove_stale_figure_pairs, save_figure_pair
from experiments.scenarios.adversarial import ENVIRONMENT_LABELS, FEEDBACK_MODE_LABELS
from experiments.scenarios.adversarial_scaling import (
    adversarial_scaling_environment_detail,
    load_adversarial_scaling_rows,
)
from experiments.result_schema import regret_sources
from metrics.confidence import mean_confidence_interval_half_width


logger = logging.getLogger(__name__)


def aggregate_scaling_regret(
    rows: list[dict[str, str]],
    source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source not in {"expected", "realized"}:
        raise ValueError(f"unknown regret source: {source}")
    action_counts = sorted({int(row["n_actions"]) for row in rows})
    samples = np.asarray(
        [
            [
                float(row[f"{source}_regret"])
                for row in rows
                if int(row["n_actions"]) == n_actions
            ]
            for n_actions in action_counts
        ]
    )
    return (
        np.asarray(action_counts),
        np.mean(samples, axis=1),
        mean_confidence_interval_half_width(samples, axis=1),
    )


def _plot_scaling(rows: list[dict[str, str]], source: str, output_path: Path) -> None:
    first = rows[0]
    action_counts, means, confidence = aggregate_scaling_regret(rows, source)
    figure, axes = plt.subplots(figsize=(7.2, 4.8))
    axes.plot(
        action_counts,
        means,
        color="#2563eb",
        marker="o",
        linewidth=2.0,
        label="Replicate mean",
    )
    axes.set_xticks(action_counts)
    axes.set_xlabel("Number of actions (K)")
    axes.set_ylabel(f"Final {source} {first['target_regret']} regret")
    environment_detail = adversarial_scaling_environment_detail(first)
    axes.set_title(
        f"{ENVIRONMENT_LABELS[first['environment']]} · {environment_detail} · "
        f"{algorithm_label(first['algorithm'])}\n"
        f"{FEEDBACK_MODE_LABELS[first['feedback_mode']]} · T={int(first['horizon']):,} · "
        f"{first['replicates']} replicates · base learner seed {first['base_learner_seed']}"
    )
    axes.grid(alpha=0.25)
    axes.legend(frameon=False)
    figure.tight_layout()
    if int(first["replicates"]) > 1:
        save_figure_pair(figure, confidence_free_figure_path(output_path), png_dpi=150, bbox_inches="tight")
        axes.fill_between(
            action_counts,
            means - confidence,
            means + confidence,
            color="#2563eb",
            alpha=0.2,
            label="Student-t 95% CI",
        )
        axes.legend(frameon=False)
        figure.tight_layout()
    save_figure_pair(figure, output_path, png_dpi=150, bbox_inches="tight")
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
        for source in regret_sources(rows[0]["regret_evaluation"]):
            output_path = output_dir / f"{input_path.stem}_{source}_regret_by_actions.png"
            _plot_scaling(rows, source, output_path)
            generated.append(output_path)

    generated_paths = generated + [
        confidence_free_figure_path(path)
        for path in generated
        if confidence_free_figure_path(path).is_file()
    ]
    remove_stale_figure_pairs(output_dir, generated_paths)
    return generated


def main() -> None:
    plot_adversarial_scaling_results()


if __name__ == "__main__":
    main()
