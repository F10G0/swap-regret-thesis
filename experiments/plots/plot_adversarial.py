import csv
from collections import defaultdict
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import ADVERSARIAL_FIGURE_DIR, ADVERSARIAL_RAW_DIR
from experiments.algorithm_labels import algorithm_label
from experiments.plots import confidence_free_figure_path, remove_stale_figure_pairs, save_figure_pair
from experiments.result_schema import regret_sources
from experiments.scenarios.adversarial import (
    ENVIRONMENT_LABELS,
    FEEDBACK_MODE_LABELS,
    TARGET_REGRET_BY_ALGORITHM,
    load_adversarial_rows,
)
from metrics.confidence import mean_confidence_interval_half_width


logger = logging.getLogger(__name__)
MAX_PLOT_POINTS = 2_000
ALGORITHM_COLORS = {
    "hedge": "#0072B2",
    "exp3": "#56B4E9",
    "exp3_ix": "#332288",
    "bm": "#D55E00",
    "ito": "#009E73",
    "lce_ix": "#882255",
    "regret_matching": "#CC79A7",
    "stationary_regret_matching": "#E69F00",
}
LINE_STYLES = ("-", "--", "-.", ":")


def _group_key(rows: list[dict[str, str]]) -> tuple:
    first = rows[0]
    replicate = int(first["replicate"])
    environment_seed = first["environment_seed"]
    return (
        first["environment"],
        first["initialization_mode"],
        first["reward_step"],
        first["feedback_mode"],
        first["regret_evaluation"],
        first["implementation_version"],
        first["n_actions"],
        first["algorithm"],
        first["horizon"],
        int(environment_seed) - replicate if environment_seed else None,
        int(first["learner_seed"]) - replicate,
    )


def group_adversarial_results(
    results: list[tuple[Path, list[dict[str, str]]]],
) -> list[list[list[dict[str, str]]]]:
    groups = defaultdict(list)
    for _, rows in results:
        groups[_group_key(rows)].append(rows)
    return [
        sorted(group, key=lambda rows: int(rows[0]["replicate"]))
        for _, group in sorted(groups.items())
    ]


def aggregate_adversarial_regret(
    trajectories: list[list[dict[str, str]]],
    column: str,
    scale_by_sqrt_time: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray([int(row["t"]) for row in trajectories[0]])
    values = np.asarray(
        [[float(row[column]) for row in trajectory] for trajectory in trajectories]
    )
    if scale_by_sqrt_time:
        values = values / np.sqrt(times)
    return (
        times,
        np.mean(values, axis=0),
        mean_confidence_interval_half_width(values, axis=0),
    )


def collect_adversarial_results(
    input_dir: str | Path,
    skip_invalid: bool = False,
) -> list[tuple[Path, list[dict[str, str]]]]:
    results = []
    for path in sorted(Path(input_dir).glob("*.csv")):
        try:
            rows = load_adversarial_rows(path, max_points=MAX_PLOT_POINTS)
        except (OSError, TypeError, ValueError, csv.Error) as error:
            if not skip_invalid:
                raise
            logger.warning("Skipping invalid adversarial result %s: %s", path, error)
            continue
        results.append((path, rows))
    return results


def _plot_regret(
    results: list[tuple[Path, list[dict[str, str]]]],
    environment: str,
    feedback_mode: str,
    n_actions: int,
    source: str,
    regret_name: str,
    average: bool,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(figsize=(10, 5.2))
    algorithm_occurrences = defaultdict(int)
    confidence_bands = []
    selected = [
        group
        for group in group_adversarial_results(results)
        if group[0][0]["environment"] == environment
        and group[0][0]["feedback_mode"] == feedback_mode
        and int(group[0][0]["n_actions"]) == n_actions
        and source in regret_sources(group[0][0]["regret_evaluation"])
    ]
    sort_fields = (
        "algorithm",
        "horizon",
        "learner_seed",
    )
    for trajectories in sorted(
        selected,
        key=lambda group: tuple(group[0][0][field] for field in sort_fields),
    ):
        first = trajectories[0][0]
        algorithm = first["algorithm"]
        if average:
            column = f"average_{source}_{regret_name}_regret"
        else:
            column = f"{source}_{regret_name}_regret"
        times, values, confidence = aggregate_adversarial_regret(
            trajectories,
            column,
            scale_by_sqrt_time=not average,
        )
        color = ALGORITHM_COLORS[algorithm]
        occurrence = algorithm_occurrences[algorithm]
        algorithm_occurrences[algorithm] += 1
        target = TARGET_REGRET_BY_ALGORITHM.get(algorithm) == regret_name
        replicate_count = len(trajectories)
        axes.plot(
            times,
            values,
            color=color,
            linestyle=LINE_STYLES[occurrence % len(LINE_STYLES)],
            linewidth=2.4 if target else 1.5,
            label=algorithm_label(algorithm) if occurrence == 0 else "_nolegend_",
        )
        if replicate_count > 1:
            confidence_bands.append((times, values, confidence, color))

    axes.set_xscale("log")
    axes.axhline(0.0, color="#7b8580", linewidth=0.8, linestyle="--")
    axes.set_xlabel("Round")
    view_label = "Average" if average else "Scaled"
    ylabel = (
        f"Average {source} {regret_name} regret"
        if average
        else f"{source.title()} {regret_name} regret / sqrt(t)"
    )
    axes.set_ylabel(ylabel)
    axes.set_title(
        f"{ENVIRONMENT_LABELS[environment]} · {FEEDBACK_MODE_LABELS[feedback_mode]} · "
        f"{n_actions} actions · "
        f"{view_label.lower()} {source} {regret_name} regret"
    )
    axes.grid(True)
    axes.legend(loc="best", fontsize="small", frameon=False)
    figure.tight_layout()
    if confidence_bands:
        save_figure_pair(figure, confidence_free_figure_path(output_path), png_dpi=150, bbox_inches="tight")
        for times, values, confidence, color in confidence_bands:
            axes.fill_between(times, values - confidence, values + confidence, color=color, alpha=0.14)
    save_figure_pair(figure, output_path, png_dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_adversarial_results(
    input_dir: str | Path = ADVERSARIAL_RAW_DIR,
    output_dir: str | Path = ADVERSARIAL_FIGURE_DIR,
    skip_invalid: bool = False,
) -> list[Path]:
    results = collect_adversarial_results(input_dir, skip_invalid=skip_invalid)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    environment_feedback_action_counts = {
        (
            rows[0]["environment"],
            rows[0]["feedback_mode"],
            int(rows[0]["n_actions"]),
        )
        for _, rows in results
    }
    for environment, feedback_mode, n_actions in sorted(environment_feedback_action_counts):
        sources = {
            source
            for _, rows in results
            if rows[0]["environment"] == environment
            and rows[0]["feedback_mode"] == feedback_mode
            and int(rows[0]["n_actions"]) == n_actions
            for source in regret_sources(rows[0]["regret_evaluation"])
        }
        for source in (
            source for source in ("expected", "realized") if source in sources
        ):
            for regret_name in ("external", "internal", "swap"):
                for average in (True, False):
                    if average:
                        filename = (
                            f"adversarial_{environment}_{feedback_mode}_{n_actions}_actions_average_"
                            f"{source}_{regret_name}_regret.png"
                        )
                    else:
                        filename = (
                            f"adversarial_{environment}_{feedback_mode}_{n_actions}_actions_{source}_"
                            f"{regret_name}_regret_over_sqrt_t.png"
                        )
                    output_path = output_dir / filename
                    _plot_regret(
                        results,
                        environment,
                        feedback_mode,
                        n_actions,
                        source,
                        regret_name,
                        average,
                        output_path,
                    )
                    generated.append(output_path)

    generated_paths = generated + [
        confidence_free_figure_path(path)
        for path in generated
        if confidence_free_figure_path(path).is_file()
    ]
    remove_stale_figure_pairs(output_dir, generated_paths)
    return generated


def main() -> None:
    plot_adversarial_results()


if __name__ == "__main__":
    main()
