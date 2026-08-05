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
from experiments.plots import FIGURE_SUFFIXES, save_figure_pair
from experiments.scenarios.adversarial import (
    ENVIRONMENT_LABELS,
    FEEDBACK_MODE_LABELS,
    HISTORICAL_FREQUENCY_ENVIRONMENT,
    TARGET_REGRET_BY_ALGORITHM,
    adversarial_environment_detail,
    load_adversarial_rows,
)


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
    n_actions: int,
    source: str,
    regret_name: str,
    average: bool,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(figsize=(10, 5.2))
    algorithm_occurrences = defaultdict(int)
    selected = [rows for _, rows in results if int(rows[0]["n_actions"]) == n_actions]
    sort_fields = (
        "environment",
        "feedback_mode",
        "algorithm",
        "horizon",
        "learner_seed",
    )
    for rows in sorted(selected, key=lambda item: tuple(item[0][field] for field in sort_fields)):
        first = rows[0]
        algorithm = first["algorithm"]
        environment_detail = adversarial_environment_detail(first, include_environment_seed=True)
        times = np.asarray([int(row["t"]) for row in rows])
        if average:
            column = f"average_{source}_{regret_name}_regret"
            values = np.asarray([float(row[column]) for row in rows])
        else:
            column = f"{source}_{regret_name}_regret"
            values = np.asarray([float(row[column]) for row in rows]) / np.sqrt(times)
        color = ALGORITHM_COLORS[algorithm]
        occurrence = algorithm_occurrences[algorithm]
        algorithm_occurrences[algorithm] += 1
        target = TARGET_REGRET_BY_ALGORITHM.get(algorithm) == regret_name
        label = (
            f"{algorithm_label(algorithm)} · "
            f"{FEEDBACK_MODE_LABELS[first['feedback_mode']]} · "
            f"{ENVIRONMENT_LABELS[first['environment']]} · "
            f"{environment_detail} · "
            f"T={int(first['horizon']):,} · learner seed {first['learner_seed']}"
        )
        axes.plot(
            times,
            values,
            color=color,
            linestyle=LINE_STYLES[occurrence % len(LINE_STYLES)],
            linewidth=2.4 if target else 1.5,
            label=label,
        )

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
        f"Adversarial experiments · {n_actions} actions · "
        f"{view_label.lower()} {source} {regret_name} regret"
    )
    axes.grid(True)
    axes.legend(loc="best", fontsize="small", frameon=False)
    figure.tight_layout()
    save_figure_pair(figure, output_path, png_dpi=150, bbox_inches="tight")
    plt.close(figure)


def _frequency_curves(
    input_path: Path,
    metadata: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_actions = int(metadata["n_actions"])
    horizon = int(metadata["horizon"])
    action_counts = np.zeros(n_actions, dtype=np.int64)
    reference_counts = np.zeros(n_actions, dtype=np.int64)
    times = []
    action_frequencies = []
    reference_frequencies = []
    reference_field = (
        "punished_action"
        if metadata["environment"] == HISTORICAL_FREQUENCY_ENVIRONMENT
        else "current_best_action"
    )
    stride = max(1, (horizon + MAX_PLOT_POINTS - 1) // MAX_PLOT_POINTS)

    with input_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            time = int(row["t"])
            action = int(row["action"])
            reference_action = int(row[reference_field])
            action_counts[action] += 1
            reference_counts[reference_action] += 1
            if time == 1 or time == horizon or time % stride == 0:
                times.append(time)
                action_frequencies.append(action_counts / time)
                reference_frequencies.append(reference_counts / time)

    return (
        np.asarray(times, dtype=int),
        np.asarray(action_frequencies, dtype=float),
        np.asarray(reference_frequencies, dtype=float),
    )


def _plot_frequencies(
    metadata: dict[str, str],
    times: np.ndarray,
    frequencies: np.ndarray,
    kind: str,
    output_path: Path,
) -> None:
    n_actions = frequencies.shape[1]
    figure, axes = plt.subplots(figsize=(10, 5.2))
    colors = plt.colormaps["tab20"].resampled(n_actions)
    for action in range(n_actions):
        axes.plot(
            times,
            frequencies[:, action],
            color=colors(action),
            linewidth=1.8,
            label=f"Action {action}",
        )

    environment_detail = adversarial_environment_detail(metadata, include_environment_seed=True)
    kind_label = {
        "action": "Learner action",
        "punished_action": "Punished-action",
        "best_action": "Best-action",
    }[kind]
    axes.set_xscale("log")
    axes.set_ylim(-0.02, 1.02)
    axes.set_xlabel("Round")
    axes.set_ylabel("Cumulative frequency")
    axes.set_title(
        f"{kind_label} frequency · "
        f"{algorithm_label(metadata['algorithm'])} · "
        f"{FEEDBACK_MODE_LABELS[metadata['feedback_mode']]} · "
        f"{environment_detail}"
    )
    axes.grid(True)
    if n_actions <= 12:
        axes.legend(loc="best", fontsize="small", frameon=False, ncol=2)
    figure.tight_layout()
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
    for n_actions in sorted({int(rows[0]["n_actions"]) for _, rows in results}):
        for source in ("expected", "realized"):
            for regret_name in ("external", "internal", "swap"):
                for average in (True, False):
                    if average:
                        filename = (
                            f"adversarial_{n_actions}_actions_average_"
                            f"{source}_{regret_name}_regret.png"
                        )
                    else:
                        filename = (
                            f"adversarial_{n_actions}_actions_{source}_"
                            f"{regret_name}_regret_over_sqrt_t.png"
                        )
                    output_path = output_dir / filename
                    _plot_regret(
                        results,
                        n_actions,
                        source,
                        regret_name,
                        average,
                        output_path,
                    )
                    generated.append(output_path)

    for input_path, rows in results:
        metadata = rows[0]
        times, action_frequencies, reference_frequencies = _frequency_curves(
            input_path, metadata
        )
        reference_kind = (
            "punished_action"
            if metadata["environment"] == HISTORICAL_FREQUENCY_ENVIRONMENT
            else "best_action"
        )
        for kind, frequencies in (
            ("action", action_frequencies),
            (reference_kind, reference_frequencies),
        ):
            output_path = output_dir / f"{input_path.stem}_{kind}_frequency.png"
            _plot_frequencies(
                metadata,
                times,
                frequencies,
                kind,
                output_path,
            )
            generated.append(output_path)

    generated_names = {
        path.with_suffix(suffix).name
        for path in generated
        for suffix in FIGURE_SUFFIXES
    }
    for suffix in FIGURE_SUFFIXES:
        for old_path in output_dir.glob(f"*{suffix}"):
            if old_path.name not in generated_names:
                old_path.unlink()
    return generated


def main() -> None:
    plot_adversarial_results()


if __name__ == "__main__":
    main()
