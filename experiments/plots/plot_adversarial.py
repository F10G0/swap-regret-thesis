from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.plots import save_figure_pair
from experiments.result_catalog import adversarial_plot_key
from experiments.plots.style import publication_plot, curve_labels, algorithm_style, regret_axis_label, finish_line_figure
from experiments.scenarios.adversarial import load_adversarial_rows


def group_adversarial_results(
    results: list[tuple[Path, list[dict[str, str]]]],
) -> list[list[list[dict[str, str]]]]:
    groups = defaultdict(list)
    for _, rows in results:
        groups[adversarial_plot_key(rows[0])].append(rows)
    return [
        sorted(group, key=lambda rows: int(rows[0]["replicate"]))
        for _, group in sorted(groups.items())
    ]


def aggregate_adversarial_regret(
    trajectories: list[list[dict[str, str]]],
    column: str,
    scale_by_sqrt_time: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rows_by_time = [{int(row["t"]): row for row in trajectory} for trajectory in trajectories]
    # Use exactly observed, shared timestamps when recording budgets differ.
    times = np.asarray(sorted(set.intersection(*(set(rows) for rows in rows_by_time))), dtype=int)
    values = np.asarray(
        [[float(rows[time][column]) for time in times] for rows in rows_by_time]
    )
    if scale_by_sqrt_time:
        values = values / np.sqrt(times)
    return (
        times,
        np.mean(values, axis=0),
    )


@publication_plot
def _plot_regret(
    results: list[tuple[Path, list[dict[str, str]]]],
    environment: str,
    feedback_mode: str,
    n_actions: int,
    regret_name: str,
    average: bool,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots()
    selected = [
        group
        for group in group_adversarial_results(results)
        if group[0][0]["environment"] == environment
        and group[0][0]["feedback_mode"] == feedback_mode
        and int(group[0][0]["n_actions"]) == n_actions
    ]
    sort_fields = (
        "algorithm",
        "horizon",
        "learner_seed",
    )
    selected = sorted(selected, key=lambda group: tuple(group[0][0][field] for field in sort_fields))
    labels = curve_labels([group[0][0] | {"replicate_count": len(group)} for group in selected])
    for index, (trajectories, label) in enumerate(zip(selected, labels)):
        first = trajectories[0][0]
        algorithm = first["algorithm"]
        if average:
            column = f"average_{regret_name}_regret"
        else:
            column = f"{regret_name}_regret"
        times, values = aggregate_adversarial_regret(
            trajectories,
            column,
            scale_by_sqrt_time=not average,
        )
        axes.plot(
            times,
            values,
            **algorithm_style(algorithm, index, len(selected)),
            label=label,
        )

    axes.set_xscale("log")
    axes.axhline(0.0, color="#7b8580", linewidth=0.8, linestyle="--")
    axes.set_xlabel(r"Round $T$")
    axes.set_ylabel(regret_axis_label(regret_name, "average" if average else "sqrt_scaling"))
    finish_line_figure(figure, axes)
    save_figure_pair(figure, output_path)
    plt.close(figure)
