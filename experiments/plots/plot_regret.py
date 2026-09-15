from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import FIGURE_DIR
from experiments.plots import save_figure_pair
from experiments.plots.style import publication_plot, curve_labels, algorithm_style, regret_axis_label, finish_line_figure
from experiments.recording import MAX_RECORDED_POINTS
from experiments.result_schema import JOINT_ACTION_HISTOGRAM_FIELD
from experiments.results import average_regret_column, iter_result_rows, regret_column
from experiments.sampling import CheckpointRows


MAX_PLOT_POINTS_PER_PLAYER = MAX_RECORDED_POINTS


@dataclass(frozen=True)
class RegretCurve:
    x: np.ndarray
    y: np.ndarray
    label: str
    style: dict


def load_rows(
    input_path: str | Path,
    max_points_per_player: int = MAX_PLOT_POINTS_PER_PLAYER,
) -> list[dict]:
    if max_points_per_player <= 0:
        raise ValueError("max_points_per_player must be positive")
    input_path = Path(input_path)
    sampler = None
    for row in iter_result_rows(input_path):
        if sampler is None:
            sampler = CheckpointRows(int(row["horizon"]), max_points_per_player)
        sampler.add({key: value for key, value in row.items() if key != JOINT_ACTION_HISTOGRAM_FIELD})
    return sampler.rows() if sampler is not None else []


def aggregate_metric_curve(replicate_runs: list[list[dict]], player: int, column: str, divide_by_sqrt_time: bool = False) -> tuple[np.ndarray, np.ndarray]:
    values_by_time = defaultdict(list)

    for rows in replicate_runs:
        for row in rows:
            if int(row["player"]) != player:
                continue
            time = int(row["t"])
            value = float(row[column])
            if divide_by_sqrt_time:
                value /= np.sqrt(time)
            values_by_time[time].append(value)

    # Never interpolate regret or average over a changing subset of replicates.
    times = np.array(sorted(time for time, values in values_by_time.items() if len(values) == len(replicate_runs)), dtype=int)
    means = np.empty(len(times), dtype=float)

    for index, time in enumerate(times):
        values = np.asarray(values_by_time[time], dtype=float)
        if len(values) != len(replicate_runs):
            raise ValueError("replicate runs contain inconsistent time points")
        means[index] = np.mean(values)

    return times, means


@publication_plot
def plot_regret_curves(curves: list[RegretCurve], y_label: str, output_path: str | Path) -> None:
    figure, axes = plt.subplots()
    for curve in curves:
        axes.plot(curve.x, curve.y, label=curve.label, **curve.style)
    axes.set_xscale("log")
    axes.axhline(0.0, color="#7b8580", linewidth=0.8, linestyle="--")
    axes.set_xlabel(r"Round $T$")
    axes.set_ylabel(y_label)
    finish_line_figure(figure, axes)
    save_figure_pair(figure, output_path)
    plt.close(figure)


@publication_plot
def plot_regret(game_name: str, replicate_groups: list[list[list[dict]]], regret_name: str, player: int, average: bool, output_dir: str | Path = FIGURE_DIR) -> None:
    if average:
        column = average_regret_column(regret_name)
        filename = f"{game_name}_average_{regret_name}_regret_player_{player}.png"
    else:
        column = regret_column(regret_name)
        filename = f"{game_name}_{regret_name}_regret_over_sqrt_t_player_{player}.png"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots()
    labels = curve_labels([group[0][0] | {"replicate_count": len(group)} for group in replicate_groups])
    curves = []
    for group_index, replicate_runs in enumerate(replicate_groups):
        times, means = aggregate_metric_curve(replicate_runs, player, column, divide_by_sqrt_time=not average)
        if len(times) == 0:
            continue
        algorithm = replicate_runs[0][0]["algorithm"].split("_vs_")[player]
        curves.append((times, means, algorithm, labels[group_index]))

    if not curves:
        plt.close(figure)
        return
    for index, (times, means, algorithm, label) in enumerate(curves):
        axes.plot(times, means, **algorithm_style(algorithm, index, len(curves)), label=label)

    axes.set_xscale("log")
    axes.axhline(0.0, color="#7b8580", linewidth=0.8, linestyle="--")
    axes.set_xlabel(r"Round $T$")
    axes.set_ylabel(regret_axis_label(regret_name, "average" if average else "sqrt_scaling"))
    finish_line_figure(figure, axes)
    output_path = output_dir / filename
    save_figure_pair(figure, output_path)
    plt.close(figure)
