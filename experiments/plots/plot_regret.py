from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.plots import save_figure_pair
from experiments.plots.style import publication_plot, finish_line_figure
from experiments.recording import MAX_RECORDED_POINTS
from experiments.result_schema import JOINT_ACTION_HISTOGRAM_FIELD
from experiments.results import iter_result_rows
from experiments.sampling import CheckpointRows


@dataclass(frozen=True)
class RegretCurve:
    x: np.ndarray
    y: np.ndarray
    label: str
    style: dict


def load_rows(
    input_path: str | Path,
    max_points_per_player: int = MAX_RECORDED_POINTS,
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
