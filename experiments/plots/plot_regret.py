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


INSUFFICIENT_LOG_LOG_POINTS = "Insufficient positive tail checkpoints for log-log fit"
NON_POSITIVE_LOG_LOG_REGRET = "Log-log fit unavailable: replicate-mean cumulative action regret is non-positive within the fit window."
INSUFFICIENT_HORIZONS = "Horizon scaling unavailable: at least three distinct horizons are required."
NON_POSITIVE_HORIZON_REGRET = "Horizon scaling unavailable: replicate-mean final cumulative action regret is non-positive at one or more horizons."


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


def aggregate_final_metric(replicate_runs: list[list[dict]], player: int, column: str) -> float:
    values = []
    for rows in replicate_runs:
        horizon = int(rows[0]["horizon"])
        final = [row for row in rows if int(row["player"]) == player and int(row["t"]) == horizon]
        if len(final) != 1:
            raise ValueError("replicate run has no unique final player observation")
        values.append(float(final[0][column]))
    return float(np.mean(values))


def regret_log_log_fit(times: np.ndarray, mean_regret: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, tuple[float, float] | None, str | None]:
    tail = times >= horizon / 10
    tail_times = times[tail]
    tail_regret = mean_regret[tail]
    if len(tail_times) < 3:
        if np.all(tail_regret > 0):
            return np.log(tail_times), np.log(tail_regret), None, INSUFFICIENT_LOG_LOG_POINTS
        return np.array([]), np.array([]), None, INSUFFICIENT_LOG_LOG_POINTS
    if np.any(tail_regret <= 0):
        return np.array([]), np.array([]), None, NON_POSITIVE_LOG_LOG_REGRET
    x = np.log(tail_times)
    y = np.log(tail_regret)
    return x, y, tuple(np.polyfit(x, y, 1)), None


def horizon_scaling_fit(horizons: np.ndarray, mean_regret: np.ndarray) -> tuple[tuple[float, float] | None, str | None]:
    if horizons.ndim != 1 or mean_regret.shape != horizons.shape or not np.all(np.isfinite(mean_regret)):
        raise ValueError("horizon scaling data must be finite one-dimensional arrays of equal length")
    if len(horizons) < 3:
        return None, INSUFFICIENT_HORIZONS
    if np.any(horizons <= 0) or np.any(np.diff(horizons) <= 0):
        raise ValueError("horizons must be positive, distinct, and strictly increasing")
    if np.any(mean_regret <= 0):
        return None, NON_POSITIVE_HORIZON_REGRET
    return tuple(np.polyfit(np.log(horizons), np.log(mean_regret), 1)), None


def _log_log_r_squared(horizons: np.ndarray, mean_regret: np.ndarray,
                       fit: tuple[float, float]) -> float | None:
    x = np.log(horizons)
    y = np.log(mean_regret)
    slope, intercept = fit
    residuals = y - (slope * x + intercept)
    sse = float(np.sum(residuals ** 2))
    centered = y - np.mean(y)
    sst = float(np.sum(centered ** 2))
    tolerance = np.finfo(float).eps * max(1.0, float(np.sum(y ** 2))) * len(y) * 16
    if sst <= tolerance:
        return 1.0 if sse <= tolerance else None
    return 1.0 - sse / sst


@publication_plot
def plot_regret_curves(curves: list[RegretCurve], y_label: str, output_path: str | Path,
                       information_rows: list[tuple[str, str]] | None = None, legend_ncol=None) -> None:
    figure, axes = plt.subplots()
    for curve in curves:
        axes.plot(curve.x, curve.y, label=curve.label, **curve.style)
    axes.set_xscale("log")
    axes.axhline(0.0, color="#7b8580", linewidth=0.8, linestyle="--")
    axes.set_xlabel(r"Round $T$")
    axes.set_ylabel(y_label)
    finish_line_figure(figure, axes, legend_ncol=legend_ncol)
    if information_rows is None:
        save_figure_pair(figure, output_path)
    else:
        save_figure_pair(figure, output_path, information_rows=information_rows)
    plt.close(figure)


@publication_plot
def plot_regret_log_log(curve: RegretCurve, metric: str, horizon: int, output_path: str | Path,
                        information_rows: list[tuple[str, str]] | None = None) -> None:
    x, y, fit, message = regret_log_log_fit(curve.x, curve.y, horizon)
    figure, axes = plt.subplots()
    if fit is None:
        axes.set_axis_off()
        axes.text(0.5, 0.5, message, ha="center", va="center", transform=axes.transAxes)
        if information_rows is not None:
            information_rows = [*information_rows, ("Fit", "Unavailable")]
    else:
        axes.plot(x, y, label=curve.label, **curve.style)
        slope, intercept = fit
        axes.plot(x, slope * x + intercept, color="#333333", linestyle=":",
                  label=f"OLS fit (slope = {slope:.3f})")
        if information_rows is not None:
            information_rows = [*information_rows, ("Fitted slope", f"{slope:.3f}")]
        axes.set_xlabel(r"$\log t$")
        axes.set_ylabel(f"log replicate-mean cumulative {metric} action regret")
        finish_line_figure(figure, axes)
    if information_rows is None:
        save_figure_pair(figure, output_path)
    else:
        save_figure_pair(figure, output_path, information_rows=information_rows)
    plt.close(figure)


@publication_plot
def plot_horizon_scaling(curves: list[RegretCurve], output_path: str | Path,
                         information_rows: list[tuple[str, str]] | None = None) -> None:
    figure, axes = plt.subplots()
    valid = False
    invalid = []
    rows = list(information_rows) if information_rows is not None else None
    for curve in curves:
        fit, _ = horizon_scaling_fit(curve.x, curve.y)
        notion = curve.label.removesuffix(" regret")
        if fit is None:
            invalid.append(f"{notion}: invalid")
            if rows is not None:
                rows.append((notion, "invalid"))
            continue
        valid = True
        slope, intercept = fit
        coefficient = np.exp(intercept)
        r_squared = _log_log_r_squared(curve.x, curve.y, fit)
        axes.plot(curve.x, curve.y, color=curve.style["color"], marker=curve.style["marker"],
                  linestyle="None", zorder=curve.style["zorder"])
        axes.plot(curve.x, coefficient * curve.x ** slope, color=curve.style["color"],
                  linestyle=curve.style["linestyle"], zorder=curve.style["zorder"],
                  label=f"{notion}: α = {slope:.3f}, c = {coefficient:.4g}")
        if rows is not None:
            r_squared_text = "unavailable" if r_squared is None else f"{r_squared:.3f}"
            rows.append((notion, f"α = {slope:.3f}, c = {coefficient:.4g}, R² = {r_squared_text}"))
    if not valid:
        axes.set_axis_off()
        axes.text(0.5, 0.5, "Horizon scaling unavailable: no regret notion has a valid fit.",
                  ha="center", va="center", transform=axes.transAxes)
    else:
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_xlabel(r"Configured horizon $T$")
        axes.set_ylabel("Replicate-mean final cumulative action regret")
    if invalid:
        axes.text(0.5, -0.18, " · ".join(invalid), ha="center", va="top", transform=axes.transAxes)
    finish_line_figure(figure, axes, legend_ncol=1)
    if rows is None:
        save_figure_pair(figure, output_path)
    else:
        save_figure_pair(figure, output_path, information_rows=rows)
    plt.close(figure)
