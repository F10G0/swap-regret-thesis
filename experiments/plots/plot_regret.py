import csv
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import FIGURE_DIR, RAW_DIR
from experiments.results import average_regret_column, iter_result_rows, regret_column
from experiments.scenarios.fieldnames import REGRET_NAMES


logger = logging.getLogger(__name__)

MAX_PLOT_POINTS_PER_PLAYER = 2000

REGRET_TYPES = {
    "expected": "Expected",
    "realized": "Realized",
}

REPLICATE_GROUP_COLUMNS = (
    "game",
    "feedback_mode",
    "algorithm",
    "algorithm_player_0",
    "algorithm_player_1",
    "horizon",
    "seed",
    "stationary_method",
)


def load_rows(input_path: str | Path, max_points_per_player: int = MAX_PLOT_POINTS_PER_PLAYER) -> list[dict]:
    if max_points_per_player <= 0:
        raise ValueError("max_points_per_player must be positive")

    sampled_rows = []
    for row in iter_result_rows(input_path):
        horizon = int(row["horizon"])
        time = int(row["t"])
        stride = max(1, (horizon + max_points_per_player - 1) // max_points_per_player)
        if time == 1 or time == horizon or time % stride == 0:
            sampled_rows.append(row)

    return sampled_rows


def collect_results(input_dir: str | Path = RAW_DIR, game_name: str | None = None, skip_invalid: bool = False) -> dict[str, dict[str, list[dict]]]:
    input_dir = Path(input_dir)
    results = defaultdict(dict)
    paths = sorted(input_dir.glob("*.csv"))
    if game_name is not None:
        paths = [path for path in paths if path.name.startswith(f"{game_name}_")]

    for path in paths:
        try:
            rows = load_rows(path)
        except (OSError, KeyError, TypeError, ValueError, csv.Error) as error:
            if not skip_invalid:
                raise
            logger.warning("Skipping invalid result %s: %s", path, error)
            continue
        if not rows:
            continue

        result_game = rows[0]["game"]
        if game_name is not None and result_game != game_name:
            continue
        run_id = rows[0]["run_id"]
        if run_id in results[result_game]:
            raise ValueError(f"duplicate run_id {run_id} in {input_dir}")
        results[result_game][run_id] = rows

    return {result_game: dict(rows_by_run) for result_game, rows_by_run in results.items()}


def available_players(rows_by_run: dict[str, list[dict]]) -> list[int]:
    return sorted({int(row["player"]) for rows in rows_by_run.values() for row in rows})


def rows_for_player(rows: list[dict], player: int) -> list[dict]:
    return [row for row in rows if int(row["player"]) == player]


def group_replicate_runs(rows_by_run: dict[str, list[dict]]) -> list[list[list[dict]]]:
    groups = defaultdict(list)
    for rows in rows_by_run.values():
        first_row = rows[0]
        key = tuple(first_row[column] for column in REPLICATE_GROUP_COLUMNS)
        groups[key].append(rows)

    return [sorted(group, key=lambda rows: int(rows[0]["replicate"])) for group in groups.values()]


def aggregate_metric_curve(replicate_runs: list[list[dict]], player: int, column: str, divide_by_sqrt_time: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_by_time = defaultdict(list)

    for rows in replicate_runs:
        for row in rows_for_player(rows, player):
            time = int(row["t"])
            value = float(row[column])
            if divide_by_sqrt_time:
                value /= np.sqrt(time)
            values_by_time[time].append(value)

    times = np.array(sorted(values_by_time), dtype=int)
    means = np.empty(len(times), dtype=float)
    confidence = np.zeros(len(times), dtype=float)

    for index, time in enumerate(times):
        values = np.asarray(values_by_time[time], dtype=float)
        if len(values) != len(replicate_runs):
            raise ValueError("replicate runs contain inconsistent time points")
        means[index] = np.mean(values)
        if len(values) > 1:
            confidence[index] = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))

    return times, means, confidence


def run_label(rows: list[dict], n_replicates: int, include_solver: bool = False) -> str:
    first_row = rows[0]
    profile = first_row["algorithm"].replace("_vs_", " vs ").replace("_", " ")
    label = f"{profile} · seed {first_row['seed']}"
    if include_solver:
        label += f" · solver {first_row['stationary_method']}"
    if n_replicates > 1:
        label += f" · {n_replicates} replicates"
    return label


def plot_regret(game_name: str, replicate_groups: list[list[list[dict]]], regret_type: str, regret_name: str, player: int, average: bool, output_dir: str | Path = FIGURE_DIR) -> None:
    regret_type_label = REGRET_TYPES[regret_type]
    if average:
        column = average_regret_column(regret_type, regret_name)
        ylabel = f"Average {regret_type_label.lower()} {regret_name} regret"
        title = f"{game_name}: average {regret_type_label.lower()} {regret_name} regret, player {player}"
        filename = f"{game_name}_average_{regret_type}_{regret_name}_regret_player_{player}.png"
    else:
        column = regret_column(regret_type, regret_name)
        ylabel = f"{regret_type_label} {regret_name} regret / sqrt(t)"
        title = f"{game_name}: {regret_type_label.lower()} {regret_name} regret scaling, player {player}"
        filename = f"{game_name}_{regret_type}_{regret_name}_regret_over_sqrt_t_player_{player}.png"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots()
    plotted = False

    color_map = plt.get_cmap("tab20")
    include_solver = len({group[0][0]["stationary_method"] for group in replicate_groups}) > 1
    for group_index, replicate_runs in enumerate(replicate_groups):
        times, means, confidence = aggregate_metric_curve(replicate_runs, player, column, divide_by_sqrt_time=not average)
        if len(times) == 0:
            continue
        plotted = True
        color = color_map(group_index % color_map.N)
        axes.plot(times, means, color=color, label=run_label(replicate_runs[0], len(replicate_runs), include_solver))
        if len(replicate_runs) > 1:
            axes.fill_between(times, means - confidence, means + confidence, color=color, alpha=0.2)

    if not plotted:
        plt.close(figure)
        return

    if not average:
        axes.set_xscale("log")
    axes.axhline(0.0, color="#7b8580", linewidth=0.8, linestyle="--")
    axes.set_xlabel("Round")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(True)
    handles, labels = axes.get_legend_handles_labels()
    legend_columns = min(2, len(handles))
    legend_rows = (len(handles) + legend_columns - 1) // legend_columns
    legend_height = 0.22 * legend_rows + 0.25
    figure_height = 4.4 + legend_height
    figure.set_size_inches(11, figure_height)
    figure.subplots_adjust(left=0.09, right=0.98, top=1.0 - 0.45 / figure_height, bottom=(legend_height + 0.35) / figure_height)
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=legend_columns, frameon=False, fontsize="small")
    figure.savefig(output_dir / filename, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def clear_game_figures(game_name: str, output_dir: str | Path = FIGURE_DIR) -> None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return

    for path in output_dir.glob(f"{game_name}_*.png"):
        path.unlink()


def plot_game_results(game_name: str, rows_by_run: dict[str, list[dict]], output_dir: str | Path) -> None:
    clear_game_figures(game_name, output_dir)
    replicate_groups = group_replicate_runs(rows_by_run)
    groups_by_regret_type = {
        source: [group for group in replicate_groups if regret_column(source, REGRET_NAMES[0]) in group[0][0]]
        for source in REGRET_TYPES
    }
    for player in available_players(rows_by_run):
        for regret_type, source_groups in groups_by_regret_type.items():
            if not source_groups:
                continue
            for regret_name in REGRET_NAMES:
                plot_regret(game_name, source_groups, regret_type, regret_name, player, average=True, output_dir=output_dir)
                plot_regret(game_name, source_groups, regret_type, regret_name, player, average=False, output_dir=output_dir)


def plot_selected_results(game_name: str, input_dir: str | Path = RAW_DIR, output_dir: str | Path = FIGURE_DIR, skip_invalid: bool = False) -> None:
    results = collect_results(input_dir, game_name, skip_invalid)
    if game_name not in results:
        return
    plot_game_results(game_name, results[game_name], output_dir)


def plot_all_results(input_dir: str | Path = RAW_DIR, output_dir: str | Path = FIGURE_DIR, skip_invalid: bool = False) -> None:
    results = collect_results(input_dir, skip_invalid=skip_invalid)
    for game_name, rows_by_run in results.items():
        plot_game_results(game_name, rows_by_run, output_dir)


def main() -> None:
    plot_all_results()


if __name__ == "__main__":
    main()
