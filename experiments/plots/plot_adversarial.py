import csv
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import ADVERSARIAL_FIGURE_DIR, ADVERSARIAL_RAW_DIR
from experiments.plots import remove_stale_figure_pairs, save_figure_pair
from experiments.plots.style import publication_plot, curve_labels, algorithm_style, regret_axis_label, finish_line_figure
from experiments.scenarios.adversarial import (
    ADVERSARIAL_BASE_FIELDNAMES,
    load_adversarial_rows,
)


logger = logging.getLogger(__name__)
MAX_PLOT_POINTS = 2_000
PLOT_ROW_CACHE_VERSION = 3


def _group_key(rows: list[dict[str, str]]) -> tuple:
    first = rows[0]
    environment_seed = first["environment_seed"]
    return (
        first["environment"],
        first["reward_step"],
        first["feedback_mode"],
        first["implementation_version"],
        first["runtime_fingerprint"],
        first["n_actions"],
        first["algorithm"],
        first["horizon"],
        int(first["base_environment_seed"]) if environment_seed else None,
        int(first["base_learner_seed"]),
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


def _source_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "version": PLOT_ROW_CACHE_VERSION,
        "source": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "max_points": MAX_PLOT_POINTS,
    }


def _load_plot_rows(path: Path, cache_dir: Path) -> list[dict[str, str]]:
    identity = _source_identity(path)
    cache_path = cache_dir / f"{path.stem}.json"
    try:
        with cache_path.open(encoding="utf-8") as file:
            cached = json.load(file)
        if isinstance(cached, dict) and cached.get("identity") == identity:
            rows = cached.get("rows")
            if isinstance(rows, list) and rows and all(
                isinstance(row, dict) and set(ADVERSARIAL_BASE_FIELDNAMES) <= row.keys()
                for row in rows
            ):
                return rows
    except (OSError, ValueError, TypeError):
        pass

    rows = load_adversarial_rows(path, max_points=MAX_PLOT_POINTS)
    if _source_identity(path) != identity:
        raise ValueError(f"{path} changed while its trajectory was being loaded")
    temporary_path = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=cache_dir, delete=False) as file:
            temporary_path = Path(file.name)
            json.dump({"identity": identity, "rows": rows}, file, separators=(",", ":"))
        os.replace(temporary_path, cache_path)
    except OSError as error:
        logger.warning("Could not cache adversarial plot rows for %s: %s", path, error)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return rows


def adversarial_figure_prefix(scope: tuple[str, str, int]) -> str:
    environment, feedback_mode, n_actions = scope
    return f"adversarial_{environment}_{feedback_mode}_{n_actions}_actions_"


def collect_adversarial_results(
    input_dir: str | Path,
    skip_invalid: bool = False,
    *,
    scope: tuple[str, str, int] | None = None,
    cache_dir: str | Path | None = None,
) -> list[tuple[Path, list[dict[str, str]]]]:
    input_dir = Path(input_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else input_dir.parent / "cache" / "plot_rows"
    results = []
    for path in sorted(input_dir.glob("*.csv")):
        try:
            if scope is not None:
                # Inspect only one row of unrelated files, never their trajectories.
                # Do not infer scope from names: legacy/renamed CSVs remain supported.
                with path.open(encoding="utf-8", newline="") as file:
                    first = next(csv.DictReader(file), None)
                if first is None:
                    raise ValueError(f"{path} is empty")
                if (first["environment"], first["feedback_mode"], int(first["n_actions"])) != scope:
                    continue
            rows = _load_plot_rows(path, cache_dir)
        except (OSError, KeyError, TypeError, ValueError, csv.Error) as error:
            if not skip_invalid:
                raise
            logger.warning("Skipping invalid adversarial result %s: %s", path, error)
            continue
        results.append((path, rows))
    return results


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
    for trajectories, label in zip(selected, labels):
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
            **algorithm_style(algorithm),
            label=label,
        )

    axes.set_xscale("log")
    axes.axhline(0.0, color="#7b8580", linewidth=0.8, linestyle="--")
    axes.set_xlabel(r"Round $T$")
    axes.set_ylabel(regret_axis_label(regret_name, "average" if average else "sqrt_scaling"))
    finish_line_figure(figure, axes)
    save_figure_pair(figure, output_path)
    plt.close(figure)


def plot_adversarial_results(
    input_dir: str | Path = ADVERSARIAL_RAW_DIR,
    output_dir: str | Path = ADVERSARIAL_FIGURE_DIR,
    skip_invalid: bool = False,
    *,
    scope: tuple[str, str, int] | None = None,
) -> list[Path]:
    results = collect_adversarial_results(input_dir, skip_invalid=skip_invalid, scope=scope)
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
        for regret_name in ("external", "internal", "swap"):
            for average in (True, False):
                if average:
                    filename = (
                        f"adversarial_{environment}_{feedback_mode}_{n_actions}_actions_average_"
                        f"{regret_name}_regret.png"
                    )
                else:
                    filename = (
                        f"adversarial_{environment}_{feedback_mode}_{n_actions}_actions_"
                        f"{regret_name}_regret_over_sqrt_t.png"
                    )
                output_path = output_dir / filename
                _plot_regret(
                    results,
                    environment,
                    feedback_mode,
                    n_actions,
                    regret_name,
                    average,
                    output_path,
                )
                generated.append(output_path)

    remove_stale_figure_pairs(output_dir, generated,
                             filename_prefix=adversarial_figure_prefix(scope) if scope is not None else None)
    return generated


def main() -> None:
    plot_adversarial_results()


if __name__ == "__main__":
    main()
