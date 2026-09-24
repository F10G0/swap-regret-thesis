"""Core full-space CE/CCE distance-convergence plotting."""

from collections.abc import Iterable
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import CUSTOM_GAME_DIR, EQUILIBRIUM_LP_TOLERANCE
from experiments.game_catalog import load_game_payoffs, payoff_tensor_digest
from experiments.plots import save_figure_pair
from experiments.plots.pdf_information import ENDPOINT_STATISTICS_DESCRIPTION, format_value_summary
from experiments.plots.style import publication_plot, finish_line_figure
from experiments.result_trajectories import load_result_empirical_distribution_trajectory
from experiments.results import iter_result_rows, result_game_payoff_digest
from metrics.equilibrium_distance import (
    EquilibriumDistanceTrajectory,
    ReplicateEquilibriumDistanceTrajectory,
    aggregate_equilibrium_distance_trajectories,
    equilibrium_distance_trajectory,
    preflight_equilibrium_analysis,
)


logger = logging.getLogger(__name__)


def _distance_cache_identity(path: Path, payoff_digest: str) -> dict:
    stat = path.stat()
    return {
        "source": str(path.resolve()), "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns, "size": stat.st_size,
        "payoff_digest": payoff_digest,
    }


def _load_result_distances(path: Path, payoff_tensor: np.ndarray, cache_dir: Path) -> EquilibriumDistanceTrajectory:
    identity = _distance_cache_identity(path, payoff_tensor_digest(payoff_tensor))
    cache_path = cache_dir / f"{sha256(str(path.resolve()).encode()).hexdigest()}.json"
    try:
        with cache_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        if payload["identity"] == identity:
            horizons = np.asarray(payload["horizons"])
            ce, cce = np.asarray(payload["ce"], dtype=float), np.asarray(payload["cce"], dtype=float)
            if (horizons.ndim == 1 and np.issubdtype(horizons.dtype, np.integer)
                    and len(horizons) > 0
                    and horizons[0] > 0 and np.all(np.diff(horizons) > 0)
                    and ce.shape == cce.shape == horizons.shape
                    and np.all(np.isfinite(ce)) and np.all(np.isfinite(cce))
                    and np.all(cce >= -EQUILIBRIUM_LP_TOLERANCE)
                    and np.all(ce <= 2 + EQUILIBRIUM_LP_TOLERANCE)
                    and np.all(cce <= ce + EQUILIBRIUM_LP_TOLERANCE)):
                return EquilibriumDistanceTrajectory(horizons, ce, cce)
    except (OSError, ValueError, KeyError, TypeError):
        pass

    empirical = load_result_empirical_distribution_trajectory(path, payoff_tensor.shape[1:])
    distances = equilibrium_distance_trajectory(payoff_tensor, empirical)
    if _distance_cache_identity(path, identity["payoff_digest"]) != identity:
        raise RuntimeError("result changed while computing equilibrium distances")
    payload = {"identity": identity, "horizons": distances.horizons.tolist(),
               "ce": distances.ce.tolist(), "cce": distances.cce.tolist()}
    temporary_path = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir, suffix=".tmp", delete=False) as file:
            temporary_path = Path(file.name)
            json.dump(payload, file, allow_nan=False)
        os.replace(temporary_path, cache_path)
    except OSError as error:
        logger.warning("Could not cache equilibrium distances for %s: %s", path, error)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return distances


@publication_plot
def _plot_equilibrium_distance(
    distances: ReplicateEquilibriumDistanceTrajectory,
    output_path: str | Path,
    information_rows: list[tuple[str, str]] | None = None,
) -> None:
    output_path = Path(output_path)
    figure, axes = plt.subplots()
    axes.plot(
        distances.horizons,
        distances.ce_mean,
        color="#d97706",
        marker="o",
        linestyle="-",
        linewidth=2.0,
        label="CE",
    )
    axes.plot(
        distances.horizons,
        distances.cce_mean,
        color="#2563eb",
        marker="s",
        linestyle="--",
        linewidth=2.0,
        label="CCE",
    )
    if (
        len(distances.horizons) > 1
        and distances.horizons[-1] / distances.horizons[0] >= 10
    ):
        axes.set_xscale("log")
    axes.set_xlabel(r"Round $T$")
    axes.set_ylabel(r"$L^1$ distance")
    axes.set_ylim(bottom=0.0)
    finish_line_figure(figure, axes)
    if information_rows is None:
        save_figure_pair(figure, output_path)
    else:
        save_figure_pair(figure, output_path, information_rows=information_rows)
    plt.close(figure)


def _load_equilibrium_game(paths: list[Path], custom_game_dir: str | Path) -> tuple[str, np.ndarray, list[dict]]:
    if not paths:
        raise ValueError("at least one result file is required")
    first_rows = []
    for path in paths:
        row = next(iter_result_rows(path), None)
        if row is None:
            raise ValueError("result file has no rows")
        first_rows.append(row)
    game_names = {row["game"] for row in first_rows}
    if len(game_names) != 1:
        raise ValueError(
            "equilibrium-convergence results must use the same game"
        )
    game_name = game_names.pop()
    payoff_tensor = load_game_payoffs(game_name, custom_game_dir)
    current_digest = payoff_tensor_digest(payoff_tensor)
    recorded_digests = {
        result_game_payoff_digest(row) for row in first_rows
    }
    recorded_digests.discard("")
    if len(recorded_digests) > 1:
        raise ValueError(
            "equilibrium-convergence results use different payoff tensors"
        )
    if recorded_digests and recorded_digests != {current_digest}:
        raise ValueError(
            "recorded payoff tensor does not match the current game definition"
        )
    return game_name, payoff_tensor, first_rows


def plot_result_equilibrium_distance(
    input_paths: str | Path | Iterable[str | Path],
    output_path: str | Path,
    custom_game_dir: str | Path = CUSTOM_GAME_DIR,
    *,
    cache_dir: str | Path | None = None,
    information_rows: list[tuple[str, str]] | None = None,
) -> None:
    paths = [Path(input_paths)] if isinstance(input_paths, (str, Path)) else [Path(path) for path in input_paths]
    _, payoff_tensor, _ = _load_equilibrium_game(paths, custom_game_dir)
    for equilibrium in ("ce", "cce"):
        preflight_equilibrium_analysis(payoff_tensor.shape[1:], equilibrium)
    replicate_distances = [
        _load_result_distances(
            path, payoff_tensor,
            Path(cache_dir) if cache_dir is not None else
            (path.parent.parent if path.parent.name == "raw" else path.parent) / "cache" / "equilibrium_distance",
        )
        for path in paths
    ]
    distances = aggregate_equilibrium_distance_trajectories(
        replicate_distances
    )
    rows = list(information_rows) if information_rows is not None else None
    if rows is not None:
        rows.extend([
            ("Final equilibrium distance at T", f"{int(distances.horizons[-1]):,}"),
            ("Endpoint statistics", ENDPOINT_STATISTICS_DESCRIPTION),
            ("CE", format_value_summary([trajectory.ce[-1] for trajectory in replicate_distances])),
            ("CCE", format_value_summary([trajectory.cce[-1] for trajectory in replicate_distances])),
        ])
    _plot_equilibrium_distance(
        distances,
        output_path,
        rows,
    )
