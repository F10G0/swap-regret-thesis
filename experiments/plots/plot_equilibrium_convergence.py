"""Core full-space CE/CCE distance-convergence plotting."""

from collections.abc import Iterable
from hashlib import sha256
import json
import logging
import os
from operator import index
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import CUSTOM_GAME_DIR, EQUILIBRIUM_LP_TOLERANCE
from experiments.game_catalog import load_game_payoffs, payoff_tensor_digest
from experiments.plots import FIGURE_SUFFIXES, save_figure_pair
from experiments.result_trajectories import load_result_action_profiles
from experiments.results import iter_result_rows, result_game_payoff_digest
from metrics.empirical_distribution import (
    EmpiricalDistributionTrajectory,
    empirical_distribution_trajectory,
)
from metrics.equilibrium_convergence import (
    EquilibriumDistanceTrajectory,
    ReplicateEquilibriumDistanceTrajectory,
    aggregate_equilibrium_distance_trajectories,
    equilibrium_distance_trajectory,
)
from metrics.equilibrium_distance import EQUILIBRIUM_DISTANCE_IMPLEMENTATION_VERSION


MAX_EQUILIBRIUM_DISTANCE_POINTS = 160
DISTANCE_CACHE_VERSION = 1
DISTANCE_CHECKPOINT_POLICY = "nearest-log-horizon-v1"
logger = logging.getLogger(__name__)


def equilibrium_distance_point_indices(horizons: np.ndarray) -> np.ndarray:
    """Select existing exact horizons, approximately uniformly in log time."""
    if len(horizons) <= MAX_EQUILIBRIUM_DISTANCE_POINTS:
        return np.arange(len(horizons))
    logarithms = np.log(horizons)
    targets = np.linspace(logarithms[0], logarithms[-1], MAX_EQUILIBRIUM_DISTANCE_POINTS)
    right = np.searchsorted(logarithms, targets).clip(0, len(horizons) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(targets - logarithms[left] <= logarithms[right] - targets, left, right)
    return np.unique(np.concatenate(([0], nearest, [len(horizons) - 1])))


def _distance_cache_identity(path: Path, payoff_digest: str, checkpoints) -> dict:
    stat = path.stat()
    return {
        "version": DISTANCE_CACHE_VERSION,
        "source": str(path.resolve()), "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns, "size": stat.st_size,
        "payoff_digest": payoff_digest,
        "metric_version": EQUILIBRIUM_DISTANCE_IMPLEMENTATION_VERSION,
        "max_points": MAX_EQUILIBRIUM_DISTANCE_POINTS,
        "checkpoint_policy": DISTANCE_CHECKPOINT_POLICY,
        "requested_checkpoints": [index(point) for point in checkpoints] if checkpoints is not None else None,
    }


def _load_result_distances(path: Path, payoff_tensor: np.ndarray, checkpoints, cache_dir: Path) -> EquilibriumDistanceTrajectory:
    identity = _distance_cache_identity(path, payoff_tensor_digest(payoff_tensor), checkpoints)
    cache_path = cache_dir / f"{sha256(str(path.resolve()).encode()).hexdigest()}.json"
    try:
        with cache_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        if payload["identity"] == identity:
            horizons = np.asarray(payload["horizons"])
            ce, cce = np.asarray(payload["ce"], dtype=float), np.asarray(payload["cce"], dtype=float)
            if (horizons.ndim == 1 and np.issubdtype(horizons.dtype, np.integer)
                    and 0 < len(horizons) <= MAX_EQUILIBRIUM_DISTANCE_POINTS
                    and horizons[0] > 0 and np.all(np.diff(horizons) > 0)
                    and ce.shape == cce.shape == horizons.shape
                    and np.all(np.isfinite(ce)) and np.all(np.isfinite(cce))
                    and np.all(cce >= -EQUILIBRIUM_LP_TOLERANCE)
                    and np.all(ce <= 2 + EQUILIBRIUM_LP_TOLERANCE)
                    and np.all(cce <= ce + EQUILIBRIUM_LP_TOLERANCE)):
                return EquilibriumDistanceTrajectory(horizons, ce, cce)
    except (OSError, ValueError, KeyError, TypeError):
        pass

    profiles = load_result_action_profiles(path, payoff_tensor.shape[1:])
    empirical = empirical_distribution_trajectory(profiles, payoff_tensor.shape[1:], checkpoints)
    indices = equilibrium_distance_point_indices(empirical.horizons)
    selected = EmpiricalDistributionTrajectory(empirical.action_shape, empirical.horizons[indices], empirical.vectors[indices])
    distances = equilibrium_distance_trajectory(payoff_tensor, selected)
    if _distance_cache_identity(path, identity["payoff_digest"], checkpoints) != identity:
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


def remove_equilibrium_distance_figures(detail_figure_dir: str | Path) -> list[Path]:
    """Remove only result CE/CCE distance figures, never unrelated detail files."""
    directory = Path(detail_figure_dir)
    removed = []
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in FIGURE_SUFFIXES and path.stem.endswith("_equilibrium_distance"):
                path.unlink()
                removed.append(path)
    return removed


def _plot_equilibrium_distance(
    distances: ReplicateEquilibriumDistanceTrajectory,
    output_path: str | Path,
    game_name: str,
) -> None:
    output_path = Path(output_path)
    figure, axes = plt.subplots(figsize=(7.2, 4.6))
    axes.plot(
        distances.horizons,
        distances.ce_mean,
        color="#d97706",
        marker="o",
        linewidth=2.0,
        label="CE",
    )
    axes.plot(
        distances.horizons,
        distances.cce_mean,
        color="#2563eb",
        marker="s",
        linewidth=2.0,
        label="CCE",
    )
    if distances.n_replicates > 1:
        axes.fill_between(
            distances.horizons,
            np.maximum(
                0.0,
                distances.ce_mean - distances.ce_confidence,
            ),
            distances.ce_mean + distances.ce_confidence,
            color="#d97706",
            alpha=0.18,
        )
        axes.fill_between(
            distances.horizons,
            np.maximum(
                0.0,
                distances.cce_mean - distances.cce_confidence,
            ),
            distances.cce_mean + distances.cce_confidence,
            color="#2563eb",
            alpha=0.18,
        )
    if (
        len(distances.horizons) > 1
        and distances.horizons[-1] / distances.horizons[0] >= 10
    ):
        axes.set_xscale("log")
    axes.set_xlabel("Horizon")
    axes.set_ylabel("L1 distance")
    axes.set_ylim(bottom=0.0)
    axes.grid(alpha=0.25)
    axes.legend()
    title = (
        "Equilibrium Distance"
        if distances.n_replicates == 1
        else (
            "Mean Equilibrium Distance "
            f"({distances.n_replicates} replicates)"
        )
    )
    axes.set_title(f"{game_name}: {title}")
    figure.tight_layout()
    save_figure_pair(figure, output_path)
    plt.close(figure)


def load_equilibrium_result_inputs(
    input_paths: str | Path | Iterable[str | Path],
    custom_game_dir: str | Path = CUSTOM_GAME_DIR,
) -> tuple[str, np.ndarray, list[np.ndarray]]:
    paths = (
        [Path(input_paths)]
        if isinstance(input_paths, (str, Path))
        else [Path(path) for path in input_paths]
    )
    game_name, payoff_tensor = _load_equilibrium_game(paths, custom_game_dir)
    profiles = [
        load_result_action_profiles(path, payoff_tensor.shape[1:])
        for path in paths
    ]
    return game_name, payoff_tensor, profiles


def _load_equilibrium_game(paths: list[Path], custom_game_dir: str | Path) -> tuple[str, np.ndarray]:
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
    return game_name, payoff_tensor


def empirical_distribution_trajectories(
    profiles: list[np.ndarray],
    action_shape: tuple[int, ...],
    checkpoints: Iterable[int] | None,
) -> list[EmpiricalDistributionTrajectory]:
    return [
        empirical_distribution_trajectory(
            action_profiles,
            action_shape,
            checkpoints,
        )
        for action_profiles in profiles
    ]


def plot_result_equilibrium_distance(
    input_paths: str | Path | Iterable[str | Path],
    output_path: str | Path,
    checkpoints: Iterable[int] | None = None,
    game_label: str | None = None,
    custom_game_dir: str | Path = CUSTOM_GAME_DIR,
    *,
    cache_dir: str | Path | None = None,
) -> None:
    paths = [Path(input_paths)] if isinstance(input_paths, (str, Path)) else [Path(path) for path in input_paths]
    game_name, payoff_tensor = _load_equilibrium_game(paths, custom_game_dir)
    checkpoints = tuple(checkpoints) if checkpoints is not None else None
    replicate_distances = [
        _load_result_distances(
            path, payoff_tensor, checkpoints,
            Path(cache_dir) if cache_dir is not None else
            (path.parent.parent if path.parent.name == "raw" else path.parent) / "cache" / "equilibrium_distance",
        )
        for path in paths
    ]
    distances = aggregate_equilibrium_distance_trajectories(
        replicate_distances
    )
    _plot_equilibrium_distance(
        distances,
        output_path,
        game_label or game_name,
    )
