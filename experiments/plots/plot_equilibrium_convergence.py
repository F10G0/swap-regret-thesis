from collections.abc import Iterable
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import CUSTOM_GAME_DIR
from experiments.game_catalog import load_game_payoffs
from experiments.result_trajectories import load_result_action_profiles
from metrics.empirical_distribution import (
    EmpiricalDistributionTrajectory,
    empirical_distribution_trajectory,
    mean_empirical_distribution_trajectory,
    uniform_checkpoints,
)
from metrics.equilibrium_convergence import (
    EquilibriumTrajectoryProjection,
    ReplicateEquilibriumDistanceTrajectory,
    aggregate_equilibrium_distance_trajectories,
    equilibrium_distance_trajectory,
    project_equilibrium_trajectory,
)
from experiments.results import iter_result_rows
from metrics.equilibrium_projection import ProjectedEquilibriumRegion


def _plot_equilibrium_distance(distances: ReplicateEquilibriumDistanceTrajectory, output_path: str | Path, game_name: str) -> None:
    output_path = Path(output_path)
    figure, axes = plt.subplots(figsize=(7.2, 4.6))
    axes.plot(distances.horizons, distances.ce_mean, color="#d97706", marker="o", linewidth=2.0, label="CE")
    axes.plot(distances.horizons, distances.cce_mean, color="#2563eb", marker="s", linewidth=2.0, label="CCE")
    if distances.n_replicates > 1:
        axes.fill_between(distances.horizons, np.maximum(0.0, distances.ce_mean - distances.ce_confidence),
                          distances.ce_mean + distances.ce_confidence, color="#d97706", alpha=0.18)
        axes.fill_between(distances.horizons, np.maximum(0.0, distances.cce_mean - distances.cce_confidence),
                          distances.cce_mean + distances.cce_confidence, color="#2563eb", alpha=0.18)
    if len(distances.horizons) > 1 and distances.horizons[-1] / distances.horizons[0] >= 10:
        axes.set_xscale("log")
    axes.set_xlabel("Horizon")
    axes.set_ylabel("L1 distance")
    axes.set_ylim(bottom=0.0)
    axes.grid(alpha=0.25)
    axes.legend()
    title = "Equilibrium Distance" if distances.n_replicates == 1 else f"Mean Equilibrium Distance ({distances.n_replicates} replicates)"
    axes.set_title(f"{game_name}: {title}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def _draw_region(axes, region: ProjectedEquilibriumRegion, color: str, label: str, alpha: float) -> None:
    boundary = region.boundary
    if region.affine_dimension == 0:
        axes.scatter(boundary[:, 0], boundary[:, 1], color=color, s=62, label=label, zorder=3)
    elif region.affine_dimension == 1:
        axes.plot(boundary[:, 0], boundary[:, 1], color=color, linewidth=5, alpha=alpha + 0.2, label=label)
        axes.scatter(boundary[:, 0], boundary[:, 1], color=color, s=24, zorder=3)
    else:
        closed = np.vstack((boundary, boundary[0]))
        axes.fill(closed[:, 0], closed[:, 1], color=color, alpha=alpha, label=label)
        axes.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.4)


def _horizon_label(horizon: int) -> str:
    if horizon >= 1_000_000 and horizon % 1_000_000 == 0:
        return f"{horizon // 1_000_000}m"
    if horizon >= 1_000 and horizon % 1_000 == 0:
        return f"{horizon // 1_000}k"
    return str(horizon)


def _plot_equilibrium_trajectory(analysis: EquilibriumTrajectoryProjection, output_path: str | Path, game_name: str,
                                 n_replicates: int) -> None:
    output_path = Path(output_path)
    figure, axes = plt.subplots(figsize=(7.2, 5.8))
    _draw_region(axes, analysis.cce_region, "#60a5fa", "CCE", 0.2)
    _draw_region(axes, analysis.ce_region, "#f59e0b", "CE", 0.3)

    trajectory = analysis.projected_trajectory
    trajectory_label = "Empirical trajectory" if n_replicates == 1 else "Mean empirical trajectory"
    axes.plot(trajectory[:, 0], trajectory[:, 1], color="#111827", marker="o", markersize=4, linewidth=1.7,
              label=trajectory_label, zorder=4)
    axes.scatter(trajectory[0, 0], trajectory[0, 1], color="#16a34a", marker="o", s=70, label="First checkpoint", zorder=5)
    axes.scatter(trajectory[-1, 0], trajectory[-1, 1], color="#dc2626", marker="X", s=76, label="End", zorder=5)
    if len(trajectory) <= 12:
        for point, horizon in zip(trajectory, analysis.empirical.horizons):
            axes.annotate(_horizon_label(int(horizon)), point, xytext=(5, 5), textcoords="offset points", fontsize=8)

    axes.set_xlabel("Projected component 1")
    axes.set_ylabel("Projected component 2")
    axes.grid(alpha=0.2)
    axes.legend()
    title = "Projected Joint-Distribution Trajectory" if n_replicates == 1 else f"Projected Mean Joint-Distribution Trajectory ({n_replicates} replicates)"
    axes.set_title(f"{game_name}: {title}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def _load_result_inputs(input_paths: str | Path | Iterable[str | Path],
                        custom_game_dir: str | Path = CUSTOM_GAME_DIR) -> tuple[str, np.ndarray, list[np.ndarray]]:
    paths = [Path(input_paths)] if isinstance(input_paths, (str, Path)) else [Path(path) for path in input_paths]
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
        raise ValueError("equilibrium-convergence results must use the same game")
    game_name = game_names.pop()
    payoff_tensor = load_game_payoffs(game_name, custom_game_dir)
    profiles = [
        load_result_action_profiles(path, payoff_tensor.shape[1:])
        for path in paths
    ]
    return game_name, payoff_tensor, profiles


def _empirical_trajectories(profiles: list[np.ndarray], action_shape: tuple[int, ...],
                            checkpoints: Iterable[int] | None) -> list[EmpiricalDistributionTrajectory]:
    return [
        empirical_distribution_trajectory(action_profiles, action_shape, checkpoints)
        for action_profiles in profiles
    ]


def _plot_result_equilibrium_distance(game_name: str, payoff_tensor: np.ndarray, profiles: list[np.ndarray],
                                      output_path: str | Path, checkpoints: Iterable[int] | None,
                                      game_label: str | None) -> None:
    empirical_trajectories = _empirical_trajectories(profiles, payoff_tensor.shape[1:], checkpoints)
    title = game_label or game_name
    replicate_distances = [
        equilibrium_distance_trajectory(payoff_tensor, trajectory)
        for trajectory in empirical_trajectories
    ]
    distances = aggregate_equilibrium_distance_trajectories(replicate_distances)
    _plot_equilibrium_distance(distances, output_path, title)


def _plot_result_equilibrium_trajectory(game_name: str, payoff_tensor: np.ndarray, profiles: list[np.ndarray],
                                        output_path: str | Path, trajectory_points: int, direction_count: int,
                                        game_label: str | None, hide_first: bool) -> None:
    checkpoints = uniform_checkpoints(len(profiles[0]), trajectory_points)
    empirical_trajectories = _empirical_trajectories(profiles, payoff_tensor.shape[1:], checkpoints)
    empirical = mean_empirical_distribution_trajectory(empirical_trajectories)
    if hide_first and len(empirical.horizons) > 1:
        empirical = EmpiricalDistributionTrajectory(empirical.action_shape, empirical.horizons[1:], empirical.vectors[1:])
    analysis = project_equilibrium_trajectory(payoff_tensor, empirical, direction_count)
    _plot_equilibrium_trajectory(analysis, output_path, game_label or game_name, len(profiles))


def plot_result_equilibrium_distance(input_paths: str | Path | Iterable[str | Path], output_path: str | Path,
                                     checkpoints: Iterable[int] | None = None, game_label: str | None = None,
                                     custom_game_dir: str | Path = CUSTOM_GAME_DIR) -> None:
    game_name, payoff_tensor, profiles = _load_result_inputs(input_paths, custom_game_dir)
    _plot_result_equilibrium_distance(game_name, payoff_tensor, profiles, output_path, checkpoints, game_label)


def plot_result_equilibrium_trajectory(input_paths: str | Path | Iterable[str | Path], output_path: str | Path,
                                       trajectory_points: int = 10, direction_count: int = 128,
                                       game_label: str | None = None,
                                       custom_game_dir: str | Path = CUSTOM_GAME_DIR,
                                       hide_first: bool = False) -> None:
    game_name, payoff_tensor, profiles = _load_result_inputs(input_paths, custom_game_dir)
    _plot_result_equilibrium_trajectory(
        game_name, payoff_tensor, profiles, output_path, trajectory_points, direction_count, game_label, hide_first
    )


def plot_result_equilibrium_convergence(input_paths: str | Path | Iterable[str | Path], distance_output_path: str | Path,
                                        trajectory_output_path: str | Path, checkpoints: Iterable[int] | None = None,
                                        trajectory_points: int = 10, direction_count: int = 128,
                                        game_label: str | None = None,
                                        distance_ready: Callable[[], None] | None = None,
                                        custom_game_dir: str | Path = CUSTOM_GAME_DIR,
                                        hide_first: bool = False) -> None:
    game_name, payoff_tensor, profiles = _load_result_inputs(input_paths, custom_game_dir)
    _plot_result_equilibrium_distance(game_name, payoff_tensor, profiles, distance_output_path, checkpoints, game_label)
    if distance_ready is not None:
        distance_ready()
    _plot_result_equilibrium_trajectory(
        game_name, payoff_tensor, profiles, trajectory_output_path, trajectory_points, direction_count, game_label, hide_first
    )
