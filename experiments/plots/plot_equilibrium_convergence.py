"""Core full-space CE/CCE distance-convergence plotting."""

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import CUSTOM_GAME_DIR
from experiments.game_catalog import load_game_payoffs, payoff_tensor_digest
from experiments.plots import save_figure_pair
from experiments.result_trajectories import load_result_action_profiles
from experiments.results import iter_result_rows, result_game_payoff_digest
from metrics.empirical_distribution import (
    EmpiricalDistributionTrajectory,
    empirical_distribution_trajectory,
)
from metrics.equilibrium_convergence import (
    ReplicateEquilibriumDistanceTrajectory,
    aggregate_equilibrium_distance_trajectories,
    equilibrium_distance_trajectory,
)


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
    profiles = [
        load_result_action_profiles(path, payoff_tensor.shape[1:])
        for path in paths
    ]
    return game_name, payoff_tensor, profiles


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
) -> None:
    game_name, payoff_tensor, profiles = load_equilibrium_result_inputs(
        input_paths,
        custom_game_dir,
    )
    empirical_trajectories = empirical_distribution_trajectories(
        profiles,
        payoff_tensor.shape[1:],
        checkpoints,
    )
    replicate_distances = [
        equilibrium_distance_trajectory(payoff_tensor, trajectory)
        for trajectory in empirical_trajectories
    ]
    distances = aggregate_equilibrium_distance_trajectories(
        replicate_distances
    )
    _plot_equilibrium_distance(
        distances,
        output_path,
        game_label or game_name,
    )
