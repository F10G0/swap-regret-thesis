import csv

import numpy as np
import pytest

from experiments.result_schema import regret_fieldnames
from experiments.result_trajectories import load_result_action_profiles
from experiments.results import load_final_result_rows
from experiments.plots.plot_joint_actions import joint_action_distribution, mean_joint_action_distribution
from experiments.scenarios.full_information_cross_play import run_full_information_cross_play_experiment
from metrics.equilibrium_convergence import EquilibriumDistanceTrajectory


def test_result_loader_keeps_three_player_profiles_and_final_rows(tmp_path) -> None:
    output_path = tmp_path / "three-player.csv"
    fieldnames = regret_fieldnames("expected")
    base = {field: 0 for field in fieldnames}
    base.update({
        "run_id": "three-player",
        "feedback_mode": "full_information",
        "regret_evaluation": "expected",
        "seed": 42,
        "replicate": 0,
        "stationary_method": "solve",
        "game": "test_three_player",
        "algorithm": "test",
        "algorithm_player_0": "test",
        "algorithm_player_1": "test",
        "horizon": 2,
    })
    profiles = [(0, 2, 1), (1, 0, 0)]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for horizon, profile in enumerate(profiles, start=1):
            for player, action in enumerate(profile):
                writer.writerow(base | {"t": horizon, "player": player, "action": action})

    assert np.array_equal(load_result_action_profiles(output_path, (2, 3, 2)), profiles)
    assert [int(row["player"]) for row in load_final_result_rows(output_path)] == [0, 1, 2]


def test_result_loader_infers_regret_evaluation_for_legacy_csv(tmp_path) -> None:
    output_path = tmp_path / "legacy.csv"
    fieldnames = [field for field in regret_fieldnames("expected") if field != "regret_evaluation"]
    base = {field: 0 for field in fieldnames}
    base.update({
        "run_id": "legacy",
        "feedback_mode": "full_information",
        "seed": 42,
        "replicate": 0,
        "stationary_method": "solve",
        "game": "rps",
        "algorithm": "hedge_vs_hedge",
        "algorithm_player_0": "hedge",
        "algorithm_player_1": "hedge",
        "horizon": 1,
        "t": 1,
    })
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for player in range(2):
            writer.writerow(base | {"player": player, "action": player})

    rows = load_final_result_rows(output_path)

    assert {row["regret_evaluation"] for row in rows} == {"expected"}


def test_result_loader_rejects_declared_evaluation_that_disagrees_with_columns(tmp_path) -> None:
    output_path = tmp_path / "mismatch.csv"
    fieldnames = regret_fieldnames("expected")
    row = {field: 0 for field in fieldnames}
    row.update({
        "run_id": "mismatch",
        "feedback_mode": "full_information",
        "regret_evaluation": "realized",
        "seed": 42,
        "replicate": 0,
        "stationary_method": "solve",
        "game": "rps",
        "algorithm": "hedge_vs_hedge",
        "algorithm_player_0": "hedge",
        "algorithm_player_1": "hedge",
        "horizon": 1,
        "t": 1,
        "player": 0,
        "action": 0,
    })
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match="does not match regret columns"):
        load_final_result_rows(output_path)


def test_joint_action_distributions_are_averaged_across_replicates(tmp_path) -> None:
    paths = [
        run_full_information_cross_play_experiment(
            "rps", ["hedge", "hedge"], horizon=4, seed=42, replicate=replicate, output_dir=tmp_path
        )
        for replicate in range(2)
    ]

    first_game, first = joint_action_distribution(paths[0])
    second_game, second = joint_action_distribution(paths[1])
    game, mean, n_replicates = mean_joint_action_distribution(paths)

    assert first_game == second_game == game == "rps"
    assert n_replicates == 2
    assert np.allclose(mean, (first + second) / 2)
    assert np.sum(mean) == pytest.approx(1.0)


def test_trajectory_uses_uniform_points_while_distance_keeps_logarithmic_checkpoints(tmp_path, monkeypatch) -> None:
    from experiments.plots import plot_equilibrium_convergence

    path = run_full_information_cross_play_experiment(
        "rps", ["hedge", "hedge"], horizon=100, seed=42, output_dir=tmp_path
    )
    distance_horizons = []
    trajectory_horizons = []

    def capture_distance(payoff_tensor, empirical):
        distance_horizons.extend(empirical.horizons)
        zeros = np.zeros(len(empirical.horizons))
        return EquilibriumDistanceTrajectory(empirical.horizons, zeros, zeros)

    def capture_trajectory(payoff_tensor, empirical, direction_count):
        trajectory_horizons.append(empirical.horizons.tolist())
        return object()

    monkeypatch.setattr(plot_equilibrium_convergence, "equilibrium_distance_trajectory", capture_distance)
    monkeypatch.setattr(plot_equilibrium_convergence, "project_equilibrium_trajectory", capture_trajectory)
    monkeypatch.setattr(plot_equilibrium_convergence, "_plot_equilibrium_distance", lambda *args: None)
    monkeypatch.setattr(plot_equilibrium_convergence, "_plot_equilibrium_trajectory", lambda *args: None)

    plot_equilibrium_convergence.plot_result_equilibrium_distance(path, tmp_path / "distance.png")
    plot_equilibrium_convergence.plot_result_equilibrium_trajectory(path, tmp_path / "trajectory.png", trajectory_points=4)
    plot_equilibrium_convergence.plot_result_equilibrium_trajectory(
        path, tmp_path / "trajectory-hidden.png", trajectory_points=4, hide_first=True
    )

    assert distance_horizons == [1, 100]
    assert trajectory_horizons == [[1, 34, 67, 100], [34, 67, 100]]
