import csv

import numpy as np
import pytest

from experiments.scenarios.cross_play import run_cross_play_experiment
from experiments.plots.plot_joint_actions import (
    joint_action_distribution,
    mean_joint_action_distribution,
)
from experiments.recording import encode_joint_action_histograms
from experiments.result_schema import JOINT_ACTION_HISTOGRAM_FIELD, regret_fieldnames
from experiments.result_trajectories import load_result_empirical_distribution_trajectory
from experiments.results import iter_result_rows, load_final_result_rows
from experiments.runtime_environment import runtime_environment_fingerprint, runtime_environment_json


def test_result_loader_keeps_three_player_histograms_and_final_rows(
    tmp_path,
) -> None:
    output_path = tmp_path / "three-player.csv"
    fieldnames = regret_fieldnames()
    base = {field: 0 for field in fieldnames}
    base[JOINT_ACTION_HISTOGRAM_FIELD] = ""
    runtime_environment = runtime_environment_json()
    base.update({
        "run_id": "three-player",
        "feedback_mode": "full_information",
        "runtime_environment": runtime_environment,
        "runtime_fingerprint": runtime_environment_fingerprint(runtime_environment),
        "seed": 42,
        "replicate": 0,
        "stationary_method": "solve",
        "game": "test_three_player",
        "algorithm": "test",
        "algorithm_profile": '["test","test","test"]',
        "horizon": 2,
    })
    profiles = [(0, 2, 1), (1, 0, 0)]
    first_counts = np.zeros((2, 3, 2), dtype=int)
    first_counts[profiles[0]] += 1
    final_counts = first_counts.copy()
    final_counts[profiles[1]] += 1
    histogram_payload = encode_joint_action_histograms([1, 2], [first_counts, final_counts])
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for horizon, profile in enumerate(profiles, start=1):
            for player, action in enumerate(profile):
                writer.writerow(
                    base
                    | {
                        "t": horizon,
                        "player": player,
                        "action": action,
                        JOINT_ACTION_HISTOGRAM_FIELD: histogram_payload if horizon == 2 and player == 0 else "",
                    }
                )

    trajectory = load_result_empirical_distribution_trajectory(output_path, (2, 3, 2))
    assert trajectory.distributions.shape == (2, 2, 3, 2)
    np.testing.assert_array_equal(trajectory.distributions[0], first_counts)
    np.testing.assert_array_equal(trajectory.distributions[1], final_counts / 2)
    assert [
        int(row["player"])
        for row in load_final_result_rows(output_path)
    ] == [0, 1, 2]


def test_streaming_result_loader_accepts_checkpoint_gaps(tmp_path) -> None:
    output_path = run_cross_play_experiment(
        "rps",
        ["hedge", "hedge"],
        horizon=3,
        output_dir=tmp_path,
        feedback_mode="full_information",
    )
    with output_path.open(newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row["t"] != "2"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    assert [int(row["t"]) for row in iter_result_rows(output_path)] == [1, 1, 3, 3]
    trajectory = load_result_empirical_distribution_trajectory(output_path, (3, 3))
    assert trajectory.horizons.tolist() == [1, 3]
    np.testing.assert_allclose(trajectory.vectors.sum(axis=1), 1.0)


def test_joint_action_distributions_are_averaged_across_replicates(
    tmp_path,
) -> None:
    paths = [
        run_cross_play_experiment(
            "rps",
            ["hedge", "hedge"],
            horizon=4,
            seed=42,
            replicate=replicate,
            output_dir=tmp_path,
            feedback_mode="full_information",
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
