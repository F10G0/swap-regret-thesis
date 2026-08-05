import csv

import numpy as np
import pytest

from experiments.plots.plot_joint_actions import (
    joint_action_distribution,
    mean_joint_action_distribution,
)
from experiments.result_schema import regret_fieldnames
from experiments.result_trajectories import load_result_action_profiles
from experiments.results import iter_result_rows, load_final_result_rows
from experiments.scenarios.full_information_cross_play import (
    run_full_information_cross_play_experiment,
)


def test_result_loader_keeps_three_player_profiles_and_final_rows(
    tmp_path,
) -> None:
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
        "algorithm_profile": '["test","test","test"]',
        "horizon": 2,
    })
    profiles = [(0, 2, 1), (1, 0, 0)]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for horizon, profile in enumerate(profiles, start=1):
            for player, action in enumerate(profile):
                writer.writerow(
                    base
                    | {"t": horizon, "player": player, "action": action}
                )

    assert np.array_equal(
        load_result_action_profiles(output_path, (2, 3, 2)),
        profiles,
    )
    assert [
        int(row["player"])
        for row in load_final_result_rows(output_path)
    ] == [0, 1, 2]


def test_streaming_result_loader_rejects_missing_round(tmp_path) -> None:
    output_path = run_full_information_cross_play_experiment(
        "rps",
        ["hedge", "hedge"],
        horizon=3,
        output_dir=tmp_path,
    )
    with output_path.open(newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row["t"] != "2"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="gap between rounds"):
        list(iter_result_rows(output_path))


def test_joint_action_distributions_are_averaged_across_replicates(
    tmp_path,
) -> None:
    paths = [
        run_full_information_cross_play_experiment(
            "rps",
            ["hedge", "hedge"],
            horizon=4,
            seed=42,
            replicate=replicate,
            output_dir=tmp_path,
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
