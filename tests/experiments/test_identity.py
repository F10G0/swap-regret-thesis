import csv
from hashlib import sha256
import json
import os

import numpy as np
import pytest

from experiments.scenarios.cross_play import run_cross_play_experiment, player_seed
from experiments.game_catalog import GameCatalog, payoff_tensor_digest
from experiments.plots.plot_regret import aggregate_metric_curve
from experiments.recording import encode_joint_action_histograms, joint_action_histogram_checkpoints
from experiments.result_schema import JOINT_ACTION_HISTOGRAM_FIELD, regret_fieldnames
from experiments.results import (
    iter_result_rows,
    result_runtime_environment,
    result_runtime_fingerprint,
)
from experiments.runtime_environment import (
    runtime_environment_fingerprint,
    runtime_environment_json,
)
from experiments.spec import MAX_RUN_ID_BYTES, ExperimentSpec


def make_spec(
    feedback_mode: str = "full_information",
    seed: int = 7,
    replicate: int = 0,
) -> ExperimentSpec:
    return ExperimentSpec(
        game_name="rps",
        feedback_mode=feedback_mode,
        algorithm_names=("bm", "bm"),
        horizon=10,
        seed=seed,
        replicate=replicate,
    )


def write_result(path, spec: ExperimentSpec) -> None:
    fieldnames = regret_fieldnames()
    row = {field: 0 for field in fieldnames}
    row[JOINT_ACTION_HISTOGRAM_FIELD] = ""
    row.update(spec.metadata())
    row.update(
        {
            "game": spec.game_name,
            "algorithm": spec.algorithm_profile_name,
            "horizon": spec.horizon,
        }
    )
    histogram_horizons = list(joint_action_histogram_checkpoints(spec.horizon))
    histogram_counts = []
    for horizon in histogram_horizons:
        counts = np.zeros((3, 3), dtype=int)
        counts[0, 0] = horizon
        histogram_counts.append(counts)
    histogram_payload = encode_joint_action_histograms(histogram_horizons, histogram_counts)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for time in range(1, spec.horizon + 1):
            for player in range(len(spec.algorithm_names)):
                writer.writerow(
                    row
                    | {
                        "t": time,
                        "player": player,
                        JOINT_ACTION_HISTOGRAM_FIELD: histogram_payload if time == spec.horizon and player == 0 else "",
                    }
                )


def test_run_id_changes_with_experiment_configuration() -> None:
    baseline = make_spec()

    assert baseline.run_id != make_spec(feedback_mode="bandit").run_id
    assert baseline.run_id != make_spec(seed=8).run_id
    changed_horizon = ExperimentSpec(
        game_name="rps",
        feedback_mode="full_information",
        algorithm_names=("bm", "bm"),
        horizon=20,
        seed=7,
    )
    assert baseline.run_id != changed_horizon.run_id
    assert baseline.run_id != ExperimentSpec("rps", "full_information", ("bm", "bm"), 10, 7, stationary_method="pinv").run_id


def test_runtime_environment_changes_identity_and_is_recorded(tmp_path) -> None:
    baseline = make_spec()
    changed = ExperimentSpec(
        "rps",
        "full_information",
        ("bm", "bm"),
        10,
        7,
        runtime_environment='{"packages":{"numpy":"different"},"python":"3.10"}',
    )
    result_path = tmp_path / f"{changed.run_id}.csv"
    write_result(result_path, changed)
    row = next(iter_result_rows(result_path))

    assert baseline.run_id != changed.run_id
    assert result_runtime_environment(row) == changed.runtime_environment
    assert result_runtime_fingerprint(row) == runtime_environment_fingerprint(
        changed.runtime_environment
    )


def test_runtime_environment_records_the_dependency_lock() -> None:
    environment = json.loads(runtime_environment_json())
    lock_path = os.path.join(os.getcwd(), "requirements.lock")

    with open(lock_path, "rb") as lock_file:
        expected = sha256(lock_file.read()).hexdigest()

    assert environment["lock_sha256"] == expected
    assert environment["packages"]["numpy"]["version"] == np.__version__


def test_experiment_spec_preserves_positional_stationary_method_compatibility() -> None:
    spec = ExperimentSpec("rps", "full_information", ("bm", "bm"), 10, 7, 0, "pinv")

    assert spec.stationary_method == "pinv"


def test_long_srm_profile_uses_readable_abbreviated_run_id() -> None:
    spec = ExperimentSpec("custom__eight-player", "full_information", ("stationary_regret_matching",) * 8, 10, 7)

    assert len(spec.run_id.encode("utf-8")) <= MAX_RUN_ID_BYTES
    assert "_srm_vs_srm_vs_srm_vs_srm_vs_srm_vs_srm_vs_srm_vs_srm_" in spec.run_id
    assert "stationary_regret_matching" not in spec.run_id
    assert spec.metadata()["algorithm_profile"].count("stationary_regret_matching") == 8


def test_unabbreviated_long_profile_uses_compact_run_id() -> None:
    long_name = "algorithm_with_an_exceptionally_long_internal_identifier"
    spec = ExperimentSpec("custom__eight-player", "full_information", (long_name,) * 8, 10, 7)

    assert len(spec.run_id.encode("utf-8")) <= MAX_RUN_ID_BYTES
    assert "_8p_" in spec.run_id
    assert long_name not in spec.run_id


def test_experiment_spec_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_spec(seed=-1)


def test_payoff_tensor_fingerprint_changes_run_identity_and_csv_metadata(
    tmp_path,
) -> None:
    first_games = tmp_path / "first-games"
    second_games = tmp_path / "second-games"
    first_definition = GameCatalog(first_games).create_random("same", 2, [2, 2], 1)
    second_definition = GameCatalog(second_games).create_random("same", 2, [2, 2], 2)
    assert first_definition.id == second_definition.id

    first_path = run_cross_play_experiment(
        first_definition.id,
        ["hedge", "hedge"],
        horizon=1,
        output_dir=tmp_path / "first-results",
        custom_game_dir=first_games,
        feedback_mode="full_information",
    )
    second_path = run_cross_play_experiment(
        second_definition.id,
        ["hedge", "hedge"],
        horizon=1,
        output_dir=tmp_path / "second-results",
        custom_game_dir=second_games,
        feedback_mode="full_information",
    )

    assert first_path.name != second_path.name
    with first_path.open(newline="") as file:
        first_digest = next(csv.DictReader(file))["game_payoff_digest"]
    with second_path.open(newline="") as file:
        second_digest = next(csv.DictReader(file))["game_payoff_digest"]
    assert first_digest != second_digest
    assert first_digest == payoff_tensor_digest(GameCatalog(first_games).load(first_definition.id))
    assert second_digest == payoff_tensor_digest(GameCatalog(second_games).load(second_definition.id))


@pytest.mark.parametrize("feedback_mode", ["full_information", "bandit"])
def test_replicates_derive_distinct_reproducible_player_seeds(
    feedback_mode: str,
) -> None:
    first = make_spec(feedback_mode=feedback_mode, replicate=0)
    second = make_spec(feedback_mode=feedback_mode, replicate=1)

    assert [player_seed(first, player) for player in range(2)] == [7, 8]
    assert [player_seed(second, player) for player in range(2)] == [9, 10]


def test_metric_curves_are_averaged_across_replicates() -> None:
    replicate_runs = [
        [
            {"player": "0", "t": "1", "regret": "1.0"},
            {"player": "0", "t": "2", "regret": "2.0"},
        ],
        [
            {"player": "0", "t": "1", "regret": "3.0"},
            {"player": "0", "t": "2", "regret": "4.0"},
        ],
    ]

    times, means = aggregate_metric_curve(
        replicate_runs,
        player=0,
        column="regret",
    )

    assert np.array_equal(times, [1, 2])
    assert np.array_equal(means, [2.0, 3.0])


def test_fixed_game_loader_rejects_missing_runtime_identity(tmp_path) -> None:
    spec = make_spec()
    result_path = tmp_path / f"{spec.run_id}.csv"
    write_result(result_path, spec)
    with result_path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    missing_fields = {"runtime_environment", "runtime_fingerprint"}
    fieldnames = [field for field in reader.fieldnames if field not in missing_fields]
    with result_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fieldnames} for row in rows)

    with pytest.raises(ValueError, match="missing required columns"):
        list(iter_result_rows(result_path))

