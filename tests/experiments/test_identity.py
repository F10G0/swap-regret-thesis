import csv
from dataclasses import replace
from hashlib import sha256
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.scenarios.cross_play import run_cross_play_experiment, player_seed
from experiments.game_catalog import GameCatalog, payoff_tensor_digest
from experiments.plots.plot_regret import aggregate_metric_curve, plot_regret
from experiments.plots.style import curve_labels
from experiments.result_schema import RESULT_IMPLEMENTATION_VERSION, regret_fieldnames
from experiments.results import (
    iter_result_rows,
    result_implementation_version,
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
    row.update(spec.metadata())
    row.update(
        {
            "game": spec.game_name,
            "algorithm": spec.algorithm_profile_name,
            "horizon": spec.horizon,
        }
    )

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
    assert baseline.run_id != ExperimentSpec("rps", "full_information", ("bm", "bm"), 10, 7, implementation_version=2).run_id


@pytest.mark.parametrize("version", [3, 4, 5])
def test_v6_identity_rejects_but_does_not_modify_legacy_results(tmp_path, version) -> None:
    current = make_spec()
    legacy = replace(current, implementation_version=version)
    assert RESULT_IMPLEMENTATION_VERSION == current.implementation_version == 6
    assert current.run_id != legacy.run_id
    legacy_path = tmp_path / f"{legacy.run_id}.csv"
    current_path = tmp_path / f"{current.run_id}.csv"
    write_result(legacy_path, legacy)
    legacy_bytes = legacy_path.read_bytes()
    write_result(current_path, current)

    with pytest.raises(ValueError, match=f"incompatible result implementation_version {version}"):
        list(iter_result_rows(legacy_path))
    assert {result_implementation_version(row) for row in iter_result_rows(current_path)} == {6}
    assert legacy_path.read_bytes() == legacy_bytes


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
    assert "regret_evaluation" not in spec.metadata()


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


def test_fixed_game_loader_rejects_missing_version(tmp_path) -> None:
    spec = make_spec()
    result_path = tmp_path / f"{spec.run_id}.csv"
    write_result(result_path, spec)
    with result_path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    legacy_fields = {
        "implementation_version",
        "runtime_environment",
        "runtime_fingerprint",
    }
    fieldnames = [field for field in reader.fieldnames if field not in legacy_fields]
    with result_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fieldnames} for row in rows)

    with pytest.raises(ValueError, match="incompatible result implementation_version 0"):
        list(iter_result_rows(result_path))


def test_plot_legend_uses_algorithm_abbreviations() -> None:
    rows = [{"algorithm": "regret_matching_vs_stationary_regret_matching_vs_hedge", "seed": "7"}]

    assert curve_labels(rows) == ["RM vs SRM vs Hedge"]


@pytest.mark.parametrize("algorithm,label", [("exp3_ix", "EXP3-IX"), ("exp3", "EXP3")])
def test_plot_legend_can_distinguish_feedback(algorithm, label) -> None:
    # Retired algorithms must keep their own labels when reading old results.
    rows = [{
        "algorithm": f"{algorithm}_vs_{algorithm}",
        "seed": "7",
        "feedback_mode": "bandit",
    }]

    assert curve_labels(rows) == [label]
    assert curve_labels([rows[0], rows[0] | {"feedback_mode": "full_information"}]) == [
        f"{label} · bandit", f"{label} · full info",
    ]


def test_plot_legend_stays_below_the_data_axes(tmp_path, monkeypatch) -> None:
    replicate_groups = []
    for index in range(25):
        row = {
            "player": "0",
            "t": "1",
            "average_external_regret": str(index),
            "feedback_mode": "full_information",
            "algorithm": f"stationary_regret_matching_{index}_vs_stationary_regret_matching_{index}",
            "seed": "7",
            "stationary_method": "solve",
        }
        replicate_groups.append([[row]])

    close_figure = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: None)
    plot_regret("rps", replicate_groups, "external", player=0, average=True, output_dir=tmp_path)
    figure = plt.gcf()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_box = figure.axes[0].get_window_extent(renderer)
    legend_box = figure.legends[0].get_window_extent(renderer)

    assert legend_box.y1 < axes_box.y0
    assert 0.0 <= legend_box.x0 < legend_box.x1 <= figure.bbox.width
    assert figure.get_figheight() > 4.4
    assert (tmp_path / "rps_average_external_regret_player_0.png").is_file()
    close_figure(figure)
