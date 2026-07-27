import csv

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.plots.plot_regret import aggregate_metric_curve, collect_results, plot_regret, run_label
from experiments.scenarios.cross_play import player_seed
from experiments.result_schema import regret_fieldnames
from experiments.spec import MAX_RUN_ID_BYTES, ExperimentSpec


def make_spec(
    feedback_mode: str = "full_information",
    seed: int = 7,
    replicate: int = 0,
    regret_evaluation: str = "feedback_aligned",
) -> ExperimentSpec:
    return ExperimentSpec(
        game_name="rps",
        feedback_mode=feedback_mode,
        algorithm_names=("bm", "bm"),
        horizon=10,
        seed=seed,
        replicate=replicate,
        regret_evaluation=regret_evaluation,
    )


def write_result(path, spec: ExperimentSpec) -> None:
    fieldnames = regret_fieldnames(spec.regret_evaluation)
    row = {field: 0 for field in fieldnames}
    row.update(spec.metadata())
    row.update(
        {
            "game": spec.game_name,
            "algorithm": spec.algorithm_profile_name,
            "algorithm_player_0": spec.algorithm_names[0],
            "algorithm_player_1": spec.algorithm_names[1],
            "horizon": spec.horizon,
            "t": 1,
            "player": 0,
        }
    )

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def test_run_id_changes_with_experiment_configuration() -> None:
    baseline = make_spec()

    assert baseline.run_id != make_spec(feedback_mode="bandit").run_id
    assert baseline.run_id != make_spec(regret_evaluation="both").run_id
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


def test_experiment_spec_preserves_positional_stationary_method_compatibility() -> None:
    spec = ExperimentSpec("rps", "full_information", ("bm", "bm"), 10, 7, 0, "pinv")

    assert spec.stationary_method == "pinv"
    assert spec.regret_evaluation == "expected"


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


def test_replicates_derive_distinct_reproducible_player_seeds() -> None:
    first = make_spec(feedback_mode="bandit", replicate=0)
    second = make_spec(feedback_mode="bandit", replicate=1)

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

    times, means, confidence = aggregate_metric_curve(
        replicate_runs,
        player=0,
        column="regret",
    )

    assert np.array_equal(times, [1, 2])
    assert np.array_equal(means, [2.0, 3.0])
    assert np.all(confidence > 0.0)


def test_plot_collection_keeps_feedback_modes_separate(tmp_path) -> None:
    full_spec = make_spec(feedback_mode="full_information")
    bandit_spec = make_spec(feedback_mode="bandit")
    write_result(tmp_path / f"{full_spec.run_id}.csv", full_spec)
    write_result(tmp_path / f"{bandit_spec.run_id}.csv", bandit_spec)

    results = collect_results(tmp_path)

    assert set(results["rps"]) == {full_spec.run_id, bandit_spec.run_id}


def test_plot_legend_uses_algorithm_abbreviations() -> None:
    rows = [{"algorithm": "regret_matching_vs_stationary_regret_matching_vs_hedge", "seed": "7"}]

    assert run_label(rows, 1) == "RM vs SRM vs Hedge · seed 7"


def test_plot_legend_can_distinguish_feedback_and_evaluation() -> None:
    rows = [{
        "algorithm": "exp3_vs_exp3",
        "seed": "7",
        "feedback_mode": "bandit",
        "regret_evaluation": "expected",
    }]

    assert run_label(rows, 1, include_feedback=True, include_evaluation=True) == "EXP3 vs EXP3 · seed 7 · bandit · evaluation expected"


def test_plot_legend_stays_below_the_data_axes(tmp_path, monkeypatch) -> None:
    replicate_groups = []
    for index in range(25):
        row = {
            "player": "0",
            "t": "1",
            "average_expected_external_regret": str(index),
            "feedback_mode": "full_information",
            "regret_evaluation": "expected",
            "algorithm": f"stationary_regret_matching_{index}_vs_stationary_regret_matching_{index}",
            "seed": "7",
            "stationary_method": "solve",
        }
        replicate_groups.append([[row]])

    close_figure = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: None)
    plot_regret("rps", replicate_groups, "expected", "external", player=0, average=True, output_dir=tmp_path)
    figure = plt.gcf()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_box = figure.axes[0].get_window_extent(renderer)
    legend_box = figure.legends[0].get_window_extent(renderer)

    assert legend_box.y1 < axes_box.y0
    assert 0.0 <= legend_box.x0 < legend_box.x1 <= figure.bbox.width
    assert figure.get_figheight() > 4.4
    assert (tmp_path / "rps_average_expected_external_regret_player_0.png").is_file()
    close_figure(figure)
