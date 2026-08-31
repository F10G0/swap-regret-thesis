import csv

import pytest

from experiments.plots import confidence_free_figure_path
from experiments.plots.plot_adversarial import (
    aggregate_adversarial_regret,
    plot_adversarial_results,
)
from experiments.runner import ExperimentCancelled
from experiments.scenarios.adversarial import (
    AdversarialExperimentSpec,
    HISTORICAL_FREQUENCY_ENVIRONMENT,
    RANDOM_WALK_ENVIRONMENT,
    load_final_adversarial_row,
    load_adversarial_rows,
    run_adversarial_experiment,
)
from tests.support import read_csv_rows as _rows


def test_adversarial_experiment_records_both_regret_sources(tmp_path) -> None:
    output_path = run_adversarial_experiment(
        "hedge",
        n_actions=3,
        horizon=5,
        seed=7,
        output_dir=tmp_path,
    )

    rows = _rows(output_path)

    assert len(rows) == 5
    assert {row["environment"] for row in rows} == {
        HISTORICAL_FREQUENCY_ENVIRONMENT
    }
    assert {row["feedback_mode"] for row in rows} == {"full_information"}
    assert {row["learner_seed"] for row in rows} == {"7"}
    assert {row["replicate"] for row in rows} == {"0"}
    assert rows[0]["punished_actions"] == "0 1"
    assert rows[-1]["t"] == "5"
    assert "average_expected_swap_regret" in rows[0]
    assert "average_realized_swap_regret" in rows[0]
    assert rows[0]["current_best_reward"] == "1.0"
    assert load_adversarial_rows(output_path)[-1] == rows[-1]


def test_bandit_adversarial_experiment_uses_scalar_learner_feedback(
    tmp_path,
) -> None:
    output_path = run_adversarial_experiment(
        "exp3",
        feedback_mode="bandit",
        n_actions=3,
        horizon=5,
        seed=7,
        output_dir=tmp_path,
    )

    rows = _rows(output_path)

    assert {row["feedback_mode"] for row in rows} == {"bandit"}
    assert len(rows) == 5
    assert "average_expected_external_regret" in rows[-1]
    assert "average_realized_external_regret" in rows[-1]


@pytest.mark.parametrize(
    ("regret_evaluation", "present", "absent"),
    [
        ("expected", "expected", "realized"),
        ("realized", "realized", "expected"),
        ("both", "expected", None),
    ],
)
def test_adversarial_regret_evaluation_controls_recorded_sources(
    tmp_path,
    regret_evaluation: str,
    present: str,
    absent: str | None,
) -> None:
    output_path = run_adversarial_experiment(
        "exp3",
        feedback_mode="bandit",
        horizon=5,
        seed=7,
        regret_evaluation=regret_evaluation,
        output_dir=tmp_path,
    )

    rows = _rows(output_path)

    assert {row["regret_evaluation"] for row in rows} == {regret_evaluation}
    assert f"average_{present}_external_regret" in rows[0]
    if regret_evaluation == "both":
        assert "average_realized_external_regret" in rows[0]
    else:
        assert f"average_{absent}_external_regret" not in rows[0]


def test_adversarial_regret_evaluation_does_not_change_play(tmp_path) -> None:
    paths = [
        run_adversarial_experiment(
            "exp3",
            feedback_mode="bandit",
            horizon=20,
            seed=7,
            regret_evaluation=evaluation,
            output_dir=tmp_path,
        )
        for evaluation in ("expected", "both")
    ]
    trajectories = [
        [(row["action"], row["payoff"]) for row in _rows(path)]
        for path in paths
    ]

    assert paths[0] != paths[1]
    assert trajectories[0] == trajectories[1]


def test_random_walk_experiment_records_environment_metadata(tmp_path) -> None:
    output_path = run_adversarial_experiment(
        "exp3",
        feedback_mode="bandit",
        environment=RANDOM_WALK_ENVIRONMENT,
        initialization_mode="uniform_grid",
        environment_seed=11,
        n_actions=3,
        horizon=5,
        seed=7,
        output_dir=tmp_path,
    )
    rows = _rows(output_path)

    assert {row["environment"] for row in rows} == {RANDOM_WALK_ENVIRONMENT}
    assert {row["initialization_mode"] for row in rows} == {"uniform_grid"}
    assert {row["reward_step"] for row in rows} == {"0.1"}
    assert {row["environment_seed"] for row in rows} == {"11"}
    assert {row["learner_seed"] for row in rows} == {"7"}
    assert all(row["punished_actions"] == "" for row in rows)
    assert all(0 <= int(row["current_best_action"]) < 3 for row in rows)
    assert all(0.0 <= float(row["current_best_reward"]) <= 1.0 for row in rows)
    assert load_adversarial_rows(output_path)[-1] == rows[-1]


def test_algorithms_share_random_walk_environment_trajectory(tmp_path) -> None:
    paths = [
        run_adversarial_experiment(
            algorithm,
            environment=RANDOM_WALK_ENVIRONMENT,
            initialization_mode="uniform_grid",
            environment_seed=11,
            n_actions=3,
            horizon=30,
            seed=learner_seed,
            output_dir=tmp_path,
        )
        for algorithm, learner_seed in (("hedge", 7), ("regret_matching", 19))
    ]
    trajectories = [
        [
            (row["current_best_action"], row["current_best_reward"])
            for row in _rows(path)
        ]
        for path in paths
    ]

    assert trajectories[0] == trajectories[1]


def test_adversarial_seeds_have_distinct_identity_and_streams(tmp_path) -> None:
    first = run_adversarial_experiment(
        "hedge",
        n_actions=3,
        horizon=20,
        seed=7,
        output_dir=tmp_path,
    )
    second = run_adversarial_experiment(
        "hedge",
        n_actions=3,
        horizon=20,
        seed=8,
        output_dir=tmp_path,
    )

    first_rows = _rows(first)
    second_rows = _rows(second)

    assert first != second
    assert [row["action"] for row in first_rows] != [
        row["action"] for row in second_rows
    ]


def test_adversarial_replicate_offsets_both_random_seeds(tmp_path) -> None:
    output_path = run_adversarial_experiment(
        "exp3",
        feedback_mode="bandit",
        environment=RANDOM_WALK_ENVIRONMENT,
        environment_seed=11,
        n_actions=3,
        horizon=5,
        seed=7,
        replicate=2,
        output_dir=tmp_path,
    )

    rows = _rows(output_path)

    assert {row["replicate"] for row in rows} == {"2"}
    assert {row["environment_seed"] for row in rows} == {"13"}
    assert {row["learner_seed"] for row in rows} == {"9"}


def test_adversarial_replicate_is_part_of_run_identity() -> None:
    common = {
        "algorithm_name": "hedge",
        "n_actions": 3,
        "horizon": 10,
        "seed": 7,
    }

    assert AdversarialExperimentSpec(**common).run_id != (
        AdversarialExperimentSpec(**common, replicate=1).run_id
    )


def test_adversarial_loader_accepts_legacy_csv_without_replicate(tmp_path) -> None:
    generated = run_adversarial_experiment(
        "hedge",
        horizon=3,
        output_dir=tmp_path,
    )
    rows = _rows(generated)
    legacy_path = tmp_path / "legacy.csv"
    legacy_fields = {"replicate", "regret_evaluation", "implementation_version"}
    fieldnames = [field for field in rows[0] if field not in legacy_fields]
    with legacy_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {field: value for field, value in row.items() if field not in legacy_fields}
            for row in rows
        )

    loaded = load_adversarial_rows(legacy_path)

    assert {row["replicate"] for row in loaded} == {"0"}
    assert {row["regret_evaluation"] for row in loaded} == {"both"}
    assert {row["implementation_version"] for row in loaded} == {"0"}
    assert load_final_adversarial_row(legacy_path)["replicate"] == "0"


def test_adversarial_implementation_version_changes_run_identity() -> None:
    common = {"algorithm_name": "hedge", "n_actions": 3, "horizon": 10, "seed": 7}

    assert AdversarialExperimentSpec(**common).run_id != AdversarialExperimentSpec(**common, implementation_version=2).run_id


def test_adversarial_regret_aggregation_uses_student_t_intervals() -> None:
    trajectories = [
        [
            {"t": "1", "average_expected_external_regret": "1"},
            {"t": "2", "average_expected_external_regret": "2"},
        ],
        [
            {"t": "1", "average_expected_external_regret": "3"},
            {"t": "2", "average_expected_external_regret": "6"},
        ],
    ]

    times, means, confidence = aggregate_adversarial_regret(
        trajectories,
        "average_expected_external_regret",
    )

    assert times.tolist() == [1, 2]
    assert means.tolist() == [2, 4]
    assert confidence == pytest.approx([12.706204736, 25.412409472])


def test_adversarial_experiment_is_atomic_on_cancellation(tmp_path) -> None:
    with pytest.raises(ExperimentCancelled):
        run_adversarial_experiment(
            "hedge",
            horizon=3,
            output_dir=tmp_path,
            should_cancel=lambda: True,
        )

    assert list(tmp_path.iterdir()) == []


def test_adversarial_plotter_creates_average_and_scaled_regret_figures(
    tmp_path,
) -> None:
    raw_dir = tmp_path / "raw"
    figure_dir = tmp_path / "figures"
    run_adversarial_experiment(
        "hedge",
        n_actions=3,
        horizon=5,
        seed=7,
        output_dir=raw_dir,
    )
    run_adversarial_experiment(
        "exp3",
        feedback_mode="bandit",
        environment=RANDOM_WALK_ENVIRONMENT,
        environment_seed=17,
        n_actions=3,
        horizon=5,
        seed=8,
        output_dir=raw_dir,
    )

    generated = plot_adversarial_results(raw_dir, figure_dir)

    expected_regret_figures = {
        f"adversarial_{environment}_{feedback}_3_actions_average_{source}_{regret}_regret.png"
        for environment, feedback in (
            (HISTORICAL_FREQUENCY_ENVIRONMENT, "full_information"),
            (RANDOM_WALK_ENVIRONMENT, "bandit"),
        )
        for source in ("expected", "realized")
        for regret in ("external", "internal", "swap")
    }
    expected_scaling_figures = {
        f"adversarial_{environment}_{feedback}_3_actions_{source}_{regret}_regret_over_sqrt_t.png"
        for environment, feedback in (
            (HISTORICAL_FREQUENCY_ENVIRONMENT, "full_information"),
            (RANDOM_WALK_ENVIRONMENT, "bandit"),
        )
        for source in ("expected", "realized")
        for regret in ("external", "internal", "swap")
    }
    assert len(generated) == 24
    assert {path.name for path in generated} == expected_regret_figures | expected_scaling_figures
    assert all(path.with_suffix(".pdf").is_file() for path in generated)


def test_adversarial_plotter_only_generates_selected_regret_source(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    figure_dir = tmp_path / "figures"
    run_adversarial_experiment(
        "hedge",
        n_actions=3,
        horizon=5,
        regret_evaluation="expected",
        output_dir=raw_dir,
    )

    generated = plot_adversarial_results(raw_dir, figure_dir)

    assert len(generated) == 6
    assert not any("realized" in path.name for path in generated)


def test_adversarial_plotter_caches_mean_only_figures_for_replicates(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    figure_dir = tmp_path / "figures"
    run_adversarial_experiment("hedge", horizon=5, seed=7, output_dir=raw_dir)
    run_adversarial_experiment(
        "hedge",
        horizon=5,
        seed=7,
        replicate=1,
        output_dir=raw_dir,
    )

    generated = plot_adversarial_results(raw_dir, figure_dir)

    regret_path = next(path for path in generated if "average_expected_external" in path.name)
    confidence_free_path = confidence_free_figure_path(regret_path)
    assert confidence_free_path.is_file()
    assert confidence_free_path.with_suffix(".pdf").is_file()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_actions": 1}, "between 2"),
        ({"horizon": 0}, "positive"),
        ({"seed": -1}, "non-negative"),
        ({"replicate": -1}, "replicate"),
        ({"regret_evaluation": "unknown"}, "regret evaluation"),
        ({"environment": RANDOM_WALK_ENVIRONMENT, "environment_seed": -1}, "environment seed"),
        ({"environment": "unknown"}, "unknown adversarial environment"),
        ({"environment": RANDOM_WALK_ENVIRONMENT, "initialization_mode": "unknown"}, "unknown random-walk initialization"),
    ],
)
def test_adversarial_spec_validation(changes, message) -> None:
    values = {
        "algorithm_name": "hedge",
        "n_actions": 3,
        "horizon": 10,
        "seed": 7,
    }
    with pytest.raises(ValueError, match=message):
        AdversarialExperimentSpec(**(values | changes))


def test_adversarial_feedback_mode_is_part_of_run_identity() -> None:
    common = {
        "algorithm_name": "bm",
        "n_actions": 3,
        "horizon": 10,
        "seed": 7,
    }

    assert AdversarialExperimentSpec(
        **common,
        feedback_mode="full_information",
    ).run_id != AdversarialExperimentSpec(
        **common,
        feedback_mode="bandit",
    ).run_id


def test_adversarial_regret_evaluation_is_part_of_run_identity() -> None:
    common = {
        "algorithm_name": "hedge",
        "n_actions": 3,
        "horizon": 10,
        "seed": 7,
    }

    assert AdversarialExperimentSpec(
        **common,
        regret_evaluation="expected",
    ).run_id != AdversarialExperimentSpec(
        **common,
        regret_evaluation="realized",
    ).run_id


def test_random_walk_seeds_and_initialization_are_part_of_identity() -> None:
    common = {
        "algorithm_name": "hedge",
        "n_actions": 3,
        "horizon": 10,
        "seed": 7,
        "environment": RANDOM_WALK_ENVIRONMENT,
    }

    baseline = AdversarialExperimentSpec(**common)
    assert baseline.run_id != AdversarialExperimentSpec(
        **(common | {"environment_seed": 8})
    ).run_id
    assert baseline.run_id != AdversarialExperimentSpec(
        **(common | {"seed": 8})
    ).run_id
    assert baseline.run_id != AdversarialExperimentSpec(
        **(common | {"initialization_mode": "uniform_grid"})
    ).run_id


def test_adversarial_algorithm_must_match_feedback_mode() -> None:
    with pytest.raises(ValueError, match="not available for bandit"):
        AdversarialExperimentSpec(
            algorithm_name="hedge",
            feedback_mode="bandit",
            n_actions=3,
            horizon=10,
            seed=7,
        )
