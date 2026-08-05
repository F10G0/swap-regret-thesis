import pytest

from experiments.plots.plot_adversarial import plot_adversarial_results
from experiments.runner import ExperimentCancelled
from experiments.scenarios.adversarial import (
    AdversarialExperimentSpec,
    HISTORICAL_FREQUENCY_ENVIRONMENT,
    RANDOM_WALK_ENVIRONMENT,
    adversarial_memory_label,
    adversarial_memory_window,
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
    assert {row["memory"] for row in rows} == {"full_history"}
    assert {row["feedback_mode"] for row in rows} == {"full_information"}
    assert {row["learner_seed"] for row in rows} == {"7"}
    assert "replicate" not in rows[0]
    assert rows[0]["punished_action"] == "0"
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
    assert all(row["punished_action"] == "" for row in rows)
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
        f"adversarial_3_actions_average_{source}_{regret}_regret.png"
        for source in ("expected", "realized")
        for regret in ("external", "internal", "swap")
    }
    expected_scaling_figures = {
        f"adversarial_3_actions_{source}_{regret}_regret_over_sqrt_t.png"
        for source in ("expected", "realized")
        for regret in ("external", "internal", "swap")
    }
    historical_path = next(raw_dir.glob("historical_frequency_*.csv"))
    random_walk_path = next(raw_dir.glob("lazy_random_walk_*.csv"))
    expected_behavior_figures = {
        f"{historical_path.stem}_action_frequency.png",
        f"{historical_path.stem}_punished_action_frequency.png",
        f"{random_walk_path.stem}_action_frequency.png",
        f"{random_walk_path.stem}_best_action_frequency.png",
    }
    assert len(generated) == 16
    assert {path.name for path in generated} == (
        expected_regret_figures
        | expected_scaling_figures
        | expected_behavior_figures
    )
    assert all(path.with_suffix(".pdf").is_file() for path in generated)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_actions": 1}, "between 2"),
        ({"horizon": 0}, "positive"),
        ({"seed": -1}, "non-negative"),
        ({"memory_window": -1}, "non-negative"),
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
        "memory_window": 0,
    }
    with pytest.raises(ValueError, match=message):
        AdversarialExperimentSpec(**(values | changes))


def test_adversarial_memory_is_part_of_run_identity() -> None:
    common = {
        "algorithm_name": "hedge",
        "n_actions": 3,
        "horizon": 10,
        "seed": 7,
    }

    assert AdversarialExperimentSpec(
        **common,
        memory_window=10,
    ).run_id != AdversarialExperimentSpec(
        **common,
        memory_window=0,
    ).run_id


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


def test_arbitrary_adversarial_memory_window() -> None:
    assert adversarial_memory_window("full_history") == 0
    assert adversarial_memory_window("last_37") == 37
    assert adversarial_memory_label(37) == "Last 37 rounds"


def test_unknown_adversarial_memory_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown adversarial memory"):
        adversarial_memory_window("unknown")
