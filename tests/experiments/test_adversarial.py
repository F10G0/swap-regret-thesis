import csv

import pytest

from experiments.plots.plot_adversarial import (
    aggregate_adversarial_regret,
    plot_adversarial_results,
)
from experiments.runner import ExperimentCancelled
from experiments.seeding import (
    ENVIRONMENT_SEED_DOMAIN,
    LEARNER_SEED_DOMAIN,
    domain_separated_seed,
)
from experiments.scenarios.adversarial import (
    AdversarialExperimentSpec,
    HISTORICAL_FREQUENCY_ENVIRONMENT,
    RANDOM_WALK_ENVIRONMENT,
    load_final_adversarial_row,
    load_adversarial_rows,
    run_adversarial_experiment,
)
from tests.support import read_csv_rows as _rows


def test_removed_exp3_is_rejected_for_new_adversarial_runs(tmp_path) -> None:
    with pytest.raises(ValueError, match="algorithm exp3 is not available"):
        run_adversarial_experiment(
            "exp3", feedback_mode="bandit", horizon=3, output_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("algorithm,feedback_mode", [
    ("exp3", "bandit"), ("exp3", "full_information"), ("unknown", "bandit"),
])
def test_invalid_adversarial_algorithm_identity(tmp_path, algorithm, feedback_mode) -> None:
    # Use a valid trajectory as a schema fixture, not as an Exp3 reproduction.
    path = run_adversarial_experiment(
        "exp3_ix", feedback_mode="bandit", horizon=3, output_dir=tmp_path,
    )
    rows = _rows(path)
    for row in rows:
        row["algorithm"] = algorithm
        row["feedback_mode"] = feedback_mode
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for loader in (load_adversarial_rows, load_final_adversarial_row):
        with pytest.raises(ValueError, match="invalid algorithm"):
            loader(path)


def test_adversarial_experiment_records_canonical_regret(tmp_path) -> None:
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
    assert {row["base_learner_seed"] for row in rows} == {"7"}
    assert {row["learner_seed"] for row in rows} == {
        str(domain_separated_seed(7, 0, LEARNER_SEED_DOMAIN))
    }
    assert {row["replicate"] for row in rows} == {"0"}
    assert rows[0]["punished_actions"] == "0 1"
    assert rows[-1]["t"] == "5"
    assert "average_swap_regret" in rows[0]
    assert rows[0]["current_best_reward"] == "1.0"
    assert load_adversarial_rows(output_path)[-1] == rows[-1]


@pytest.mark.parametrize("algorithm", ["auer_exp3", "exp3_ix"])
def test_bandit_adversarial_experiment_uses_scalar_learner_feedback(
    tmp_path, algorithm,
) -> None:
    output_path = run_adversarial_experiment(
        algorithm,
        feedback_mode="bandit",
        n_actions=3,
        horizon=5,
        seed=7,
        output_dir=tmp_path,
    )

    rows = _rows(output_path)

    assert {row["feedback_mode"] for row in rows} == {"bandit"}
    assert {row["algorithm"] for row in rows} == {algorithm}
    assert load_adversarial_rows(output_path)[-1] == rows[-1]
    assert len(rows) == 5
    assert "average_external_regret" in rows[-1]


def test_random_walk_experiment_records_environment_metadata(tmp_path) -> None:
    output_path = run_adversarial_experiment(
        "exp3_ix",
        feedback_mode="bandit",
        environment=RANDOM_WALK_ENVIRONMENT,
        environment_seed=11,
        n_actions=3,
        horizon=5,
        seed=7,
        output_dir=tmp_path,
    )
    rows = _rows(output_path)

    assert {row["environment"] for row in rows} == {RANDOM_WALK_ENVIRONMENT}
    assert all("initialization_mode" not in row for row in rows)
    assert {row["reward_step"] for row in rows} == {"0.1"}
    assert {row["base_environment_seed"] for row in rows} == {"11"}
    assert {row["base_learner_seed"] for row in rows} == {"7"}
    assert {row["environment_seed"] for row in rows} == {
        str(domain_separated_seed(11, 0, ENVIRONMENT_SEED_DOMAIN))
    }
    assert {row["learner_seed"] for row in rows} == {
        str(domain_separated_seed(7, 0, LEARNER_SEED_DOMAIN))
    }
    assert all(row["punished_actions"] == "" for row in rows)
    assert all(0 <= int(row["current_best_action"]) < 3 for row in rows)
    assert all(0.0 <= float(row["current_best_reward"]) <= 1.0 for row in rows)
    assert load_adversarial_rows(output_path)[-1] == rows[-1]


def test_algorithms_share_random_walk_environment_trajectory(tmp_path) -> None:
    paths = [
        run_adversarial_experiment(
            algorithm,
            environment=RANDOM_WALK_ENVIRONMENT,
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


def test_equal_base_seeds_still_create_distinct_random_streams() -> None:
    spec = AdversarialExperimentSpec(
        algorithm_name="hedge",
        environment=RANDOM_WALK_ENVIRONMENT,
        environment_seed=42,
        n_actions=3,
        horizon=10,
        seed=42,
    )

    assert spec.learner_seed != spec.replicate_environment_seed
    assert spec.configuration()["base_learner_seed"] == 42
    assert spec.configuration()["base_environment_seed"] == 42


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
        "exp3_ix",
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
    assert {row["base_environment_seed"] for row in rows} == {"11"}
    assert {row["base_learner_seed"] for row in rows} == {"7"}
    assert {row["environment_seed"] for row in rows} == {
        str(domain_separated_seed(11, 2, ENVIRONMENT_SEED_DOMAIN))
    }
    assert {row["learner_seed"] for row in rows} == {
        str(domain_separated_seed(7, 2, LEARNER_SEED_DOMAIN))
    }


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


def test_adversarial_loader_rejects_unversioned_csv(tmp_path) -> None:
    generated = run_adversarial_experiment(
        "hedge",
        horizon=3,
        output_dir=tmp_path,
    )
    rows = _rows(generated)
    legacy_path = tmp_path / "legacy.csv"
    legacy_fields = {
        "base_environment_seed",
        "base_learner_seed",
        "replicate",
        "implementation_version",
        "runtime_environment",
        "runtime_fingerprint",
    }
    fieldnames = [field for field in rows[0] if field not in legacy_fields]
    with legacy_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {field: value for field, value in row.items() if field not in legacy_fields}
            for row in rows
        )

    for loader in (load_adversarial_rows, load_final_adversarial_row):
        with pytest.raises(ValueError, match="incompatible result implementation_version 0"):
            loader(legacy_path)


def test_adversarial_implementation_version_changes_run_identity() -> None:
    common = {"algorithm_name": "hedge", "n_actions": 3, "horizon": 10, "seed": 7}

    assert AdversarialExperimentSpec(**common).implementation_version == 6
    assert AdversarialExperimentSpec(**common).run_id != AdversarialExperimentSpec(**common, implementation_version=2).run_id


@pytest.mark.parametrize("version", [3, 4, 5])
def test_adversarial_loader_rejects_stale_results_without_modifying_them(tmp_path, version) -> None:
    common = {"algorithm_name": "hedge", "n_actions": 3, "horizon": 3, "seed": 7}
    current_path = run_adversarial_experiment(**common, output_dir=tmp_path)
    legacy = AdversarialExperimentSpec(**common, implementation_version=version)
    legacy_path = tmp_path / f"{legacy.run_id}.csv"
    # A schema fixture, not a reproduction of a legacy learner implementation.
    rows = _rows(current_path)
    with legacy_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(row | {"implementation_version": str(version), "run_id": legacy.run_id} for row in rows)
    legacy_bytes = legacy_path.read_bytes()

    assert {row["implementation_version"] for row in load_adversarial_rows(current_path)} == {"6"}
    for loader in (load_adversarial_rows, load_final_adversarial_row):
        with pytest.raises(ValueError, match=f"incompatible result implementation_version {version}"):
            loader(legacy_path)
    assert legacy_path.read_bytes() == legacy_bytes


def test_adversarial_regret_aggregation_uses_replicate_means() -> None:
    trajectories = [
        [
            {"t": "1", "average_external_regret": "1"},
            {"t": "2", "average_external_regret": "2"},
        ],
        [
            {"t": "1", "average_external_regret": "3"},
            {"t": "2", "average_external_regret": "6"},
        ],
    ]

    times, means = aggregate_adversarial_regret(
        trajectories,
        "average_external_regret",
    )

    assert times.tolist() == [1, 2]
    assert means.tolist() == [2, 4]


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
        "exp3_ix",
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
        f"adversarial_{environment}_{feedback}_3_actions_average_{regret}_regret.png"
        for environment, feedback in (
            (HISTORICAL_FREQUENCY_ENVIRONMENT, "full_information"),
            (RANDOM_WALK_ENVIRONMENT, "bandit"),
        )
        for regret in ("external", "internal", "swap")
    }
    expected_scaling_figures = {
        f"adversarial_{environment}_{feedback}_3_actions_{regret}_regret_over_sqrt_t.png"
        for environment, feedback in (
            (HISTORICAL_FREQUENCY_ENVIRONMENT, "full_information"),
            (RANDOM_WALK_ENVIRONMENT, "bandit"),
        )
        for regret in ("external", "internal", "swap")
    }
    assert len(generated) == 12
    assert {path.name for path in generated} == expected_regret_figures | expected_scaling_figures
    assert all(path.with_suffix(".pdf").is_file() for path in generated)


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

    regret_path = next(path for path in generated if "average_external" in path.name)
    assert regret_path.is_file()
    assert regret_path.with_suffix(".pdf").is_file()
    assert len(list(figure_dir.glob("*.png"))) == len(generated)
    assert len(list(figure_dir.glob("*.pdf"))) == len(generated)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_actions": 1}, "between 2"),
        ({"horizon": 0}, "positive"),
        ({"seed": -1}, "non-negative"),
        ({"replicate": -1}, "replicate"),
        ({"environment": RANDOM_WALK_ENVIRONMENT, "environment_seed": -1}, "environment seed"),
        ({"environment": "unknown"}, "unknown adversarial environment"),
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


def test_random_walk_seeds_are_part_of_identity() -> None:
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


def test_adversarial_algorithm_must_match_feedback_mode() -> None:
    with pytest.raises(ValueError, match="not available for bandit"):
        AdversarialExperimentSpec(
            algorithm_name="hedge",
            feedback_mode="bandit",
            n_actions=3,
            horizon=10,
            seed=7,
        )
