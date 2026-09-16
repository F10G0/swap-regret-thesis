import csv

import pytest

from experiments.plots.plot_adversarial import aggregate_adversarial_regret
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
        n_actions=3,
        horizon=5,
        seed=7,
        output_dir=tmp_path,
    )
    rows = _rows(output_path)

    assert {row["environment"] for row in rows} == {RANDOM_WALK_ENVIRONMENT}
    assert all("initialization_mode" not in row for row in rows)
    assert {row["reward_step"] for row in rows} == {"0.1"}
    assert {row["base_environment_seed"] for row in rows} == {"7"}
    assert {row["base_learner_seed"] for row in rows} == {"7"}
    assert {row["environment_seed"] for row in rows} == {
        str(domain_separated_seed(7, 0, ENVIRONMENT_SEED_DOMAIN))
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
            n_actions=3,
            horizon=30,
            seed=11,
            output_dir=tmp_path,
        )
        for algorithm in ("hedge", "regret_matching")
    ]
    trajectories = [
        [
            (row["current_best_action"], row["current_best_reward"])
            for row in _rows(path)
        ]
        for path in paths
    ]

    assert trajectories[0] == trajectories[1]


def test_one_base_seed_deterministically_separates_roles_and_replicates() -> None:
    spec = AdversarialExperimentSpec(
        algorithm_name="hedge",
        environment=RANDOM_WALK_ENVIRONMENT,
        n_actions=3,
        horizon=10,
        seed=42,
    )
    same = AdversarialExperimentSpec(
        algorithm_name="hedge", environment=RANDOM_WALK_ENVIRONMENT,
        n_actions=3, horizon=10, seed=42)
    next_replicate = AdversarialExperimentSpec(
        algorithm_name="hedge", environment=RANDOM_WALK_ENVIRONMENT,
        n_actions=3, horizon=10, seed=42, replicate=1)
    changed = AdversarialExperimentSpec(
        algorithm_name="hedge", environment=RANDOM_WALK_ENVIRONMENT,
        n_actions=3, horizon=10, seed=43)

    assert spec.learner_seed != spec.replicate_environment_seed
    assert (spec.learner_seed, spec.replicate_environment_seed) == (
        same.learner_seed, same.replicate_environment_seed)
    assert spec.learner_seed != next_replicate.learner_seed
    assert spec.replicate_environment_seed != next_replicate.replicate_environment_seed
    assert spec.learner_seed != changed.learner_seed
    assert spec.replicate_environment_seed != changed.replicate_environment_seed
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
        n_actions=3,
        horizon=5,
        seed=7,
        replicate=2,
        output_dir=tmp_path,
    )

    rows = _rows(output_path)

    assert {row["replicate"] for row in rows} == {"2"}
    assert {row["base_environment_seed"] for row in rows} == {"7"}
    assert {row["base_learner_seed"] for row in rows} == {"7"}
    assert {row["environment_seed"] for row in rows} == {
        str(domain_separated_seed(7, 2, ENVIRONMENT_SEED_DOMAIN))
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


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_actions": 1}, "between 2"),
        ({"horizon": 0}, "positive"),
        ({"seed": -1}, "non-negative"),
        ({"replicate": -1}, "replicate"),
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


def test_random_walk_base_seed_is_part_of_identity() -> None:
    common = {
        "algorithm_name": "hedge",
        "n_actions": 3,
        "horizon": 10,
        "seed": 7,
        "environment": RANDOM_WALK_ENVIRONMENT,
    }

    assert AdversarialExperimentSpec(**common).run_id != AdversarialExperimentSpec(
        **(common | {"seed": 8})).run_id


def test_adversarial_algorithm_must_match_feedback_mode() -> None:
    with pytest.raises(ValueError, match="not available for bandit"):
        AdversarialExperimentSpec(
            algorithm_name="hedge",
            feedback_mode="bandit",
            n_actions=3,
            horizon=10,
            seed=7,
        )
