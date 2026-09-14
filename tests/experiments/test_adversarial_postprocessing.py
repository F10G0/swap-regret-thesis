import csv

import pytest

import experiments.scenarios.adversarial as scenario
from tests.support import read_csv_rows


def create_run(directory, **kwargs):
    return scenario.run_adversarial_experiment(
        "auer_exp3", feedback_mode="bandit", environment=scenario.RANDOM_WALK_ENVIRONMENT,
        horizon=20, output_dir=directory, **kwargs,
    )


def rewrite_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_constant_metadata_is_validated_once_but_every_observation_is_checked(tmp_path, monkeypatch):
    path = create_run(tmp_path)
    metadata_calls, observations = [], []
    validate_metadata = scenario._validate_adversarial_metadata
    validate_observation = scenario._validate_adversarial_observation

    def metadata(row, path):
        metadata_calls.append(row["t"])
        return validate_metadata(row, path)

    def observation(row, path, n_actions):
        observations.append(row["t"])
        return validate_observation(row, path, n_actions)

    monkeypatch.setattr(scenario, "_validate_adversarial_metadata", metadata)
    monkeypatch.setattr(scenario, "_validate_adversarial_observation", observation)
    expected = read_csv_rows(path)
    assert scenario.load_adversarial_rows(path) == expected
    assert metadata_calls == ["1"]
    assert observations == [str(t) for t in range(1, 21)]


@pytest.mark.parametrize("field", scenario.ADVERSARIAL_IDENTITY_FIELDS)
def test_metadata_changes_on_unplotted_rows_are_rejected(tmp_path, field):
    path = create_run(tmp_path)
    rows = read_csv_rows(path)
    rows[1][field] = "corrupted"
    rewrite_rows(path, rows)
    with pytest.raises(ValueError, match="inconsistent metadata"):
        scenario.load_adversarial_rows(path, max_points=2)


@pytest.mark.parametrize("field,value", [("action", "100"), ("current_best_action", "100"), ("current_best_reward", "nan")])
def test_invalid_observations_on_unplotted_rows_are_rejected(tmp_path, field, value):
    path = create_run(tmp_path)
    rows = read_csv_rows(path)
    rows[1][field] = value
    rewrite_rows(path, rows)
    with pytest.raises(ValueError, match="invalid"):
        scenario.load_adversarial_rows(path, max_points=2)
