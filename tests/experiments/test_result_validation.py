from collections import Counter
import csv
import json

import pytest

import experiments.results as results
import experiments.runtime_environment as runtime
from experiments.recording import encode_action_block
from experiments.result_trajectories import load_result_action_profiles
from experiments.scenarios.full_information_cross_play import run_full_information_cross_play_experiment
from tests.support import read_csv_rows


def create_run(directory, max_recorded_points=6):
    return run_full_information_cross_play_experiment(
        "rps", ["hedge", "bm"], horizon=30, regret_evaluation="both",
        output_dir=directory, max_recorded_points=max_recorded_points,
    )


def rewrite_rows(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("loader", [lambda path: list(results.iter_result_rows(path)), results.load_final_result_rows])
def test_fixed_metadata_is_parsed_and_hashed_once_per_file(tmp_path, monkeypatch, loader):
    path = create_run(tmp_path)
    calls = Counter()
    names = ["result_regret_evaluation", "result_algorithm_profile", "result_game_payoff_digest",
             "result_implementation_version", "validate_runtime_environment", "runtime_environment_fingerprint"]
    for name in names:
        original = getattr(results, name)

        def count(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(results, name, count)
    original_loads = json.loads

    def loads(*args, **kwargs):
        calls["json.loads"] += 1
        return original_loads(*args, **kwargs)

    monkeypatch.setattr(json, "loads", loads)
    assert loader(path)
    assert calls == Counter({**dict.fromkeys(names, 1), "json.loads": 2})


@pytest.mark.parametrize("field", results.CONSTANT_RESULT_COLUMNS)
def test_fixed_loader_rejects_constant_changes_on_every_row(tmp_path, field):
    path = create_run(tmp_path)
    rows = read_csv_rows(path)
    for row in rows:
        row.update(n_players="2", algorithm_player_0="hedge", algorithm_player_1="bm")
    rows[1][field] = "corrupted"
    rewrite_rows(path, rows)
    with pytest.raises(ValueError, match="inconsistent run metadata"):
        list(results.iter_result_rows(path))


@pytest.mark.parametrize("field,value", [
    ("feedback_mode", "invalid"), ("implementation_version", "-1"),
    ("runtime_environment", "[]"), ("runtime_fingerprint", "f" * 64),
    ("game_payoff_digest", "not-a-digest"), ("regret_evaluation", "realized"),
    ("algorithm_profile", '["hedge"]'), ("algorithm_profile", "not JSON"),
    ("horizon", "0"), ("seed", "-1"), ("replicate", "-1"),
])
def test_first_row_still_establishes_valid_file_metadata(tmp_path, field, value):
    path = create_run(tmp_path)
    rows = read_csv_rows(path)
    for row in rows:
        row[field] = value
    rewrite_rows(path, rows)
    with pytest.raises(ValueError):
        list(results.iter_result_rows(path))


def test_legacy_profiles_and_inferred_regret_sources_still_load(tmp_path):
    path = create_run(tmp_path)
    rows = read_csv_rows(path)
    for row in rows:
        del row["algorithm_profile"]
        del row["regret_evaluation"]
        row.update(algorithm_player_0="hedge", algorithm_player_1="bm")
    rewrite_rows(path, rows)
    loaded = list(results.iter_result_rows(path))
    assert loaded == [row | {"regret_evaluation": "both"} for row in rows]


def test_runtime_fingerprint_does_not_reparse_canonical_input(monkeypatch):
    canonical = runtime.validate_runtime_environment('{"z": 2, "a": 1}')
    expected = runtime.runtime_environment_fingerprint(canonical)

    def unexpected(*args, **kwargs):
        pytest.fail("already-canonical runtime JSON was parsed again")

    monkeypatch.setattr(json, "loads", unexpected)
    assert runtime.runtime_environment_fingerprint(canonical) == expected


@pytest.mark.parametrize("empty_runtime", ["", "0"])
def test_fingerprint_without_runtime_is_not_treated_as_valid_legacy_metadata(tmp_path, empty_runtime):
    path = create_run(tmp_path)
    rows = read_csv_rows(path)
    for row in rows:
        row["runtime_environment"] = empty_runtime
        row["runtime_fingerprint"] = runtime.runtime_environment_fingerprint("")
    rewrite_rows(path, rows)
    with pytest.raises(ValueError, match="requires runtime_environment"):
        list(results.iter_result_rows(path))


@pytest.mark.parametrize("budget", [6, 100])
@pytest.mark.parametrize("shape", [(3,), (3, 3, 3)])
def test_action_shape_must_match_file_player_count(tmp_path, budget, shape):
    path = create_run(tmp_path, max_recorded_points=budget)
    with pytest.raises(ValueError, match="action shape"):
        load_result_action_profiles(path, shape)


@pytest.mark.parametrize("budget", [6, 100])
@pytest.mark.parametrize("corruption", ["duplicate_player", "missing_player", "missing_final", "action_bounds"])
def test_action_reconstruction_keeps_file_boundary_checks(tmp_path, budget, corruption):
    path = create_run(tmp_path, max_recorded_points=budget)
    rows = read_csv_rows(path)
    if corruption == "duplicate_player":
        rows[1]["player"] = rows[0]["player"]
    elif corruption == "missing_player":
        rows.pop(1)
    elif corruption == "missing_final":
        rows = rows[:-2]
    else:
        rows[0]["action"] = "3"
        if budget == 6:
            rows[0]["action_history"] = encode_action_block([3])
    rewrite_rows(path, rows)
    with pytest.raises(ValueError):
        load_result_action_profiles(path, (3, 3))
