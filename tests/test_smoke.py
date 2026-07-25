import csv
import importlib

import pytest

from experiments.runner import ExperimentCancelled
from experiments.scenarios import bandit_cross_play, full_information_cross_play


def _read_rows(path):
    with path.open("r", newline="") as file:
        return list(csv.DictReader(file))


def test_main_entrypoint_is_importable() -> None:
    module = importlib.import_module("main")
    assert callable(module.main)


def test_full_information_experiment_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(full_information_cross_play, "RAW_DIR", tmp_path)

    output_path = full_information_cross_play.run_full_information_cross_play_experiment(
        game_name="rps",
        algorithm_names=["hedge", "hedge"],
        horizon=3,
        seed=7,
    )

    rows = _read_rows(output_path)
    assert len(rows) == 6
    assert {row["feedback_mode"] for row in rows} == {"full_information"}
    assert {row["stationary_method"] for row in rows} == {"solve"}
    assert "expected_swap_regret" in rows[0]
    assert "realized_swap_regret" not in rows[0]

    with pytest.raises(FileExistsError, match="already exists"):
        full_information_cross_play.run_full_information_cross_play_experiment(
            game_name="rps",
            algorithm_names=["hedge", "hedge"],
            horizon=3,
            seed=7,
        )


def test_regret_matching_experiment_uses_fixed_normalization(tmp_path) -> None:
    output_path = full_information_cross_play.run_full_information_cross_play_experiment(
        game_name="rps",
        algorithm_names=["regret_matching", "hedge"],
        horizon=3,
        seed=7,
        output_dir=tmp_path,
    )

    rows = _read_rows(output_path)
    assert len(rows) == 6
    assert "learning_rate_player_0" not in rows[0]


def test_stationary_regret_matching_experiment_smoke(tmp_path) -> None:
    output_path = full_information_cross_play.run_full_information_cross_play_experiment(
        game_name="rps",
        algorithm_names=["stationary_regret_matching", "hedge"],
        horizon=3,
        seed=7,
        output_dir=tmp_path,
    )

    rows = _read_rows(output_path)
    assert len(rows) == 6
    assert "learning_rate_player_0" not in rows[0]


def test_bandit_experiment_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bandit_cross_play, "RAW_DIR", tmp_path)

    output_path = bandit_cross_play.run_bandit_cross_play_experiment(
        game_name="rps",
        algorithm_names=["exp3", "exp3"],
        horizon=3,
        seed=7,
    )

    rows = _read_rows(output_path)
    assert len(rows) == 6
    assert {row["feedback_mode"] for row in rows} == {"bandit"}
    assert "realized_swap_regret" in rows[0]
    assert "expected_swap_regret" not in rows[0]


def test_cancelled_experiment_does_not_publish_partial_result(tmp_path) -> None:
    with pytest.raises(ExperimentCancelled):
        bandit_cross_play.run_bandit_cross_play_experiment(
            game_name="rps", algorithm_names=["exp3", "exp3"], horizon=3, seed=7, output_dir=tmp_path, should_cancel=lambda: True,
        )

    assert list(tmp_path.iterdir()) == []


def test_exp3_ix_experiment_smoke(tmp_path) -> None:
    output_path = bandit_cross_play.run_bandit_cross_play_experiment(
        game_name="rps",
        algorithm_names=["exp3_ix", "exp3"],
        horizon=3,
        seed=7,
        output_dir=tmp_path,
    )

    rows = _read_rows(output_path)
    assert len(rows) == 6
    assert "learning_rate_player_0" not in rows[0]


def test_lce_ix_experiment_uses_theoretical_default_schedule(tmp_path) -> None:
    output_path = bandit_cross_play.run_bandit_cross_play_experiment(
        game_name="rps",
        algorithm_names=["lce_ix", "exp3"],
        horizon=3,
        seed=7,
        output_dir=tmp_path,
    )

    rows = _read_rows(output_path)
    assert len(rows) == 6
    assert "learning_rate_player_0" not in rows[0]


def test_bandit_replicates_use_distinct_runs_and_random_streams(tmp_path) -> None:
    output_paths = bandit_cross_play.run_bandit_cross_play_replicates(
        game_name="rps",
        algorithm_names=["exp3", "exp3"],
        n_replicates=3,
        horizon=3,
        seed=7,
        output_dir=tmp_path,
    )

    assert len(set(output_paths)) == 3
    rows_by_replicate = [_read_rows(path) for path in output_paths]
    assert [
        {int(row["replicate"]) for row in rows}
        for rows in rows_by_replicate
    ] == [{0}, {1}, {2}]
