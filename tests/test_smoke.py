import csv

import pytest

from experiments.runner import ExperimentCancelled
from experiments.scenarios import bandit_cross_play, full_information_cross_play


def _read_rows(path):
    with path.open("r", newline="") as file:
        return list(csv.DictReader(file))
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


@pytest.mark.parametrize(
    ("feedback_mode", "regret_evaluation", "expected_sources"),
    [
        ("full_information", "expected", {"expected"}),
        ("full_information", "realized", {"realized"}),
        ("full_information", "both", {"expected", "realized"}),
        ("bandit", "expected", {"expected"}),
        ("bandit", "realized", {"realized"}),
        ("bandit", "both", {"expected", "realized"}),
    ],
)
def test_regret_evaluation_is_independent_of_feedback_mode(
    tmp_path,
    feedback_mode: str,
    regret_evaluation: str,
    expected_sources: set[str],
) -> None:
    if feedback_mode == "full_information":
        output_path = full_information_cross_play.run_full_information_cross_play_experiment(
            game_name="rps", algorithm_names=["hedge", "hedge"], horizon=2, seed=7,
            output_dir=tmp_path, regret_evaluation=regret_evaluation,
        )
    else:
        output_path = bandit_cross_play.run_bandit_cross_play_experiment(
            game_name="rps", algorithm_names=["exp3", "exp3"], horizon=2, seed=7,
            output_dir=tmp_path, regret_evaluation=regret_evaluation,
        )

    rows = _read_rows(output_path)
    assert {row["regret_evaluation"] for row in rows} == {regret_evaluation}
    for source in ("expected", "realized"):
        assert (f"{source}_swap_regret" in rows[0]) == (source in expected_sources)


def test_regret_evaluation_does_not_change_bandit_play(tmp_path) -> None:
    expected_path = bandit_cross_play.run_bandit_cross_play_experiment(
        game_name="rps", algorithm_names=["exp3", "exp3"], horizon=10, seed=7,
        output_dir=tmp_path, regret_evaluation="expected",
    )
    both_path = bandit_cross_play.run_bandit_cross_play_experiment(
        game_name="rps", algorithm_names=["exp3", "exp3"], horizon=10, seed=7,
        output_dir=tmp_path, regret_evaluation="both",
    )

    expected_rows = _read_rows(expected_path)
    both_rows = _read_rows(both_path)
    behavior_fields = ("t", "player", "action", "payoff")
    assert expected_path != both_path
    assert [
        tuple(row[field] for field in behavior_fields)
        for row in expected_rows
    ] == [
        tuple(row[field] for field in behavior_fields)
        for row in both_rows
    ]


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


def test_o1_stationary_regret_matching_handles_solver_roundoff(tmp_path) -> None:
    output_path = (
        full_information_cross_play.run_full_information_cross_play_experiment(
            game_name="bertrand_standard_o1",
            algorithm_names=[
                "stationary_regret_matching",
                "stationary_regret_matching",
            ],
            horizon=1300,
            seed=42,
            output_dir=tmp_path,
        )
    )

    rows = _read_rows(output_path)

    assert len(rows) == 2600


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
