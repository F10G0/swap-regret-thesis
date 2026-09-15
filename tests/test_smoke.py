import pytest

from experiments.scenarios.cross_play import run_cross_play_experiment
from experiments.runner import ExperimentCancelled
from experiments.scenarios import cross_play
from tests.support import read_csv_rows as _read_rows


@pytest.mark.parametrize("feedback_mode,algorithm", [("full_information", "hedge"), ("bandit", "exp3_ix")])
def test_fixed_experiment_records_feedback_and_refuses_overwrite(tmp_path, monkeypatch, feedback_mode, algorithm):
    monkeypatch.setattr(cross_play, "RAW_DIR", tmp_path)
    options = dict(game_name="rps", algorithm_names=[algorithm, algorithm], horizon=3,
                   seed=7, feedback_mode=feedback_mode)
    output_path = run_cross_play_experiment(**options)
    rows = _read_rows(output_path)
    assert len(rows) == 6
    assert {row["feedback_mode"] for row in rows} == {feedback_mode}
    assert {row["stationary_method"] for row in rows} == {"solve"}
    assert "swap_regret" in rows[0]
    original = output_path.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        run_cross_play_experiment(**options)
    assert output_path.read_bytes() == original


@pytest.mark.parametrize("feedback_mode,names", [
    ("full_information", ["regret_matching", "hedge"]),
    ("full_information", ["stationary_regret_matching", "hedge"]),
    ("bandit", ["exp3_ix", "bm"]),
    ("bandit", ["lce_ix", "exp3_ix"]),
])
def test_registered_profiles_record_without_learning_rate_columns(tmp_path, feedback_mode, names):
    output_path = run_cross_play_experiment(
        "rps", names, horizon=3, seed=7, output_dir=tmp_path, feedback_mode=feedback_mode,
    )

    rows = _read_rows(output_path)
    assert len(rows) == 6
    assert "learning_rate_player_0" not in rows[0]


def test_removed_exp3_is_rejected_for_new_cross_play_runs(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown algorithm: exp3"):
        run_cross_play_experiment(
            game_name="rps", algorithm_names=["exp3", "exp3_ix"],
            horizon=3, output_dir=tmp_path,
            feedback_mode="bandit",
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("feedback_mode", ["full_information", "bandit"])
def test_all_feedback_modes_record_the_canonical_schema(tmp_path, feedback_mode):
    from experiments.result_schema import REGRET_FIELDNAMES
    if feedback_mode == "full_information":
        names = ["hedge", "bm"]
    else:
        names = ["exp3_ix", "bm"]
    rows = _read_rows(run_cross_play_experiment("rps", names, feedback_mode=feedback_mode, horizon=4, seed=7, output_dir=tmp_path))
    for row in rows:
        assert {key for key in row if key.endswith("_regret")} == set(REGRET_FIELDNAMES)
        assert "regret_evaluation" not in row
        assert "implementation_version" not in row


def test_cancelled_experiment_does_not_publish_partial_result(tmp_path) -> None:
    with pytest.raises(ExperimentCancelled):
        run_cross_play_experiment(
            game_name="rps", algorithm_names=["exp3_ix", "exp3_ix"], horizon=3, seed=7, output_dir=tmp_path, should_cancel=lambda: True,
            feedback_mode="bandit",
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("feedback_mode", ["full_information", "bandit"])
def test_retired_games_are_rejected_for_new_cross_play_runs(
    tmp_path, feedback_mode
) -> None:
    game_name = "bertrand_standard_o1"
    if feedback_mode == "full_information":
        algorithm = "hedge"
    else:
        algorithm = "exp3_ix"
    with pytest.raises(ValueError, match=f"unknown game: {game_name}"):
        run_cross_play_experiment(
            game_name=game_name,
            feedback_mode=feedback_mode,
            algorithm_names=[algorithm, algorithm],
            horizon=3,
            output_dir=tmp_path / "raw",
            custom_game_dir=tmp_path / "custom-games",
        )
    assert not (tmp_path / "raw").exists()
