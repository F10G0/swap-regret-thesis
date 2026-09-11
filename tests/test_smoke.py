import pytest

from experiments.runner import ExperimentCancelled
from experiments.scenarios import bandit_cross_play, full_information_cross_play
from tests.support import read_csv_rows as _read_rows


RETIRED_GAME_IDS = (
    "bertrand_standard_o1",
    "bertrand_linear_o2",
    "bertrand_logit_o3",
    "bertrand_linear_o2_prime",
    "bertrand_logit_o3_prime",
)


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
    assert "swap_regret" in rows[0]

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


def test_removed_exp3_is_rejected_for_new_cross_play_runs(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown algorithm: exp3"):
        bandit_cross_play.run_bandit_cross_play_experiment(
            game_name="rps", algorithm_names=["exp3", "exp3_ix"],
            horizon=3, output_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_bandit_experiment_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bandit_cross_play, "RAW_DIR", tmp_path)

    output_path = bandit_cross_play.run_bandit_cross_play_experiment(
        game_name="rps",
        algorithm_names=["exp3_ix", "exp3_ix"],
        horizon=3,
        seed=7,
    )

    rows = _read_rows(output_path)
    assert len(rows) == 6
    assert {row["feedback_mode"] for row in rows} == {"bandit"}
    assert "swap_regret" in rows[0]


@pytest.mark.parametrize("feedback_mode", ["full_information", "bandit"])
def test_all_feedback_modes_record_the_canonical_schema(tmp_path, feedback_mode):
    from experiments.result_schema import REGRET_FIELDNAMES, RESULT_IMPLEMENTATION_VERSION
    if feedback_mode == "full_information":
        runner = full_information_cross_play.run_full_information_cross_play_experiment
        names = ["hedge", "bm"]
    else:
        runner = bandit_cross_play.run_bandit_cross_play_experiment
        names = ["exp3_ix", "bm"]
    rows = _read_rows(runner("rps", names, horizon=4, seed=7, output_dir=tmp_path))
    for row in rows:
        assert {key for key in row if key.endswith("_regret")} == set(REGRET_FIELDNAMES)
        assert "regret_evaluation" not in row
        assert int(row["implementation_version"]) == RESULT_IMPLEMENTATION_VERSION == 5


def test_cancelled_experiment_does_not_publish_partial_result(tmp_path) -> None:
    with pytest.raises(ExperimentCancelled):
        bandit_cross_play.run_bandit_cross_play_experiment(
            game_name="rps", algorithm_names=["exp3_ix", "exp3_ix"], horizon=3, seed=7, output_dir=tmp_path, should_cancel=lambda: True,
        )

    assert list(tmp_path.iterdir()) == []


def test_exp3_ix_and_bm_experiment_smoke(tmp_path) -> None:
    output_path = bandit_cross_play.run_bandit_cross_play_experiment(
        game_name="rps",
        algorithm_names=["exp3_ix", "bm"],
        horizon=3,
        seed=7,
        output_dir=tmp_path,
    )

    rows = _read_rows(output_path)
    assert len(rows) == 6
    assert "learning_rate_player_0" not in rows[0]


@pytest.mark.parametrize("game_name", RETIRED_GAME_IDS)
@pytest.mark.parametrize("feedback_mode", ["full_information", "bandit"])
def test_retired_games_are_rejected_for_new_cross_play_runs(
    tmp_path, game_name, feedback_mode
) -> None:
    if feedback_mode == "full_information":
        runner = full_information_cross_play.run_full_information_cross_play_experiment
        algorithm = "hedge"
    else:
        runner = bandit_cross_play.run_bandit_cross_play_experiment
        algorithm = "exp3_ix"
    with pytest.raises(ValueError, match=f"unknown game: {game_name}"):
        runner(
            game_name=game_name,
            algorithm_names=[algorithm, algorithm],
            horizon=3,
            output_dir=tmp_path / "raw",
            custom_game_dir=tmp_path / "custom-games",
        )
    assert not (tmp_path / "raw").exists()


def test_lce_ix_experiment_uses_theoretical_default_schedule(tmp_path) -> None:
    output_path = bandit_cross_play.run_bandit_cross_play_experiment(
        game_name="rps",
        algorithm_names=["lce_ix", "exp3_ix"],
        horizon=3,
        seed=7,
        output_dir=tmp_path,
    )

    rows = _read_rows(output_path)
    assert len(rows) == 6
    assert "learning_rate_player_0" not in rows[0]
