import csv
from collections import Counter

import numpy as np
import pytest

from environments import RepeatedGame, BanditRepeatedGame
from experiments.game_catalog import load_game_payoffs
from experiments.plots.plot_joint_actions import joint_action_distribution
from experiments.plots.plot_regret import load_rows, aggregate_metric_curve
from experiments.plots.plot_adversarial import aggregate_adversarial_regret
from experiments.recording import decode_action_block, recording_checkpoints
from experiments.result_trajectories import load_result_action_profiles
from experiments.results import iter_result_rows, load_final_result_rows
from experiments.runner import run_game
from experiments.scenarios.adversarial import (
    RANDOM_WALK_ENVIRONMENT, HISTORICAL_FREQUENCY_ENVIRONMENT,
    load_adversarial_rows, run_adversarial_experiment,
)
from experiments.scenarios.bandit_cross_play import ALGORITHMS as BANDIT, run_bandit_cross_play_experiment
from experiments.scenarios.full_information_cross_play import ALGORITHMS as FULL, run_full_information_cross_play_experiment
from metrics.regret import BaseReplacementRegretBundle, ExpectedRegretBundle, RealizedRegretBundle, RegretBundles
from tests.support import read_csv_rows


class MemoryRecorder:
    def __init__(self):
        self.rows = []

    def record(self, row):
        self.rows.append(row)


@pytest.mark.parametrize("horizon", [1, 2, 100, 2000, 2001, 1_000_000])
def test_recording_budget_and_endpoints(horizon):
    points = recording_checkpoints(horizon)
    assert points[0] == 1 and points[-1] == horizon
    assert len(points) <= 2000
    assert tuple(sorted(set(points))) == points


@pytest.mark.parametrize("mode,name", [*(('full_information', name) for name in FULL), *(('bandit', name) for name in BANDIT)])
@pytest.mark.parametrize("evaluation", ["expected", "realized", "both"])
def test_sparse_runner_matches_original_dense_loop(mode, name, evaluation):
    horizon = 23
    factory = (FULL if mode == "full_information" else BANDIT)[name]
    game_type = RepeatedGame if mode == "full_information" else BanditRepeatedGame
    payoff_tensor = load_game_payoffs("rps")

    def players():
        return [factory.create(3, horizon, seed) for seed in (7, 8)]

    # Reference the original loop: both trackers, every round, every summary.
    reference_players = players()
    game = game_type(payoff_tensor)
    regrets = [RegretBundles(3) for _ in reference_players]
    reference_rows = {}
    history = []
    sources = ("expected", "realized") if evaluation == "both" else (evaluation,)
    for t in range(1, horizon + 1):
        actions = tuple(player.sample_action() for player in reference_players)
        history.append(actions)
        game.step(actions)
        for i, (player, action) in enumerate(zip(reference_players, actions)):
            strategy = player.strategy()
            feedback = game.feedback(i)
            deviations = feedback if mode == "full_information" else game.deviation_payoffs(i)
            regrets[i].update(strategy, action, deviations)
            player.update(feedback)
            summary = {key: value for source in sources for key, value in getattr(regrets[i], source).summary(t).items()}
            reference_rows[t, i] = dict(game="rps", algorithm=name, t=t, player=i, action=action,
                                       payoff=float(deviations[action]), **summary)

    actual_players = players()
    recorder = MemoryRecorder()
    run_game("rps", mode, game_type(payoff_tensor), name, actual_players, recorder, horizon,
             regret_evaluation=evaluation, max_recorded_points=6)
    blocks = [[], []]
    previous = [0, 0]
    for stored in recorder.rows:
        row = stored.copy()
        block = row.pop("action_history")
        t, i = row["t"], row["player"]
        assert row == reference_rows[t, i]
        blocks[i].extend(decode_action_block(block, t - previous[i]))
        previous[i] = t
    assert np.array_equal(np.asarray(blocks).T, history)
    for actual, reference in zip(actual_players, reference_players):
        assert actual.t == reference.t == horizon
        assert np.array_equal(actual.strategy(), reference.strategy())


@pytest.mark.parametrize("evaluation", ["expected", "realized", "both"])
def test_only_selected_trackers_update_and_only_checkpoints_are_summarized(monkeypatch, evaluation):
    updates, summaries = Counter(), Counter()
    for cls in (ExpectedRegretBundle, RealizedRegretBundle):
        original = cls.update

        def update(self, *args, _original=original):
            updates[self.regret_type] += 1
            return _original(self, *args)

        monkeypatch.setattr(cls, "update", update)
    original_summary = BaseReplacementRegretBundle.summary

    def summary(self, time):
        summaries[self.regret_type, time] += 1
        return original_summary(self, time)

    monkeypatch.setattr(BaseReplacementRegretBundle, "summary", summary)
    bundles = RegretBundles(3, evaluation)
    assert (bundles.expected is None) == (evaluation == "realized")
    assert (bundles.realized is None) == (evaluation == "expected")
    run_game("rps", "bandit", BanditRepeatedGame(load_game_payoffs("rps")), "auer_exp3",
             [BANDIT["auer_exp3"].create(3, 100, i) for i in range(2)], MemoryRecorder(), 100,
             regret_evaluation=evaluation, max_recorded_points=6)
    selected = {"expected", "realized"} if evaluation == "both" else {evaluation}
    assert updates == Counter({source: 200 for source in selected})
    assert summaries == Counter({(source, t): 2 for source in selected for t in recording_checkpoints(100, 6)})


@pytest.mark.parametrize("runner,names", [
    (run_full_information_cross_play_experiment, ["hedge", "bm"]),
    (run_bandit_cross_play_experiment, ["auer_exp3", "ito"]),
])
def test_sparse_csv_preserves_all_actions_regrets_and_joint_distribution(tmp_path, runner, names):
    kwargs = dict(game_name="rps", algorithm_names=names, horizon=101, seed=7, regret_evaluation="both")
    dense = runner(**kwargs, output_dir=tmp_path / "dense", max_recorded_points=200)
    sparse = runner(**kwargs, output_dir=tmp_path / "sparse", max_recorded_points=8)
    assert dense.name == sparse.name
    dense_rows = {(row["t"], row["player"]): row for row in iter_result_rows(dense)}
    sparse_rows = list(iter_result_rows(sparse))
    for row in sparse_rows:
        assert {k: v for k, v in row.items() if k != "action_history"} == dense_rows[row["t"], row["player"]]
    assert len(sparse_rows) <= 16
    assert np.array_equal(load_result_action_profiles(dense, (3, 3)), load_result_action_profiles(sparse, (3, 3)))
    assert np.array_equal(joint_action_distribution(dense)[1], joint_action_distribution(sparse)[1])
    assert len(load_final_result_rows(sparse)) == 2
    assert {row["player"] for row in load_rows(sparse)} == {"0", "1"}


@pytest.mark.parametrize("environment", [RANDOM_WALK_ENVIRONMENT, HISTORICAL_FREQUENCY_ENVIRONMENT])
@pytest.mark.parametrize("evaluation", ["expected", "realized", "both"])
def test_adversarial_sparse_rows_equal_dense_checkpoints(tmp_path, environment, evaluation):
    kwargs = dict(algorithm_name="auer_exp3", feedback_mode="bandit", horizon=101, seed=17,
                  environment=environment, regret_evaluation=evaluation)
    dense = run_adversarial_experiment(**kwargs, output_dir=tmp_path / "dense", max_recorded_points=200)
    sparse = run_adversarial_experiment(**kwargs, output_dir=tmp_path / "sparse", max_recorded_points=8)
    dense_rows = {row["t"]: row for row in load_adversarial_rows(dense)}
    sparse_rows = load_adversarial_rows(sparse)
    assert dense.name == sparse.name
    assert len(sparse_rows) <= 8
    assert all(row == dense_rows[row["t"]] for row in sparse_rows)
    assert sparse_rows[-1] == dense_rows["101"]


@pytest.mark.parametrize("evaluation", ["expected", "realized", "both"])
def test_adversarial_summaries_are_only_extracted_at_checkpoints(tmp_path, monkeypatch, evaluation):
    calls = Counter()
    original = BaseReplacementRegretBundle.summary

    def summary(self, time):
        calls[self.regret_type, time] += 1
        return original(self, time)

    monkeypatch.setattr(BaseReplacementRegretBundle, "summary", summary)
    run_adversarial_experiment("auer_exp3", feedback_mode="bandit", horizon=51, output_dir=tmp_path,
                              regret_evaluation=evaluation, max_recorded_points=6)
    selected = {"expected", "realized"} if evaluation == "both" else {evaluation}
    assert calls == Counter({(source, t): 1 for source in selected for t in recording_checkpoints(51, 6)})


@pytest.mark.parametrize("mutate", ["missing_first", "missing_last", "duplicate", "reverse", "missing_player", "missing_block"])
def test_sparse_validation_still_rejects_corrupt_trajectories(tmp_path, mutate):
    path = run_bandit_cross_play_experiment("rps", ["auer_exp3"] * 2, horizon=25,
                                          output_dir=tmp_path, max_recorded_points=6)
    rows = read_csv_rows(path)
    if mutate == "missing_first":
        rows = rows[2:]
    elif mutate == "missing_last":
        rows = rows[:-2]
    elif mutate == "duplicate":
        rows.insert(0, rows[0])
    elif mutate == "reverse":
        rows = rows[::-1]
    elif mutate == "missing_player":
        rows.pop(1)
    else:
        rows[2]["action_history"] = ""
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        list(iter_result_rows(path))


def test_plot_loaders_align_legacy_dense_and_default_sparse_checkpoints(tmp_path):
    for runner, kwargs, loader in [
        (run_adversarial_experiment, dict(algorithm_name="auer_exp3", feedback_mode="bandit"),
         lambda path: load_adversarial_rows(path, max_points=2000)),
        (run_bandit_cross_play_experiment, dict(game_name="rps", algorithm_names=["auer_exp3"] * 2), load_rows),
    ]:
        dense = runner(**kwargs, horizon=2100, output_dir=tmp_path / "dense", max_recorded_points=3000)
        sparse = runner(**kwargs, horizon=2100, output_dir=tmp_path / "sparse")
        assert [row["t"] for row in loader(dense)] == [row["t"] for row in loader(sparse)]


def test_replicate_aggregation_uses_only_shared_observed_times():
    trajectories = [
        [dict(t=str(t), player="0", expected_external_regret=str(t)) for t in (1, 2, 3, 10)],
        [dict(t=str(t), player="0", expected_external_regret=str(2 * t)) for t in (1, 4, 10)],
    ]
    for aggregate in (
        lambda: aggregate_metric_curve(trajectories, 0, "expected_external_regret"),
        lambda: aggregate_adversarial_regret(trajectories, "expected_external_regret"),
    ):
        times, means, _ = aggregate()
        assert times.tolist() == [1, 10]
        assert means.tolist() == [1.5, 15.0]
