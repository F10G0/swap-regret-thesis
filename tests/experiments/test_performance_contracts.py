from functools import partial
import csv
from collections import Counter
import json

import numpy as np
import pytest

from environments import RepeatedGame, BanditRepeatedGame
from experiments.scenarios.cross_play import ALGORITHMS_BY_FEEDBACK_MODE, run_cross_play_experiment
from experiments.game_catalog import load_game_payoffs
from experiments.plots.plot_joint_actions import joint_action_distribution
from experiments.plots.plot_regret import load_rows, aggregate_metric_curve
from experiments.plots.plot_adversarial import aggregate_adversarial_regret
from experiments.recording import MAX_RECORDED_POINTS, joint_action_histogram_checkpoints, recording_checkpoints
from experiments.result_schema import JOINT_ACTION_HISTOGRAM_FIELD
from experiments.result_trajectories import load_result_empirical_distribution_trajectory
from experiments.results import iter_result_rows, load_final_result_rows
from experiments.runner import run_game
from experiments.scenarios.adversarial import (
    RANDOM_WALK_ENVIRONMENT, HISTORICAL_FREQUENCY_ENVIRONMENT,
    load_adversarial_rows, run_adversarial_experiment,
)
from metrics.regret import RegretBundle
from tests.support import read_csv_rows

BANDIT = ALGORITHMS_BY_FEEDBACK_MODE["bandit"]
FULL = ALGORITHMS_BY_FEEDBACK_MODE["full_information"]


class MemoryRecorder:
    def __init__(self):
        self.rows = []

    def record(self, row):
        self.rows.append(row)


@pytest.mark.parametrize("horizon", [1, 2, 100, 500, 501, 1_000_000])
def test_recording_budget_and_endpoints(horizon):
    points = recording_checkpoints(horizon)
    assert points[0] == 1 and points[-1] == horizon
    assert len(points) <= MAX_RECORDED_POINTS
    assert tuple(sorted(set(points))) == points
    if horizon <= MAX_RECORDED_POINTS:
        assert points == tuple(range(1, horizon + 1))
    else:
        expected = np.geomspace(1, horizon, MAX_RECORDED_POINTS).astype(int)
        expected[0], expected[-1] = 1, horizon
        assert points == tuple(sorted(set(map(int, expected))))
        assert recording_checkpoints(horizon, MAX_RECORDED_POINTS * 2) == points


@pytest.mark.parametrize(("horizon", "expected"), [
    (1, (1,)),
    (9, (1, 9)),
    (10, (1, 10)),
    (99, (1, 10, 99)),
    (100, (1, 10, 100)),
    (12_345, (1, 10, 100, 1_000, 10_000, 12_345)),
    (1_000_000, (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000)),
])
def test_joint_action_histogram_checkpoint_policy(horizon, expected):
    assert joint_action_histogram_checkpoints(horizon) == expected


@pytest.mark.parametrize("mode,name", [*(('full_information', name) for name in FULL), *(('bandit', name) for name in BANDIT)])
def test_sparse_runner_matches_original_dense_loop(mode, name):
    horizon = 23
    factory = (FULL if mode == "full_information" else BANDIT)[name]
    game_type = RepeatedGame if mode == "full_information" else BanditRepeatedGame
    payoff_tensor = load_game_payoffs("rps")

    def players():
        return [factory.create(3, horizon, seed) for seed in (7, 8)]

    # Reference the dense loop, extracting the mathematical summaries every round.
    reference_players = players()
    game = game_type(payoff_tensor)
    gains = [np.zeros((3, 3)) for _ in reference_players]
    reference_rows = {}
    history = []
    for t in range(1, horizon + 1):
        actions = tuple(player.sample_action() for player in reference_players)
        history.append(actions)
        game.step(actions)
        for i, (player, action) in enumerate(zip(reference_players, actions)):
            feedback = game.feedback(i)
            deviations = feedback if mode == "full_information" else game.deviation_payoffs(i)
            gains[i][action] += deviations - deviations[action]
            player.update(feedback)
            values = {
                "external": float(np.max(np.sum(gains[i], axis=0))),
                "internal": float(np.max(gains[i])),
                "swap": float(np.sum(np.max(gains[i], axis=1))),
            }
            summary = {
                field: value for name, regret in values.items()
                for field, value in ((f"{name}_regret", regret), (f"average_{name}_regret", regret / t))
            }
            reference_rows[t, i] = dict(game="rps", algorithm=name, t=t, player=i, action=action,
                                       payoff=float(deviations[action]), **summary)

    actual_players = players()
    recorder = MemoryRecorder()
    run_game("rps", mode, game_type(payoff_tensor), name, actual_players, recorder, horizon,
             max_recorded_points=6)
    histogram_payloads = []
    for stored in recorder.rows:
        row = stored.copy()
        payload = row.pop(JOINT_ACTION_HISTOGRAM_FIELD, "")
        if payload:
            histogram_payloads.append(payload)
        t, i = row["t"], row["player"]
        assert row == reference_rows[t, i]
    assert len(histogram_payloads) == 1
    histograms = json.loads(histogram_payloads[0])
    assert histograms["horizons"] == [1, 10, horizon]
    for checkpoint, values in zip(histograms["horizons"], histograms["counts"]):
        expected = np.zeros((3, 3), dtype=int)
        np.add.at(expected, tuple(np.asarray(history[:checkpoint]).T), 1)
        np.testing.assert_array_equal(np.asarray(values).reshape(3, 3), expected)
    for actual, reference in zip(actual_players, reference_players):
        assert actual.t == reference.t == horizon
        assert np.array_equal(actual.strategy(), reference.strategy())


def test_tracker_updates_every_round_and_summarizes_only_checkpoints(monkeypatch):
    updates, summaries = [], Counter()
    original_update = RegretBundle.update
    original_summary = RegretBundle.summary

    def update(self, *args):
        updates.append(id(self))
        return original_update(self, *args)

    def summary(self, time):
        summaries[time] += 1
        return original_summary(self, time)

    monkeypatch.setattr(RegretBundle, "update", update)
    monkeypatch.setattr(RegretBundle, "summary", summary)
    run_game("rps", "bandit", BanditRepeatedGame(load_game_payoffs("rps")), "auer_exp3",
             [BANDIT["auer_exp3"].create(3, 100, i) for i in range(2)], MemoryRecorder(), 100,
             max_recorded_points=6)
    assert len(updates) == 200
    assert len(set(updates)) == 2
    assert summaries == Counter({t: 2 for t in recording_checkpoints(100, 6)})


@pytest.mark.parametrize("runner,names", [
    (partial(run_cross_play_experiment, feedback_mode="full_information"), ["hedge", "bm"]),
    (partial(run_cross_play_experiment, feedback_mode="bandit"), ["auer_exp3", "ito"]),
])
def test_regret_recording_budget_does_not_change_histograms_or_joint_distribution(tmp_path, runner, names):
    kwargs = dict(game_name="rps", algorithm_names=names, horizon=101, seed=7)
    dense = runner(**kwargs, output_dir=tmp_path / "dense", max_recorded_points=200)
    sparse = runner(**kwargs, output_dir=tmp_path / "sparse", max_recorded_points=8)
    assert dense.name == sparse.name
    dense_rows = {(row["t"], row["player"]): row for row in iter_result_rows(dense)}
    sparse_rows = list(iter_result_rows(sparse))
    for row in sparse_rows:
        expected = dense_rows[row["t"], row["player"]]
        assert {k: v for k, v in row.items() if k != JOINT_ACTION_HISTOGRAM_FIELD} == {
            k: v for k, v in expected.items() if k != JOINT_ACTION_HISTOGRAM_FIELD
        }
    assert len(sparse_rows) <= 16
    assert all("action_history" not in row for row in sparse_rows)
    dense_histograms = load_result_empirical_distribution_trajectory(dense, (3, 3))
    sparse_histograms = load_result_empirical_distribution_trajectory(sparse, (3, 3))
    np.testing.assert_array_equal(dense_histograms.horizons, sparse_histograms.horizons)
    np.testing.assert_array_equal(dense_histograms.vectors, sparse_histograms.vectors)
    assert np.array_equal(joint_action_distribution(dense)[1], joint_action_distribution(sparse)[1])
    assert len(load_final_result_rows(sparse)) == 2
    assert {row["player"] for row in load_rows(sparse)} == {"0", "1"}


@pytest.mark.parametrize("environment", [RANDOM_WALK_ENVIRONMENT, HISTORICAL_FREQUENCY_ENVIRONMENT])
def test_adversarial_sparse_rows_equal_dense_checkpoints(tmp_path, environment):
    kwargs = dict(algorithm_name="auer_exp3", feedback_mode="bandit", horizon=101, seed=17,
                  environment=environment)
    dense = run_adversarial_experiment(**kwargs, output_dir=tmp_path / "dense", max_recorded_points=200)
    sparse = run_adversarial_experiment(**kwargs, output_dir=tmp_path / "sparse", max_recorded_points=8)
    dense_rows = {row["t"]: row for row in load_adversarial_rows(dense)}
    sparse_rows = load_adversarial_rows(sparse)
    assert dense.name == sparse.name
    assert len(sparse_rows) <= 8
    assert all(row == dense_rows[row["t"]] for row in sparse_rows)
    assert sparse_rows[-1] == dense_rows["101"]


def test_adversarial_summaries_are_only_extracted_at_checkpoints(tmp_path, monkeypatch):
    calls = Counter()
    original = RegretBundle.summary

    def summary(self, time):
        calls[time] += 1
        return original(self, time)

    monkeypatch.setattr(RegretBundle, "summary", summary)
    run_adversarial_experiment("auer_exp3", feedback_mode="bandit", horizon=51, output_dir=tmp_path,
                              max_recorded_points=6)
    assert calls == Counter({t: 1 for t in recording_checkpoints(51, 6)})


@pytest.mark.parametrize("mutate", ["missing_first", "missing_last", "duplicate", "reverse", "missing_player", "missing_histogram"])
def test_fixed_result_validation_rejects_corrupt_trajectories(tmp_path, mutate):
    path = run_cross_play_experiment("rps", ["auer_exp3"] * 2, horizon=25,
                                          output_dir=tmp_path, max_recorded_points=6, feedback_mode="bandit")
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
        next(row for row in rows if row[JOINT_ACTION_HISTOGRAM_FIELD])[JOINT_ACTION_HISTOGRAM_FIELD] = ""
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        load_result_empirical_distribution_trajectory(path, (3, 3))


def test_regret_plots_use_default_log_checkpoints_and_axes(tmp_path, monkeypatch):
    from experiments.plots import plot_regret
    from experiments.plots.style import profile_series_style

    def check_points(figure, output_path):
        curves = [line for line in figure.axes[0].lines if not line.get_label().startswith("_")]
        assert all(list(curve.get_xdata()) == list(recording_checkpoints(2100)) for curve in curves)
        assert all(len(curve.get_ydata()) <= MAX_RECORDED_POINTS for curve in curves)
        assert figure.axes[0].get_xscale() == "log"

    monkeypatch.setattr(plot_regret, "save_figure_pair", check_points)
    adversarial_path = run_adversarial_experiment("auer_exp3", feedback_mode="bandit", horizon=2100, output_dir=tmp_path)
    fixed_path = run_cross_play_experiment("rps", ["auer_exp3"] * 2, feedback_mode="bandit", horizon=2100, output_dir=tmp_path)
    adversarial_rows = load_adversarial_rows(adversarial_path)
    fixed_rows = load_rows(fixed_path)
    curves = []
    for index, (times, means) in enumerate((
        aggregate_adversarial_regret([adversarial_rows], "average_external_regret"),
        aggregate_metric_curve([fixed_rows], 0, "average_external_regret"),
    )):
        curves.append(plot_regret.RegretCurve(times, means, str(index), profile_series_style(index, 2)))
    plot_regret.plot_regret_curves(curves, "$R_T/T$", tmp_path / "regret.png")


def test_replicate_aggregation_uses_only_shared_observed_times():
    trajectories = [
        [dict(t=str(t), player="0", external_regret=str(t)) for t in (1, 2, 3, 10)],
        [dict(t=str(t), player="0", external_regret=str(2 * t)) for t in (1, 4, 10)],
    ]
    for aggregate in (
        lambda: aggregate_metric_curve(trajectories, 0, "external_regret"),
        lambda: aggregate_adversarial_regret(trajectories, "external_regret"),
    ):
        times, means = aggregate()
        assert times.tolist() == [1, 10]
        assert means.tolist() == [1.5, 15.0]
