"""Contracts shared by every empirical feedback setting and result family."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from environments import BanditRepeatedGame, RepeatedGame
from experiments.scenarios.cross_play import run_cross_play_experiment
from experiments.plots import plot_adversarial, plot_regret
from experiments.result_schema import REGRET_FIELDNAMES, RESULT_IMPLEMENTATION_VERSION
from experiments.results import iter_result_rows, load_final_result_rows
from experiments.runner import run_game
from experiments.scenarios import adversarial, adversarial_scaling
from metrics.regret import RegretBundle
from tests.support import read_csv_rows
from experiments.result_catalog import ResultRepository


class ObservedLearner:
    n_actions = 2

    def __init__(self, feedback_mode):
        self.feedback_mode = feedback_mode
        self.feedbacks = []
        self.probabilities = np.array([0.25, 0.75])

    def strategy(self):
        return self.probabilities

    def sample_action(self):
        return 0

    def update(self, feedback):
        if self.feedback_mode == "bandit":
            assert isinstance(feedback, float)
        else:
            assert isinstance(feedback, np.ndarray) and feedback.shape == (2,)
        self.feedbacks.append(np.copy(feedback))


def capture_evaluator(monkeypatch):
    observations = []
    original = RegretBundle.update

    def update(self, strategy, payoff_vector):
        observations.append((strategy.copy(), payoff_vector.copy()))
        return original(self, strategy, payoff_vector)

    monkeypatch.setattr(RegretBundle, "update", update)
    return observations


@pytest.mark.parametrize("feedback_mode", ["full_information", "bandit"])
def test_fixed_game_feedback_boundary_and_counterfactual_evaluation(monkeypatch, feedback_mode):
    # Against either realized opponent action, player 0 earns [0, 1].
    payoffs = np.array([[[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]])
    game_type = RepeatedGame if feedback_mode == "full_information" else BanditRepeatedGame
    players = [ObservedLearner(feedback_mode) for _ in range(2)]
    rows = []
    observations = capture_evaluator(monkeypatch)
    run_game("fixture", feedback_mode, game_type(payoffs), "fixture", players,
             SimpleNamespace(record=rows.append), horizon=4)
    assert len(observations) == 8
    for strategy, vector in observations[::2]:
        np.testing.assert_array_equal(strategy, [0.25, 0.75])
        np.testing.assert_array_equal(vector, [0.0, 1.0])
    if feedback_mode == "bandit":
        np.testing.assert_array_equal(players[0].feedbacks, np.zeros(4))
    else:
        np.testing.assert_array_equal(players[0].feedbacks, [[0.0, 1.0]] * 4)
    final = next(row for row in rows if row["t"] == 4 and row["player"] == 0)
    # The sampled action always loses, but strategy-weighted regret is only 4 * 0.25.
    for name in ("external", "internal", "swap"):
        assert final[f"{name}_regret"] == 1.0
        assert final[f"average_{name}_regret"] == 0.25
    np.testing.assert_array_equal(players[0].strategy(), [0.25, 0.75])


@pytest.mark.parametrize("feedback_mode", ["full_information", "bandit"])
@pytest.mark.parametrize("environment", [
    adversarial.HISTORICAL_FREQUENCY_ENVIRONMENT, adversarial.RANDOM_WALK_ENVIRONMENT,
])
def test_adversarial_feedback_boundary_and_evaluator_information(
    tmp_path, monkeypatch, feedback_mode, environment,
):
    learner = ObservedLearner(feedback_mode)
    algorithm = "hedge" if feedback_mode == "full_information" else "auer_exp3"
    monkeypatch.setitem(adversarial.ALGORITHMS_BY_FEEDBACK_MODE[feedback_mode], algorithm,
                        SimpleNamespace(create=lambda *args: learner))
    observations = capture_evaluator(monkeypatch)
    path = adversarial.run_adversarial_experiment(
        algorithm, n_actions=2, horizon=7, feedback_mode=feedback_mode,
        environment=environment, output_dir=tmp_path,
    )
    gains = np.zeros((2, 2))
    rows = adversarial.load_adversarial_rows(path)
    for (strategy, payoffs), feedback, row in zip(observations, learner.feedbacks, rows):
        np.testing.assert_array_equal(feedback, payoffs[0] if feedback_mode == "bandit" else payoffs)
        gains += strategy[:, None] * (payoffs[None, :] - payoffs[:, None])
        assert float(row["external_regret"]) == float(np.max(np.sum(gains, axis=0)))
        assert float(row["internal_regret"]) == float(np.max(gains))
        assert float(row["swap_regret"]) == float(np.sum(np.max(gains, axis=1)))
        assert {key for key in row if key.endswith("_regret")} == set(REGRET_FIELDNAMES)
    assert len(learner.feedbacks) == len(observations) == 7


@pytest.mark.parametrize("kind", ["fixed", "adversarial", "scaling"])
@pytest.mark.parametrize("version", [2, 3, None])
def test_stale_csv_is_rejected_without_rewriting(tmp_path, kind, version):
    if kind == "fixed":
        path = run_cross_play_experiment("rps", ["auer_exp3"] * 2, horizon=3, output_dir=tmp_path, feedback_mode="bandit")
        loaders = [lambda p: list(iter_result_rows(p)), load_final_result_rows]
    elif kind == "adversarial":
        path = adversarial.run_adversarial_experiment("hedge", horizon=3, output_dir=tmp_path)
        loaders = [adversarial.load_adversarial_rows, adversarial.load_final_adversarial_row]
    else:
        spec = adversarial_scaling.AdversarialScalingSpec(
            adversarial.RANDOM_WALK_ENVIRONMENT, "bandit", "auer_exp3",
            (2, 3), 1, 3, 7, 11,
        )
        path = adversarial_scaling.run_adversarial_scaling_experiment(spec, tmp_path)
        loaders = [adversarial_scaling.load_adversarial_scaling_rows]
    rows = read_csv_rows(path)
    assert {row["implementation_version"] for row in rows} == {str(RESULT_IMPLEMENTATION_VERSION)}
    for row in rows:
        for name in REGRET_FIELDNAMES:
            old_name = name.replace("average_", "average_expected_") if name.startswith("average_") else "expected_" + name
            row[old_name] = row.pop(name)
        row["regret_evaluation"] = "expected"
        if version is None:
            row.pop("implementation_version")
        else:
            row["implementation_version"] = str(version)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = path.read_bytes()
    for loader in loaders:
        with pytest.raises(ValueError, match="incompatible result implementation_version"):
            loader(path)
    if kind == "fixed":
        snapshot = ResultRepository(tmp_path).snapshot()
        assert snapshot.summaries() == []
        assert "incompatible result implementation_version" in snapshot.warnings[0]
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", ["fixed", "adversarial"])
def test_previous_plot_cache_version_cannot_bypass_csv_validation(tmp_path, monkeypatch, kind):
    cache = tmp_path / "cache"
    if kind == "fixed":
        module = plot_regret
        path = run_cross_play_experiment("rps", ["auer_exp3"] * 2, horizon=3, output_dir=tmp_path, feedback_mode="bandit")
        collect = lambda: module.collect_results(tmp_path, cache_dir=cache)
        loader_name = "iter_result_rows"
    else:
        module = plot_adversarial
        path = adversarial.run_adversarial_experiment("hedge", horizon=3, output_dir=tmp_path)
        collect = lambda: module.collect_adversarial_results(tmp_path, cache_dir=cache)
        loader_name = "load_adversarial_rows"
    collect()
    cache_path = cache / f"{path.stem}.json"
    payload = json.loads(cache_path.read_text())
    identity = payload if kind == "fixed" else payload["identity"]
    identity["version"] = module.PLOT_ROW_CACHE_VERSION - 1
    cache_path.write_text(json.dumps(payload))
    original = getattr(module, loader_name)
    calls = []

    def load(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, loader_name, load)
    collect()
    assert calls == [path]
