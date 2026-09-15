"""Contracts shared by every empirical feedback setting and result family."""

from types import SimpleNamespace

import numpy as np
import pytest

from environments import BanditRepeatedGame, RepeatedGame
from experiments.scenarios.cross_play import run_cross_play_experiment
from experiments.result_schema import REGRET_FIELDNAMES
from experiments.runner import run_game
from experiments.scenarios import adversarial
from metrics.regret import RegretBundle


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
