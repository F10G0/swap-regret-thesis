import numpy as np
import pytest

from algorithms.internal_regret import RegretMatching, StationaryRegretMatching
from algorithms.internal_regret.base import RegretMatchingBase


def _play(learner: RegretMatching, action: int, reward_vector: np.ndarray) -> None:
    strategy = np.zeros(learner.n_actions)
    strategy[action] = 1.0
    learner.current_strategy = strategy
    assert learner.sample_action() == action
    learner.update(reward_vector)


def test_regret_matching_starts_uniformly() -> None:
    learner = RegretMatching(3, seed=0)

    assert np.allclose(learner.strategy(), np.full(3, 1.0 / 3.0))


def test_regret_matching_updates_only_the_played_action_row() -> None:
    learner = RegretMatching(3, seed=0)

    _play(learner, 1, np.array([0.2, 0.4, 0.9]))

    expected_matrix = np.zeros((3, 3))
    expected_matrix[1] = [-0.2, 0.0, 0.5]
    assert np.allclose(learner.cumulative_regret, expected_matrix)
    assert np.allclose(learner.strategy(), np.array([0.0, 5.0 / 6.0, 1.0 / 6.0]))


def test_regret_matching_uses_positive_average_regrets_from_current_action() -> None:
    learner = RegretMatching(3, seed=0)
    _play(learner, 0, np.array([0.2, 0.5, 1.0]))
    _play(learner, 0, np.array([0.6, 0.4, 0.7]))

    expected_switching = np.array([0.0, 0.05, 0.45]) / learner.normalization
    expected_strategy = expected_switching.copy()
    expected_strategy[0] = 1.0 - np.sum(expected_switching)

    assert learner.t == 2
    assert np.allclose(learner.strategy(), expected_strategy)


def test_regret_matching_preserves_inertia() -> None:
    learner = RegretMatching(4, seed=0)

    _play(learner, 0, np.array([0.0, 1.0, 1.0, 1.0]))

    assert learner.strategy()[0] > 0.0
    assert np.isclose(np.sum(learner.strategy()), 1.0)


@pytest.mark.parametrize("n_actions", [1, 3, 9, 50])
@pytest.mark.parametrize("round_count", [1, 1000, 1_000_000])
def test_regret_matching_row_equals_previous_full_matrix(n_actions, round_count) -> None:
    learner = RegretMatching(n_actions)
    learner.t = round_count
    random = np.random.default_rng(42)
    for regrets in [
        np.zeros((n_actions, n_actions)),
        -random.random((n_actions, n_actions)) * round_count,
        random.uniform(-1.0, 1.0, size=(n_actions, n_actions)) * round_count,
    ]:
        np.fill_diagonal(regrets, 0.0)
        learner.cumulative_regret = regrets.copy()
        previous = np.maximum(regrets, 0.0) / (n_actions * round_count)
        np.fill_diagonal(previous, 1.0 - previous.sum(axis=1))
        for action in range(n_actions):
            learner.current_action = action
            np.testing.assert_array_equal(learner._compute_strategy(), previous[action])
        np.testing.assert_array_equal(learner.cumulative_regret, regrets)


def test_regret_matching_does_not_construct_full_transition_matrix(monkeypatch) -> None:
    def unexpected_matrix(_learner):
        pytest.fail("ordinary RegretMatching must only construct its selected row")

    monkeypatch.setattr(RegretMatchingBase, "_regret_transition_matrix", property(unexpected_matrix))
    learner = RegretMatching(9, seed=42)
    random = np.random.default_rng(7)
    for _ in range(20):
        learner.sample_action()
        learner.update(random.random(9))
        assert learner.strategy().shape == (9,)
        assert learner.strategy().sum() == pytest.approx(1.0)


@pytest.mark.parametrize("learner_type", [RegretMatching, StationaryRegretMatching])
def test_regret_matching_reset_clears_sampled_action(learner_type) -> None:
    learner = learner_type(3, seed=0)
    learner.sample_action()
    learner.update(np.array([0.2, 0.5, 0.8]))

    learner.reset()

    assert learner.current_action is None
    assert learner.t == 0
    assert np.array_equal(learner.cumulative_regret, np.zeros((3, 3)))
    assert np.allclose(learner.strategy(), np.full(3, 1.0 / 3.0))
