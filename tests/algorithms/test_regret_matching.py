import numpy as np
import pytest

from algorithms.internal_regret import RegretMatching, StationaryRegretMatching


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

    expected_regret = np.zeros((3, 3))
    expected_regret[1] = [-0.2, 0.0, 0.5]
    assert np.allclose(learner.cumulative_regret, expected_regret)
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
