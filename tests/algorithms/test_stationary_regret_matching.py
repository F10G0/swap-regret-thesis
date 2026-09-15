import numpy as np

from algorithms.internal_regret import StationaryRegretMatching
import algorithms.internal_regret.regret_matching as regret_matching_module
from algorithms.stationary import stationary_distribution


def test_stationary_regret_matching_starts_uniformly() -> None:
    learner = StationaryRegretMatching(3, seed=0)

    assert np.allclose(learner.strategy(), np.full(3, 1.0 / 3.0))


def test_stationary_regret_matching_still_solves_the_complete_transition_matrix(monkeypatch) -> None:
    learner = StationaryRegretMatching(9, seed=42)
    learner.t = 1000
    regrets = np.random.default_rng(7).uniform(-100.0, 100.0, size=(9, 9))
    np.fill_diagonal(regrets, 0.0)
    learner.cumulative_regret = regrets.copy()
    expected_matrix = np.maximum(regrets, 0.0) / (9 * learner.t)
    np.fill_diagonal(expected_matrix, 1.0 - expected_matrix.sum(axis=1))
    expected_strategy = stationary_distribution(expected_matrix)
    calls = []

    def check_matrix(matrix):
        np.testing.assert_array_equal(matrix, expected_matrix)
        calls.append(matrix.shape)
        return stationary_distribution(matrix)

    monkeypatch.setattr(regret_matching_module, "stationary_distribution", check_matrix)
    for action in range(9):
        learner.current_action = action
        np.testing.assert_array_equal(learner._compute_strategy(), expected_strategy)
    assert calls == [(9, 9)] * 9
    np.testing.assert_array_equal(learner.cumulative_regret, regrets)


def test_stationary_regret_matching_updates_only_the_sampled_action_row() -> None:
    learner = StationaryRegretMatching(3, seed=0)
    reward_vector = np.array([0.2, 0.8, 0.5])
    learner.current_strategy = np.array([0.0, 1.0, 0.0])

    assert learner.sample_action() == 1
    learner.update(reward_vector)

    expected_matrix = np.zeros((3, 3))
    expected_matrix[1] = [-0.6, 0.0, -0.3]
    assert np.allclose(learner.cumulative_regret, expected_matrix)


def test_stationary_regret_matching_satisfies_regret_flow_balance() -> None:
    learner = StationaryRegretMatching(3, seed=0)
    learner.t = 1
    learner.cumulative_regret = np.array(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    strategy = learner._compute_strategy()
    positive_regret = np.maximum(learner.cumulative_regret, 0.0)
    incoming_regret = strategy @ positive_regret
    outgoing_regret = strategy * np.sum(positive_regret, axis=1)

    assert np.allclose(strategy, [0.2, 0.4, 0.4])
    assert np.allclose(incoming_regret, outgoing_regret)

    learner.t = 100
    assert np.allclose(learner._compute_strategy(), strategy)
