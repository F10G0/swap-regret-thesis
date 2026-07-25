import numpy as np

from algorithms.internal_regret import StationaryRegretMatching


def test_stationary_regret_matching_starts_uniformly() -> None:
    learner = StationaryRegretMatching(3, seed=0)

    assert np.allclose(learner.strategy(), np.full(3, 1.0 / 3.0))


def test_stationary_regret_matching_updates_only_the_sampled_action_row() -> None:
    learner = StationaryRegretMatching(3, seed=0)
    reward_vector = np.array([0.2, 0.8, 0.5])
    learner.current_strategy = np.array([0.0, 1.0, 0.0])

    assert learner.sample_action() == 1
    learner.update(reward_vector)

    expected_regret = np.zeros((3, 3))
    expected_regret[1] = [-0.6, 0.0, -0.3]
    assert np.allclose(learner.cumulative_regret, expected_regret)


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
