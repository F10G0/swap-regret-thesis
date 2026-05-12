import math
import pytest

from algorithms import make_etc, make_etc_doubling
from algorithms.etc_wrappers import optimal_etc_m


def test_round_robin_exploration():
    algo = make_etc(n_arms=3, m=2)

    actions = []
    for _ in range(6):
        action = algo.select_action()
        actions.append(action)
        algo.update(action, reward=0.0)

    assert actions == [0, 1, 2, 0, 1, 2]


def test_commit_after_exploration():
    algo = make_etc(n_arms=3, m=1)

    rewards = {
        0: 0.1,
        1: 0.2,
        2: 0.9,
    }

    for _ in range(3):
        action = algo.select_action()
        algo.update(action, rewards[action])

    assert algo.select_action() == 2


def test_committed_arm_stays_fixed():
    algo = make_etc(n_arms=3, m=1)

    rewards = {
        0: 0.1,
        1: 0.8,
        2: 0.2,
    }

    for _ in range(3):
        action = algo.select_action()
        algo.update(action, rewards[action])

    actions = []
    for _ in range(5):
        action = algo.select_action()
        actions.append(action)
        algo.update(action, reward=0.0)

    assert actions == [1, 1, 1, 1, 1]
    assert algo.committed_arm == 1


def test_empirical_means_after_exploration():
    algo = make_etc(n_arms=2, m=2)

    updates = [
        (0, 1.0),
        (1, 0.0),
        (0, 0.5),
        (1, 1.0),
    ]

    for expected_action, reward in updates:
        action = algo.select_action()
        assert action == expected_action
        algo.update(action, reward)

    assert algo.counts.tolist() == [2, 2]
    assert algo.reward_sums.tolist() == [1.5, 1.0]
    assert algo.empirical_means().tolist() == [0.75, 0.5]
    assert algo.select_action() == 0


def test_in_exploration_phase():
    algo = make_etc(n_arms=2, m=2)

    assert algo.in_exploration_phase()

    for _ in range(3):
        action = algo.select_action()
        algo.update(action, reward=0.0)

    assert algo.in_exploration_phase()

    action = algo.select_action()
    algo.update(action, reward=0.0)

    assert not algo.in_exploration_phase()


def test_reset():
    algo = make_etc(n_arms=3, m=2)

    for _ in range(4):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    algo.reset()

    assert algo.t == 0
    assert algo.committed_arm is None
    assert algo.counts.tolist() == [0, 0, 0]
    assert algo.reward_sums.tolist() == [0.0, 0.0, 0.0]


def test_invalid_m():
    with pytest.raises(ValueError, match="m must be positive"):
        make_etc(n_arms=3, m=0)

    with pytest.raises(ValueError, match="m must be positive"):
        make_etc(n_arms=3, m=-1)


def test_invalid_n_arms():
    with pytest.raises(ValueError, match="n_arms must be positive"):
        make_etc(n_arms=0, m=1)

    with pytest.raises(ValueError, match="n_arms must be positive"):
        make_etc(n_arms=-1, m=1)


def test_invalid_update_action():
    algo = make_etc(n_arms=3, m=1)

    with pytest.raises(IndexError, match="invalid action index"):
        algo.update(-1, reward=0.0)

    with pytest.raises(IndexError, match="invalid action index"):
        algo.update(3, reward=0.0)


def test_argmax_tie_breaks_to_smallest_index():
    algo = make_etc(n_arms=3, m=1)

    for _ in range(3):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    assert algo.select_action() == 0
    assert algo.committed_arm == 0


def test_repr_contains_state():
    algo = make_etc(n_arms=3, m=2)
    text = repr(algo)

    assert "ExploreThenCommit" in text
    assert "n_arms=3" in text
    assert "m=2" in text
    assert "t=0" in text
    assert "committed_arm=None" in text


def test_optimal_etc_m():
    assert optimal_etc_m(n_arms=4, horizon=32) == math.ceil((32 / 4) ** (2.0 / 3.0))


def test_optimal_etc_m_invalid_arguments():
    with pytest.raises(ValueError, match="n_arms must be positive"):
        optimal_etc_m(n_arms=0, horizon=10)

    with pytest.raises(ValueError, match="horizon must be positive"):
        optimal_etc_m(n_arms=2, horizon=0)


def test_make_etc_doubling_initial_state():
    algo = make_etc_doubling(n_arms=3)

    assert algo.n_arms == 3
    assert algo.t == 0
    assert algo.epoch == 0
    assert algo.epoch_t == 0
    assert algo.epoch_length == 1


def test_make_etc_doubling_uses_default_m_fn():
    algo = make_etc_doubling(n_arms=2, initial_epoch_length=8)

    assert algo.algorithm.m == optimal_etc_m(n_arms=2, horizon=8)


def test_make_etc_doubling_uses_custom_m_fn():
    algo = make_etc_doubling(
        n_arms=2,
        m_fn=lambda horizon: 3,
        initial_epoch_length=8,
    )

    assert algo.algorithm.m == 3


def test_make_etc_doubling_advances_epoch():
    algo = make_etc_doubling(
        n_arms=2,
        m_fn=lambda horizon: 1,
        initial_epoch_length=2,
    )

    assert algo.epoch == 0
    assert algo.epoch_length == 2

    for _ in range(2):
        action = algo.select_action()
        algo.update(action, reward=0.0)

    assert algo.epoch == 1
    assert algo.epoch_t == 0
    assert algo.epoch_length == 4
    assert algo.algorithm.t == 0


def test_make_etc_doubling_resets_inner_algorithm_each_epoch():
    algo = make_etc_doubling(
        n_arms=2,
        m_fn=lambda horizon: 1,
        initial_epoch_length=1,
    )

    action = algo.select_action()
    algo.update(action, reward=1.0)

    assert algo.epoch == 1
    assert algo.algorithm.t == 0
    assert algo.algorithm.counts.tolist() == [0, 0]
    assert algo.algorithm.reward_sums.tolist() == [0.0, 0.0]


def test_make_etc_doubling_invalid_initial_epoch_length():
    with pytest.raises(ValueError, match="initial_epoch_length must be positive"):
        make_etc_doubling(n_arms=2, initial_epoch_length=0)


def test_make_etc_doubling_invalid_n_arms():
    with pytest.raises(ValueError, match="n_arms must be positive"):
        make_etc_doubling(n_arms=0)
        