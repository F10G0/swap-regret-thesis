import math

import pytest

from algorithms import (
    make_ucb_standard,
    make_ucb_delta,
    make_ucb_asymptotically_optimal,
)


def test_initialization_phase_plays_each_arm_once():
    algo = make_ucb_standard(n_arms=3)

    actions = []
    for _ in range(3):
        action = algo.select_action()
        actions.append(action)
        algo.update(action, reward=0.0)

    assert actions == [0, 1, 2]
    assert algo.t == 3
    assert algo.counts.tolist() == [1, 1, 1]
    assert not algo.in_initialization_phase()


def test_selects_arm_with_highest_ucb_value_after_initialization():
    algo = make_ucb_delta(n_arms=3, delta=0.1)

    rewards = {
        0: 0.1,
        1: 0.9,
        2: 0.2,
    }

    for _ in range(3):
        action = algo.select_action()
        algo.update(action, rewards[action])

    assert algo.select_action() == 1


def test_ucb_exploration_bonus_can_change_choice():
    algo = make_ucb_delta(n_arms=2, delta=0.1)

    # Initialization
    action = algo.select_action()
    algo.update(action, reward=1.0)

    action = algo.select_action()
    algo.update(action, reward=0.0)

    # Arm 0 has higher mean, but after many pulls of arm 0,
    # arm 1 should eventually get a larger bonus.
    for _ in range(20):
        algo.update(0, reward=1.0)

    assert algo.select_action() == 1


def test_best_ucb_arm_matches_manual_computation():
    algo = make_ucb_delta(n_arms=3, delta=0.2)

    rewards = [0.2, 0.5, 0.9]
    for reward in rewards:
        action = algo.select_action()
        algo.update(action, reward)

    beta = 2.0 * math.log(1.0 / 0.2)
    means = algo.empirical_means()
    bonuses = (beta / algo.counts) ** 0.5
    expected = int((means + bonuses).argmax())

    assert algo.best_ucb_arm() == expected


def test_empirical_statistics_are_updated_correctly():
    algo = make_ucb_standard(n_arms=2)

    algo.update(0, 1.0)
    algo.update(0, 0.5)
    algo.update(1, 0.0)

    assert algo.t == 3
    assert algo.counts.tolist() == [2, 1]
    assert algo.reward_sums.tolist() == [1.5, 0.0]
    assert algo.empirical_means().tolist() == [0.75, 0.0]


def test_reset():
    algo = make_ucb_standard(n_arms=3)

    for _ in range(3):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    algo.reset()

    assert algo.t == 0
    assert algo.counts.tolist() == [0, 0, 0]
    assert algo.reward_sums.tolist() == [0.0, 0.0, 0.0]
    assert algo.in_initialization_phase()
    assert algo.select_action() == 0


def test_invalid_n_arms():
    with pytest.raises(ValueError, match="n_arms must be positive"):
        make_ucb_standard(n_arms=0)

    with pytest.raises(ValueError, match="n_arms must be positive"):
        make_ucb_delta(n_arms=-1, delta=0.1)

    with pytest.raises(ValueError, match="n_arms must be positive"):
        make_ucb_asymptotically_optimal(n_arms=0)


def test_invalid_delta():
    with pytest.raises(ValueError, match="delta must be in"):
        make_ucb_delta(n_arms=2, delta=0.0)

    with pytest.raises(ValueError, match="delta must be in"):
        make_ucb_delta(n_arms=2, delta=1.0)

    with pytest.raises(ValueError, match="delta must be in"):
        make_ucb_delta(n_arms=2, delta=-0.1)


def test_invalid_update_action():
    algo = make_ucb_standard(n_arms=3)

    with pytest.raises(IndexError, match="invalid action index"):
        algo.update(-1, reward=0.0)

    with pytest.raises(IndexError, match="invalid action index"):
        algo.update(3, reward=0.0)


def test_negative_beta_raises_error():
    from algorithms.ucb import UpperConfidenceBound

    algo = UpperConfidenceBound(
        n_arms=2,
        beta_fn=lambda t: -1.0,
    )

    for _ in range(2):
        action = algo.select_action()
        algo.update(action, reward=0.0)

    with pytest.raises(ValueError, match="beta_fn\\(t\\) must be nonnegative"):
        algo.select_action()


def test_none_beta_fn_raises_error():
    from algorithms.ucb import UpperConfidenceBound

    with pytest.raises(ValueError, match="beta_fn must not be None"):
        UpperConfidenceBound(n_arms=2, beta_fn=None)


def test_standard_beta_function():
    algo = make_ucb_standard(n_arms=2)

    assert algo.beta_fn(0) == pytest.approx(2.0 * math.log(2))
    assert algo.beta_fn(1) == pytest.approx(2.0 * math.log(2))
    assert algo.beta_fn(10) == pytest.approx(2.0 * math.log(10))


def test_delta_beta_function():
    algo = make_ucb_delta(n_arms=2, delta=0.05)

    assert algo.beta_fn(0) == pytest.approx(2.0 * math.log(1.0 / 0.05))
    assert algo.beta_fn(100) == pytest.approx(2.0 * math.log(1.0 / 0.05))


def test_asymptotically_optimal_beta_function():
    algo = make_ucb_asymptotically_optimal(n_arms=2)

    expected_t2 = 2.0 * math.log(1.0 + 2 * (math.log(2) ** 2))
    expected_t10 = 2.0 * math.log(1.0 + 10 * (math.log(10) ** 2))

    assert algo.beta_fn(0) == pytest.approx(expected_t2)
    assert algo.beta_fn(2) == pytest.approx(expected_t2)
    assert algo.beta_fn(10) == pytest.approx(expected_t10)


def test_argmax_tie_breaks_to_smallest_index():
    algo = make_ucb_delta(n_arms=3, delta=0.1)

    for _ in range(3):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    assert algo.select_action() == 0


def test_repr_contains_state():
    algo = make_ucb_standard(n_arms=4)
    text = repr(algo)

    assert "UpperConfidenceBound" in text
    assert "n_arms=4" in text
    assert "t=0" in text
    assert "beta_fn=" in text
    