import math

import pytest

from algorithms import (
    make_phased_ucb_exponential,
    make_phased_ucb_count_doubling,
)


def test_initialization_phase_plays_each_arm_once():
    algo = make_phased_ucb_exponential(n_arms=3, delta=0.1)

    actions = []
    for _ in range(3):
        action = algo.select_action()
        actions.append(action)
        algo.update(action, reward=0.0)

    assert actions == [0, 1, 2]
    assert algo.t == 3
    assert algo.counts.tolist() == [1, 1, 1]
    assert not algo.in_initialization_phase()


def test_exponential_phase_length_first_phase():
    algo = make_phased_ucb_exponential(n_arms=2, delta=0.1)

    # Initialization
    for _ in range(2):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    action = algo.select_action()

    assert algo.phase == 1
    assert algo.phase_arm == action
    assert algo.phase_remaining_pulls == 2  # 2^1


def test_exponential_phase_plays_same_arm_until_phase_ends():
    algo = make_phased_ucb_exponential(n_arms=2, delta=0.1)

    rewards = {
        0: 1.0,
        1: 0.0,
    }

    # Initialization
    for _ in range(2):
        action = algo.select_action()
        algo.update(action, rewards[action])

    first_phase_arm = algo.select_action()
    assert first_phase_arm == 0

    actions = []
    for _ in range(2):
        action = algo.select_action()
        actions.append(action)
        algo.update(action, reward=1.0)

    assert actions == [0, 0]
    assert algo.phase == 2
    assert algo.phase_arm is None
    assert algo.phase_remaining_pulls == 0


def test_exponential_second_phase_length():
    algo = make_phased_ucb_exponential(n_arms=2, delta=0.1)

    # Initialization
    for _ in range(2):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    # Finish phase 1: length 2
    for _ in range(2):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    assert algo.phase == 2

    action = algo.select_action()

    assert algo.phase_arm == action
    assert algo.phase_remaining_pulls == 4  # 2^2


def test_count_doubling_phase_length_alpha_2():
    algo = make_phased_ucb_count_doubling(n_arms=2, delta=0.1, alpha=2.0)

    # Initialization: each arm count is 1
    for _ in range(2):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    action = algo.select_action()

    assert algo.phase_arm == action
    assert algo.phase_remaining_pulls == 1


def test_count_doubling_phase_length_alpha_3():
    algo = make_phased_ucb_count_doubling(n_arms=2, delta=0.1, alpha=3.0)

    # Initialization: each arm count is 1
    for _ in range(2):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    action = algo.select_action()

    assert algo.phase_arm == action
    assert algo.phase_remaining_pulls == 2


def test_count_doubling_uses_snapshot_count_at_phase_start():
    algo = make_phased_ucb_count_doubling(n_arms=2, delta=0.1, alpha=3.0)

    # Initialization
    for _ in range(2):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    phase_arm = algo.select_action()
    initial_remaining = algo.phase_remaining_pulls

    algo.update(phase_arm, reward=1.0)

    assert initial_remaining == 2
    assert algo.phase_remaining_pulls == 1


def test_best_ucb_arm_matches_manual_computation():
    algo = make_phased_ucb_exponential(n_arms=3, delta=0.2)

    rewards = [0.2, 0.5, 0.9]

    # Initialization
    for reward in rewards:
        action = algo.select_action()
        algo.update(action, reward)

    beta = 2.0 * math.log(1.0 / 0.2)
    means = algo.empirical_means()
    bonuses = (beta / algo.counts) ** 0.5
    expected = int((means + bonuses).argmax())

    assert algo.best_ucb_arm() == expected


def test_compute_phase_length_rejects_nonpositive_value():
    from algorithms.phased_ucb import PhasedUCB

    algo = PhasedUCB(
        n_arms=2,
        beta_fn=lambda t: 1.0,
        phase_length_fn=lambda phase, arm, counts, t: 0,
    )

    for _ in range(2):
        action = algo.select_action()
        algo.update(action, reward=0.0)

    with pytest.raises(ValueError, match="phase_length_fn must return a positive value"):
        algo.select_action()


def test_negative_beta_raises_error():
    from algorithms.phased_ucb import PhasedUCB

    algo = PhasedUCB(
        n_arms=2,
        beta_fn=lambda t: -1.0,
        phase_length_fn=lambda phase, arm, counts, t: 1,
    )

    for _ in range(2):
        action = algo.select_action()
        algo.update(action, reward=0.0)

    with pytest.raises(ValueError, match="beta_fn\\(t\\) must be nonnegative"):
        algo.select_action()


def test_none_beta_fn_raises_error():
    from algorithms.phased_ucb import PhasedUCB

    with pytest.raises(ValueError, match="beta_fn must not be None"):
        PhasedUCB(
            n_arms=2,
            beta_fn=None,
            phase_length_fn=lambda phase, arm, counts, t: 1,
        )


def test_none_phase_length_fn_raises_error():
    from algorithms.phased_ucb import PhasedUCB

    with pytest.raises(ValueError, match="phase_length_fn must not be None"):
        PhasedUCB(
            n_arms=2,
            beta_fn=lambda t: 1.0,
            phase_length_fn=None,
        )


def test_invalid_n_arms():
    with pytest.raises(ValueError, match="n_arms must be positive"):
        make_phased_ucb_exponential(n_arms=0, delta=0.1)

    with pytest.raises(ValueError, match="n_arms must be positive"):
        make_phased_ucb_count_doubling(n_arms=-1, delta=0.1)


def test_invalid_delta():
    with pytest.raises(ValueError, match="delta must be in"):
        make_phased_ucb_exponential(n_arms=2, delta=0.0)

    with pytest.raises(ValueError, match="delta must be in"):
        make_phased_ucb_exponential(n_arms=2, delta=1.0)

    with pytest.raises(ValueError, match="delta must be in"):
        make_phased_ucb_count_doubling(n_arms=2, delta=-0.1)


def test_invalid_alpha():
    with pytest.raises(ValueError, match="alpha must be greater than 1"):
        make_phased_ucb_count_doubling(n_arms=2, delta=0.1, alpha=1.0)

    with pytest.raises(ValueError, match="alpha must be greater than 1"):
        make_phased_ucb_count_doubling(n_arms=2, delta=0.1, alpha=0.5)


def test_invalid_update_action():
    algo = make_phased_ucb_exponential(n_arms=3, delta=0.1)

    with pytest.raises(IndexError, match="invalid action index"):
        algo.update(-1, reward=0.0)

    with pytest.raises(IndexError, match="invalid action index"):
        algo.update(3, reward=0.0)


def test_reset():
    algo = make_phased_ucb_exponential(n_arms=3, delta=0.1)

    for _ in range(4):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    algo.reset()

    assert algo.t == 0
    assert algo.phase == 1
    assert algo.phase_arm is None
    assert algo.phase_remaining_pulls == 0
    assert algo.counts.tolist() == [0, 0, 0]
    assert algo.reward_sums.tolist() == [0.0, 0.0, 0.0]


def test_argmax_tie_breaks_to_smallest_index():
    algo = make_phased_ucb_exponential(n_arms=3, delta=0.1)

    for _ in range(3):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    assert algo.select_action() == 0
    assert algo.phase_arm == 0


def test_exponential_beta_function():
    algo = make_phased_ucb_exponential(n_arms=2, delta=0.05)

    expected = 2.0 * math.log(1.0 / 0.05)

    assert algo.beta_fn(0) == pytest.approx(expected)
    assert algo.beta_fn(100) == pytest.approx(expected)


def test_count_doubling_beta_function():
    algo = make_phased_ucb_count_doubling(n_arms=2, delta=0.05)

    expected = 2.0 * math.log(1.0 / 0.05)

    assert algo.beta_fn(0) == pytest.approx(expected)
    assert algo.beta_fn(100) == pytest.approx(expected)


def test_repr_contains_state():
    algo = make_phased_ucb_exponential(n_arms=4, delta=0.1)
    text = repr(algo)

    assert "PhasedUCB" in text
    assert "n_arms=4" in text
    assert "t=0" in text
    assert "phase=1" in text
    assert "phase_arm=None" in text
    assert "phase_remaining_pulls=0" in text
    