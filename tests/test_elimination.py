import math

import pytest

from algorithms import (
    make_elimination,
    make_elimination_standard,
)


def test_initial_state():
    algo = make_elimination(n_arms=3, phase_length_fn=lambda phase: 2)

    assert algo.t == 0
    assert algo.phase == 1
    assert algo.active_arms == [0, 1, 2]
    assert algo.active_index == 0
    assert algo.phase_remaining_pulls == 6
    assert algo.counts.tolist() == [0, 0, 0]
    assert algo.reward_sums.tolist() == [0.0, 0.0, 0.0]
    assert algo.phase_counts.tolist() == [0, 0, 0]
    assert algo.phase_reward_sums.tolist() == [0.0, 0.0, 0.0]


def test_round_robin_over_active_arms():
    algo = make_elimination(n_arms=3, phase_length_fn=lambda phase: 2)

    actions = []
    for _ in range(6):
        action = algo.select_action()
        actions.append(action)
        algo.update(action, reward=0.0)

    assert actions == [0, 1, 2, 0, 1, 2]


def test_global_and_phase_statistics_are_updated():
    algo = make_elimination(n_arms=2, phase_length_fn=lambda phase: 2)

    updates = [
        (0, 1.0),
        (1, 0.5),
        (0, 0.0),
    ]

    for expected_action, reward in updates:
        action = algo.select_action()
        assert action == expected_action
        algo.update(action, reward)

    assert algo.t == 3
    assert algo.counts.tolist() == [2, 1]
    assert algo.reward_sums.tolist() == [1.0, 0.5]
    assert algo.phase_counts.tolist() == [2, 1]
    assert algo.phase_reward_sums.tolist() == [1.0, 0.5]
    assert algo.phase_remaining_pulls == 1


def test_phase_empirical_means():
    algo = make_elimination(n_arms=3, phase_length_fn=lambda phase: 2)

    algo.update(0, 1.0)
    algo.update(1, 0.5)
    algo.update(0, 0.0)

    means = algo.phase_empirical_means()

    assert means.tolist() == [0.5, 0.5, 0.0]


def test_advance_phase_after_phase_is_complete():
    algo = make_elimination(n_arms=2, phase_length_fn=lambda phase: 1)

    actions = []
    for _ in range(2):
        action = algo.select_action()
        actions.append(action)
        algo.update(action, reward=1.0)

    assert actions == [0, 1]
    assert algo.phase == 2
    assert algo.active_arms == [0, 1]
    assert algo.active_index == 0
    assert algo.phase_remaining_pulls == 2
    assert algo.phase_counts.tolist() == [0, 0]
    assert algo.phase_reward_sums.tolist() == [0.0, 0.0]


def test_eliminates_arm_with_low_phase_mean():
    algo = make_elimination(n_arms=2, phase_length_fn=lambda phase: 1)

    action = algo.select_action()
    assert action == 0
    algo.update(action, reward=1.0)

    action = algo.select_action()
    assert action == 1
    algo.update(action, reward=0.0)

    assert algo.phase == 2
    assert algo.active_arms == [0]
    assert algo.is_commitment_phase()


def test_does_not_eliminate_arm_within_threshold():
    algo = make_elimination(n_arms=2, phase_length_fn=lambda phase: 1)

    # In phase 1, threshold = 2^(-1) = 0.5.
    # Arm 1 has mean 0.6, best mean is 1.0, so it survives.
    action = algo.select_action()
    assert action == 0
    algo.update(action, reward=1.0)

    action = algo.select_action()
    assert action == 1
    algo.update(action, reward=0.6)

    assert algo.phase == 2
    assert algo.active_arms == [0, 1]


def test_commitment_phase_always_plays_remaining_arm():
    algo = make_elimination(n_arms=2, phase_length_fn=lambda phase: 1)

    # Eliminate arm 1.
    algo.update(0, reward=1.0)
    algo.update(1, reward=0.0)

    assert algo.active_arms == [0]
    assert algo.is_commitment_phase()

    actions = []
    for _ in range(5):
        action = algo.select_action()
        actions.append(action)
        algo.update(action, reward=0.0)

    assert actions == [0, 0, 0, 0, 0]
    assert algo.phase == 2
    assert algo.active_arms == [0]


def test_commitment_phase_does_not_decrease_phase_remaining_pulls():
    algo = make_elimination(n_arms=2, phase_length_fn=lambda phase: 1)

    # Eliminate arm 1.
    algo.update(0, reward=1.0)
    algo.update(1, reward=0.0)

    remaining = algo.phase_remaining_pulls

    for _ in range(3):
        action = algo.select_action()
        algo.update(action, reward=0.0)

    assert algo.phase_remaining_pulls == remaining


def test_compute_phase_remaining_pulls_uses_active_arms():
    algo = make_elimination(n_arms=4, phase_length_fn=lambda phase: 3)

    assert algo.compute_phase_remaining_pulls() == 12

    algo.active_arms = [0, 2]
    assert algo.compute_phase_remaining_pulls() == 6


def test_compute_phase_remaining_pulls_rejects_nonpositive_value():
    algo = make_elimination(n_arms=2, phase_length_fn=lambda phase: 1)

    algo.phase_length_fn = lambda phase: 0

    with pytest.raises(ValueError, match="phase_length_fn must return a positive value"):
        algo.compute_phase_remaining_pulls()


def test_make_elimination_standard_phase_length():
    delta = 0.1
    algo = make_elimination_standard(n_arms=3, delta=delta)

    expected_m1 = math.ceil((2.0 ** 2) * math.log(1.0 / delta))
    expected_total = expected_m1 * 3

    assert algo.phase_remaining_pulls == expected_total


def test_make_elimination_standard_later_phase_length():
    delta = 0.2
    algo = make_elimination_standard(n_arms=2, delta=delta)

    algo.phase = 3
    expected_m3 = math.ceil((2.0 ** 6) * math.log(1.0 / delta))

    assert algo.compute_phase_remaining_pulls() == expected_m3 * 2


def test_invalid_delta():
    with pytest.raises(ValueError, match="delta must be in"):
        make_elimination_standard(n_arms=2, delta=0.0)

    with pytest.raises(ValueError, match="delta must be in"):
        make_elimination_standard(n_arms=2, delta=1.0)

    with pytest.raises(ValueError, match="delta must be in"):
        make_elimination_standard(n_arms=2, delta=-0.1)


def test_invalid_phase_length_fn():
    with pytest.raises(ValueError, match="phase_length_fn must not be None"):
        make_elimination(n_arms=2, phase_length_fn=None)


def test_invalid_n_arms():
    with pytest.raises(ValueError, match="n_arms must be positive"):
        make_elimination(n_arms=0, phase_length_fn=lambda phase: 1)

    with pytest.raises(ValueError, match="n_arms must be positive"):
        make_elimination_standard(n_arms=-1, delta=0.1)


def test_invalid_update_action():
    algo = make_elimination(n_arms=3, phase_length_fn=lambda phase: 1)

    with pytest.raises(AssertionError):
        algo.update(4, reward=0.0)


def test_select_action_no_active_arms_raises_runtime_error():
    algo = make_elimination(n_arms=2, phase_length_fn=lambda phase: 1)
    algo.active_arms = []

    with pytest.raises(RuntimeError, match="no active arms remain"):
        algo.select_action()


def test_reset():
    algo = make_elimination(n_arms=3, phase_length_fn=lambda phase: 2)

    for _ in range(4):
        action = algo.select_action()
        algo.update(action, reward=1.0)

    algo.reset()

    assert algo.t == 0
    assert algo.phase == 1
    assert algo.active_arms == [0, 1, 2]
    assert algo.active_index == 0
    assert algo.phase_remaining_pulls == 6
    assert algo.counts.tolist() == [0, 0, 0]
    assert algo.reward_sums.tolist() == [0.0, 0.0, 0.0]
    assert algo.phase_counts.tolist() == [0, 0, 0]
    assert algo.phase_reward_sums.tolist() == [0.0, 0.0, 0.0]


def test_repr_contains_state():
    algo = make_elimination(n_arms=3, phase_length_fn=lambda phase: 2)
    text = repr(algo)

    assert "EliminationAlgorithm" in text
    assert "n_arms=3" in text
    assert "t=0" in text
    assert "phase=1" in text
    assert "phase_remaining_pulls=6" in text
    assert "active_arms=[0, 1, 2]" in text
    assert "active_index=0" in text
    