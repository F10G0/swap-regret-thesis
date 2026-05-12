import numpy as np
import pytest

from algorithms import (
    make_exp3,
    make_exp3_doubling,
    make_exp3_ix_doubling,
    make_exp3_adaptive,
    make_exp3_ix_adaptive,
)


def test_exp3_selects_valid_action():
    alg = make_exp3(
        n_arms=3,
        learning_rate=0.1,
        implicit_exploration=0.0,
    )

    action = alg.select_action()

    assert 0 <= action < 3


def test_exp3_probabilities_sum_to_one():
    alg = make_exp3(
        n_arms=4,
        learning_rate=0.1,
        implicit_exploration=0.0,
    )

    alg.select_action()

    assert np.isclose(np.sum(alg.probabilities), 1.0)
    assert np.all(alg.probabilities >= 0.0)


def test_exp3_update_increases_time():
    alg = make_exp3(
        n_arms=3,
        learning_rate=0.1,
        implicit_exploration=0.0,
    )

    action = alg.select_action()
    alg.update(action, reward=1.0)

    assert alg.t == 1


def test_exp3_update_changes_selected_arm_loss():
    alg = make_exp3(
        n_arms=3,
        learning_rate=0.1,
        implicit_exploration=0.0,
    )

    action = alg.select_action()
    old_loss = alg.estimated_cumulative_loss.copy()

    alg.update(action, reward=0.0)

    assert alg.estimated_cumulative_loss[action] > old_loss[action]


def test_exp3_rejects_invalid_reward():
    alg = make_exp3(
        n_arms=3,
        learning_rate=0.1,
        implicit_exploration=0.0,
    )

    action = alg.select_action()

    with pytest.raises(ValueError):
        alg.update(action, reward=1.5)


def test_exp3_doubling_runs_one_round():
    alg = make_exp3_doubling(n_arms=3)

    action = alg.select_action()
    alg.update(action, reward=1.0)

    assert alg.t == 1


def test_exp3_ix_doubling_uses_implicit_exploration():
    alg = make_exp3_ix_doubling(n_arms=3)

    assert alg.algorithm.implicit_exploration > 0.0
    assert np.isclose(
        alg.algorithm.implicit_exploration,
        0.5 * alg.algorithm.learning_rate,
    )


def test_exp3_adaptive_lr_runs_one_round():
    alg = make_exp3_adaptive(n_arms=3)

    action = alg.select_action()
    alg.update(action, reward=1.0)

    assert alg.t == 1


def test_exp3_ix_adaptive_lr_sets_implicit_exploration():
    alg = make_exp3_ix_adaptive(n_arms=3)

    alg.select_action()

    assert alg.implicit_exploration > 0.0
    assert np.isclose(
        alg.implicit_exploration,
        0.5 * alg.learning_rate,
    )
    