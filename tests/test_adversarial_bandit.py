import numpy as np
import pytest

from environments.adversarial_bandit import AdversarialBandit


def test_adversarial_bandit_n_arms_and_horizon():
    rewards = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
    ])

    env = AdversarialBandit(rewards)

    assert env.n_arms == 3
    assert env.horizon == 2


def test_adversarial_bandit_pull_returns_correct_reward():
    rewards = np.array([
        [1.0, 0.0],
        [0.5, 1.0],
    ])

    env = AdversarialBandit(rewards)

    assert env.pull(0) == 1.0
    assert env.pull(1) == 1.0


def test_adversarial_bandit_best_fixed_arm_reward():
    rewards = np.array([
        [1.0, 0.0],
        [0.5, 1.0],
        [0.5, 0.0],
    ])

    env = AdversarialBandit(rewards)

    assert env.best_fixed_arm_reward() == 2.0
    assert env.best_fixed_arm() == 0


def test_adversarial_bandit_reset():
    rewards = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    env = AdversarialBandit(rewards)

    env.pull(0)
    env.reset()

    assert env.pull(0) == 1.0


def test_adversarial_bandit_rejects_invalid_action():
    rewards = np.array([
        [1.0, 0.0],
    ])

    env = AdversarialBandit(rewards)

    with pytest.raises(IndexError):
        env.pull(2)


def test_adversarial_bandit_rejects_rewards_outside_unit_interval():
    rewards = np.array([
        [1.2, 0.0],
    ])

    with pytest.raises(ValueError):
        AdversarialBandit(rewards)
        