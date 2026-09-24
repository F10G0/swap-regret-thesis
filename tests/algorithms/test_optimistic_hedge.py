import numpy as np
import pytest

from algorithms.external_regret import OptimisticHedge
from algorithms.swap_regret import BMOptimisticHedge


def normalized(weights):
    return weights / np.sum(weights)


def test_optimistic_hedge_initialization_and_standalone_rate() -> None:
    learner = OptimisticHedge(3, horizon=100, seed=0)

    np.testing.assert_allclose(learner.current_strategy, np.full(3, 1.0 / 3.0))
    np.testing.assert_array_equal(learner.cumulative_score, np.zeros(3))
    np.testing.assert_array_equal(learner.previous_reward, np.zeros(3))
    assert learner.t == 0
    assert learner.current_action is None
    assert learner.learning_rate == pytest.approx((np.log(3) / 100) ** (1.0 / 6.0))


def test_reward_updates_match_the_multiplicative_and_loss_forms() -> None:
    learner = OptimisticHedge(3, horizon=100, seed=0)
    eta = learner.learning_rate
    r1 = np.array([0.2, 0.5, 0.9])
    r2 = np.array([0.8, 0.1, 0.4])
    x1 = learner.current_strategy.copy()

    learner.update(r1)
    x2 = normalized(x1 * np.exp(eta * 2.0 * r1))
    np.testing.assert_allclose(learner.current_strategy, x2)
    np.testing.assert_allclose(learner.cumulative_score, 2.0 * r1)
    np.testing.assert_array_equal(learner.previous_reward, r1)
    assert learner.t == 1

    learner.update(r2)
    x3 = normalized(x2 * np.exp(eta * (2.0 * r2 - r1)))
    np.testing.assert_allclose(learner.current_strategy, x3)
    np.testing.assert_allclose(learner.cumulative_score, r1 + 2.0 * r2)
    np.testing.assert_array_equal(learner.previous_reward, r2)
    assert learner.t == 2

    zero = np.zeros(3)
    loss1, loss2 = 1.0 - r1, 1.0 - r2
    loss_x2 = normalized(x1 * np.exp(-eta * (2.0 * loss1 - zero)))
    loss_x3 = normalized(loss_x2 * np.exp(-eta * (2.0 * loss2 - loss1)))
    np.testing.assert_allclose(loss_x2, x2)
    np.testing.assert_allclose(loss_x3, x3)


def test_explicit_rate_and_reset_are_fixed_and_copy_feedback() -> None:
    learner = OptimisticHedge(3, horizon=7, learning_rate=0.125, seed=0)
    reward = np.array([0.2, 0.5, 0.9])
    learner.sample_action()
    learner.update(reward)
    reward[:] = 0.0
    np.testing.assert_array_equal(learner.previous_reward, [0.2, 0.5, 0.9])

    learner.reset()

    assert learner.learning_rate == 0.125
    assert learner.t == 0
    assert learner.current_action is None
    np.testing.assert_allclose(learner.current_strategy, np.full(3, 1.0 / 3.0))
    np.testing.assert_array_equal(learner.cumulative_score, np.zeros(3))
    np.testing.assert_array_equal(learner.previous_reward, np.zeros(3))


@pytest.mark.parametrize("learning_rate", [0.0, -0.1, np.inf, -np.inf, np.nan])
def test_explicit_learning_rate_must_be_finite_and_strictly_positive(learning_rate) -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        OptimisticHedge(3, horizon=7, learning_rate=learning_rate)


def test_bm_optimistic_hedge_uses_theorem_rate_and_inner_learners() -> None:
    learner = BMOptimisticHedge(3, horizon=1, n_players=2, seed=0)
    expected = (3.0 * np.log(3.0) / 4.0) ** 0.25

    assert expected > 1.0 / 6.0
    assert learner.inner_learning_rate == pytest.approx(expected)
    assert len(learner.learners) == 3
    assert all(isinstance(inner, OptimisticHedge) for inner in learner.learners)
    assert all(inner.learning_rate == pytest.approx(expected) for inner in learner.learners)
    assert expected != pytest.approx((np.log(3.0)) ** (1.0 / 6.0))


@pytest.mark.parametrize("n_actions", [-1, 0, 1])
def test_bm_optimistic_hedge_requires_at_least_two_actions(n_actions) -> None:
    with pytest.raises(ValueError, match="n_actions must be at least 2"):
        BMOptimisticHedge(n_actions, horizon=100)


def test_bm_optimistic_hedge_uses_current_and_previous_weighted_rewards() -> None:
    learner = BMOptimisticHedge(3, horizon=100, n_players=2, seed=0)
    r1 = np.array([0.2, 0.5, 0.9])
    r2 = np.array([0.8, 0.1, 0.4])
    outer_x1 = learner.current_strategy.copy()

    learner.update(r1)
    first_scores = [inner.cumulative_score.copy() for inner in learner.learners]
    for index, inner in enumerate(learner.learners):
        weighted = outer_x1[index] * r1
        np.testing.assert_allclose(inner.previous_reward, weighted)
        np.testing.assert_allclose(inner.cumulative_score, 2.0 * weighted)

    outer_x2 = learner.current_strategy.copy()
    learner.update(r2)
    for index, inner in enumerate(learner.learners):
        previous = outer_x1[index] * r1
        current = outer_x2[index] * r2
        np.testing.assert_allclose(inner.cumulative_score - first_scores[index], 2.0 * current - previous)
        np.testing.assert_allclose(inner.previous_reward, current)


def test_bm_optimistic_hedge_reset_recreates_initialized_learners() -> None:
    learner = BMOptimisticHedge(3, horizon=100, n_players=4, seed=0)
    original_learners = tuple(learner.learners)
    learner.update(np.array([0.2, 0.5, 0.9]))

    learner.reset()

    assert learner.t == 0
    assert learner.current_action is None
    np.testing.assert_allclose(learner.current_strategy, np.full(3, 1.0 / 3.0))
    assert len(learner.learners) == 3
    assert all(inner is not original for inner, original in zip(learner.learners, original_learners))
    assert all(inner.t == 0 and inner.current_action is None for inner in learner.learners)
    assert all(inner.learning_rate == pytest.approx(learner.inner_learning_rate) for inner in learner.learners)
    assert all(np.array_equal(inner.cumulative_score, np.zeros(3)) for inner in learner.learners)
    assert all(np.array_equal(inner.previous_reward, np.zeros(3)) for inner in learner.learners)


def test_optimistic_strategies_remain_finite_normalized_distributions() -> None:
    learners = [OptimisticHedge(5, horizon=1000), BMOptimisticHedge(5, horizon=1000, n_players=3)]
    rewards = np.random.default_rng(7).random((1000, 5))

    for learner in learners:
        for reward in rewards:
            learner.update(reward)
            assert np.all(np.isfinite(learner.current_strategy))
            assert np.all(learner.current_strategy >= 0.0)
            assert np.sum(learner.current_strategy) == pytest.approx(1.0)


@pytest.mark.parametrize("n_players", [0, -1, 1.5, np.inf, np.nan, True])
def test_bm_optimistic_hedge_requires_a_positive_integer_player_count(n_players) -> None:
    with pytest.raises(ValueError, match="n_players"):
        BMOptimisticHedge(3, horizon=100, n_players=n_players)
