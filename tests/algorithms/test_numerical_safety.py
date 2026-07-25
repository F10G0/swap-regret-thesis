from functools import partial

import numpy as np
import pytest

from algorithms.external_regret import Exp3, Exp3IX, Hedge
from algorithms.swap_regret import BanditBM, LCEIX
from config import NUMERICAL_TOLERANCE


@pytest.mark.parametrize("factory", [partial(Exp3IX, 1, 0, seed=0), partial(LCEIX, 1, seed=0)])
def test_implicit_exploration_algorithms_allow_one_action(factory) -> None:
    learner = factory()

    assert learner.sample_action() == 0
    learner.update(0.5)
    assert np.array_equal(learner.strategy(), [1.0])


@pytest.mark.parametrize("reward", [np.nan, np.inf, -np.inf])
def test_exp3_rejects_non_finite_rewards(reward: float) -> None:
    learner = Exp3(2, horizon=10, seed=0)
    learner.sample_action()

    with pytest.raises(ValueError, match="finite"):
        learner.update(reward)


def test_exp3_ix_uses_implicit_exploration_loss_estimate() -> None:
    learner = Exp3IX(2, horizon=0, seed=0)
    action = learner.sample_action()

    learner.update(0.25)

    eta_1 = np.sqrt(np.log(2))
    eta_2 = np.sqrt(np.log(2) / 2)
    expected_loss = 0.75 / (0.5 + eta_1 / 2.0)
    assert learner.t == 1
    assert learner.learning_rate == pytest.approx(eta_2)
    assert learner.implicit_exploration == pytest.approx(eta_2 / 2.0)
    assert learner.cumulative_score[action] == pytest.approx(-expected_loss)
    assert np.count_nonzero(learner.cumulative_score) == 1


def test_exp3_uses_importance_weighted_reward() -> None:
    learner = Exp3(2, horizon=10, seed=0)
    action = learner.sample_action()

    learner.update(0.25)

    assert learner.current_action == action
    assert learner.cumulative_score[action] == pytest.approx(0.25 / 0.5)
    assert np.count_nonzero(learner.cumulative_score) == 1


def test_hedge_rejects_non_finite_reward_vectors() -> None:
    learner = Hedge(2, horizon=10, seed=0)

    with pytest.raises(ValueError, match="finite"):
        learner.update(np.array([np.nan, 0.5]))


def test_hedge_updates_cumulative_score() -> None:
    learner = Hedge(2, horizon=10, seed=0)

    learner.update(np.array([0.25, 1.0]))

    assert np.array_equal(learner.cumulative_score, [0.25, 1.0])
    assert learner.strategy()[1] > learner.strategy()[0]


def test_exponential_weights_remain_normalized_for_extreme_scores() -> None:
    learner = Hedge(2, horizon=10, seed=0)
    learner.cumulative_score = np.array([0.0, 3000.0])
    learner.current_strategy = learner._compute_strategy()

    strategy = learner.strategy()
    assert np.all(np.isfinite(strategy))
    assert np.all(strategy >= NUMERICAL_TOLERANCE / (1.0 + NUMERICAL_TOLERANCE))
    assert np.isclose(np.sum(strategy), 1.0)


def test_bandit_blum_mansour_survives_large_inner_scores() -> None:
    learner = BanditBM(2, horizon=10, seed=0)
    learner.learners[0].cumulative_score = np.array([1000.0, 0.0])
    learner.learners[1].cumulative_score = np.array([0.0, 1000.0])

    for inner_learner in learner.learners:
        inner_learner.current_strategy = inner_learner._compute_strategy()
    learner.current_strategy = learner._compute_strategy()

    learner.sample_action()
    learner.update(1.0)

    assert np.all(np.isfinite(learner.strategy()))


def test_bandit_blum_mansour_updates_reward_based_learners() -> None:
    learner = BanditBM(2, horizon=10, seed=0)
    action = learner.sample_action()

    learner.update(0.5)

    for inner_learner in learner.learners:
        assert inner_learner.current_action == action
        assert inner_learner.cumulative_score[action] == pytest.approx(0.5)
        assert np.count_nonzero(inner_learner.cumulative_score) == 1


def test_lce_ix_uses_theoretical_learning_rate_schedule() -> None:
    learner = LCEIX(2, seed=0)
    action = learner.sample_action()

    learner.update(0.25)

    eta_1 = np.sqrt(np.log(2))
    eta_2 = np.sqrt(np.log(2) / 2)
    expected_observed_loss = 0.5 * 0.5 * 0.75 / 0.5
    expected_estimated_loss = expected_observed_loss / (0.5 + eta_1 / 2.0)

    for inner_learner in learner.learners:
        assert inner_learner.current_action == action
        assert inner_learner.t == 1
        assert inner_learner.learning_rate == pytest.approx(eta_2)
        assert inner_learner.implicit_exploration == pytest.approx(eta_2 / 2.0)
        assert inner_learner.cumulative_score[action] == pytest.approx(-expected_estimated_loss)


def test_lce_ix_reset_restores_the_first_round() -> None:
    learner = LCEIX(2, seed=0)
    learner.sample_action()
    learner.update(0.25)

    learner.reset()

    assert learner.current_action is None
    assert all(inner_learner.t == 0 for inner_learner in learner.learners)
    assert np.allclose(learner.strategy(), [0.5, 0.5])


def test_seed_reproduces_sampled_action_sequence() -> None:
    first = Hedge(3, horizon=10, seed=17)
    second = Hedge(3, horizon=10, seed=17)

    assert [first.sample_action() for _ in range(20)] == [
        second.sample_action() for _ in range(20)
    ]


def test_known_horizon_learning_rates_are_fixed() -> None:
    hedge = Hedge(3, horizon=100, seed=0)
    exp3 = Exp3(3, horizon=100, seed=0)
    exp3_ix = Exp3IX(3, horizon=100, seed=0)

    assert hedge.learning_rate == pytest.approx(np.sqrt(8.0 * np.log(3) / 100))
    assert exp3.learning_rate == pytest.approx(np.sqrt(np.log(3) / 300))
    assert exp3_ix.learning_rate == pytest.approx(np.sqrt(np.log(3) / 100))

    hedge.update(np.array([0.2, 0.5, 0.8]))
    exp3.sample_action()
    exp3.update(0.5)
    exp3_ix.sample_action()
    exp3_ix.update(0.5)

    assert hedge.learning_rate == pytest.approx(np.sqrt(8.0 * np.log(3) / 100))
    assert exp3.learning_rate == pytest.approx(np.sqrt(np.log(3) / 300))
    assert exp3_ix.learning_rate == pytest.approx(np.sqrt(np.log(3) / 100))


def test_unknown_horizon_learning_rates_follow_local_updates() -> None:
    hedge = Hedge(3, horizon=0, seed=0)
    exp3 = Exp3(3, horizon=0, seed=0)

    assert hedge.learning_rate == pytest.approx(np.sqrt(8.0 * np.log(3)))
    assert exp3.learning_rate == pytest.approx(np.sqrt(np.log(3) / 3))

    hedge.update(np.array([0.2, 0.5, 0.8]))
    exp3.sample_action()
    exp3.update(0.5)

    assert hedge.learning_rate == pytest.approx(np.sqrt(8.0 * np.log(3) / 2))
    assert exp3.learning_rate == pytest.approx(np.sqrt(np.log(3) / 6))


def test_learning_rate_schedule_continues_beyond_known_horizon() -> None:
    learner = Hedge(2, horizon=2, seed=0)

    learner.update(np.array([0.2, 0.8]))
    assert learner._rate_horizon == 2

    learner.update(np.array([0.2, 0.8]))
    assert learner._rate_horizon == 3


@pytest.mark.parametrize("learner_type", [Hedge, Exp3, Exp3IX])
def test_exponential_weights_reject_negative_horizons(learner_type) -> None:
    with pytest.raises(ValueError, match="horizon"):
        learner_type(3, horizon=-1)
