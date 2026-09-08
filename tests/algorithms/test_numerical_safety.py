from functools import partial

import numpy as np
import pytest

from algorithms.external_regret import AuerExp3, Exp3IX, Hedge, TsallisINF
from algorithms.swap_regret import BanditBM, FullBM, LCEIX
from config import NUMERICAL_TOLERANCE


class RecordingAuerExp3(AuerExp3):
    def _reset_state(self) -> None:
        super()._reset_state()
        self.received_gains = []

    def _update_state(self, reward: float) -> None:
        self.received_gains.append(reward)
        super()._update_state(reward)


@pytest.mark.parametrize("factory", [partial(Exp3IX, 1, horizon=10, seed=0), partial(LCEIX, 1, seed=0)])
def test_implicit_exploration_algorithms_allow_one_action(factory) -> None:
    learner = factory()

    assert learner.sample_action() == 0
    learner.update(0.5)
    assert np.array_equal(learner.strategy(), [1.0])


def test_exp3_ix_uses_implicit_exploration_loss_estimate() -> None:
    learner = Exp3IX(2, horizon=10, seed=0)
    action = learner.sample_action()

    learner.update(0.25)

    eta = np.sqrt(2.0 * np.log(2) / 20)
    expected_loss = 0.75 / (0.5 + eta / 2.0)
    assert learner.t == 1
    assert learner.learning_rate == pytest.approx(eta)
    assert learner.implicit_exploration == pytest.approx(eta / 2.0)
    assert learner.cumulative_score[action] == pytest.approx(-expected_loss)
    assert np.count_nonzero(learner.cumulative_score) == 1


def test_auer_exp3_uses_importance_weighted_gain() -> None:
    learner = AuerExp3(2, horizon=10, seed=0)
    action = learner.sample_action()

    learner.update(0.25)

    expected_score = np.zeros(2)
    expected_score[action] = 0.5
    assert np.array_equal(learner.cumulative_score, expected_score)


def test_auer_exp3_uses_literature_exploration_rate() -> None:
    learner = AuerExp3(3, horizon=100, seed=0)

    expected_gamma = np.sqrt(3.0 * np.log(3.0) / 100.0)
    assert learner.explicit_exploration == pytest.approx(expected_gamma)


def test_auer_exp3_mixes_weights_with_explicit_uniform_exploration() -> None:
    learner = AuerExp3(3, horizon=100, seed=0)
    learner.cumulative_score = np.array([0.0, 4.0, 1.0])

    strategy = learner._compute_strategy()
    gamma = learner.explicit_exploration
    logits = gamma * learner.cumulative_score / 3.0
    weights = np.exp(logits - np.max(logits))
    expected = (1.0 - gamma) * weights / np.sum(weights) + gamma / 3.0

    assert np.allclose(strategy, expected)
    assert np.all(strategy >= gamma / 3.0)


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


def test_bandit_blum_mansour_constructs_paper_observed_gains() -> None:
    learner = BanditBM(
        2,
        horizon=10,
        inner_algorithm_factory=RecordingAuerExp3,
        seed=0,
    )
    transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
    for inner_learner, strategy in zip(learner.learners, transition_matrix):
        inner_learner.current_strategy = strategy
    outer_strategy = learner._compute_strategy()
    assert np.allclose(outer_strategy, [0.6, 0.4])
    learner.current_strategy = outer_strategy
    action = 1
    reward = 0.25
    learner.current_action = action

    learner.update(reward)

    probability = outer_strategy[action]
    expected_observed_gains = []
    for i, inner_learner in enumerate(learner.learners):
        observed_gain = outer_strategy[i] * transition_matrix[i, action] * reward / probability
        expected_observed_gains.append(observed_gain)
        assert inner_learner.received_gains == pytest.approx([observed_gain])
        assert inner_learner.t == 1

    assert expected_observed_gains == pytest.approx([0.075, 0.175])
    assert np.allclose(
        [inner_learner.cumulative_score for inner_learner in learner.learners],
        [[0.0, 0.375], [0.0, 0.25]],
    )


def test_lce_ix_uses_theoretical_learning_rate_schedule() -> None:
    learner = LCEIX(2, seed=0)
    action = learner.sample_action()

    learner.update(0.25)

    eta_1 = np.sqrt(np.log(2))
    eta_2 = np.sqrt(np.log(2) / 2)
    expected_observed_loss = 0.5 * 0.5 * 0.75 / 0.5
    expected_estimated_loss = expected_observed_loss / (0.5 + eta_1 / 2.0)

    for inner_learner in learner.learners:
        assert inner_learner.t == 1
        assert inner_learner.learning_rate == pytest.approx(eta_2)
        assert inner_learner.implicit_exploration == pytest.approx(eta_2 / 2.0)
        assert inner_learner.cumulative_score[action] == pytest.approx(-expected_estimated_loss)


def test_tsallis_inf_strategy_satisfies_the_half_tsallis_kkt_equation() -> None:
    learner = TsallisINF(3, seed=0)
    learner.t = 4
    learner.cumulative_loss = np.array([0.0, 1.0, 3.0])

    strategy = learner._compute_strategy()
    lagrange_values = 1.0 / (learner.learning_rate * np.sqrt(strategy)) - learner.cumulative_loss

    assert np.allclose(lagrange_values, lagrange_values[0])
    assert strategy[0] > strategy[1] > strategy[2]


def test_tsallis_inf_distributions_remain_valid() -> None:
    learner = TsallisINF(5, seed=0)

    for round_index in range(100):
        learner.sample_action()
        learner.update(float(round_index % 2))
        strategy = learner.strategy()
        assert np.all(np.isfinite(strategy))
        assert np.all(strategy >= 0.0)
        assert np.isclose(np.sum(strategy), 1.0)


def test_tsallis_inf_reset_restores_initial_local_time() -> None:
    learner = TsallisINF(3, seed=0)
    learner.sample_action()
    learner.update(0.25)

    learner.reset()

    assert learner.t == 0
    assert learner.learning_rate == pytest.approx(1.0)
    assert np.allclose(learner.cumulative_loss, 0.0)
    assert np.allclose(learner.strategy(), np.full(3, 1.0 / 3.0))


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
    auer_exp3 = AuerExp3(3, horizon=100, seed=0)
    exp3_ix = Exp3IX(3, horizon=100, seed=0)

    assert hedge.learning_rate == pytest.approx(np.sqrt(8.0 * np.log(3) / 100))
    assert auer_exp3.learning_rate == pytest.approx(np.sqrt(3 * np.log(3) / 100) / 3)
    assert exp3_ix.learning_rate == pytest.approx(np.sqrt(2.0 * np.log(3) / 300))

    hedge.update(np.array([0.2, 0.5, 0.8]))
    auer_exp3.sample_action()
    auer_exp3.update(0.5)
    exp3_ix.sample_action()
    exp3_ix.update(0.5)

    assert hedge.learning_rate == pytest.approx(np.sqrt(8.0 * np.log(3) / 100))
    assert auer_exp3.learning_rate == pytest.approx(np.sqrt(3 * np.log(3) / 100) / 3)
    assert exp3_ix.learning_rate == pytest.approx(np.sqrt(2.0 * np.log(3) / 300))


def test_unknown_horizon_learning_rates_follow_local_updates() -> None:
    hedge = Hedge(3, horizon=None, seed=0)

    assert hedge.learning_rate == pytest.approx(np.sqrt(8.0 * np.log(3)))

    hedge.update(np.array([0.2, 0.5, 0.8]))

    assert hedge.learning_rate == pytest.approx(np.sqrt(8.0 * np.log(3) / 2))


@pytest.mark.parametrize("learner_type", [Hedge, AuerExp3, Exp3IX])
def test_fixed_schedule_does_not_switch_after_horizon(learner_type) -> None:
    learner = learner_type(2, horizon=2, seed=0)
    initial_rate = learner.learning_rate
    for _ in range(4):
        learner.sample_action()
        learner.update(np.array([0.2, 0.8]) if learner_type is Hedge else 0.5)
        assert learner.learning_rate == initial_rate


@pytest.mark.parametrize("learner_type", [Hedge, AuerExp3, Exp3IX, FullBM, BanditBM])
@pytest.mark.parametrize("horizon", [0, -1, 1.5, np.inf, np.nan, True])
def test_known_horizon_must_be_a_positive_integer(learner_type, horizon) -> None:
    with pytest.raises(ValueError, match="horizon"):
        learner_type(3, horizon=horizon)


@pytest.mark.parametrize("learner_type", [AuerExp3, Exp3IX, FullBM, BanditBM])
def test_known_horizon_is_required(learner_type) -> None:
    with pytest.raises(TypeError, match="horizon"):
        learner_type(3)
    with pytest.raises(ValueError, match="horizon"):
        learner_type(3, horizon=None)
