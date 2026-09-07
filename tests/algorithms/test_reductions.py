import numpy as np
import pytest

from algorithms.external_regret import Hedge, TsallisINF
from algorithms.swap_regret import BanditBM, BanditIto, FullBM, FullIto, LCEIX


def test_full_blum_mansour_updates_every_learner() -> None:
    learner = FullBM(2, horizon=10, seed=0)

    learner.update(np.array([0.2, 0.8]))

    for inner_learner in learner.learners:
        assert np.allclose(inner_learner.cumulative_score, [0.1, 0.4])


def test_full_ito_uses_outer_strategy_and_updates_only_selected_learner() -> None:
    learner = FullIto(3, seed=0)
    learner.current_strategy = np.array([0.0, 1.0, 0.0])

    learner.sample_action()
    selected_learner = learner.selected_learner
    learner.update(np.array([0.2, 0.5, 0.8]))

    assert selected_learner is learner.learners[1]
    assert np.allclose(selected_learner.cumulative_score, [0.2, 0.5, 0.8])
    assert all(np.allclose(inner_learner.cumulative_score, 0.0) for inner_learner in [learner.learners[0], learner.learners[2]])
    assert learner.selected_learner is None


def test_bandit_ito_updates_only_selected_learner() -> None:
    learner = BanditIto(2, seed=0)
    learner.current_strategy = np.array([1.0, 0.0])

    action = learner.sample_action()
    selected_learner = learner.selected_learner
    learner.update(0.5)

    assert selected_learner is learner.learners[0]
    assert isinstance(selected_learner, TsallisINF)
    expected_loss = np.zeros(2)
    expected_loss[action] = 1.0
    assert np.array_equal(selected_learner.cumulative_loss, expected_loss)
    assert selected_learner.t == 1
    assert selected_learner.learning_rate == pytest.approx(1.0 / np.sqrt(2.0))
    assert np.allclose(learner.learners[1].cumulative_loss, 0.0)
    assert learner.learners[1].t == 0
    assert learner.learners[1].learning_rate == pytest.approx(1.0)
    assert learner.selected_learner is None


@pytest.mark.parametrize("learner_type", [FullIto, BanditIto])
def test_ito_reset_clears_selected_learner(learner_type) -> None:
    learner = learner_type(3, seed=0)
    learner.sample_action()
    assert learner.selected_learner is not None

    learner.reset()

    assert learner.current_action is None
    assert learner.selected_learner is None
    assert np.allclose(learner.strategy(), np.full(3, 1.0 / 3.0))
    if learner_type is BanditIto:
        assert all(isinstance(inner, TsallisINF) for inner in learner.learners)
        assert all(inner.t == 0 for inner in learner.learners)
        assert all(np.allclose(inner.cumulative_loss, 0.0) for inner in learner.learners)


def test_lce_ix_uses_exact_schedule_over_multiple_rounds() -> None:
    learner = LCEIX(3, seed=0)

    learner.sample_action()
    learner.update(0.25)
    learner.sample_action()
    learner.update(0.75)

    eta_3 = np.sqrt(np.log(3) / 3)
    for inner_learner in learner.learners:
        assert inner_learner.t == 2
        assert inner_learner.learning_rate == pytest.approx(eta_3)
        assert inner_learner.implicit_exploration == pytest.approx(eta_3 / 2.0)


def test_lce_ix_has_no_external_learning_rate() -> None:
    with pytest.raises(TypeError, match="learning_rate"):
        LCEIX(2, learning_rate=0.3, seed=0)


@pytest.mark.parametrize("learner_type", [FullBM, BanditBM])
def test_blum_mansour_passes_the_known_horizon_to_inner_learners(learner_type) -> None:
    learner = learner_type(3, horizon=100, seed=0)

    assert all(inner_learner.horizon == 100 for inner_learner in learner.learners)


def test_full_ito_hedge_uses_selected_learners_local_time() -> None:
    learner = FullIto(3, seed=0)
    assert all(isinstance(inner, Hedge) and inner.horizon is None for inner in learner.learners)
    for _ in range(3):
        learner.current_strategy = np.array([0.0, 1.0, 0.0])
        learner.sample_action()
        learner.update(np.array([0.2, 0.5, 0.8]))
    assert [inner.t for inner in learner.learners] == [0, 3, 0]
    assert learner.learners[1].learning_rate == pytest.approx(np.sqrt(8 * np.log(3) / 4))
    assert learner.learners[0].learning_rate == pytest.approx(np.sqrt(8 * np.log(3)))


def test_bandit_ito_supports_a_factory_without_a_horizon_parameter() -> None:
    def factory(n_actions, *, seed):
        return TsallisINF(n_actions, seed=seed)

    learner = BanditIto(3, inner_algorithm_factory=factory, seed=0)
    learner.sample_action()
    learner.update(0.5)
    assert sum(inner.t for inner in learner.learners) == 1
