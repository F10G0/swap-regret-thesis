import numpy as np
import pytest

from algorithms.external_regret import AuerExp3, Exp3IX, TsallisINF
from experiments.scenarios.cross_play import ALGORITHMS_BY_FEEDBACK_MODE

BANDIT_ALGORITHMS = ALGORITHMS_BY_FEEDBACK_MODE["bandit"]
FULL_ALGORITHMS = ALGORITHMS_BY_FEEDBACK_MODE["full_information"]


def test_registered_bandit_learners_use_literature_specific_inner_learners() -> None:
    auer_exp3 = BANDIT_ALGORITHMS["auer_exp3"].create(n_actions=3, horizon=100, seed=0)
    exp3_ix = BANDIT_ALGORITHMS["exp3_ix"].create(n_actions=3, horizon=100, seed=0)
    bandit_bm = BANDIT_ALGORITHMS["bm"].create(n_actions=3, horizon=100, seed=0)
    bandit_ito = BANDIT_ALGORITHMS["ito"].create(n_actions=3, horizon=100, seed=0)

    assert list(BANDIT_ALGORITHMS) == ["auer_exp3", "exp3_ix", "bm", "ito", "lce_ix"]
    assert list(FULL_ALGORITHMS) == ["hedge", "bm", "ito", "regret_matching", "stationary_regret_matching"]
    assert isinstance(auer_exp3, AuerExp3) and auer_exp3.horizon == 100
    assert isinstance(exp3_ix, Exp3IX) and exp3_ix.horizon == 100
    assert all(isinstance(inner, AuerExp3) and inner.horizon == 100 for inner in bandit_bm.learners)
    assert all(isinstance(inner, TsallisINF) for inner in bandit_ito.learners)


@pytest.mark.parametrize("mode,registry", [("bandit", BANDIT_ALGORITHMS), ("full", FULL_ALGORITHMS)])
def test_registered_learners_execute_with_explicit_horizon_contracts(mode, registry) -> None:
    for factory in registry.values():
        learner = factory.create(n_actions=3, horizon=4, seed=11)
        if factory.uses_horizon:
            assert learner.horizon == 4
        else:
            assert not hasattr(learner, "horizon")
        for _ in range(4):
            learner.sample_action()
            learner.update(0.5 if mode == "bandit" else np.array([0.2, 0.5, 0.8]))
        assert learner.t == 4
        assert np.isclose(learner.strategy().sum(), 1.0)


@pytest.mark.parametrize("name", ["ito", "lce_ix"])
def test_local_schedules_do_not_depend_on_experiment_horizon(name) -> None:
    first = BANDIT_ALGORITHMS[name].create(n_actions=3, horizon=10, seed=7)
    second = BANDIT_ALGORITHMS[name].create(n_actions=3, horizon=1000, seed=7)
    for reward in [0.2, 0.8, 0.5]:
        assert first.sample_action() == second.sample_action()
        first.update(reward)
        second.update(reward)
        assert np.array_equal(first.strategy(), second.strategy())
