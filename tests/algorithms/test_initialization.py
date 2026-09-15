from collections.abc import Callable
from functools import partial

import numpy as np
import pytest

from algorithms.base import Algorithm
from algorithms.external_regret import AuerExp3, Exp3IX, Hedge, OptimisticHedge, TsallisINF
from algorithms.internal_regret import RegretMatching, StationaryRegretMatching
from algorithms.swap_regret import BMOptimisticHedge, BanditBM, BanditIto, FullBM, FullIto, LCEIX, LCEIXInner


@pytest.mark.parametrize(
    ("factory", "feedback"),
    [
        pytest.param(partial(Hedge, 3, 10, seed=0), np.array([0.2, 0.5, 0.8]), id="hedge"),
        pytest.param(partial(OptimisticHedge, 3, 10, seed=0), np.array([0.2, 0.5, 0.8]), id="optimistic-hedge"),
        pytest.param(partial(Exp3IX, 3, horizon=10, seed=0), 0.5, id="exp3-ix"),
        pytest.param(partial(AuerExp3, 3, horizon=10, seed=0), 0.5, id="auer-exp3"),
        pytest.param(partial(TsallisINF, 3, seed=0), 0.5, id="tsallis-inf"),
        pytest.param(partial(Hedge, 3, horizon=None, seed=0), np.array([0.2, 0.5, 0.8]), id="anytime-hedge"),
        pytest.param(partial(RegretMatching, 3, seed=0), np.array([0.2, 0.5, 0.8]), id="regret-matching"),
        pytest.param(partial(StationaryRegretMatching, 3, seed=0), np.array([0.2, 0.5, 0.8]), id="stationary-regret-matching"),
        pytest.param(partial(FullBM, 3, 10, seed=0), np.array([0.2, 0.5, 0.8]), id="full-bm"),
        pytest.param(partial(BMOptimisticHedge, 3, 10, seed=0), np.array([0.2, 0.5, 0.8]), id="bm-optimistic-hedge"),
        pytest.param(partial(BanditBM, 3, 10, seed=0), 0.5, id="bandit-bm"),
        pytest.param(partial(FullIto, 3, seed=0), np.array([0.2, 0.5, 0.8]), id="full-ito"),
        pytest.param(partial(BanditIto, 3, seed=0), 0.5, id="bandit-ito"),
        pytest.param(partial(LCEIX, 3, seed=0), 0.5, id="lce-ix"),
    ],
)
def test_algorithms_start_and_reset_uniformly(factory: Callable[[], Algorithm], feedback: float | np.ndarray) -> None:
    learner = factory()
    uniform_strategy = np.full(learner.n_actions, 1.0 / learner.n_actions)

    assert np.allclose(learner.strategy(), uniform_strategy)

    learner.sample_action()
    learner.update(feedback)
    learner.reset()

    assert learner.current_action is None
    assert np.allclose(learner.strategy(), uniform_strategy)


@pytest.mark.parametrize("learner_type", [TsallisINF, RegretMatching, StationaryRegretMatching, FullIto, BanditIto, LCEIX, LCEIXInner])
def test_horizon_free_learners_reject_a_horizon_argument(learner_type) -> None:
    learner = learner_type(3, seed=0)
    assert not hasattr(learner, "horizon")
    with pytest.raises(TypeError, match="horizon"):
        learner_type(3, horizon=10, seed=0)
