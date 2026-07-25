from collections.abc import Callable
from functools import partial

import numpy as np
import pytest

from algorithms.base import Algorithm
from algorithms.external_regret import Exp3, Exp3IX, Hedge
from algorithms.internal_regret import RegretMatching, StationaryRegretMatching
from algorithms.swap_regret import BanditBM, BanditIto, FullBM, FullIto, LCEIX


@pytest.mark.parametrize(
    ("factory", "feedback"),
    [
        pytest.param(partial(Hedge, 3, 10, seed=0), np.array([0.2, 0.5, 0.8]), id="hedge"),
        pytest.param(partial(Exp3, 3, 10, seed=0), 0.5, id="exp3"),
        pytest.param(partial(Exp3IX, 3, 0, seed=0), 0.5, id="exp3-ix"),
        pytest.param(partial(RegretMatching, 3, seed=0), np.array([0.2, 0.5, 0.8]), id="regret-matching"),
        pytest.param(partial(StationaryRegretMatching, 3, seed=0), np.array([0.2, 0.5, 0.8]), id="stationary-regret-matching"),
        pytest.param(partial(FullBM, 3, 10, seed=0), np.array([0.2, 0.5, 0.8]), id="full-bm"),
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
