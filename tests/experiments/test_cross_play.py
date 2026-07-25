from algorithms.external_regret import Hedge
from algorithms.internal_regret import RegretMatching
from experiments.scenarios.cross_play import AlgorithmFactory


def test_algorithm_factory_passes_the_experiment_horizon() -> None:
    learner = AlgorithmFactory(Hedge).create(n_actions=3, horizon=100, seed=0)

    assert learner.horizon == 100


def test_algorithm_factory_omits_unused_horizon() -> None:
    learner = AlgorithmFactory(RegretMatching, uses_horizon=False).create(n_actions=3, horizon=100, seed=0)

    assert learner.horizon == 0
