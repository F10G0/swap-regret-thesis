from experiments.scenarios.cross_play import ALGORITHMS_BY_FEEDBACK_MODE, AlgorithmFactory
import numpy as np
import pytest

from algorithms.base import Algorithm
from algorithms.external_regret import AuerExp3, Exp3IX, Hedge, TsallisINF
from algorithms.internal_regret import StationaryRegretMatching
from algorithms.swap_regret import BanditBM, BanditIto, FullIto, LCEIX
from algorithms.swap_regret.base import StationaryReduction
from config import NUMERICAL_TOLERANCE
from environments import BanditRepeatedGame, HistoricalFrequencyAdversary, LazyRandomWalkEnvironment, RepeatedGame
from metrics.regret import RegretBundle

BANDIT = ALGORITHMS_BY_FEEDBACK_MODE["bandit"]
FULL = ALGORITHMS_BY_FEEDBACK_MODE["full_information"]


CASES = [
    *[("full_information", name, factory) for name, factory in FULL.items()],
    *[("bandit", name, factory) for name, factory in BANDIT.items()],
    ("bandit", "tsallis", AlgorithmFactory(TsallisINF, uses_horizon=False)),
    ("full_information", "anytime_hedge", AlgorithmFactory(Hedge, uses_horizon=False)),
]


def assert_distribution(strategy):
    assert np.all(np.isfinite(strategy))
    assert np.all(strategy >= 0)
    assert abs(np.sum(strategy) - 1) <= NUMERICAL_TOLERANCE


@pytest.mark.parametrize("mode,name,factory", CASES, ids=[f"{mode}-{name}" for mode, name, _ in CASES])
@pytest.mark.parametrize("n_actions", [3, 9])
@pytest.mark.parametrize("environment", ["fixed", "historical", "random_walk"])
def test_valid_runs_preserve_trusted_core_invariants(monkeypatch, mode, name, factory, n_actions, environment):
    original_update = Algorithm.update

    def checked_update(learner, feedback):
        # Assertions belong in tests, including those on reduction-generated feedback.
        assert np.all(np.isfinite(feedback))
        assert np.all(np.asarray(feedback) >= -NUMERICAL_TOLERANCE)
        assert np.all(np.asarray(feedback) <= 1 + NUMERICAL_TOLERANCE)
        before_feedback = np.copy(feedback)
        previous = learner.strategy()
        before_strategy = previous.copy()
        assert previous is learner.current_strategy
        assert_distribution(previous)
        original_update(learner, feedback)
        np.testing.assert_array_equal(feedback, before_feedback)
        np.testing.assert_array_equal(previous, before_strategy)
        assert_distribution(learner.strategy())
        if isinstance(learner, StationaryReduction):
            matrix = learner._transition_matrix
        elif isinstance(learner, StationaryRegretMatching):
            matrix = learner._regret_transition_matrix
        else:
            return
        assert np.linalg.norm(learner.strategy() @ matrix - learner.strategy(), ord=1) <= NUMERICAL_TOLERANCE

    monkeypatch.setattr(Algorithm, "update", checked_update)
    learner = factory.create(n_actions, 100, seed=7)
    regrets = RegretBundle(n_actions)
    if environment == "fixed":
        payoff_tensor = np.random.default_rng(17).random((2, n_actions, n_actions))
        game = (RepeatedGame if mode == "full_information" else BanditRepeatedGame)(payoff_tensor)
    elif environment == "historical":
        game = HistoricalFrequencyAdversary(n_actions)
    else:
        game = LazyRandomWalkEnvironment(n_actions, 100, seed=17)

    for time in range(100):
        action = learner.sample_action()
        assert learner.current_action == action
        if environment == "fixed":
            game.step((action, time % n_actions))
            payoffs = game.deviation_payoffs(0)
            feedback = game.feedback(0)
        else:
            game.step((action,)) if environment == "historical" else game.step()
            payoffs = game.feedback()
            feedback = payoffs if mode == "full_information" else float(payoffs[action])
        regrets.update(action, payoffs)
        learner.update(feedback)
        assert np.all(np.isfinite(regrets.cumulative_replacement_gains))
    assert learner.t == 100
    if isinstance(learner, (FullIto, BanditIto)):
        assert sum(child.t for child in learner.learners) == 100
    if environment == "fixed":
        np.testing.assert_array_equal(game.payoff_tensor, payoff_tensor)
    assert all(np.isfinite(value) for value in regrets.summary(100).values())


@pytest.mark.parametrize("learner_class", [BanditBM, LCEIX])
def test_bandit_reduction_builds_matrix_only_for_stationary_solve(monkeypatch, learner_class):
    learner = learner_class(3, horizon=10) if learner_class is BanditBM else learner_class(3)
    original = StationaryReduction._transition_matrix.fget
    calls = []

    def matrix(learner):
        calls.append(learner.t)
        return original(learner)

    monkeypatch.setattr(StationaryReduction, "_transition_matrix", property(matrix))
    for _ in range(10):
        learner.sample_action()
        learner.update(0.25)
    assert calls == list(range(1, 11))


@pytest.mark.parametrize("learner_class", [Hedge, AuerExp3, Exp3IX])
def test_fixed_constants_are_not_recomputed_after_construction(monkeypatch, learner_class):
    learner = learner_class(3, horizon=10, seed=7)
    rate = learner.learning_rate

    def unexpected(*args):
        pytest.fail("a fixed-horizon constant was recomputed")

    monkeypatch.setattr(np, "sqrt", unexpected)
    monkeypatch.setattr(np, "log", unexpected)
    for _ in range(3):
        # NumPy's choice itself uses sqrt for its own probability tolerance;
        # this unit test isolates learner parameter computation from sampling.
        learner.current_action = 1
        learner.update(np.array([0.2, 0.5, 0.8]) if learner_class is Hedge else 0.5)
        assert learner.learning_rate == rate
    learner.reset()
    assert learner.learning_rate == rate
