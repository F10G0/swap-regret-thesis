import numpy as np
import pytest

from algorithms.external_regret import TsallisINF


def old_bisection_strategy(losses, eta):
    """The pre-cleanup 100-step solver, including its arithmetic and normalization."""
    if len(losses) == 1:
        return np.ones(1)
    shifted = losses - np.min(losses)
    lower, upper = 0.0, np.sqrt(len(losses)) / eta
    for _ in range(100):
        normalizer = (lower + upper) / 2.0
        probabilities = 1.0 / np.square(eta * (shifted + normalizer))
        if np.sum(probabilities) > 1.0:
            lower = normalizer
        else:
            upper = normalizer
    probabilities = 1.0 / np.square(eta * (shifted + upper))
    return probabilities / np.sum(probabilities)


@pytest.mark.parametrize("n_actions", [1, 3, 5, 9, 100])
@pytest.mark.parametrize("local_time", [0, 1, 4, 100, 1_000_000, 10**12])
def test_scaled_newton_matches_old_bisection(n_actions, local_time):
    learner = TsallisINF(n_actions, seed=7)
    learner.t = local_time
    random = np.random.default_rng(17)
    cases = [
        np.zeros(n_actions),
        np.full(n_actions, 1e12),
        np.arange(n_actions) * 2.5 + 100,
        np.arange(n_actions) * 1e16,
        np.arange(n_actions) % 2 * 1e8,  # Multiple minimum-loss actions.
        *[random.uniform(0, 1000, n_actions) for _ in range(5)],
        *[random.lognormal(0, 15, n_actions) for _ in range(5)],
    ]
    for losses in cases:
        learner.cumulative_loss = losses
        probabilities = learner._compute_strategy()
        expected = old_bisection_strategy(losses, learner.learning_rate)
        np.testing.assert_allclose(probabilities, expected, atol=2e-14, rtol=2e-13)
        assert np.all(np.isfinite(probabilities))
        assert np.all(probabilities >= 0)
        assert abs(np.sum(probabilities) - 1) < 1e-14
        a = learner.learning_rate * (losses - np.min(losses))
        z = 1 / np.sqrt(probabilities[np.argmin(losses)])
        assert abs(np.sum(1 / np.square(a + z)) - 1) < 1e-13


@pytest.mark.parametrize("offset", [0.0, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8])
def test_normalizer_near_lower_endpoint_remains_accurate(offset):
    learner = TsallisINF(9)
    # Near z=1, safeguarded steps may need bisection before Newton is admissible.
    learner.cumulative_loss = np.full(9, np.sqrt(8 / max(offset, 1e-300)))
    learner.cumulative_loss[0] = 0
    np.testing.assert_allclose(learner._compute_strategy(),
                               old_bisection_strategy(learner.cumulative_loss, learner.learning_rate),
                               atol=2e-14, rtol=2e-13)
