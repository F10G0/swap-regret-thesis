import numpy as np
import pytest

from experiments.games import PAYOFF_FACTORIES, create_cyclic_dominance_payoffs, normalize_payoffs


def test_benchmark_payoffs_are_valid_two_player_games() -> None:
    for factory in PAYOFF_FACTORIES.values():
        payoffs = factory()
        assert payoffs.shape[0] == 2
        assert payoffs.ndim == 3
        assert np.all((0.0 <= payoffs) & (payoffs <= 1.0))


def test_cyclic_dominance_is_balanced() -> None:
    payoffs = create_cyclic_dominance_payoffs(5)
    assert np.all(np.sum(payoffs[0] == 1.0, axis=1) == 2)
    assert np.all(np.sum(payoffs[0] == 0.0, axis=1) == 2)
    assert np.allclose(payoffs[0] + payoffs[1], 1.0)


def test_normalize_payoffs_rejects_constant_values() -> None:
    with pytest.raises(ValueError, match="constant"):
        normalize_payoffs(np.ones((2, 2)))
