import numpy as np
import pytest

from algorithms.stationary import _validate_distribution, stationary_distribution


@pytest.mark.parametrize("method", ["solve", "pinv", "iteration"])
def test_stationary_distribution_has_small_residual(method: str) -> None:
    transition_matrix = np.array(
        [
            [0.8, 0.2],
            [0.4, 0.6],
        ]
    )

    distribution = stationary_distribution(transition_matrix, method=method)

    assert np.linalg.norm(distribution @ transition_matrix - distribution, ord=1) < 1e-10


def test_iteration_raises_when_periodic_chain_does_not_converge() -> None:
    transition_matrix = np.array(
        [
            [0.0, 0.25, 0.75],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        stationary_distribution(transition_matrix, method="iteration", max_iterations=100)


def test_stationary_distribution_handles_reducible_chain() -> None:
    transition_matrix = np.eye(3)

    distribution = stationary_distribution(transition_matrix)

    assert np.all(distribution >= 0.0)
    assert np.isclose(np.sum(distribution), 1.0)
    assert np.linalg.norm(distribution @ transition_matrix - distribution, ord=1) < 1e-10


def test_stationary_distribution_rejects_non_finite_matrix() -> None:
    transition_matrix = np.array(
        [
            [np.nan, np.nan],
            [0.5, 0.5],
        ]
    )

    with pytest.raises(ValueError, match="finite"):
        stationary_distribution(transition_matrix)


def test_stationary_distribution_uses_absolute_row_tolerance() -> None:
    transition_matrix = np.array([[0.5, 0.50000001], [0.5, 0.5]])

    with pytest.raises(ValueError, match="sum"):
        stationary_distribution(transition_matrix)


def test_solve_handles_sparse_chain_without_falling_back(monkeypatch) -> None:
    transition_matrix = np.array([[0.0, 1.0], [0.5, 0.5]])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("pseudoinverse should not be used")

    monkeypatch.setattr(np.linalg, "pinv", fail_if_called)

    distribution = stationary_distribution(transition_matrix, method="solve")

    assert np.allclose(distribution, [1.0 / 3.0, 2.0 / 3.0])


def test_solve_falls_back_to_pseudoinverse(monkeypatch) -> None:
    transition_matrix = np.eye(3)
    original_pinv = np.linalg.pinv
    calls = 0

    def fail_solve(*args, **kwargs):
        raise np.linalg.LinAlgError

    def record_pinv(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_pinv(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "solve", fail_solve)
    monkeypatch.setattr(np.linalg, "pinv", record_pinv)

    distribution = stationary_distribution(transition_matrix, method="solve")

    assert calls == 1
    assert np.allclose(distribution, np.full(3, 1.0 / 3.0))


def test_stationary_distribution_cleans_roundoff_sized_probability_errors() -> None:
    transition_matrix = np.eye(2)

    distribution = _validate_distribution(np.array([-5e-13, 1.0 + 5e-13]), transition_matrix)

    assert np.array_equal(distribution, [0.0, 1.0])


def test_stationary_distribution_rejects_material_probability_errors() -> None:
    transition_matrix = np.eye(2)

    with pytest.raises(FloatingPointError, match="probabilities"):
        _validate_distribution(np.array([-1e-6, 1.0 + 1e-6]), transition_matrix)
