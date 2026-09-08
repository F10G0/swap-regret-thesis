import numpy as np
import pytest

from algorithms.stationary import _validate_distribution, stationary_distribution
from config import NUMERICAL_TOLERANCE


@pytest.fixture
def srm_round_531988_matrix() -> np.ndarray:
    """Captured SRM failure: K=9, replicate index 3, both base seeds 42, centered walk."""
    edges = {
        (0, 7): 2.5063221977438067e-6,
        (1, 8): 1.0234148974120463e-6,
        (2, 7): 1.8797416482747128e-7,
        (3, 6): 2.527208216058424e-6,
        (4, 0): 2.0050577581949992e-6,
        (4, 6): 2.1721459047116563e-6,
        (5, 2): 1.691767483477309e-6,
        (6, 2): 1.420249245388115e-6,
        (6, 8): 1.6499954468482765e-6,
        (7, 5): 1.6499954468500135e-6,
        (8, 6): 5.639224944924297e-7,
    }
    matrix = np.zeros((9, 9))
    for (source, target), probability in edges.items():
        matrix[source, target] = probability
    np.fill_diagonal(matrix, 1.0 - matrix.sum(axis=1))
    return matrix


def _assert_stationary_distribution(distribution, matrix) -> None:
    assert np.all(np.isfinite(distribution))
    assert np.all(distribution >= 0.0)
    assert np.all(distribution <= 1.0)
    assert abs(distribution.sum() - 1.0) <= NUMERICAL_TOLERANCE
    assert np.linalg.norm(distribution @ matrix - distribution, ord=1) <= NUMERICAL_TOLERANCE


@pytest.mark.parametrize("method", ["solve", "pinv"])
def test_stationary_solver_handles_captured_srm_failure(srm_round_531988_matrix, method) -> None:
    matrix = srm_round_531988_matrix
    original = matrix.copy()
    distribution = stationary_distribution(matrix, method=method)

    _assert_stationary_distribution(distribution, matrix)
    np.testing.assert_array_equal(matrix, original)
    # The only recurrent class is the cycle 2 -> 7 -> 5 -> 2. Its stationary
    # masses are inversely proportional to its outgoing transition rates.
    expected = np.zeros(9)
    expected[[2, 7, 5]] = 1.0 / matrix[[2, 7, 5], [7, 5, 2]]
    expected /= expected.sum()
    np.testing.assert_allclose(distribution, expected, atol=NUMERICAL_TOLERANCE, rtol=0.0)


def test_srm_failure_matrix_uses_direct_solve_with_scaled_equations(srm_round_531988_matrix, monkeypatch) -> None:
    original_solve = np.linalg.solve
    calls = 0

    def check_solve(A, b):
        nonlocal calls
        calls += 1
        np.testing.assert_allclose(np.max(np.abs(A[:-1]), axis=1), 1.0)
        np.testing.assert_array_equal(A[-1], np.ones(9))
        np.testing.assert_array_equal(b, np.r_[np.zeros(8), 1.0])
        return original_solve(A, b)

    def fail_pinv(*args, **kwargs):
        raise AssertionError("the captured SRM matrix should no longer need a fallback")

    monkeypatch.setattr(np.linalg, "solve", check_solve)
    monkeypatch.setattr(np.linalg, "pinv", fail_pinv)
    distribution = stationary_distribution(srm_round_531988_matrix)
    assert calls == 1
    _assert_stationary_distribution(distribution, srm_round_531988_matrix)


def test_srm_failure_matrix_supports_pseudoinverse_fallback(srm_round_531988_matrix, monkeypatch) -> None:
    def fail_solve(*args, **kwargs):
        raise np.linalg.LinAlgError

    monkeypatch.setattr(np.linalg, "solve", fail_solve)
    distribution = stationary_distribution(srm_round_531988_matrix)
    _assert_stationary_distribution(distribution, srm_round_531988_matrix)


@pytest.mark.parametrize("method", ["solve", "pinv"])
@pytest.mark.parametrize("rate", [1e-6, 1e-12, 1e-18])
def test_stationary_solver_preserves_tiny_off_diagonal_flows(method, rate) -> None:
    matrix = np.array([[1.0 - rate, rate], [3.0 * rate, 1.0 - 3.0 * rate]])
    distribution = stationary_distribution(matrix, method=method)
    _assert_stationary_distribution(distribution, matrix)
    np.testing.assert_allclose(distribution, [0.75, 0.25], atol=NUMERICAL_TOLERANCE, rtol=0.0)


@pytest.mark.parametrize("method", ["solve", "pinv"])
@pytest.mark.parametrize("n_actions", [2, 3, 9])
def test_stationary_solver_matches_previous_well_conditioned_system(method, n_actions) -> None:
    random = np.random.default_rng(42)
    for _ in range(10):
        matrix = random.uniform(0.1, 1.0, size=(n_actions, n_actions))
        matrix /= matrix.sum(axis=1, keepdims=True)
        A = matrix.T - np.eye(n_actions)
        if method == "solve":
            A[-1] = 1.0
            b = np.zeros(n_actions)
            b[-1] = 1.0
            previous = np.linalg.solve(A, b)
        else:
            A = np.vstack([A, np.ones(n_actions)])
            previous = np.linalg.pinv(A)[:, -1]
        distribution = stationary_distribution(matrix, method=method)
        _assert_stationary_distribution(distribution, matrix)
        np.testing.assert_allclose(distribution, previous, atol=NUMERICAL_TOLERANCE, rtol=0.0)


@pytest.mark.parametrize("method", ["solve", "pinv", "iteration"])
@pytest.mark.parametrize("n_actions", [1, 3, 9])
def test_identity_stationary_distribution_remains_uniform(method, n_actions) -> None:
    matrix = np.eye(n_actions)
    with np.errstate(divide="raise", invalid="raise"):
        distribution = stationary_distribution(matrix, method=method)
    _assert_stationary_distribution(distribution, matrix)
    np.testing.assert_allclose(distribution, np.full(n_actions, 1.0 / n_actions), atol=NUMERICAL_TOLERANCE, rtol=0.0)


@pytest.mark.parametrize("method", ["solve", "pinv"])
def test_reducible_stationary_distribution_preserves_minimum_norm_semantics(method) -> None:
    matrix = np.array([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    distribution = stationary_distribution(matrix, method=method)
    _assert_stationary_distribution(distribution, matrix)
    np.testing.assert_allclose(distribution, [0.0, 0.5, 0.5], atol=NUMERICAL_TOLERANCE, rtol=0.0)


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
    roundoff = NUMERICAL_TOLERANCE / 2.0
    distribution = _validate_distribution(np.array([-roundoff, 1.0 + roundoff]), transition_matrix)

    assert np.array_equal(distribution, [0.0, 1.0])


def test_stationary_distribution_rejects_material_probability_errors() -> None:
    transition_matrix = np.eye(2)

    with pytest.raises(FloatingPointError, match="probabilities"):
        _validate_distribution(np.array([-1e-6, 1.0 + 1e-6]), transition_matrix)


def test_solve_falls_back_when_numerical_residual_is_too_large(monkeypatch) -> None:
    transition_matrix = np.array([[0.8, 0.2], [0.4, 0.6]])
    monkeypatch.setattr(np.linalg, "solve", lambda *args: np.array([0.5, 0.5]))
    distribution = stationary_distribution(transition_matrix)
    assert np.linalg.norm(distribution @ transition_matrix - distribution, ord=1) < NUMERICAL_TOLERANCE


@pytest.mark.parametrize("method", ["solve", "pinv", "iteration"])
def test_stationary_solver_does_not_mutate_transition_matrix(method) -> None:
    transition_matrix = np.array([[0.8, 0.2], [0.4, 0.6]])
    original = transition_matrix.copy()
    stationary_distribution(transition_matrix, method=method)
    np.testing.assert_array_equal(transition_matrix, original)
