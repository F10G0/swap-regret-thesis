import numpy as np

from config import NUMERICAL_TOLERANCE, STATIONARITY_TOLERANCE, STATIONARY_METHOD


def stationary_distribution(transition_matrix: np.ndarray, method: str = STATIONARY_METHOD, max_iterations: int = 1000) -> np.ndarray:
    if transition_matrix.ndim != 2:
        raise ValueError("transition_matrix must be a 2D array")
    dim, columns = transition_matrix.shape
    if dim != columns:
        raise ValueError("transition_matrix must be square")
    if not np.all(np.isfinite(transition_matrix)):
        raise ValueError("transition_matrix must contain only finite entries")
    if np.any(transition_matrix < 0.0):
        raise ValueError("transition_matrix must not contain negative entries")
    if not np.allclose(np.sum(transition_matrix, axis=1), 1.0, atol=NUMERICAL_TOLERANCE, rtol=0.0):
        raise ValueError("transition_matrix rows must sum to 1")

    if method == "solve":
        return _stationary_distribution_solve(transition_matrix, dim)
    if method == "pinv":
        return _stationary_distribution_pinv(transition_matrix, dim)
    if method == "iteration":
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        return _stationary_distribution_iteration(transition_matrix, dim, max_iterations)
    raise ValueError(f"unknown stationary distribution method: {method}")


def _stationary_distribution_solve(transition_matrix: np.ndarray, dim: int) -> np.ndarray:
    # Solve pQ = p, equivalently (Q^T - I) p^T = 0.
    A = transition_matrix.T - np.eye(dim)
    b = np.zeros(dim)

    # Replace one redundant equation by sum(p) = 1.
    A[-1] = np.ones(dim)
    b[-1] = 1.0

    try:
        distribution = np.linalg.solve(A, b)
        return _validate_distribution(distribution, transition_matrix)
    except (np.linalg.LinAlgError, FloatingPointError):
        return _stationary_distribution_pinv(transition_matrix, dim)


def _stationary_distribution_pinv(transition_matrix: np.ndarray, dim: int) -> np.ndarray:
    A = transition_matrix.T - np.eye(dim)
    A = np.vstack([A, np.ones(dim)])
    b = np.zeros(dim + 1)
    b[-1] = 1.0

    distribution = np.linalg.pinv(A) @ b
    return _validate_distribution(distribution, transition_matrix)


def _stationary_distribution_iteration(transition_matrix: np.ndarray, dim: int, max_iterations: int) -> np.ndarray:
    distribution = np.full(dim, 1.0 / dim)
    for _ in range(max_iterations):
        next_distribution = distribution @ transition_matrix
        if np.linalg.norm(next_distribution - distribution, ord=1) < NUMERICAL_TOLERANCE:
            return _validate_distribution(next_distribution, transition_matrix)
        distribution = next_distribution

    raise RuntimeError(f"stationary-distribution iteration did not converge after {max_iterations} iterations")


def _validate_distribution(distribution: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    if distribution.shape != (transition_matrix.shape[0],):
        raise FloatingPointError("stationary distribution has the wrong shape")
    if not np.all(np.isfinite(distribution)):
        raise FloatingPointError("stationary distribution is not finite")
    if np.any((distribution < -NUMERICAL_TOLERANCE) | (distribution > 1.0 + NUMERICAL_TOLERANCE)):
        raise FloatingPointError("stationary probabilities must be in [0, 1]")
    if not np.isclose(np.sum(distribution), 1.0, atol=NUMERICAL_TOLERANCE, rtol=0.0):
        raise FloatingPointError("stationary distribution is not normalized")

    distribution = np.clip(distribution, 0.0, 1.0)
    distribution /= np.sum(distribution)
    residual = np.linalg.norm(distribution @ transition_matrix - distribution, ord=1)
    if not np.isfinite(residual) or residual > STATIONARITY_TOLERANCE:
        raise FloatingPointError(f"stationary-distribution residual {residual} exceeds {STATIONARITY_TOLERANCE}")
    return distribution
