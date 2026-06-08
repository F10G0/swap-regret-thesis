import numpy as np

from config import NUMERICAL_TOLERANCE


def stationary_distribution(transition_matrix: np.ndarray, method: str = "solve", max_iterations: int = 1000) -> np.ndarray:
    transition_matrix = np.asarray(transition_matrix, dtype=float)

    if transition_matrix.ndim != 2:
        raise ValueError("transition_matrix must be a 2D array")
    dim, columns = transition_matrix.shape
    if dim != columns:
        raise ValueError("transition_matrix must be square")
    if np.any(transition_matrix < 0.0):
        raise ValueError("transition_matrix must not contain negative entries")
    if not np.allclose(np.sum(transition_matrix, axis=1), 1.0, atol=NUMERICAL_TOLERANCE):
        raise ValueError("transition_matrix rows must sum to 1")

    if method == "solve":
        return _stationary_distribution_solve(transition_matrix, dim=dim)
    elif method == "pinv":
        return _stationary_distribution_pinv(transition_matrix, dim=dim)
    elif method == "iteration":
        return _stationary_distribution_iteration(transition_matrix, dim=dim, max_iterations=max_iterations)
    else:
        raise ValueError(f"unknown stationary distribution method: {method}")


def _stationary_distribution_solve(transition_matrix: np.ndarray, dim: int) -> np.ndarray:
    if np.any(transition_matrix < NUMERICAL_TOLERANCE):
        return _stationary_distribution_pinv(transition_matrix, dim=dim)

    # Solve pQ = p, equivalently (Q^T - I) p^T = 0.
    A = transition_matrix.T - np.eye(dim)
    b = np.zeros(dim)

    # Since Q is strictly positive, rank(A) = dim - 1.
    # Replace one redundant equation by sum(p) = 1.
    A[-1] = np.ones(dim)
    b[-1] = 1.0

    distribution = np.linalg.solve(A, b)
    return _validate_distribution(distribution)


def _stationary_distribution_pinv(transition_matrix: np.ndarray, dim: int) -> np.ndarray:
    # Keep all equations from pQ = p and add sum(p) = 1.
    A = transition_matrix.T - np.eye(dim)
    A = np.vstack([A, np.ones(dim)])

    b = np.zeros(dim + 1)
    b[-1] = 1.0

    distribution = np.linalg.pinv(A) @ b
    return _validate_distribution(distribution)


def _stationary_distribution_iteration(transition_matrix: np.ndarray, dim: int, max_iterations: int = 1000) -> np.ndarray:
    distribution = np.full(dim, 1.0 / dim)
    for _ in range(max_iterations):
        next_distribution = distribution @ transition_matrix

        if np.linalg.norm(next_distribution - distribution, ord=1) < NUMERICAL_TOLERANCE:
            return _validate_distribution(next_distribution)
        distribution = next_distribution

    return _validate_distribution(distribution)


def _validate_distribution(distribution: np.ndarray) -> np.ndarray:
    if np.any(distribution < -NUMERICAL_TOLERANCE):
        raise FloatingPointError("negative stationary probability")
    if not np.isclose(np.sum(distribution), 1.0, atol=NUMERICAL_TOLERANCE):
        raise FloatingPointError("stationary distribution is not normalized")
    
    if np.any((distribution < 0.0) | (distribution > 1.0)):
        distribution = np.clip(distribution, 0.0, 1.0)
        distribution /= np.sum(distribution)
    return distribution
