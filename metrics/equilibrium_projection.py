from collections.abc import Iterable
from dataclasses import dataclass
from operator import index

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from metrics.equilibrium import optimize_equilibrium


@dataclass(frozen=True)
class LinearProjection2D:
    center: np.ndarray
    components: np.ndarray
    rank: int

    @classmethod
    def fit(cls, trajectories: Iterable[np.ndarray]) -> "LinearProjection2D":
        arrays = [np.asarray(trajectory, dtype=float) for trajectory in trajectories]
        if not arrays or any(array.ndim != 2 or len(array) == 0 for array in arrays):
            raise ValueError("at least one non-empty matrix of distribution vectors is required")
        feature_count = arrays[0].shape[1]
        if feature_count == 0 or any(array.shape[1] != feature_count for array in arrays):
            raise ValueError("all trajectories must have the same positive vector dimension")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("trajectory vectors must contain only finite values")

        samples = np.concatenate(arrays, axis=0)
        center = np.mean(samples, axis=0)
        _, singular_values, right_vectors = np.linalg.svd(samples - center, full_matrices=False)
        threshold = np.finfo(float).eps * max(samples.shape) * (singular_values[0] if singular_values.size else 0.0)
        rank = int(np.count_nonzero(singular_values > threshold))
        components = np.zeros((2, feature_count))
        for component in range(min(2, rank)):
            vector = right_vectors[component].copy()
            pivot = int(np.argmax(np.abs(vector)))
            if vector[pivot] < 0.0:
                vector *= -1.0
            components[component] = vector
        return cls(center, components, rank)

    def transform(self, vectors) -> np.ndarray:
        array = np.asarray(vectors, dtype=float)
        if array.shape[-1:] != self.center.shape:
            raise ValueError(f"vectors must have trailing dimension {self.center.size}")
        if not np.all(np.isfinite(array)):
            raise ValueError("vectors must contain only finite values")
        return (array - self.center) @ self.components.T

    def objective(self, direction, action_shape: tuple[int, ...]) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise ValueError("direction must be a finite two-dimensional vector")
        if int(np.prod(action_shape)) != self.center.size:
            raise ValueError("action_shape does not match the projection dimension")
        return (self.components.T @ direction).reshape(action_shape, order="C")


@dataclass(frozen=True)
class ProjectedEquilibriumRegion:
    support_points: np.ndarray
    support_distributions: np.ndarray
    boundary: np.ndarray
    affine_dimension: int


def _line_boundary_indices(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    coordinate = centered @ right_vectors[0]
    return np.array([int(np.argmin(coordinate)), int(np.argmax(coordinate))])


def _boundary_indices(points: np.ndarray) -> tuple[np.ndarray, int]:
    if len(points) == 1:
        return np.array([0]), 0
    centered = points - np.mean(points, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    threshold = np.finfo(float).eps * max(points.shape) * (singular_values[0] if singular_values.size else 0.0)
    rank = min(2, int(np.count_nonzero(singular_values > threshold)))
    if rank == 0:
        return np.array([0]), 0
    if rank == 1:
        return _line_boundary_indices(points), 1
    try:
        return ConvexHull(points).vertices, 2
    except QhullError:
        return _line_boundary_indices(points), 1


def project_equilibrium_region(payoff_tensor, projection: LinearProjection2D, equilibrium: str = "ce", direction_count: int = 128) -> ProjectedEquilibriumRegion:
    try:
        direction_count = index(direction_count)
    except TypeError as error:
        raise ValueError("direction_count must be an integer") from error
    if direction_count < 3:
        raise ValueError("direction_count must be at least three")

    payoffs = np.asarray(payoff_tensor, dtype=float)
    action_shape = payoffs.shape[1:]
    if projection.rank == 0:
        directions = [np.array([1.0, 0.0])]
    elif projection.rank == 1:
        directions = [np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
    else:
        angles = np.linspace(0.0, 2.0 * np.pi, direction_count, endpoint=False)
        directions = [np.array([np.cos(angle), np.sin(angle)]) for angle in angles]

    support_distributions = np.asarray([
        optimize_equilibrium(payoffs, equilibrium, projection.objective(direction, action_shape))
        for direction in directions
    ])
    support_points = projection.transform(support_distributions.reshape(len(support_distributions), -1))
    boundary_indices, dimension = _boundary_indices(support_points)
    return ProjectedEquilibriumRegion(support_points, support_distributions, support_points[boundary_indices], dimension)
