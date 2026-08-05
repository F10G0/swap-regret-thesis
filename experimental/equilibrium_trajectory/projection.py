"""Projection and support reconstruction for trajectory visualization."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from operator import index
import warnings

import numpy as np
import pulp

from config import EQUILIBRIUM_LP_TOLERANCE
from metrics.equilibrium import create_equilibrium_lp, optimize_equilibrium
from experimental.equilibrium_trajectory.geometry import (
    EquilibriumAffineGeometry,
    EquilibriumProjectionGeometry,
    _stable_sign,
)


DEFAULT_SUPPORT_QUERY_CAP = 512
DEFAULT_RELATIVE_RENDER_TOLERANCE = 1e-3
COMPARISON_PROJECTION_VERSION = 1
UNIFIED_COMPARISON_PROJECTION_VERSION = 1
TRAJECTORY_RENDER_CACHE_VERSION = 9
_POLYGON_TOLERANCE = 10.0 * EQUILIBRIUM_LP_TOLERANCE
_EXPOSED_FACE_WIDTH_TOLERANCE = float(
    np.sqrt(EQUILIBRIUM_LP_TOLERANCE)
)


@dataclass(frozen=True)
class LinearProjection2D:
    center: np.ndarray
    components: np.ndarray
    rank: int
    axis_roles: tuple[str, str]

    def transform(self, vectors) -> np.ndarray:
        array = np.asarray(vectors, dtype=float)
        if array.shape[-1:] != self.center.shape:
            raise ValueError(
                f"vectors must have trailing dimension {self.center.size}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("vectors must contain only finite values")
        return (array - self.center) @ self.components.T

    def objective(
        self,
        direction,
        action_shape: tuple[int, ...],
    ) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise ValueError(
                "direction must be a finite two-dimensional vector"
            )
        if int(np.prod(action_shape)) != self.center.size:
            raise ValueError(
                "action_shape does not match the projection dimension"
            )
        return (self.components.T @ direction).reshape(
            action_shape,
            order="C",
        )


@dataclass(frozen=True)
class ProjectedEquilibriumSet:
    support_points: np.ndarray
    support_distributions: np.ndarray
    boundary: np.ndarray
    affine_dimension: int
    certified: bool
    support_query_count: int
    support_lp_count: int
    certification_mode: str
    render_tolerance: float | None
    support_gap_tolerance: float
    max_observed_support_gap: float
    projected_scale: float


@dataclass(frozen=True)
class _SupportResult:
    value: float
    point: np.ndarray
    payload: object
    raw_value: float | None = None


@dataclass(frozen=True)
class _PolygonVertex:
    identifier: int
    point: np.ndarray
    payload: object


@dataclass(frozen=True)
class _AdaptivePolygonResult:
    vertices: tuple[_PolygonVertex, ...]
    certified: bool
    support_query_count: int
    certification_mode: str
    render_tolerance: float | None
    support_gap_tolerance: float
    max_observed_support_gap: float
    projected_scale: float


def _canonical_span_basis(
    span_basis: np.ndarray,
    count: int,
    existing: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    basis = np.asarray(span_basis, dtype=float)
    if basis.ndim != 2:
        raise ValueError("span_basis must be a matrix")
    selected = [] if existing is None else [
        np.asarray(vector, dtype=float).copy() for vector in existing
    ]
    new_vectors: list[np.ndarray] = []
    tolerance = 1e-10
    for coordinate in range(basis.shape[0]):
        candidate = basis @ basis[coordinate]
        for vector in selected:
            candidate -= np.dot(candidate, vector) * vector
        norm = float(np.linalg.norm(candidate))
        if norm <= tolerance:
            continue
        candidate = _stable_sign(candidate / norm)
        selected.append(candidate)
        new_vectors.append(candidate)
        if len(new_vectors) == count:
            break
    return new_vectors


def _principal_directions(
    samples: np.ndarray,
    allowed_basis: np.ndarray,
    count: int,
) -> list[np.ndarray]:
    basis = np.asarray(allowed_basis, dtype=float)
    if count == 0:
        return []
    if basis.ndim != 2 or basis.shape[0] != samples.shape[1]:
        raise ValueError("allowed subspace does not match trajectory vectors")
    if basis.shape[1] < count:
        raise ValueError("admissible subspace has too few dimensions")

    centered = samples - np.mean(samples, axis=0)
    allowed_samples = centered @ basis
    _, singular_values, right_vectors = np.linalg.svd(
        allowed_samples,
        full_matrices=False,
    )
    scale = singular_values[0] if singular_values.size else 0.0
    threshold = (
        np.finfo(float).eps
        * max(allowed_samples.shape)
        * scale
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    directions: list[np.ndarray] = []
    position = 0
    while position < min(rank, count):
        end = position + 1
        tie_tolerance = max(
            threshold,
            1e-12 * max(1.0, singular_values[position]),
        )
        while (
            end < rank
            and abs(singular_values[end] - singular_values[position])
            <= tie_tolerance
        ):
            end += 1
        span = basis @ right_vectors[position:end].T
        needed = min(end - position, count - len(directions))
        directions.extend(_canonical_span_basis(span, needed, directions))
        position = end
    if len(directions) < count:
        directions.extend(
            _canonical_span_basis(
                basis,
                count - len(directions),
                directions,
            )
        )
    if len(directions) != count:
        raise RuntimeError(
            "could not construct the requested constrained PCA axes"
        )
    return directions


def _projection_from_samples(
    geometry: EquilibriumProjectionGeometry,
    samples: np.ndarray,
) -> LinearProjection2D:
    feature_count = int(np.prod(geometry.action_shape))
    components = np.zeros((2, feature_count))
    if geometry.simplex_dimension == 0:
        rank = 0
    elif geometry.simplex_dimension == 1:
        components[0] = _principal_directions(
            samples,
            geometry.x_subspace_basis,
            1,
        )[0]
        rank = 1
    elif geometry.shared_axis_subspace:
        axes = _principal_directions(
            samples,
            geometry.x_subspace_basis,
            2,
        )
        components[:] = axes
        rank = 2
    else:
        components[0] = _principal_directions(
            samples,
            geometry.x_subspace_basis,
            1,
        )[0]
        components[1] = _principal_directions(
            samples,
            geometry.y_subspace_basis,
            1,
        )[0]
        rank = 2
    return LinearProjection2D(
        geometry.ce.reference.copy(),
        components,
        rank,
        geometry.axis_roles,
    )


def _validated_projection_trajectories(
    geometry: EquilibriumProjectionGeometry,
    trajectories: Iterable[np.ndarray],
    require_matching_checkpoints: bool,
) -> list[np.ndarray]:
    arrays = [
        np.asarray(trajectory, dtype=float)
        for trajectory in trajectories
    ]
    if not arrays or any(
        array.ndim != 2 or len(array) == 0
        for array in arrays
    ):
        raise ValueError(
            "at least one non-empty matrix of distribution vectors is required"
        )
    feature_count = int(np.prod(geometry.action_shape))
    if any(array.shape[1] != feature_count for array in arrays):
        raise ValueError(
            "all trajectories must match the equilibrium geometry"
        )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(
            "trajectory vectors must contain only finite values"
        )
    if require_matching_checkpoints and len({len(array) for array in arrays}) != 1:
        raise ValueError(
            "comparison trajectories must have matching checkpoints"
        )
    return arrays


def fit_equilibrium_projection(
    geometry: EquilibriumProjectionGeometry,
    trajectories: Iterable[np.ndarray],
) -> LinearProjection2D:
    arrays = _validated_projection_trajectories(
        geometry,
        trajectories,
        require_matching_checkpoints=False,
    )
    return _projection_from_samples(
        geometry,
        np.concatenate(arrays, axis=0),
    )


def fit_equilibrium_comparison_projection(
    geometry: EquilibriumProjectionGeometry,
    trajectories: Iterable[np.ndarray],
) -> LinearProjection2D:
    """Fit shared axes balancing member separation and common path motion."""
    arrays = _validated_projection_trajectories(
        geometry,
        trajectories,
        require_matching_checkpoints=True,
    )
    member_trajectories = np.asarray(arrays)
    common_path = np.mean(member_trajectories, axis=0)
    separation = member_trajectories - common_path[None, :, :]
    path_motion = common_path - np.mean(common_path, axis=0)

    normalized_blocks = []
    for block in (
        separation.reshape((-1, separation.shape[-1])),
        path_motion,
    ):
        total_variation = float(np.linalg.norm(block))
        if total_variation > 1e-12:
            normalized_blocks.append(block / total_variation)
    if normalized_blocks:
        comparison_samples = np.vstack(normalized_blocks)
    else:
        comparison_samples = np.zeros((1, member_trajectories.shape[-1]))
    return _projection_from_samples(geometry, comparison_samples)


def fit_unified_equilibrium_comparison_direction(
    trajectories: Iterable[np.ndarray],
) -> np.ndarray:
    """Fit the balanced comparison direction in the simplex tangent space."""
    arrays = [
        np.asarray(trajectory, dtype=float)
        for trajectory in trajectories
    ]
    if not arrays or any(
        array.ndim != 2 or len(array) == 0
        for array in arrays
    ):
        raise ValueError(
            "at least one non-empty matrix of distribution vectors is required"
        )
    if len({array.shape for array in arrays}) != 1:
        raise ValueError(
            "comparison trajectories must have matching checkpoints and features"
        )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(
            "trajectory vectors must contain only finite values"
        )

    member_trajectories = np.asarray(arrays)
    common_path = np.mean(member_trajectories, axis=0)
    separation = member_trajectories - common_path[None, :, :]
    path_motion = common_path - np.mean(common_path, axis=0)
    normalized_blocks = []
    for block in (
        separation.reshape((-1, separation.shape[-1])),
        path_motion,
    ):
        total_variation = float(np.linalg.norm(block))
        if total_variation > 1e-12:
            normalized_blocks.append(block / total_variation)
    feature_count = member_trajectories.shape[-1]
    if feature_count == 1:
        return np.zeros(1)
    if normalized_blocks:
        samples = np.vstack(normalized_blocks)
        samples -= np.mean(samples, axis=1, keepdims=True)
        _, singular_values, right_vectors = np.linalg.svd(
            samples,
            full_matrices=False,
        )
        scale = singular_values[0] if singular_values.size else 0.0
        threshold = (
            np.finfo(float).eps
            * max(samples.shape)
            * scale
        )
        rank = int(np.count_nonzero(singular_values > threshold))
        if rank:
            end = 1
            tie_tolerance = max(
                threshold,
                1e-12 * max(1.0, singular_values[0]),
            )
            while (
                end < rank
                and abs(singular_values[end] - singular_values[0])
                <= tie_tolerance
            ):
                end += 1
            direction = _canonical_span_basis(
                right_vectors[:end].T,
                1,
            )[0]
            direction -= np.mean(direction)
            norm = float(np.linalg.norm(direction))
            if norm > 1e-12:
                return _stable_sign(direction / norm)

    direction = np.zeros(feature_count)
    direction[0] = 1.0 / np.sqrt(2.0)
    direction[1] = -1.0 / np.sqrt(2.0)
    return _stable_sign(direction)


def _line_direction(
    projection: LinearProjection2D,
    affine: EquilibriumAffineGeometry,
) -> np.ndarray:
    image = projection.components @ affine.direction_basis
    left_vectors, singular_values, _ = np.linalg.svd(
        image,
        full_matrices=False,
    )
    if not singular_values.size or singular_values[0] <= 1e-10:
        raise RuntimeError(
            "one-dimensional equilibrium image has no visible direction"
        )
    return _stable_sign(left_vectors[:, 0])


def _normalized_solver_distribution(distribution: np.ndarray) -> np.ndarray:
    if distribution is None:
        raise RuntimeError(
            "equilibrium optimizer did not return a distribution"
        )
    result = np.asarray(distribution, dtype=float).copy()
    result[np.abs(result) < 1e-12] = 0.0
    total = float(np.sum(result))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError(
            "equilibrium optimizer returned an invalid distribution"
        )
    return result / total


def _point_tolerance(
    first: np.ndarray,
    second: np.ndarray,
    tolerance: float,
) -> float:
    return tolerance * max(
        1.0,
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
    )


def _cross(
    origin: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_offset = first - origin
    second_offset = second - origin
    return float(
        first_offset[0] * second_offset[1]
        - first_offset[1] * second_offset[0]
    )


def _ordered_convex_hull(
    vertices: list[_PolygonVertex],
) -> tuple[_PolygonVertex, ...]:
    ordered = sorted(
        vertices,
        key=lambda vertex: (
            float(vertex.point[0]),
            float(vertex.point[1]),
            vertex.identifier,
        ),
    )
    if len(ordered) <= 2:
        return tuple(ordered)

    def half_hull(
        candidates: Iterable[_PolygonVertex],
    ) -> list[_PolygonVertex]:
        result: list[_PolygonVertex] = []
        for candidate in candidates:
            while len(result) >= 2:
                turn = _cross(
                    result[-2].point,
                    result[-1].point,
                    candidate.point,
                )
                if turn > 0.0:
                    break
                result.pop()
            result.append(candidate)
        return result

    lower = half_hull(ordered)
    upper = half_hull(reversed(ordered))
    return tuple(lower[:-1] + upper[:-1])


def _add_polygon_vertex(
    vertices: list[_PolygonVertex],
    support: _SupportResult,
    tolerance: float,
) -> bool:
    point = np.asarray(support.point, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise RuntimeError(
            "support oracle returned an invalid projected point"
        )
    for vertex in vertices:
        if (
            np.linalg.norm(point - vertex.point)
            <= _point_tolerance(point, vertex.point, tolerance)
        ):
            return False
    vertices.append(
        _PolygonVertex(
            len(vertices),
            point.copy(),
            support.payload,
        )
    )
    return True


def _edge_key(
    first: _PolygonVertex,
    second: _PolygonVertex,
) -> tuple[int, int]:
    return first.identifier, second.identifier


def _hull_edges(
    hull: tuple[_PolygonVertex, ...],
) -> dict[tuple[int, int], tuple[_PolygonVertex, _PolygonVertex]]:
    return {
        _edge_key(hull[position], hull[(position + 1) % len(hull)]): (
            hull[position],
            hull[(position + 1) % len(hull)],
        )
        for position in range(len(hull))
    }


def _edge_outward_normal(
    first: _PolygonVertex,
    second: _PolygonVertex,
) -> np.ndarray:
    edge = second.point - first.point
    norm = float(np.linalg.norm(edge))
    if norm == 0.0:
        raise RuntimeError("adaptive polygon contains a zero-length edge")
    return np.array([edge[1], -edge[0]]) / norm


def _uncertified_polygon_warning(
    label: str,
    reason: str,
) -> None:
    warnings.warn(
        f"{label} projected region reconstruction is uncertified: {reason}",
        RuntimeWarning,
        stacklevel=3,
    )


def _reconstruct_projected_polygon(
    support_oracle,
    support_extreme_oracle,
    support_query_cap: int,
    tolerance: float,
    label: str,
    relative_render_tolerance: float | None = None,
) -> _AdaptivePolygonResult:
    vertices: list[_PolygonVertex] = []
    initial_results: list[tuple[np.ndarray, _SupportResult]] = []
    support_certificates: list[
        tuple[np.ndarray, _SupportResult]
    ] = []
    support_query_count = 0

    def query(direction: np.ndarray) -> _SupportResult | None:
        nonlocal support_query_count
        if support_query_count >= support_query_cap:
            return None
        support_query_count += 1
        result = support_oracle(direction)
        support_certificates.append((direction.copy(), result))
        return result

    for direction in (
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
    ):
        result = query(direction)
        if result is None:
            break
        initial_results.append((direction, result))
        _add_polygon_vertex(vertices, result, tolerance)

    horizontal_span = max(
        0.0,
        initial_results[0][1].value + initial_results[1][1].value,
    )
    vertical_span = max(
        0.0,
        initial_results[2][1].value + initial_results[3][1].value,
    )
    projected_scale = max(horizontal_span, vertical_span, tolerance)
    render_component = (
        0.0
        if relative_render_tolerance is None
        else relative_render_tolerance * projected_scale
    )
    support_gap_tolerance = tolerance + render_component

    def incomplete_result(
        current_hull: tuple[_PolygonVertex, ...],
        mode: str,
    ) -> _AdaptivePolygonResult:
        return _AdaptivePolygonResult(
            current_hull,
            False,
            support_query_count,
            mode,
            relative_render_tolerance,
            support_gap_tolerance,
            float("nan"),
            projected_scale,
        )

    hull = _ordered_convex_hull(vertices)
    if len(hull) < 3:
        for direction, result in initial_results:
            endpoint = support_extreme_oracle(direction, result, 1.0)
            _add_polygon_vertex(vertices, endpoint, tolerance)
            hull = _ordered_convex_hull(vertices)
            if len(hull) >= 3:
                break
    if len(hull) < 3:
        for direction, result in initial_results:
            endpoint = support_extreme_oracle(direction, result, -1.0)
            _add_polygon_vertex(vertices, endpoint, tolerance)
            hull = _ordered_convex_hull(vertices)
            if len(hull) >= 3:
                break
    if len(hull) < 3:
        _uncertified_polygon_warning(
            label,
            "the initial support points were numerically degenerate",
        )
        return incomplete_result(hull, "numerical_failure")

    certified_edges: set[tuple[int, int]] = set()
    certified_edge_gaps: dict[tuple[int, int], float] = {}
    queued_edges: set[tuple[int, int]] = set()
    work_queue: deque[tuple[int, int]] = deque()
    current_edges: dict[
        tuple[int, int],
        tuple[_PolygonVertex, _PolygonVertex],
    ] = {}

    def update_edge_state() -> None:
        nonlocal current_edges
        current_edges = _hull_edges(hull)
        surviving_keys = set(current_edges)
        certified_edges.intersection_update(surviving_keys)
        for stale_key in set(certified_edge_gaps) - surviving_keys:
            certified_edge_gaps.pop(stale_key)
        queued_edges.intersection_update(surviving_keys)
        for key, edge in current_edges.items():
            if key in certified_edges:
                continue
            first, second = edge
            outward_normal = _edge_outward_normal(first, second)
            for direction, support in support_certificates:
                edge_support = max(
                    float(np.dot(direction, first.point)),
                    float(np.dot(direction, second.point)),
                )
                support_gap = max(0.0, support.value - edge_support)
                if (
                    float(np.dot(outward_normal, direction))
                    >= 1.0 - tolerance
                    and support_gap <= support_gap_tolerance
                ):
                    certified_edges.add(key)
                    certified_edge_gaps[key] = support_gap
                    break
        for key in current_edges:
            if key not in certified_edges and key not in queued_edges:
                work_queue.append(key)
                queued_edges.add(key)

    update_edge_state()
    while work_queue:
        key = work_queue.popleft()
        queued_edges.discard(key)
        edge = current_edges.get(key)
        if edge is None or key in certified_edges:
            continue
        first, second = edge
        direction = _edge_outward_normal(first, second)
        support = query(direction)
        if support is None:
            _uncertified_polygon_warning(
                label,
                f"the support-query cap of {support_query_cap} was reached",
            )
            return incomplete_result(hull, "safety_cap")

        edge_support = max(
            float(np.dot(direction, first.point)),
            float(np.dot(direction, second.point)),
        )
        support_gap = max(0.0, support.value - edge_support)
        if support_gap <= support_gap_tolerance:
            certified_edges.add(key)
            certified_edge_gaps[key] = support_gap
            continue

        endpoint = support_extreme_oracle(direction, support, 1.0)
        added = _add_polygon_vertex(vertices, endpoint, tolerance)
        if not added:
            opposite_endpoint = support_extreme_oracle(
                direction,
                support,
                -1.0,
            )
            added = _add_polygon_vertex(
                vertices,
                opposite_endpoint,
                tolerance,
            )
        if not added:
            added = _add_polygon_vertex(vertices, support, tolerance)
        if not added:
            _uncertified_polygon_warning(
                label,
                "a support gap did not produce a distinct hull vertex",
            )
            return incomplete_result(hull, "numerical_failure")

        hull = _ordered_convex_hull(vertices)
        if len(hull) < 3:
            _uncertified_polygon_warning(
                label,
                "the updated support points were numerically degenerate",
            )
            return incomplete_result(hull, "numerical_failure")
        update_edge_state()

    max_support_gap = max(certified_edge_gaps.values(), default=0.0)
    return _AdaptivePolygonResult(
        hull,
        True,
        support_query_count,
        (
            "exact"
            if relative_render_tolerance is None
            else "render"
        ),
        relative_render_tolerance,
        support_gap_tolerance,
        max_support_gap,
        projected_scale,
    )


class _EquilibriumSupportOracle2D:
    def __init__(
        self,
        payoff_tensor: np.ndarray,
        equilibrium: str,
        projection: LinearProjection2D,
        tolerance: float,
    ):
        self.payoffs = np.asarray(payoff_tensor, dtype=float)
        self.equilibrium = equilibrium
        self.projection = projection
        self.action_shape = self.payoffs.shape[1:]
        self.tolerance = tolerance
        self.lp_solve_count = 0
        self._primary_cache: dict[
            tuple[float, float],
            _SupportResult,
        ] = {}
        self._extreme_cache: dict[
            tuple[tuple[float, float], int],
            _SupportResult,
        ] = {}

    @staticmethod
    def _normalized_direction(direction: np.ndarray) -> np.ndarray:
        result = np.asarray(direction, dtype=float)
        if result.shape != (2,) or not np.all(np.isfinite(result)):
            raise ValueError(
                "support direction must be a finite two-dimensional vector"
            )
        norm = float(np.linalg.norm(result))
        if norm == 0.0:
            raise ValueError("support direction must be nonzero")
        return result / norm

    @staticmethod
    def _direction_key(direction: np.ndarray) -> tuple[float, float]:
        return tuple(np.round(direction, decimals=13))

    def support_value(self, direction: np.ndarray) -> _SupportResult:
        direction = self._normalized_direction(direction)
        key = self._direction_key(direction)
        cached = self._primary_cache.get(key)
        if cached is not None:
            return cached
        objective = self.projection.objective(
            direction,
            self.action_shape,
        )
        self.lp_solve_count += 1
        raw_distribution = optimize_equilibrium(
            self.payoffs,
            self.equilibrium,
            objective,
        )
        distribution = _normalized_solver_distribution(
            raw_distribution
        )
        point = self.projection.transform(
            distribution.reshape(1, -1)
        )[0]
        result = _SupportResult(
            float(np.dot(direction, point)),
            point,
            distribution,
            float(np.sum(objective * raw_distribution)),
        )
        self._primary_cache[key] = result
        return result

    def support_extreme_point(
        self,
        direction: np.ndarray,
        primary: _SupportResult,
        orientation: float,
    ) -> _SupportResult:
        direction = self._normalized_direction(direction)
        sign = 1 if orientation >= 0.0 else -1
        key = (self._direction_key(direction), sign)
        cached = self._extreme_cache.get(key)
        if cached is not None:
            return cached

        primary_objective = self.projection.objective(
            direction,
            self.action_shape,
        )
        primary_distribution = np.asarray(primary.payload, dtype=float)
        raw_support = (
            primary.raw_value
            if primary.raw_value is not None
            else float(np.sum(primary_objective * primary_distribution))
        )
        perpendicular = np.array([-direction[1], direction[0]])
        secondary_direction = sign * perpendicular
        secondary_objective = self.projection.objective(
            secondary_direction,
            self.action_shape,
        )
        variables, problem = create_equilibrium_lp(
            self.payoffs,
            self.equilibrium,
            secondary_objective,
        )
        profiles = tuple(variables)
        problem += (
            pulp.lpSum(
                float(primary_objective[profile])
                * variables[profile]
                for profile in profiles
            )
            >= raw_support - self.tolerance
        )
        self.lp_solve_count += 1
        status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
        if status != pulp.LpStatusOptimal:
            raise RuntimeError(
                f"{self.equilibrium.upper()} exposed-face "
                "optimization failed with solver status "
                f"{pulp.LpStatus[problem.status]}"
            )
        distribution = _normalized_solver_distribution(
            np.array(
                [
                    variables[profile].varValue
                    for profile in profiles
                ],
                dtype=float,
            ).reshape(self.action_shape)
        )
        point = self.projection.transform(
            distribution.reshape(1, -1)
        )[0]
        face_scale = max(
            1.0,
            float(np.linalg.norm(point)),
            float(np.linalg.norm(primary.point)),
        )
        result = (
            primary
            if np.linalg.norm(point - primary.point)
            <= _EXPOSED_FACE_WIDTH_TOLERANCE * face_scale
            else _SupportResult(
                float(np.dot(direction, point)),
                point,
                distribution,
            )
        )
        self._extreme_cache[key] = result
        return result


def project_equilibrium_set(
    payoff_tensor,
    projection: LinearProjection2D,
    affine: EquilibriumAffineGeometry,
    projected_dimension: int,
    equilibrium: str = "ce",
    support_query_cap: int = DEFAULT_SUPPORT_QUERY_CAP,
    relative_render_tolerance: float | None = (
        DEFAULT_RELATIVE_RENDER_TOLERANCE
    ),
) -> ProjectedEquilibriumSet:
    try:
        support_query_cap = index(support_query_cap)
    except TypeError as error:
        raise ValueError("support_query_cap must be an integer") from error
    if support_query_cap < 4:
        raise ValueError("support_query_cap must be at least four")
    if relative_render_tolerance is not None:
        try:
            relative_render_tolerance = float(
                relative_render_tolerance
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "relative_render_tolerance must be a non-negative number "
                "or None"
            ) from error
        if (
            not np.isfinite(relative_render_tolerance)
            or relative_render_tolerance < 0.0
        ):
            raise ValueError(
                "relative_render_tolerance must be a non-negative number "
                "or None"
            )
    if projected_dimension not in {0, 1, 2}:
        raise ValueError("projected_dimension must be zero, one, or two")

    payoffs = np.asarray(payoff_tensor, dtype=float)
    action_shape = payoffs.shape[1:]
    if projected_dimension == 0:
        support_distributions = affine.reference.reshape(
            (1, *action_shape),
            order="C",
        )
        support_points = projection.transform(
            affine.reference.reshape(1, -1)
        )
        return ProjectedEquilibriumSet(
            support_points,
            support_distributions,
            support_points.copy(),
            0,
            True,
            0,
            0,
            "exact",
            None,
            0.0,
            0.0,
            0.0,
        )

    if projected_dimension == 1:
        line_direction = _line_direction(projection, affine)
        directions = [line_direction, -line_direction]
        support_distributions = np.asarray([
            _normalized_solver_distribution(
                optimize_equilibrium(
                    payoffs,
                    equilibrium,
                    projection.objective(direction, action_shape),
                )
            )
            for direction in directions
        ])
        support_points = projection.transform(
            support_distributions.reshape(
                len(support_distributions),
                -1,
            )
        )
        coordinate = support_points @ line_direction
        boundary = support_points[
            [int(np.argmin(coordinate)), int(np.argmax(coordinate))]
        ]
        return ProjectedEquilibriumSet(
            support_points,
            support_distributions,
            boundary,
            1,
            True,
            2,
            2,
            "exact",
            None,
            0.0,
            0.0,
            0.0,
        )

    oracle = _EquilibriumSupportOracle2D(
        payoffs,
        equilibrium,
        projection,
        EQUILIBRIUM_LP_TOLERANCE,
    )
    polygon = _reconstruct_projected_polygon(
        oracle.support_value,
        oracle.support_extreme_point,
        support_query_cap,
        _POLYGON_TOLERANCE,
        equilibrium.upper(),
        relative_render_tolerance,
    )
    support_points = np.asarray([
        vertex.point for vertex in polygon.vertices
    ])
    support_distributions = np.asarray([
        vertex.payload for vertex in polygon.vertices
    ])
    return ProjectedEquilibriumSet(
        support_points,
        support_distributions,
        support_points.copy(),
        2,
        polygon.certified,
        polygon.support_query_count,
        oracle.lp_solve_count,
        polygon.certification_mode,
        polygon.render_tolerance,
        polygon.support_gap_tolerance,
        polygon.max_observed_support_gap,
        polygon.projected_scale,
    )
