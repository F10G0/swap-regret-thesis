"""Affine equilibrium geometry for experimental trajectory rendering."""

from concurrent.futures import Future
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from threading import Lock

import numpy as np
import pulp

from config import EQUILIBRIUM_LP_TOLERANCE
from metrics.equilibrium import create_equilibrium_lp, optimize_equilibrium


GEOMETRY_CACHE_VERSION = 1
_RANK_TOLERANCE = 1e-9
_SLACK_TOLERANCE = 10.0 * EQUILIBRIUM_LP_TOLERANCE


@dataclass(frozen=True)
class EquilibriumAffineGeometry:
    reference: np.ndarray
    direction_basis: np.ndarray
    normal_basis: np.ndarray

    @property
    def dimension(self) -> int:
        return self.direction_basis.shape[1]


@dataclass(frozen=True)
class EquilibriumProjectionGeometry:
    action_shape: tuple[int, ...]
    simplex_dimension: int
    ce: EquilibriumAffineGeometry
    cce: EquilibriumAffineGeometry
    projection_case: str
    ce_projected_dimension: int
    cce_projected_dimension: int
    x_subspace_basis: np.ndarray
    y_subspace_basis: np.ndarray
    shared_axis_subspace: bool
    axis_roles: tuple[str, str]


def _stable_sign(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=float).copy()
    if result.size:
        pivot = int(np.argmax(np.abs(result)))
        if result[pivot] < 0.0:
            result *= -1.0
    return result


def _stable_basis(basis: np.ndarray) -> np.ndarray:
    result = np.asarray(basis, dtype=float).copy()
    for column in range(result.shape[1]):
        result[:, column] = _stable_sign(result[:, column])
    return result


def _normalized_distribution(distribution: np.ndarray) -> np.ndarray:
    result = np.asarray(distribution, dtype=float).copy()
    result[np.abs(result) < 1e-12] = 0.0
    total = float(np.sum(result))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError(
            "equilibrium optimizer returned an invalid distribution"
        )
    return result / total


def _null_space(rows: np.ndarray, feature_count: int) -> np.ndarray:
    matrix = np.asarray(rows, dtype=float)
    if matrix.size == 0:
        return np.eye(feature_count)
    _, singular_values, right_vectors = np.linalg.svd(
        matrix.reshape((-1, feature_count)),
        full_matrices=True,
    )
    scale = singular_values[0] if singular_values.size else 0.0
    threshold = max(
        _RANK_TOLERANCE,
        np.finfo(float).eps * max(matrix.shape) * scale,
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    return _stable_basis(right_vectors[rank:].T)


def _orthonormal_span(vectors: np.ndarray, expected_dimension: int | None = None) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("subspace vectors must be a matrix")
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0))
    left_vectors, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    scale = singular_values[0] if singular_values.size else 0.0
    threshold = max(
        _RANK_TOLERANCE,
        np.finfo(float).eps * max(matrix.shape) * scale,
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    if expected_dimension is not None and rank != expected_dimension:
        raise RuntimeError(
            f"equilibrium subspace has numerical dimension {rank}, expected {expected_dimension}"
        )
    return _stable_basis(left_vectors[:, :rank])


def _simplex_tangent_basis(feature_count: int) -> np.ndarray:
    if feature_count <= 1:
        return np.zeros((feature_count, 0))
    ones = np.ones((1, feature_count))
    return _null_space(ones, feature_count)


def _incentive_slack_rows(payoff_tensor: np.ndarray, equilibrium: str) -> np.ndarray:
    payoffs = np.asarray(payoff_tensor, dtype=float)
    action_shape = payoffs.shape[1:]
    feature_count = int(np.prod(action_shape))
    rows: list[np.ndarray] = []
    for player, action_count in enumerate(action_shape):
        actual = payoffs[player]
        if equilibrium == "cce":
            for deviation_action in range(action_count):
                deviation = np.take(
                    actual,
                    deviation_action,
                    axis=player,
                )
                deviation = np.expand_dims(deviation, axis=player)
                deviation = np.broadcast_to(deviation, action_shape)
                row = (actual - deviation).reshape(feature_count, order="C")
                if np.linalg.norm(row) > _RANK_TOLERANCE:
                    rows.append(row)
            continue
        if equilibrium != "ce":
            raise ValueError(f"unknown equilibrium concept: {equilibrium}")
        for recommended_action in range(action_count):
            recommendation_mask = np.zeros(action_shape)
            recommendation_slice = [slice(None)] * len(action_shape)
            recommendation_slice[player] = recommended_action
            recommendation_mask[tuple(recommendation_slice)] = 1.0
            for deviation_action in range(action_count):
                if deviation_action == recommended_action:
                    continue
                deviation = np.take(
                    actual,
                    deviation_action,
                    axis=player,
                )
                deviation = np.expand_dims(deviation, axis=player)
                deviation = np.broadcast_to(deviation, action_shape)
                row = (
                    recommendation_mask * (actual - deviation)
                ).reshape(feature_count, order="C")
                if np.linalg.norm(row) > _RANK_TOLERANCE:
                    rows.append(row)
    if not rows:
        return np.zeros((0, feature_count))
    matrix = np.asarray(rows)
    norms = np.linalg.norm(matrix, axis=1)
    return matrix / norms[:, None]


def _solve_relative_interior_test(
    payoff_tensor: np.ndarray,
    equilibrium: str,
    incentive_rows: np.ndarray,
) -> tuple[float, np.ndarray]:
    payoffs = np.asarray(payoff_tensor, dtype=float)
    action_shape = payoffs.shape[1:]
    profiles = list(np.ndindex(action_shape))
    variables, problem = create_equilibrium_lp(
        payoffs,
        equilibrium,
        np.zeros(action_shape),
    )
    common_slack = pulp.LpVariable(
        f"{equilibrium}_relative_interior_slack",
        lowBound=0.0,
        upBound=1.0,
    )
    for profile in profiles:
        problem += variables[profile] >= common_slack
    for row in incentive_rows:
        problem += (
            pulp.lpSum(
                float(coefficient) * variables[profile]
                for coefficient, profile in zip(row, profiles)
            )
            >= common_slack
        )
    problem.sense = pulp.LpMaximize
    problem.setObjective(common_slack)
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if status != pulp.LpStatusOptimal:
        raise RuntimeError(
            f"{equilibrium.upper()} relative-interior optimization failed "
            f"with solver status {pulp.LpStatus[problem.status]}"
        )
    distribution = _normalized_distribution(np.array(
        [variables[profile].varValue for profile in profiles],
        dtype=float,
    ))
    return float(common_slack.varValue), distribution


def _affine_geometry(
    payoff_tensor: np.ndarray,
    equilibrium: str,
) -> EquilibriumAffineGeometry:
    payoffs = np.asarray(payoff_tensor, dtype=float)
    action_shape = payoffs.shape[1:]
    feature_count = int(np.prod(action_shape))
    simplex_dimension = max(0, feature_count - 1)
    incentive_rows = _incentive_slack_rows(payoffs, equilibrium)
    common_slack, interior_candidate = _solve_relative_interior_test(
        payoffs,
        equilibrium,
        incentive_rows,
    )
    if common_slack > _SLACK_TOLERANCE:
        direction_basis = _simplex_tangent_basis(feature_count)
        return EquilibriumAffineGeometry(
            interior_candidate,
            direction_basis,
            np.zeros((feature_count, 0)),
        )

    unresolved_profiles = np.ones(feature_count, dtype=bool)
    unresolved_incentives = np.ones(len(incentive_rows), dtype=bool)
    witnesses: list[np.ndarray] = []
    while np.any(unresolved_profiles) or np.any(unresolved_incentives):
        objective = unresolved_profiles.astype(float)
        if np.any(unresolved_incentives):
            objective += np.sum(
                incentive_rows[unresolved_incentives],
                axis=0,
            )
        distribution = _normalized_distribution(optimize_equilibrium(
            payoffs,
            equilibrium,
            objective.reshape(action_shape, order="C"),
        ).reshape(feature_count, order="C"))
        profile_slacks = distribution
        incentive_slacks = (
            incentive_rows @ distribution
            if len(incentive_rows)
            else np.zeros(0)
        )
        positive_profiles = unresolved_profiles & (
            profile_slacks > _SLACK_TOLERANCE
        )
        positive_incentives = unresolved_incentives & (
            incentive_slacks > _SLACK_TOLERANCE
        )
        if not np.any(positive_profiles) and not np.any(positive_incentives):
            break
        witnesses.append(distribution)
        unresolved_profiles[positive_profiles] = False
        unresolved_incentives[positive_incentives] = False

    if not witnesses:
        witnesses.append(_normalized_distribution(
            optimize_equilibrium(
                payoffs,
                equilibrium,
                np.zeros(action_shape),
            ).reshape(feature_count, order="C")
        ))
    reference = np.mean(witnesses, axis=0)
    universal_rows = [np.ones(feature_count)]
    universal_rows.extend(np.eye(feature_count)[unresolved_profiles])
    if np.any(unresolved_incentives):
        universal_rows.extend(incentive_rows[unresolved_incentives])
    direction_basis = _null_space(
        np.asarray(universal_rows),
        feature_count,
    )
    if direction_basis.shape[1] > simplex_dimension:
        raise RuntimeError("equilibrium affine dimension exceeds the simplex")
    normal_basis = _null_space(
        np.vstack((np.ones(feature_count), direction_basis.T)),
        feature_count,
    )
    return EquilibriumAffineGeometry(
        reference,
        direction_basis,
        normal_basis,
    )


def _difference_subspace(
    outer_basis: np.ndarray,
    inner_basis: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(outer_basis, dtype=float)
    if inner_basis.shape[1]:
        residual = residual - inner_basis @ (inner_basis.T @ residual)
    expected = outer_basis.shape[1] - inner_basis.shape[1]
    return _orthonormal_span(residual, expected)


def _projection_family(
    action_shape: tuple[int, ...],
    ce: EquilibriumAffineGeometry,
    cce: EquilibriumAffineGeometry,
) -> EquilibriumProjectionGeometry:
    feature_count = int(np.prod(action_shape))
    simplex_dimension = max(0, feature_count - 1)
    ce_dimension = ce.dimension
    cce_dimension = cce.dimension
    if ce_dimension > cce_dimension:
        raise RuntimeError("CE affine dimension exceeds CCE affine dimension")
    if ce_dimension:
        residual = ce.direction_basis - cce.direction_basis @ (
            cce.direction_basis.T @ ce.direction_basis
        )
        if np.linalg.norm(residual) > 100.0 * _RANK_TOLERANCE:
            raise RuntimeError("CE affine direction space is not contained in CCE")

    tangent_basis = _simplex_tangent_basis(feature_count)
    if simplex_dimension <= 1:
        return EquilibriumProjectionGeometry(
            action_shape,
            simplex_dimension,
            ce,
            cce,
            "ambient_rank_0" if simplex_dimension == 0 else "ambient_rank_1",
            min(ce_dimension, simplex_dimension),
            min(cce_dimension, simplex_dimension),
            tangent_basis,
            np.zeros((feature_count, 0)),
            False,
            ("simplex tangent", "unused"),
        )

    if ce_dimension < cce_dimension < simplex_dimension:
        return EquilibriumProjectionGeometry(
            action_shape,
            simplex_dimension,
            ce,
            cce,
            "nested_lower_dimensional",
            0,
            1,
            _difference_subspace(
                cce.direction_basis,
                ce.direction_basis,
            ),
            cce.normal_basis,
            False,
            ("CCE tangent outside CE", "CCE normal"),
        )

    if ce_dimension == cce_dimension < simplex_dimension:
        codimension = simplex_dimension - cce_dimension
        if codimension >= 2:
            return EquilibriumProjectionGeometry(
                action_shape,
                simplex_dimension,
                ce,
                cce,
                "common_affine_hull_point",
                0,
                0,
                cce.normal_basis,
                cce.normal_basis,
                True,
                ("common equilibrium normal 1", "common equilibrium normal 2"),
            )
        return EquilibriumProjectionGeometry(
            action_shape,
            simplex_dimension,
            ce,
            cce,
            "common_affine_hull_line",
            1,
            1,
            cce.direction_basis,
            cce.normal_basis,
            False,
            ("common equilibrium tangent", "common equilibrium normal"),
        )

    if cce_dimension != simplex_dimension:
        raise RuntimeError("unhandled equilibrium affine-dimension case")
    if ce_dimension <= simplex_dimension - 2:
        return EquilibriumProjectionGeometry(
            action_shape,
            simplex_dimension,
            ce,
            cce,
            "full_cce_point_ce",
            0,
            2,
            ce.normal_basis,
            ce.normal_basis,
            True,
            ("CE normal 1", "CE normal 2"),
        )
    if ce_dimension == simplex_dimension - 1:
        return EquilibriumProjectionGeometry(
            action_shape,
            simplex_dimension,
            ce,
            cce,
            "full_cce_line_ce",
            1,
            2,
            ce.direction_basis,
            ce.normal_basis,
            False,
            ("CE tangent", "CE normal"),
        )
    return EquilibriumProjectionGeometry(
        action_shape,
        simplex_dimension,
        ce,
        cce,
        "full_dimensional",
        2,
        2,
        tangent_basis,
        tangent_basis,
        True,
        ("simplex principal component 1", "simplex principal component 2"),
    )


def analyze_equilibrium_projection_geometry(
    payoff_tensor: np.ndarray,
) -> EquilibriumProjectionGeometry:
    payoffs = np.asarray(payoff_tensor, dtype=float)
    if payoffs.ndim < 2:
        raise ValueError("payoff_tensor must contain one action axis per player")
    action_shape = tuple(int(size) for size in payoffs.shape[1:])
    ce = _affine_geometry(payoffs, "ce")
    cce = _affine_geometry(payoffs, "cce")
    return _projection_family(action_shape, ce, cce)


def _save_geometry(
    path: Path,
    geometry: EquilibriumProjectionGeometry,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".equilibrium-geometry-",
        dir=path.parent,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        np.savez_compressed(
            temporary_path,
            version=np.array(GEOMETRY_CACHE_VERSION),
            lp_tolerance=np.array(EQUILIBRIUM_LP_TOLERANCE),
            action_shape=np.asarray(geometry.action_shape, dtype=int),
            simplex_dimension=np.array(geometry.simplex_dimension),
            ce_reference=geometry.ce.reference,
            ce_direction=geometry.ce.direction_basis,
            ce_normal=geometry.ce.normal_basis,
            cce_reference=geometry.cce.reference,
            cce_direction=geometry.cce.direction_basis,
            cce_normal=geometry.cce.normal_basis,
            projection_case=np.array(geometry.projection_case),
            ce_projected_dimension=np.array(
                geometry.ce_projected_dimension
            ),
            cce_projected_dimension=np.array(
                geometry.cce_projected_dimension
            ),
            x_subspace=geometry.x_subspace_basis,
            y_subspace=geometry.y_subspace_basis,
            shared_axis_subspace=np.array(
                int(geometry.shared_axis_subspace)
            ),
            axis_roles=np.asarray(geometry.axis_roles),
        )
        os.replace(temporary_path, path)


def _load_geometry(path: Path) -> EquilibriumProjectionGeometry:
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["version"]) != GEOMETRY_CACHE_VERSION:
            raise ValueError("unsupported equilibrium geometry cache version")
        if float(archive["lp_tolerance"]) != EQUILIBRIUM_LP_TOLERANCE:
            raise ValueError(
                "equilibrium geometry cache uses a different LP tolerance"
            )
        ce = EquilibriumAffineGeometry(
            np.asarray(archive["ce_reference"], dtype=float),
            np.asarray(archive["ce_direction"], dtype=float),
            np.asarray(archive["ce_normal"], dtype=float),
        )
        cce = EquilibriumAffineGeometry(
            np.asarray(archive["cce_reference"], dtype=float),
            np.asarray(archive["cce_direction"], dtype=float),
            np.asarray(archive["cce_normal"], dtype=float),
        )
        return EquilibriumProjectionGeometry(
            tuple(int(value) for value in archive["action_shape"]),
            int(archive["simplex_dimension"]),
            ce,
            cce,
            str(archive["projection_case"]),
            int(archive["ce_projected_dimension"]),
            int(archive["cce_projected_dimension"]),
            np.asarray(archive["x_subspace"], dtype=float),
            np.asarray(archive["y_subspace"], dtype=float),
            bool(int(archive["shared_axis_subspace"])),
            tuple(str(value) for value in archive["axis_roles"]),
        )


class EquilibriumGeometryCache:
    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._lock = Lock()
        self._futures: dict[str, Future[EquilibriumProjectionGeometry]] = {}

    def _path(self, payoff_digest: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / (
            f"v{GEOMETRY_CACHE_VERSION}_{payoff_digest}.npz"
        )

    def get(
        self,
        payoff_digest: str,
        payoff_tensor: np.ndarray,
    ) -> EquilibriumProjectionGeometry:
        with self._lock:
            future = self._futures.get(payoff_digest)
            owner = future is None
            if future is None:
                future = Future()
                self._futures[payoff_digest] = future
        if not owner:
            return future.result()
        try:
            path = self._path(payoff_digest)
            geometry = None
            if path is not None and path.is_file():
                try:
                    geometry = _load_geometry(path)
                except (KeyError, OSError, TypeError, ValueError):
                    geometry = None
            if geometry is None:
                geometry = analyze_equilibrium_projection_geometry(
                    payoff_tensor
                )
                if path is not None:
                    _save_geometry(path, geometry)
            future.set_result(geometry)
            return geometry
        except BaseException as error:
            future.set_exception(error)
            with self._lock:
                self._futures.pop(payoff_digest, None)
            raise
