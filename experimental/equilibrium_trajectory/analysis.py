"""Projection-dependent equilibrium trajectory analysis."""

from dataclasses import dataclass, replace

import numpy as np

from config import EQUILIBRIUM_LP_TOLERANCE
from metrics.empirical_distribution import EmpiricalDistributionTrajectory
from metrics.equilibrium import optimize_equilibrium
from metrics.equilibrium_convergence import (
    EquilibriumDistanceTrajectory,
    equilibrium_distance_trajectory,
)
from metrics.equilibrium_distance import equilibrium_l1_distance
from experimental.equilibrium_trajectory.geometry import (
    EquilibriumProjectionGeometry,
    analyze_equilibrium_projection_geometry,
)
from experimental.equilibrium_trajectory.projection import (
    DEFAULT_RELATIVE_RENDER_TOLERANCE,
    DEFAULT_SUPPORT_QUERY_CAP,
    LinearProjection2D,
    ProjectedEquilibriumSet,
    fit_equilibrium_comparison_projection,
    fit_equilibrium_projection,
    fit_unified_equilibrium_comparison_direction,
    project_equilibrium_set,
)


@dataclass(frozen=True)
class EquilibriumTrajectoryProjection:
    empirical: EmpiricalDistributionTrajectory
    projection: LinearProjection2D
    projected_trajectory: np.ndarray
    ce_region: ProjectedEquilibriumSet
    cce_region: ProjectedEquilibriumSet
    view_kind: str = "geometry"
    axis_labels: tuple[str, str] = (
        "Projected component 1",
        "Projected component 2",
    )


@dataclass(frozen=True)
class EquilibriumComparisonMemberProjection:
    member_id: str
    empirical: EmpiricalDistributionTrajectory
    projected_trajectory: np.ndarray


@dataclass(frozen=True)
class EquilibriumTrajectoryComparison:
    projection: LinearProjection2D
    members: tuple[EquilibriumComparisonMemberProjection, ...]
    ce_region: ProjectedEquilibriumSet
    cce_region: ProjectedEquilibriumSet
    view_kind: str = "geometry"
    axis_labels: tuple[str, str] = (
        "Projected component 1",
        "Projected component 2",
    )


@dataclass(frozen=True)
class EquilibriumConvergenceAnalysis:
    empirical: EmpiricalDistributionTrajectory
    distances: EquilibriumDistanceTrajectory
    projection: LinearProjection2D
    projected_trajectory: np.ndarray
    ce_region: ProjectedEquilibriumSet
    cce_region: ProjectedEquilibriumSet
    view_kind: str = "geometry"
    axis_labels: tuple[str, str] = (
        "Projected component 1",
        "Projected component 2",
    )


def _validated_comparison_inputs(
    empiricals: list[EmpiricalDistributionTrajectory],
    member_ids: list[str] | None,
    fit_empiricals: list[EmpiricalDistributionTrajectory] | None,
) -> tuple[
    list[str],
    EmpiricalDistributionTrajectory,
    list[EmpiricalDistributionTrajectory],
]:
    if not empiricals:
        raise ValueError("at least one comparison trajectory is required")
    resolved_member_ids = (
        [str(index) for index in range(len(empiricals))]
        if member_ids is None
        else list(member_ids)
    )
    if (
        len(resolved_member_ids) != len(empiricals)
        or len(set(resolved_member_ids)) != len(resolved_member_ids)
    ):
        raise ValueError("comparison member ids must be unique and complete")

    first = empiricals[0]
    if any(
        empirical.action_shape != first.action_shape
        or not np.array_equal(empirical.horizons, first.horizons)
        for empirical in empiricals[1:]
    ):
        raise ValueError(
            "comparison trajectories must have matching action shapes and checkpoints"
        )

    fitting = empiricals if fit_empiricals is None else fit_empiricals
    if len(fitting) != len(empiricals):
        raise ValueError(
            "one fitting trajectory is required per comparison member"
        )
    fit_horizons = fitting[0].horizons
    if any(
        empirical.action_shape != first.action_shape
        or not np.array_equal(empirical.horizons, fit_horizons)
        for empirical in fitting
    ):
        raise ValueError(
            "fitting trajectories must have matching action shapes and checkpoints"
        )
    return resolved_member_ids, first, fitting


def _cce_distance_coordinates(
    payoff_tensor,
    empirical: EmpiricalDistributionTrajectory,
) -> np.ndarray:
    return np.maximum(0.0, np.asarray([
        equilibrium_l1_distance(
            payoff_tensor,
            distribution,
            "cce",
        ).distance
        for distribution in empirical.distributions
    ]))


def project_geometry_trajectory(
    payoff_tensor,
    empirical: EmpiricalDistributionTrajectory,
    support_query_cap: int = DEFAULT_SUPPORT_QUERY_CAP,
    geometry: EquilibriumProjectionGeometry | None = None,
    relative_render_tolerance: float | None = (
        DEFAULT_RELATIVE_RENDER_TOLERANCE
    ),
    fit_empirical: EmpiricalDistributionTrajectory | None = None,
) -> EquilibriumTrajectoryProjection:
    if geometry is None:
        geometry = analyze_equilibrium_projection_geometry(payoff_tensor)
    fitting_trajectory = (
        empirical if fit_empirical is None else fit_empirical
    )
    projection = fit_equilibrium_projection(
        geometry,
        [fitting_trajectory.vectors],
    )
    return EquilibriumTrajectoryProjection(
        empirical,
        projection,
        projection.transform(empirical.vectors),
        project_equilibrium_set(
            payoff_tensor,
            projection,
            geometry.ce,
            geometry.ce_projected_dimension,
            "ce",
            support_query_cap,
            relative_render_tolerance,
        ),
        project_equilibrium_set(
            payoff_tensor,
            projection,
            geometry.cce,
            geometry.cce_projected_dimension,
            "cce",
            support_query_cap,
            relative_render_tolerance,
        ),
    )


def _equilibrium_relative_regions(
    ce_region: ProjectedEquilibriumSet,
    cce_region: ProjectedEquilibriumSet,
) -> tuple[ProjectedEquilibriumSet, ProjectedEquilibriumSet]:
    relative_ce = replace(
        ce_region,
        support_points=np.zeros_like(ce_region.support_points),
        boundary=np.zeros_like(ce_region.boundary),
    )
    relative_cce = replace(
        cce_region,
        support_points=np.column_stack((
            cce_region.support_points[:, 0],
            np.zeros(len(cce_region.support_points)),
        )),
        boundary=np.column_stack((
            cce_region.boundary[:, 0],
            np.zeros(len(cce_region.boundary)),
        )),
    )
    return relative_ce, relative_cce


def project_equilibrium_trajectory_comparison(
    payoff_tensor,
    empiricals: list[EmpiricalDistributionTrajectory],
    member_ids: list[str] | None = None,
    support_query_cap: int = DEFAULT_SUPPORT_QUERY_CAP,
    geometry: EquilibriumProjectionGeometry | None = None,
    relative_render_tolerance: float | None = (
        DEFAULT_RELATIVE_RENDER_TOLERANCE
    ),
    fit_empiricals: list[EmpiricalDistributionTrajectory] | None = None,
) -> EquilibriumTrajectoryComparison:
    member_ids, _, fitting = _validated_comparison_inputs(
        empiricals,
        member_ids,
        fit_empiricals,
    )

    if geometry is None:
        geometry = analyze_equilibrium_projection_geometry(payoff_tensor)
    projection = fit_equilibrium_comparison_projection(
        geometry,
        [empirical.vectors for empirical in fitting],
    )
    ce_region = project_equilibrium_set(
        payoff_tensor,
        projection,
        geometry.ce,
        geometry.ce_projected_dimension,
        "ce",
        support_query_cap,
        relative_render_tolerance,
    )
    cce_region = project_equilibrium_set(
        payoff_tensor,
        projection,
        geometry.cce,
        geometry.cce_projected_dimension,
        "cce",
        support_query_cap,
        relative_render_tolerance,
    )
    projected = [
        projection.transform(empirical.vectors)
        for empirical in empiricals
    ]
    view_kind = "geometry"
    axis_labels = (
        "Projected component 1",
        "Projected component 2",
    )
    if geometry.projection_case == "nested_lower_dimensional":
        ce_region, cce_region = _equilibrium_relative_regions(
            ce_region,
            cce_region,
        )
        projected = [
            np.column_stack((
                coordinates[:, 0],
                _cce_distance_coordinates(payoff_tensor, empirical),
            ))
            for empirical, coordinates in zip(empiricals, projected)
        ]
        view_kind = "equilibrium_relative"
        axis_labels = (
            "CE-relative CCE tangent coordinate",
            "L1 distance to CCE",
        )
    return EquilibriumTrajectoryComparison(
        projection,
        tuple(
            EquilibriumComparisonMemberProjection(
                member_id,
                empirical,
                coordinates,
            )
            for member_id, empirical, coordinates in zip(
                member_ids,
                empiricals,
                projected,
            )
        ),
        ce_region,
        cce_region,
        view_kind,
        axis_labels,
    )


def _normalized_equilibrium_distribution(distribution) -> np.ndarray:
    result = np.asarray(distribution, dtype=float).copy()
    result[np.abs(result) < 1e-12] = 0.0
    total = float(np.sum(result))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError(
            "equilibrium optimizer returned an invalid distribution"
        )
    return result / total


def _unified_equilibrium_interval(
    payoff_tensor,
    equilibrium: str,
    direction: np.ndarray,
    x_center: float,
) -> tuple[ProjectedEquilibriumSet, tuple[float, float]]:
    payoffs = np.asarray(payoff_tensor, dtype=float)
    action_shape = payoffs.shape[1:]
    objective = direction.reshape(action_shape, order="C")
    minimum_distribution = _normalized_equilibrium_distribution(
        optimize_equilibrium(payoffs, equilibrium, -objective)
    )
    maximum_distribution = _normalized_equilibrium_distribution(
        optimize_equilibrium(payoffs, equilibrium, objective)
    )
    support_distributions = np.asarray([
        minimum_distribution,
        maximum_distribution,
    ])
    raw_values = support_distributions.reshape(2, -1) @ direction
    order = np.argsort(raw_values, kind="stable")
    raw_values = raw_values[order]
    support_distributions = support_distributions[order]
    points = np.column_stack((raw_values - x_center, np.zeros(2)))
    scale = max(1.0, float(np.max(np.abs(raw_values))))
    width_tolerance = 10.0 * EQUILIBRIUM_LP_TOLERANCE * scale
    affine_dimension = int(raw_values[1] - raw_values[0] > width_tolerance)
    boundary = points if affine_dimension else points[:1]
    return ProjectedEquilibriumSet(
        points,
        support_distributions,
        boundary,
        affine_dimension,
        True,
        2,
        2,
        "exact",
        None,
        0.0,
        0.0,
        float(raw_values[1] - raw_values[0]),
    ), (float(raw_values[0]), float(raw_values[1]))


def project_unified_equilibrium_trajectory_comparison(
    payoff_tensor,
    empiricals: list[EmpiricalDistributionTrajectory],
    member_ids: list[str] | None = None,
    fit_empiricals: list[EmpiricalDistributionTrajectory] | None = None,
) -> EquilibriumTrajectoryComparison:
    """Project every comparison through one simplex-relative CCE view."""
    member_ids, first, fitting = _validated_comparison_inputs(
        empiricals,
        member_ids,
        fit_empiricals,
    )

    direction = fit_unified_equilibrium_comparison_direction(
        [empirical.vectors for empirical in fitting]
    )
    uncentered_ce, ce_interval = _unified_equilibrium_interval(
        payoff_tensor,
        "ce",
        direction,
        0.0,
    )
    x_center = 0.5 * (ce_interval[0] + ce_interval[1])
    ce_region = replace(
        uncentered_ce,
        support_points=(
            uncentered_ce.support_points
            - np.array([x_center, 0.0])
        ),
        boundary=(
            uncentered_ce.boundary
            - np.array([x_center, 0.0])
        ),
    )
    cce_region, cce_interval = _unified_equilibrium_interval(
        payoff_tensor,
        "cce",
        direction,
        x_center,
    )
    containment_tolerance = 20.0 * EQUILIBRIUM_LP_TOLERANCE * max(
        1.0,
        *(abs(value) for value in (*ce_interval, *cce_interval)),
    )
    if (
        ce_interval[0] < cce_interval[0] - containment_tolerance
        or ce_interval[1] > cce_interval[1] + containment_tolerance
    ):
        raise RuntimeError(
            "projected CE interval is not contained in projected CCE"
        )

    feature_count = first.vectors.shape[1]
    components = np.zeros((2, feature_count))
    components[0] = direction
    projection = LinearProjection2D(
        x_center * direction,
        components,
        int(np.linalg.norm(direction) > 0.0),
        ("simplex comparison direction", "CCE distance"),
    )
    members = []
    for member_id, empirical in zip(member_ids, empiricals):
        x_coordinates = empirical.vectors @ direction - x_center
        members.append(EquilibriumComparisonMemberProjection(
            member_id,
            empirical,
            np.column_stack((
                x_coordinates,
                _cce_distance_coordinates(payoff_tensor, empirical),
            )),
        ))
    return EquilibriumTrajectoryComparison(
        projection,
        tuple(members),
        ce_region,
        cce_region,
        "unified_equilibrium_relative",
        (
            "Shared comparison direction (CE interval centered at 0)",
            "L1 distance to CCE",
        ),
    )


def project_equilibrium_trajectory(
    payoff_tensor,
    empirical: EmpiricalDistributionTrajectory,
    support_query_cap: int = DEFAULT_SUPPORT_QUERY_CAP,
    geometry: EquilibriumProjectionGeometry | None = None,
    relative_render_tolerance: float | None = (
        DEFAULT_RELATIVE_RENDER_TOLERANCE
    ),
    fit_empirical: EmpiricalDistributionTrajectory | None = None,
) -> EquilibriumTrajectoryProjection:
    comparison = project_equilibrium_trajectory_comparison(
        payoff_tensor,
        [empirical],
        member_ids=["trajectory"],
        support_query_cap=support_query_cap,
        geometry=geometry,
        relative_render_tolerance=relative_render_tolerance,
        fit_empiricals=(
            None if fit_empirical is None else [fit_empirical]
        ),
    )
    member = comparison.members[0]
    return EquilibriumTrajectoryProjection(
        member.empirical,
        comparison.projection,
        member.projected_trajectory,
        comparison.ce_region,
        comparison.cce_region,
        comparison.view_kind,
        comparison.axis_labels,
    )


def analyze_equilibrium_convergence(
    payoff_tensor,
    empirical: EmpiricalDistributionTrajectory,
    support_query_cap: int = DEFAULT_SUPPORT_QUERY_CAP,
    distances: EquilibriumDistanceTrajectory | None = None,
    geometry: EquilibriumProjectionGeometry | None = None,
    relative_render_tolerance: float | None = (
        DEFAULT_RELATIVE_RENDER_TOLERANCE
    ),
) -> EquilibriumConvergenceAnalysis:
    if distances is None:
        distances = equilibrium_distance_trajectory(payoff_tensor, empirical)
    trajectory = project_equilibrium_trajectory(
        payoff_tensor,
        empirical,
        support_query_cap,
        geometry,
        relative_render_tolerance,
    )
    return EquilibriumConvergenceAnalysis(
        empirical,
        distances,
        trajectory.projection,
        trajectory.projected_trajectory,
        trajectory.ce_region,
        trajectory.cce_region,
        trajectory.view_kind,
        trajectory.axis_labels,
    )
