from dataclasses import dataclass
import math

import numpy as np

from metrics.empirical_distribution import EmpiricalDistributionTrajectory
from metrics.equilibrium_distance import equilibrium_l1_distance
from metrics.equilibrium_projection import LinearProjection2D, ProjectedEquilibriumRegion, project_equilibrium_region


@dataclass(frozen=True)
class EquilibriumDistanceTrajectory:
    horizons: np.ndarray
    ce: np.ndarray
    cce: np.ndarray


@dataclass(frozen=True)
class ReplicateEquilibriumDistanceTrajectory:
    horizons: np.ndarray
    ce_mean: np.ndarray
    cce_mean: np.ndarray
    ce_confidence: np.ndarray
    cce_confidence: np.ndarray
    n_replicates: int


@dataclass(frozen=True)
class EquilibriumTrajectoryProjection:
    empirical: EmpiricalDistributionTrajectory
    projection: LinearProjection2D
    projected_trajectory: np.ndarray
    ce_region: ProjectedEquilibriumRegion
    cce_region: ProjectedEquilibriumRegion


@dataclass(frozen=True)
class EquilibriumConvergenceAnalysis:
    empirical: EmpiricalDistributionTrajectory
    distances: EquilibriumDistanceTrajectory
    projection: LinearProjection2D
    projected_trajectory: np.ndarray
    ce_region: ProjectedEquilibriumRegion
    cce_region: ProjectedEquilibriumRegion


def equilibrium_distance_trajectory(payoff_tensor, empirical: EmpiricalDistributionTrajectory) -> EquilibriumDistanceTrajectory:
    ce_distances = []
    cce_distances = []
    for distribution in empirical.distributions:
        ce_distances.append(equilibrium_l1_distance(payoff_tensor, distribution, "ce").distance)
        cce_distances.append(equilibrium_l1_distance(payoff_tensor, distribution, "cce").distance)
    return EquilibriumDistanceTrajectory(empirical.horizons, np.asarray(ce_distances), np.asarray(cce_distances))


def aggregate_equilibrium_distance_trajectories(
    trajectories: list[EquilibriumDistanceTrajectory],
) -> ReplicateEquilibriumDistanceTrajectory:
    if not trajectories:
        raise ValueError("at least one equilibrium-distance trajectory is required")
    horizons = trajectories[0].horizons
    for trajectory in trajectories[1:]:
        if not np.array_equal(trajectory.horizons, horizons):
            raise ValueError("equilibrium-distance trajectories must have matching horizons")
    ce = np.asarray([trajectory.ce for trajectory in trajectories])
    cce = np.asarray([trajectory.cce for trajectory in trajectories])
    if len(trajectories) == 1:
        ce_confidence = np.zeros_like(ce[0])
        cce_confidence = np.zeros_like(cce[0])
    else:
        scale = 1.96 / math.sqrt(len(trajectories))
        ce_confidence = scale * np.std(ce, axis=0, ddof=1)
        cce_confidence = scale * np.std(cce, axis=0, ddof=1)
    return ReplicateEquilibriumDistanceTrajectory(
        horizons.copy(),
        np.mean(ce, axis=0),
        np.mean(cce, axis=0),
        ce_confidence,
        cce_confidence,
        len(trajectories),
    )


def project_equilibrium_trajectory(
    payoff_tensor,
    empirical: EmpiricalDistributionTrajectory,
    direction_count: int = 128,
) -> EquilibriumTrajectoryProjection:
    projection = LinearProjection2D.fit([empirical.vectors])
    return EquilibriumTrajectoryProjection(
        empirical,
        projection,
        projection.transform(empirical.vectors),
        project_equilibrium_region(payoff_tensor, projection, "ce", direction_count),
        project_equilibrium_region(payoff_tensor, projection, "cce", direction_count),
    )


def analyze_equilibrium_convergence(payoff_tensor, empirical: EmpiricalDistributionTrajectory, direction_count: int = 128,
                                    distances: EquilibriumDistanceTrajectory | None = None) -> EquilibriumConvergenceAnalysis:
    if distances is None:
        distances = equilibrium_distance_trajectory(payoff_tensor, empirical)
    trajectory = project_equilibrium_trajectory(payoff_tensor, empirical, direction_count)
    return EquilibriumConvergenceAnalysis(
        empirical,
        distances,
        trajectory.projection,
        trajectory.projected_trajectory,
        trajectory.ce_region,
        trajectory.cce_region,
    )
