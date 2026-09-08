"""Full-space CE/CCE convergence metrics.

Projected trajectories deliberately live in
``experimental.equilibrium_trajectory``.  This module contains only the
authoritative distance-vs-horizon evidence used by the core analysis.
"""

from dataclasses import dataclass

import numpy as np

from metrics.confidence import mean_confidence_interval_half_width
from metrics.empirical_distribution import EmpiricalDistributionTrajectory
from metrics.equilibrium_distance import _PreparedDistanceLP


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


def equilibrium_distance_trajectory(
    payoff_tensor,
    empirical: EmpiricalDistributionTrajectory,
) -> EquilibriumDistanceTrajectory:
    ce = _PreparedDistanceLP(payoff_tensor, "ce")
    cce = _PreparedDistanceLP(payoff_tensor, "cce")
    if empirical.action_shape != ce.action_shape:
        raise ValueError("empirical action shape must match the payoff tensor")
    ce_distances = []
    cce_distances = []
    for vector in empirical.vectors:
        # Figures need only the objective, not a reshaped nearest equilibrium.
        ce_distances.append(float(ce.solve(vector).fun))
        cce_distances.append(float(cce.solve(vector).fun))
    return EquilibriumDistanceTrajectory(
        empirical.horizons,
        np.asarray(ce_distances),
        np.asarray(cce_distances),
    )


def aggregate_equilibrium_distance_trajectories(
    trajectories: list[EquilibriumDistanceTrajectory],
) -> ReplicateEquilibriumDistanceTrajectory:
    if not trajectories:
        raise ValueError(
            "at least one equilibrium-distance trajectory is required"
        )
    horizons = trajectories[0].horizons
    for trajectory in trajectories[1:]:
        if not np.array_equal(trajectory.horizons, horizons):
            raise ValueError(
                "equilibrium-distance trajectories must have matching horizons"
            )
    ce = np.asarray([trajectory.ce for trajectory in trajectories])
    cce = np.asarray([trajectory.cce for trajectory in trajectories])
    return ReplicateEquilibriumDistanceTrajectory(
        horizons.copy(),
        np.mean(ce, axis=0),
        np.mean(cce, axis=0),
        mean_confidence_interval_half_width(ce),
        mean_confidence_interval_half_width(cce),
        len(trajectories),
    )
