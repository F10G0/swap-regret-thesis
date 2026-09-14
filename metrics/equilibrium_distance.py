from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from config import EQUILIBRIUM_LP_TOLERANCE
from metrics.empirical_distribution import EmpiricalDistributionTrajectory


EQUILIBRIUM_DISTANCE_IMPLEMENTATION_VERSION = 1


@dataclass(frozen=True)
class EquilibriumDistanceResult:
    distance: float
    nearest_distribution: np.ndarray


def _validated_payoff_tensor(payoff_tensor) -> np.ndarray:
    try:
        payoffs = np.asarray(payoff_tensor, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("payoff_tensor must be a rectangular numeric array") from error
    if payoffs.ndim < 2 or payoffs.shape[0] != payoffs.ndim - 1:
        raise ValueError(
            "payoff_tensor must have one action axis per player "
            "(shape[0] must equal ndim - 1)"
        )
    if any(dimension == 0 for dimension in payoffs.shape):
        raise ValueError("payoff_tensor dimensions must be non-empty")
    if not np.all(np.isfinite(payoffs)):
        raise ValueError("payoff_tensor must contain only finite values")
    return payoffs


def _validated_distribution(empirical_distribution, action_shape: tuple[int, ...]) -> np.ndarray:
    empirical = np.asarray(empirical_distribution, dtype=float)
    if empirical.shape != action_shape:
        raise ValueError(f"empirical_distribution must have shape {action_shape}")
    valid_values = np.all(np.isfinite(empirical)) and np.all(empirical >= -EQUILIBRIUM_LP_TOLERANCE)
    valid_values = valid_values and np.all(empirical <= 1.0 + EQUILIBRIUM_LP_TOLERANCE)
    valid_total = np.isclose(empirical.sum(), 1.0, atol=EQUILIBRIUM_LP_TOLERANCE, rtol=0.0)
    if not valid_values or not valid_total:
        raise ValueError("empirical_distribution must be a probability distribution")
    return empirical


class _PreparedDistanceLP:
    """Fixed CE/CCE and L1 matrices; only the empirical RHS changes per solve."""

    def __init__(self, payoff_tensor, equilibrium: str):
        if equilibrium not in {"ce", "cce"}:
            raise ValueError(f"unknown equilibrium concept {equilibrium!r}")
        self.equilibrium = equilibrium
        payoffs = np.asarray(payoff_tensor, dtype=float)
        self.action_shape = payoffs.shape[1:]
        self.n_profiles = int(np.prod(self.action_shape))
        profiles = np.array(list(np.ndindex(self.action_shape)))
        incentive_rows = []
        for player, n_actions in enumerate(self.action_shape):
            utility = payoffs[player].ravel(order="C")
            for deviation in range(n_actions):
                deviation_utility = np.broadcast_to(
                    np.expand_dims(np.take(payoffs[player], deviation, axis=player), axis=player),
                    self.action_shape,
                ).ravel(order="C")
                # Deviation minus obedience <= 0, in the upstream C-order.
                gain = deviation_utility - utility
                if equilibrium == "cce":
                    incentive_rows.append(gain)
                else:
                    for recommendation in range(n_actions):
                        if recommendation != deviation:
                            incentive_rows.append(np.where(profiles[:, player] == recommendation, gain, 0.0))

        m = self.n_profiles
        incentives = sparse.csr_matrix(np.asarray(incentive_rows).reshape(-1, m))
        identity = sparse.eye(m, format="csr")
        self.c = np.concatenate((np.zeros(m), np.ones(m)))
        self.a_ub = sparse.vstack((
            sparse.hstack((incentives, sparse.csr_matrix(incentives.shape))),
            sparse.hstack((identity, -identity)),
            sparse.hstack((-identity, -identity)),
        ), format="csr")
        self.b_ub = np.zeros(self.a_ub.shape[0])
        self.a_eq = sparse.csr_matrix(np.concatenate((np.ones(m), np.zeros(m)))[None, :])
        self.b_eq = np.ones(1)

    def solve(self, empirical_vector: np.ndarray):
        m = self.n_profiles
        self.b_ub[-2 * m:-m] = empirical_vector
        self.b_ub[-m:] = -empirical_vector
        result = linprog(
            self.c, A_ub=self.a_ub, b_ub=self.b_ub,
            A_eq=self.a_eq, b_eq=self.b_eq, bounds=(0.0, None), method="highs",
            options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9,
                     "ipm_optimality_tolerance": 1e-10},
        )
        if not result.success:
            raise RuntimeError(f"{self.equilibrium.upper()} distance optimization failed: {result.message}")
        return result


def equilibrium_l1_distance(payoff_tensor, empirical_distribution, equilibrium: str = "ce") -> EquilibriumDistanceResult:
    """Return exact full-dimensional L1 distance and a nearest CE/CCE."""
    payoffs = _validated_payoff_tensor(payoff_tensor)
    action_shape = payoffs.shape[1:]
    empirical = _validated_distribution(empirical_distribution, action_shape)
    prepared = _PreparedDistanceLP(payoffs, equilibrium)
    result = prepared.solve(empirical.ravel(order="C"))
    nearest = result.x[:prepared.n_profiles].reshape(action_shape, order="C")
    return EquilibriumDistanceResult(float(result.fun), nearest)


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
    n_replicates: int


def equilibrium_distance_trajectory(
    payoff_tensor,
    empirical: EmpiricalDistributionTrajectory,
) -> EquilibriumDistanceTrajectory:
    payoffs = _validated_payoff_tensor(payoff_tensor)
    ce = _PreparedDistanceLP(payoffs, "ce")
    cce = _PreparedDistanceLP(payoffs, "cce")
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
        len(trajectories),
    )
