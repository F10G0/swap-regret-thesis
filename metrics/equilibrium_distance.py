from dataclasses import dataclass

import numpy as np
import pulp

from config import EQUILIBRIUM_LP_TOLERANCE
from metrics.equilibrium import create_equilibrium_lp


@dataclass(frozen=True)
class EquilibriumDistanceResult:
    distance: float
    nearest_distribution: np.ndarray


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


def equilibrium_l1_distance(payoff_tensor, empirical_distribution, equilibrium: str = "ce") -> EquilibriumDistanceResult:
    """Return the full-dimensional L1 distance to the upstream CE/CCE polytope."""
    action_shape = np.asarray(payoff_tensor).shape[1:]
    empirical = _validated_distribution(empirical_distribution, action_shape)
    variables, problem = create_equilibrium_lp(payoff_tensor, equilibrium, np.zeros(action_shape))
    profiles = list(variables)
    deviations = pulp.LpVariable.dicts("l1_distance", profiles, lowBound=0.0)
    for profile in profiles:
        problem += deviations[profile] >= variables[profile] - empirical[profile]
        problem += deviations[profile] >= empirical[profile] - variables[profile]
    problem.sense = pulp.LpMinimize
    problem.setObjective(pulp.lpSum(deviations.values()))
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if status != pulp.LpStatusOptimal:
        raise RuntimeError(f"{equilibrium.upper()} distance optimization failed with solver status {pulp.LpStatus[problem.status]}")
    nearest = np.array([variables[profile].varValue for profile in profiles], dtype=float).reshape(action_shape, order="C")
    return EquilibriumDistanceResult(float(pulp.value(problem.objective)), nearest)
