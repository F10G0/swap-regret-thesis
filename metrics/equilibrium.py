from collections.abc import Iterator, Sequence
from operator import index

import numpy as np
from scipy.optimize import linprog

from config import EQUILIBRIUM_LP_TOLERANCE


EQUILIBRIUM_CONCEPTS = {"ce", "cce"}


class EquilibriumOptimizationError(RuntimeError):
    """Raised when an equilibrium-profile linear program is unsuccessful."""

    def __init__(
        self,
        equilibrium: str,
        target_profile: tuple[int, ...],
        status: int,
        message: str,
    ) -> None:
        super().__init__(
            f"{equilibrium.upper()} optimization failed for profile "
            f"{target_profile}: solver status {status}: {message}"
        )
        self.equilibrium = equilibrium
        self.target_profile = target_profile
        self.status = status
        self.solver_message = message


def _validate_payoff_tensor(payoff_tensor) -> np.ndarray:
    """Return a validated floating-point copy of a finite-game payoff tensor."""
    try:
        array = np.asarray(payoff_tensor)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "payoff tensor must be a rectangular numeric array"
        ) from error

    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype,
        np.complexfloating,
    ):
        raise ValueError("payoff tensor must contain real numeric values")

    array = np.asarray(array, dtype=float)
    if array.ndim < 2:
        raise ValueError(
            "payoff tensor must have shape "
            "(n_players, action_1, ..., action_n)"
        )
    if array.shape[0] == 0:
        raise ValueError("number of players must be non-zero")
    if array.shape[0] != array.ndim - 1:
        raise ValueError(
            "number of players must match number of action dimensions"
        )
    if any(size == 0 for size in array.shape[1:]):
        raise ValueError("each player must have at least one action")
    if not np.all(np.isfinite(array)):
        raise ValueError("payoffs must contain only finite values")
    return array.copy()


def joint_action_profiles(
    action_counts: Sequence[int],
) -> Iterator[tuple[int, ...]]:
    """Iterate over joint actions in NumPy C-order."""
    try:
        shape = tuple(index(count) for count in action_counts)
    except (TypeError, ValueError) as error:
        raise ValueError("action counts must be integers") from error
    if not shape:
        raise ValueError("at least one player action count is required")
    if any(count <= 0 for count in shape):
        raise ValueError("action counts must be positive")
    return np.ndindex(shape)


def _profiles_and_shape(
    payoff_tensor,
) -> tuple[np.ndarray, tuple[int, ...], tuple[tuple[int, ...], ...]]:
    payoffs = _validate_payoff_tensor(payoff_tensor)
    action_shape = payoffs.shape[1:]
    profiles = tuple(joint_action_profiles(action_shape))
    return payoffs, action_shape, profiles


def _build_cce_constraints(
    payoffs: np.ndarray,
    action_shape: tuple[int, ...],
    profiles: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for player, n_actions in enumerate(action_shape):
        for deviation_action in range(n_actions):
            row = np.empty(len(profiles), dtype=float)
            for flat_index, profile in enumerate(profiles):
                deviated_profile = list(profile)
                deviated_profile[player] = deviation_action
                row[flat_index] = (
                    payoffs[(player, *deviated_profile)]
                    - payoffs[(player, *profile)]
                )
            rows.append(row)

    return np.asarray(rows, dtype=float), np.zeros(len(rows), dtype=float)


def build_cce_constraints(payoff_tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return ``A_ub, b_ub`` for unconditional fixed-deviation CCE constraints."""
    payoffs, action_shape, profiles = _profiles_and_shape(payoff_tensor)
    return _build_cce_constraints(payoffs, action_shape, profiles)


def _build_ce_constraints(
    payoffs: np.ndarray,
    action_shape: tuple[int, ...],
    profiles: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for player, n_actions in enumerate(action_shape):
        for recommended_action in range(n_actions):
            for deviation_action in range(n_actions):
                if deviation_action == recommended_action:
                    continue

                row = np.zeros(len(profiles), dtype=float)
                for flat_index, profile in enumerate(profiles):
                    if profile[player] != recommended_action:
                        continue
                    deviated_profile = list(profile)
                    deviated_profile[player] = deviation_action
                    row[flat_index] = (
                        payoffs[(player, *deviated_profile)]
                        - payoffs[(player, *profile)]
                    )
                rows.append(row)

    n_profiles = len(profiles)
    matrix = (
        np.asarray(rows, dtype=float)
        if rows
        else np.empty((0, n_profiles), dtype=float)
    )
    return matrix, np.zeros(len(rows), dtype=float)


def build_ce_constraints(payoff_tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return ``A_ub, b_ub`` for recommendation-conditioned CE constraints."""
    payoffs, action_shape, profiles = _profiles_and_shape(payoff_tensor)
    return _build_ce_constraints(payoffs, action_shape, profiles)


def _validate_equilibrium(equilibrium: str) -> str:
    if equilibrium not in EQUILIBRIUM_CONCEPTS:
        choices = ", ".join(sorted(EQUILIBRIUM_CONCEPTS))
        raise ValueError(
            f"unknown equilibrium concept {equilibrium!r}; expected one of: "
            f"{choices}"
        )
    return equilibrium


def _validate_target_profile(
    target_profile: Sequence[int],
    action_shape: tuple[int, ...],
) -> tuple[int, ...]:
    try:
        profile = tuple(index(action) for action in target_profile)
    except (TypeError, ValueError) as error:
        raise ValueError("target profile actions must be integers") from error
    if len(profile) != len(action_shape):
        raise ValueError("target profile must contain one action per player")
    for player, (action, n_actions) in enumerate(zip(profile, action_shape)):
        if not 0 <= action < n_actions:
            raise ValueError(
                f"target action {action} is invalid for player {player}"
            )
    return profile


def _constraints_for(
    equilibrium: str,
    payoffs: np.ndarray,
    action_shape: tuple[int, ...],
    profiles: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    if equilibrium == "ce":
        return _build_ce_constraints(payoffs, action_shape, profiles)
    return _build_cce_constraints(payoffs, action_shape, profiles)


def _raise_invalid_solution(
    equilibrium: str,
    target_profile: tuple[int, ...],
    result,
    detail: str,
) -> None:
    raise EquilibriumOptimizationError(
        equilibrium,
        target_profile,
        int(result.status),
        f"{result.message}; invalid solution: {detail}",
    )


def _validate_solution(
    distribution: np.ndarray,
    constraints: np.ndarray,
    bounds: np.ndarray,
    equilibrium: str,
    target_profile: tuple[int, ...],
    result,
) -> None:
    tolerance = EQUILIBRIUM_LP_TOLERANCE
    n_profiles = constraints.shape[1]
    if distribution.shape != (n_profiles,):
        _raise_invalid_solution(
            equilibrium,
            target_profile,
            result,
            f"expected {n_profiles} probabilities, got {distribution.shape}",
        )
    if not np.all(np.isfinite(distribution)):
        _raise_invalid_solution(
            equilibrium,
            target_profile,
            result,
            "probabilities are not finite",
        )
    if np.any(distribution < -tolerance) or np.any(
        distribution > 1.0 + tolerance
    ):
        _raise_invalid_solution(
            equilibrium,
            target_profile,
            result,
            "probability bounds are violated",
        )

    normalization_residual = abs(float(np.sum(distribution)) - 1.0)
    if normalization_residual > tolerance:
        _raise_invalid_solution(
            equilibrium,
            target_profile,
            result,
            f"normalization residual {normalization_residual} exceeds "
            f"{tolerance}",
        )

    if constraints.shape[0]:
        incentive_residual = float(
            np.max(constraints @ distribution - bounds)
        )
        if incentive_residual > tolerance:
            _raise_invalid_solution(
                equilibrium,
                target_profile,
                result,
                f"incentive residual {incentive_residual} exceeds "
                f"{tolerance}",
            )


def _maximize_profile(
    action_shape: tuple[int, ...],
    constraints: np.ndarray,
    constraint_bounds: np.ndarray,
    equilibrium: str,
    target_profile: tuple[int, ...],
) -> float:
    n_profiles = int(np.prod(action_shape))
    target_index = int(
        np.ravel_multi_index(target_profile, action_shape)
    )
    objective = np.zeros(n_profiles, dtype=float)
    objective[target_index] = -1.0
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=constraint_bounds,
        A_eq=np.ones((1, n_profiles), dtype=float),
        b_eq=np.ones(1, dtype=float),
        bounds=[(0.0, None)] * n_profiles,
        method="highs",
    )
    if not result.success:
        raise EquilibriumOptimizationError(
            equilibrium,
            target_profile,
            int(result.status),
            str(result.message),
        )

    distribution = np.asarray(result.x, dtype=float)
    _validate_solution(
        distribution,
        constraints,
        constraint_bounds,
        equilibrium,
        target_profile,
        result,
    )
    return float(distribution[target_index])


def max_equilibrium_profile_weight(
    payoff_tensor,
    target_profile: Sequence[int],
    equilibrium: str = "ce",
) -> float:
    """Return the largest equilibrium probability of one joint action."""
    equilibrium = _validate_equilibrium(equilibrium)
    payoffs, action_shape, profiles = _profiles_and_shape(payoff_tensor)
    target = _validate_target_profile(target_profile, action_shape)
    constraints, bounds = _constraints_for(
        equilibrium,
        payoffs,
        action_shape,
        profiles,
    )
    return _maximize_profile(
        action_shape,
        constraints,
        bounds,
        equilibrium,
        target,
    )


def equilibrium_profile_weights(
    payoff_tensor,
    equilibrium: str = "ce",
) -> np.ndarray:
    """Return independently maximized equilibrium weights for every profile.

    Entry ``a`` is ``max_{mu in E} mu(a)`` for the requested equilibrium
    polytope. Different entries can be attained by different distributions,
    so the returned array is not itself an equilibrium distribution and
    generally does not sum to one.
    """
    equilibrium = _validate_equilibrium(equilibrium)
    payoffs, action_shape, profiles = _profiles_and_shape(payoff_tensor)
    constraints, bounds = _constraints_for(
        equilibrium,
        payoffs,
        action_shape,
        profiles,
    )

    weights = np.empty(action_shape, dtype=float)
    for profile in profiles:
        weights[profile] = _maximize_profile(
            action_shape,
            constraints,
            bounds,
            equilibrium,
            profile,
        )
    return weights
