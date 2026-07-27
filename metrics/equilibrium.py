from collections.abc import Sequence
from typing import cast

from games_learning.utils import equilibrium as games_learning_equilibrium
import numpy as np


_COARSE_BY_CONCEPT = {"ce": False, "cce": True}


def _coarse_for_equilibrium(equilibrium: str) -> bool:
    try:
        return _COARSE_BY_CONCEPT[equilibrium]
    except KeyError as error:
        choices = ", ".join(sorted(_COARSE_BY_CONCEPT))
        raise ValueError(f"unknown equilibrium concept {equilibrium!r}; expected one of: {choices}") from error


def optimize_equilibrium(payoff_tensor, equilibrium: str = "ce", objective: np.ndarray | None = None) -> np.ndarray:
    """Return the upstream CE/CCE maximizing the optional linear objective."""
    coarse = _coarse_for_equilibrium(equilibrium)
    if objective is None:
        return games_learning_equilibrium.get_correlated_equilibrium(payoff_matrix=payoff_tensor, coarse=coarse)
    return games_learning_equilibrium.get_correlated_equilibrium(payoff_matrix=payoff_tensor, coarse=coarse, objective=objective)


def create_equilibrium_lp(payoff_tensor, equilibrium: str, objective: np.ndarray):
    payoff_matrix = cast(tuple[np.ndarray], tuple(payoff_tensor))
    return games_learning_equilibrium.create_cce_lp(
        payoff_matrix=payoff_matrix,
        coarse=_coarse_for_equilibrium(equilibrium),
        objective=objective,
    )


def max_equilibrium_profile_weight(payoff_tensor, target_profile: Sequence[int], equilibrium: str = "ce") -> float:
    """Return the largest equilibrium probability of one joint action."""
    action_shape = np.asarray(payoff_tensor).shape[1:]
    profile = tuple(target_profile)
    objective = np.zeros(action_shape)
    objective[profile] = 1.0
    distribution = optimize_equilibrium(payoff_tensor, equilibrium, objective)
    return float(distribution[profile])


def equilibrium_profile_weights(payoff_tensor, equilibrium: str = "ce") -> np.ndarray:
    """Return independently maximized equilibrium weights for every profile.

    Different entries can be attained by different distributions, so the
    returned array is not itself an equilibrium distribution.
    """
    action_shape = np.asarray(payoff_tensor).shape[1:]
    weights = np.empty(action_shape)
    for profile in np.ndindex(action_shape):
        weights[profile] = max_equilibrium_profile_weight(payoff_tensor, profile, equilibrium)
    return weights
