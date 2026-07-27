from operator import index

from games_learning.game.econ_game import (
    BertrandLinear,
    BertrandLogit,
    BertrandStandard,
)
from games_learning.game.matrix_game import ExampleMatrixGames
import numpy as np


def normalize_payoffs(payoffs: np.ndarray) -> np.ndarray:
    payoffs = np.asarray(payoffs, dtype=float)
    if payoffs.size == 0:
        raise ValueError("cannot normalize empty payoffs")
    if not np.all(np.isfinite(payoffs)):
        raise ValueError("payoffs must contain only finite values")

    minimum = np.min(payoffs)
    scale = np.ptp(payoffs)
    if np.isclose(scale, 0.0):
        raise ValueError("cannot normalize constant payoffs")
    return (payoffs - minimum) / scale


def _normalized_game_payoffs(payoff_tensor) -> np.ndarray:
    try:
        payoffs = np.asarray(payoff_tensor, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("game payoffs must be a rectangular numeric array") from error
    if payoffs.ndim < 2 or payoffs.shape[0] != payoffs.ndim - 1:
        raise ValueError("game payoffs must have shape (n_players, action_1, ..., action_n)")
    if any(size == 0 for size in payoffs.shape):
        raise ValueError("game payoff dimensions must be non-empty")
    if not np.all(np.isfinite(payoffs)):
        raise ValueError("game payoffs must contain only finite values")
    return np.stack([normalize_payoffs(payoff_matrix) for payoff_matrix in payoffs])


def create_rock_paper_scissors_payoffs() -> np.ndarray:
    game = ExampleMatrixGames("rock_paper_scissors")
    return _normalized_game_payoffs(game.payoff_matrix)


def create_rock_paper_scissors_lizard_spock_payoffs() -> np.ndarray:
    """Create the five-action symmetric zero-sum game used by Leme et al. (2024).

    Actions are ordered Rock, Paper, Scissors, Lizard, Spock.
    """
    winning_actions = {
        0: (2, 3),
        1: (0, 4),
        2: (1, 3),
        3: (1, 4),
        4: (0, 2),
    }
    payoff_player_0 = np.zeros((5, 5))
    for winner, losers in winning_actions.items():
        payoff_player_0[winner, list(losers)] = 1.0
        payoff_player_0[list(losers), winner] = -1.0
    return _normalized_game_payoffs((payoff_player_0, -payoff_player_0))


def create_matching_pennies_payoffs() -> np.ndarray:
    game = ExampleMatrixGames("matching_pennies")
    return _normalized_game_payoffs(game.payoff_matrix)


def _finite_pair(value, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain exactly two values")
    try:
        pair = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must contain exactly two values") from error
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    try:
        result = (float(pair[0]), float(pair[1]))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite numeric values")
    return result


def _finite_scalar(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_pair_or_scalar(value, name: str) -> tuple[float, float]:
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError):
        return _finite_pair(value, name)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must contain finite numeric values")
    return (scalar, scalar)


def _positive_pair_or_scalar(value, name: str) -> tuple[float, float]:
    pair = _finite_pair_or_scalar(value, name)
    if min(pair) <= 0.0:
        raise ValueError(f"{name} must be positive")
    return pair


def _bertrand_parameters(
    n_actions: int,
    cost: tuple[float, float],
    interval: tuple[float, float],
) -> tuple[int, tuple[float, float], tuple[float, float]]:
    try:
        n_actions = index(n_actions)
    except TypeError as error:
        raise ValueError("n_actions must be an integer") from error
    if n_actions <= 1:
        raise ValueError("n_actions must be greater than 1")

    cost = _finite_pair(cost, "cost")
    interval = _finite_pair(interval, "interval")
    if interval[0] >= interval[1]:
        raise ValueError("interval lower bound must be less than upper bound")
    return n_actions, cost, interval


def _bertrand_payoffs(game_class, n_actions: int, cost, interval, **parameters) -> np.ndarray:
    game = game_class(n_agents=2, n_discr=n_actions, cost=cost, interval=interval, **parameters)
    return _normalized_game_payoffs(game.payoff_matrix)


def create_standard_bertrand_payoffs(
    n_actions: int,
    cost: tuple[float, float],
    interval: tuple[float, float],
    maximum_demand: float = 1.0,
) -> np.ndarray:
    """Create normalized payoffs through games_learning.BertrandStandard."""
    n_actions, cost, interval = _bertrand_parameters(n_actions, cost, interval)
    maximum_demand = _finite_scalar(maximum_demand, "maximum_demand")
    if maximum_demand <= 0.0:
        raise ValueError("maximum_demand must be positive")
    return _bertrand_payoffs(BertrandStandard, n_actions, cost, interval, maximum_demand=maximum_demand)


def create_linear_bertrand_payoffs(
    n_actions: int,
    cost: tuple[float, float],
    interval: tuple[float, float],
    alpha: float | tuple[float, float],
    beta: float | tuple[float, float],
    gamma: float,
) -> np.ndarray:
    """Create normalized payoffs through games_learning.BertrandLinear."""
    n_actions, cost, interval = _bertrand_parameters(n_actions, cost, interval)
    alpha = _positive_pair_or_scalar(alpha, "alpha")
    beta = _positive_pair_or_scalar(beta, "beta")
    gamma = _finite_scalar(gamma, "gamma")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    if gamma >= min(beta):
        raise ValueError("gamma must be less than each beta")
    return _bertrand_payoffs(BertrandLinear, n_actions, cost, interval, alpha=alpha, beta=beta, gamma=gamma)


def create_logit_bertrand_payoffs(
    n_actions: int,
    cost: tuple[float, float],
    interval: tuple[float, float],
    alpha: float | tuple[float, float],
    mu: float | tuple[float, float],
    alpha0: float = 0.0,
) -> np.ndarray:
    """Create normalized payoffs through games_learning.BertrandLogit."""
    n_actions, cost, interval = _bertrand_parameters(n_actions, cost, interval)
    alpha = _positive_pair_or_scalar(alpha, "alpha")
    mu = _positive_pair_or_scalar(mu, "mu")
    alpha0 = _finite_scalar(alpha0, "alpha0")
    if not np.isclose(alpha0, 0.0):
        raise ValueError("alpha0 must be zero because games_learning fixes the outside-option contribution at one")
    return _bertrand_payoffs(BertrandLogit, n_actions, cost, interval, alpha=alpha, mu=mu)


PAYOFF_FACTORIES = {
    "rps": create_rock_paper_scissors_payoffs,
    "rpsls": create_rock_paper_scissors_lizard_spock_payoffs,
    "matching_pennies": create_matching_pennies_payoffs,
    "bertrand_standard_o1": lambda: create_standard_bertrand_payoffs(
        n_actions=21,
        cost=(0.0, 0.0),
        interval=(0.05, 1.0),
        maximum_demand=1.0,
    ),
    "bertrand_linear_o2": lambda: create_linear_bertrand_payoffs(
        n_actions=21,
        cost=(0.0, 0.0),
        interval=(0.0, 1.0),
        alpha=0.48,
        beta=0.9,
        gamma=0.6,
    ),
    "bertrand_logit_o3": lambda: create_logit_bertrand_payoffs(
        n_actions=21,
        cost=(1.0, 1.0),
        interval=(1.0, 2.0),
        alpha=(2.0, 2.0),
        mu=0.25,
    ),
    "bertrand_linear_o2_prime": lambda: create_linear_bertrand_payoffs(
        n_actions=21,
        cost=(0.0, 0.2),
        interval=(0.0, 1.0),
        alpha=0.48,
        beta=0.9,
        gamma=0.6,
    ),
    "bertrand_logit_o3_prime": lambda: create_logit_bertrand_payoffs(
        n_actions=21,
        cost=(0.5, 1.0),
        interval=(0.5, 2.0),
        alpha=(1.5, 2.0),
        mu=0.25,
    ),
}
