from operator import index

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


def _two_player_payoffs(payoff_player_0: np.ndarray, payoff_player_1: np.ndarray) -> np.ndarray:
    return np.stack((normalize_payoffs(payoff_player_0), normalize_payoffs(payoff_player_1)))


def _identical_interest_payoffs(payoff_matrix: np.ndarray) -> np.ndarray:
    return _two_player_payoffs(payoff_matrix, payoff_matrix)


def _zero_sum_payoffs(payoff_player_0: np.ndarray) -> np.ndarray:
    return _two_player_payoffs(payoff_player_0, -payoff_player_0)


def create_rock_paper_scissors_payoffs() -> np.ndarray:
    payoff_player_0 = np.array(
        [
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0],
        ],
        dtype=float,
    )
    return _zero_sum_payoffs(payoff_player_0)


def create_dominant_coordination_payoffs(n_actions: int) -> np.ndarray:
    if n_actions <= 1:
        raise ValueError("n_actions must be greater than 1")

    payoff_matrix = np.zeros((n_actions, n_actions), dtype=float)
    np.fill_diagonal(payoff_matrix, 0.9)
    payoff_matrix[0, 0] = 1.0

    return _identical_interest_payoffs(payoff_matrix)


def create_cyclic_dominance_payoffs(n_actions: int) -> np.ndarray:
    if n_actions < 3:
        raise ValueError("n_actions must be at least 3")
    if n_actions % 2 == 0:
        raise ValueError("n_actions must be odd for balanced cyclic dominance")

    actions = np.arange(n_actions)
    differences = (actions[:, None] - actions) % n_actions
    payoff_player_0 = (differences <= n_actions // 2).astype(float)
    np.fill_diagonal(payoff_player_0, 0.5)
    return _zero_sum_payoffs(payoff_player_0)


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


def _bertrand_price_grid(
    n_actions: int,
    cost: tuple[float, float],
    interval: tuple[float, float],
) -> tuple[tuple[float, float], np.ndarray]:
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

    prices = np.linspace(interval[0], interval[1], n_actions)
    if not np.all(np.isfinite(prices)):
        raise ValueError("price grid must contain only finite values")
    return cost, prices


def create_standard_bertrand_payoffs(
    n_actions: int,
    cost: tuple[float, float],
    interval: tuple[float, float],
    maximum_demand: float = 1.0,
) -> np.ndarray:
    """Return normalized payoffs for a two-firm homogeneous-good Bertrand game."""
    cost, prices = _bertrand_price_grid(n_actions, cost, interval)

    maximum_demand = _finite_scalar(maximum_demand, "maximum_demand")
    if maximum_demand <= 0.0:
        raise ValueError("maximum_demand must be positive")

    prices_0 = prices[:, None]
    prices_1 = prices[None, :]

    demand_0 = np.where(
        prices_0 < prices_1,
        maximum_demand,
        np.where(
            prices_0 == prices_1,
            maximum_demand / 2.0,
            0.0,
        ),
    )
    demand_1 = np.where(
        prices_1 < prices_0,
        maximum_demand,
        np.where(
            prices_0 == prices_1,
            maximum_demand / 2.0,
            0.0,
        ),
    )
    payoff_player_0 = (prices_0 - cost[0]) * demand_0
    payoff_player_1 = (prices_1 - cost[1]) * demand_1

    if not np.all(np.isfinite(payoff_player_0)) or not np.all(
        np.isfinite(payoff_player_1)
    ):
        raise ValueError("Bertrand profits must contain only finite values")
    if np.isclose(np.ptp(payoff_player_0), 0.0) or np.isclose(
        np.ptp(payoff_player_1),
        0.0,
    ):
        raise ValueError("Bertrand payoffs must not be constant")

    return _two_player_payoffs(payoff_player_0, payoff_player_1)


def create_linear_bertrand_payoffs(
    n_actions: int,
    cost: tuple[float, float],
    interval: tuple[float, float],
    alpha: float | tuple[float, float],
    beta: float | tuple[float, float],
    gamma: float,
) -> np.ndarray:
    """Return normalized payoffs for a two-firm linear-demand Bertrand game.

    Demand is not clipped at zero. For firms ``i`` and ``j``, the implemented
    model is ``d_i = alpha_i - beta_i * price_i + gamma * price_j``.
    """
    cost, prices = _bertrand_price_grid(n_actions, cost, interval)
    alpha = _finite_pair_or_scalar(alpha, "alpha")
    beta = _finite_pair_or_scalar(beta, "beta")
    gamma = _finite_scalar(gamma, "gamma")

    if min(alpha) <= 0.0:
        raise ValueError("alpha must be positive")
    if min(beta) <= 0.0:
        raise ValueError("beta must be positive")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    if gamma >= min(beta):
        raise ValueError("gamma must be less than each beta")

    prices_0 = prices[:, None]
    prices_1 = prices[None, :]
    demand_0 = alpha[0] - beta[0] * prices_0 + gamma * prices_1
    demand_1 = alpha[1] - beta[1] * prices_1 + gamma * prices_0
    payoff_player_0 = (prices_0 - cost[0]) * demand_0
    payoff_player_1 = (prices_1 - cost[1]) * demand_1

    return _two_player_payoffs(payoff_player_0, payoff_player_1)


def create_logit_bertrand_payoffs(
    n_actions: int,
    cost: tuple[float, float],
    interval: tuple[float, float],
    alpha: tuple[float, float],
    mu: float,
    alpha0: float = 0.0,
) -> np.ndarray:
    """Return normalized payoffs for a two-firm logit-demand Bertrand game.

    Each firm's demand is its softmax share among both firms and an outside
    good whose quality is ``alpha0``.
    """
    cost, prices = _bertrand_price_grid(n_actions, cost, interval)
    alpha = _finite_pair(alpha, "alpha")
    mu = _finite_scalar(mu, "mu")
    alpha0 = _finite_scalar(alpha0, "alpha0")

    if min(alpha) <= 0.0:
        raise ValueError("alpha must be positive")
    if mu <= 0.0:
        raise ValueError("mu must be positive")

    prices_0 = prices[:, None]
    prices_1 = prices[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        utility_0 = (alpha[0] - prices_0) / mu
        utility_1 = (alpha[1] - prices_1) / mu
        outside_utility = alpha0 / mu
    if (
        not np.all(np.isfinite(utility_0))
        or not np.all(np.isfinite(utility_1))
        or not np.isfinite(outside_utility)
    ):
        raise ValueError("scaled logit utilities must contain only finite values")

    maximum_utility = np.maximum(
        np.maximum(utility_0, utility_1),
        outside_utility,
    )
    exp_0 = np.exp(utility_0 - maximum_utility)
    exp_1 = np.exp(utility_1 - maximum_utility)
    exp_outside = np.exp(outside_utility - maximum_utility)
    denominator = exp_0 + exp_1 + exp_outside
    demand_0 = exp_0 / denominator
    demand_1 = exp_1 / denominator

    payoff_player_0 = (prices_0 - cost[0]) * demand_0
    payoff_player_1 = (prices_1 - cost[1]) * demand_1
    return _two_player_payoffs(payoff_player_0, payoff_player_1)


PAYOFF_FACTORIES = {
    "rps": create_rock_paper_scissors_payoffs,
    "dominant_coordination_9": lambda: create_dominant_coordination_payoffs(9),
    "cyclic_dominance_9": lambda: create_cyclic_dominance_payoffs(9),
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
        alpha0=0.0,
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
        alpha0=0.0,
    ),
}
