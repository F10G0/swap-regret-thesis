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


PAYOFF_FACTORIES = {
    "rps": create_rock_paper_scissors_payoffs,
    "dominant_coordination_9": lambda: create_dominant_coordination_payoffs(9),
    "cyclic_dominance_9": lambda: create_cyclic_dominance_payoffs(9),
}
