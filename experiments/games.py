import numpy as np


def normalize_payoffs(payoffs: np.ndarray) -> np.ndarray:
    payoffs = np.asarray(payoffs, dtype=float)

    min_value = np.min(payoffs)
    max_value = np.max(payoffs)

    if np.isclose(min_value, max_value):
        raise ValueError("cannot normalize constant payoffs")

    return (payoffs - min_value) / (max_value - min_value)


def _two_player_payoffs(payoff_player_0: np.ndarray, payoff_player_1: np.ndarray) -> np.ndarray:
    payoff_player_0 = normalize_payoffs(payoff_player_0)
    payoff_player_1 = normalize_payoffs(payoff_player_1)

    if payoff_player_0.shape != payoff_player_1.shape:
        raise ValueError("both players must have the same payoff matrix shape")

    return np.stack([payoff_player_0, payoff_player_1], axis=0)


def _identical_interest_payoffs(payoff_matrix: np.ndarray) -> np.ndarray:
    return _two_player_payoffs(payoff_matrix, payoff_matrix.copy())


def _zero_sum_payoffs(payoff_player_0: np.ndarray) -> np.ndarray:
    return _two_player_payoffs(payoff_player_0, 0.0 - payoff_player_0)


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

    payoff_player_0 = np.full(
        (n_actions, n_actions),
        0.5,
        dtype=float,
    )

    half = n_actions // 2

    for row_action in range(n_actions):
        for column_action in range(n_actions):
            if row_action == column_action:
                payoff_player_0[row_action, column_action] = 0.5
            elif (row_action - column_action) % n_actions <= half:
                payoff_player_0[row_action, column_action] = 1.0
            else:
                payoff_player_0[row_action, column_action] = 0.0

    return _zero_sum_payoffs(payoff_player_0)


PAYOFF_FACTORIES = {
    "rps": create_rock_paper_scissors_payoffs,
    "dominant_coordination_9": lambda: create_dominant_coordination_payoffs(9),
    "cyclic_dominance_9": lambda: create_cyclic_dominance_payoffs(9),
}
