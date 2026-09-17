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
    """Create RPS payoffs with actions ordered Rock, Paper, Scissors.

    The tensor axes are (player, player-0 action, player-1 action).
    Each player's raw -1/0/1 payoffs are normalized to 0/0.5/1.
    """
    payoff_player_0 = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    return _normalized_game_payoffs((payoff_player_0, -payoff_player_0))


def create_matching_pennies_payoffs() -> np.ndarray:
    """Create Matching Pennies with actions ordered Heads, Tails."""
    payoff_player_0 = np.array([[1, -1], [-1, 1]])
    return _normalized_game_payoffs((payoff_player_0, -payoff_player_0))


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


PAYOFF_FACTORIES = {
    "matching_pennies": create_matching_pennies_payoffs,
    "rps": create_rock_paper_scissors_payoffs,
    "rpsls": create_rock_paper_scissors_lizard_spock_payoffs,
}
