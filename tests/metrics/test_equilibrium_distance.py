import numpy as np
import pytest

from config import EQUILIBRIUM_LP_TOLERANCE
from experiments.games import create_rock_paper_scissors_payoffs
import metrics.equilibrium as equilibrium_module
from metrics.equilibrium_distance import equilibrium_l1_distance
from tests.support import coordination_game_payoffs


@pytest.mark.parametrize(("equilibrium", "expected_coarse"), [("ce", False), ("cce", True)])
def test_distance_reuses_upstream_equilibrium_polytope(monkeypatch, equilibrium: str, expected_coarse: bool) -> None:
    upstream_create_lp = equilibrium_module.games_learning_equilibrium.create_cce_lp
    coarse_arguments = []

    def recording_create_lp(payoff_matrix, coarse, objective=None):
        coarse_arguments.append(coarse)
        return upstream_create_lp(payoff_matrix=payoff_matrix, coarse=coarse, objective=objective)

    monkeypatch.setattr(equilibrium_module.games_learning_equilibrium, "create_cce_lp", recording_create_lp)
    empirical = np.array([[1.0, 0.0], [0.0, 0.0]])

    result = equilibrium_l1_distance(coordination_game_payoffs(), empirical, equilibrium)

    assert coarse_arguments == [expected_coarse]
    assert result.distance == pytest.approx(0.0, abs=EQUILIBRIUM_LP_TOLERANCE)
    assert result.nearest_distribution.shape == empirical.shape


def test_rps_diagonal_distribution_has_zero_cce_but_positive_ce_distance() -> None:
    payoff_tensor = create_rock_paper_scissors_payoffs()
    diagonal_distribution = np.zeros((3, 3))
    np.fill_diagonal(diagonal_distribution, 1.0 / 3.0)

    cce_distance = equilibrium_l1_distance(payoff_tensor, diagonal_distribution, "cce").distance
    ce_distance = equilibrium_l1_distance(payoff_tensor, diagonal_distribution, "ce").distance

    assert cce_distance == pytest.approx(0.0, abs=EQUILIBRIUM_LP_TOLERANCE)
    assert ce_distance > 0.1
    assert cce_distance >= -EQUILIBRIUM_LP_TOLERANCE
    assert ce_distance >= -EQUILIBRIUM_LP_TOLERANCE
    assert cce_distance <= ce_distance + EQUILIBRIUM_LP_TOLERANCE


def test_distance_supports_heterogeneous_three_player_games() -> None:
    payoff_tensor = np.zeros((3, 2, 1, 2))
    empirical = np.array([[[0.1, 0.2]], [[0.3, 0.4]]])

    distance = equilibrium_l1_distance(payoff_tensor, empirical, "ce").distance

    assert distance == pytest.approx(0.0, abs=EQUILIBRIUM_LP_TOLERANCE)


def test_distance_rejects_unknown_equilibrium_concept() -> None:
    with pytest.raises(ValueError, match="unknown equilibrium concept"):
        equilibrium_l1_distance(coordination_game_payoffs(), np.full((2, 2), 0.25), "nash")


@pytest.mark.parametrize(
    "empirical",
    [
        np.full((2, 3), 1.0 / 6.0),
        np.array([[0.5, 0.5], [0.5, -0.5]]),
        np.array([[0.5, 0.0], [0.0, 0.0]]),
        np.array([[np.nan, 0.0], [0.0, 1.0]]),
    ],
)
def test_distance_rejects_malformed_empirical_distributions(empirical: np.ndarray) -> None:
    with pytest.raises(ValueError):
        equilibrium_l1_distance(coordination_game_payoffs(), empirical)
