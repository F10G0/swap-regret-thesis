import numpy as np
import pulp
import pytest

from config import EQUILIBRIUM_LP_TOLERANCE
from experiments.games import create_rock_paper_scissors_payoffs
import metrics.equilibrium as equilibrium_module
from metrics.equilibrium_distance import equilibrium_l1_distance
from tests.support import coordination_game_payoffs


def old_pulp_distance(payoffs, empirical, concept):
    """Pre-change production formulation, retained only as a test oracle."""
    variables, problem = equilibrium_module.create_equilibrium_lp(payoffs, concept, np.zeros(empirical.shape))
    deviations = pulp.LpVariable.dicts("l1_distance", list(variables), lowBound=0.0)
    for profile in variables:
        problem += deviations[profile] >= variables[profile] - empirical[profile]
        problem += deviations[profile] >= empirical[profile] - variables[profile]
    problem.sense = pulp.LpMinimize
    problem.setObjective(pulp.lpSum(deviations.values()))
    assert problem.solve(pulp.PULP_CBC_CMD(msg=False)) == pulp.LpStatusOptimal
    return float(pulp.value(problem.objective))


@pytest.mark.parametrize("concept", ["ce", "cce"])
@pytest.mark.parametrize("payoffs", [
    np.array([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]]),
    coordination_game_payoffs(),
    create_rock_paper_scissors_payoffs(),
    np.random.default_rng(17).random((3, 2, 1, 3)),
])
def test_highs_matches_old_polytope_and_distance(payoffs, concept):
    shape = payoffs.shape[1:]
    random = np.random.default_rng(42)
    pure = np.zeros(shape)
    pure.flat[0] = 1
    distributions = [pure, np.full(shape, 1 / np.prod(shape)),
                     *[random.dirichlet(np.ones(np.prod(shape))).reshape(shape) for _ in range(4)]]
    for empirical in distributions:
        result = equilibrium_l1_distance(payoffs, empirical, concept)
        assert result.distance == pytest.approx(old_pulp_distance(payoffs, empirical, concept), abs=1e-8, rel=1e-8)
        nearest = result.nearest_distribution
        assert nearest.shape == shape
        assert nearest.min() >= -1e-9
        assert nearest.sum() == pytest.approx(1, abs=1e-9)
        assert np.abs(nearest - empirical).sum() == pytest.approx(result.distance, abs=1e-8)
        # Verify feasibility in the upstream polytope, not just our own matrices.
        variables, problem = equilibrium_module.create_equilibrium_lp(payoffs, concept, np.zeros(shape))
        for profile, variable in variables.items():
            variable.varValue = nearest[profile]
        for constraint in problem.constraints.values():
            if constraint.sense == pulp.LpConstraintEQ:
                assert abs(constraint.value()) < 1e-8
            else:
                assert constraint.sense * constraint.value() >= -1e-8


def test_prepared_distance_reuses_all_coefficient_matrices(monkeypatch):
    import metrics.equilibrium_distance as module
    original = module.linprog
    matrices = []

    def solve(*args, **kwargs):
        assert kwargs["method"] == "highs"
        matrices.append((kwargs["A_ub"], kwargs["A_eq"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "linprog", solve)
    prepared = module._PreparedDistanceLP(create_rock_paper_scissors_payoffs(), "ce")
    for vector in (np.full(9, 1/9), np.eye(9)[0], np.eye(9)[1]):
        prepared.solve(vector)
    assert all(a is matrices[0][0] and b is matrices[0][1] for a, b in matrices)


def test_failed_distance_solve_is_not_silently_used(monkeypatch):
    from types import SimpleNamespace
    import metrics.equilibrium_distance as module
    monkeypatch.setattr(module, "linprog", lambda *args, **kwargs: SimpleNamespace(success=False, message="test failure"))
    with pytest.raises(RuntimeError, match="CE distance optimization failed"):
        equilibrium_l1_distance(coordination_game_payoffs(), np.full((2, 2), 0.25))


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
