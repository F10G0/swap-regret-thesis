import numpy as np
import pytest

from config import EQUILIBRIUM_LP_TOLERANCE
from experiments.games import create_rock_paper_scissors_payoffs
from metrics.equilibrium_distance import equilibrium_l1_distance
from tests.support import coordination_game_payoffs


# Objective baselines independently cross-validated before dependency removal.
# All 60 comparisons passed its existing abs=1e-8, rel=1e-8 tolerance.
FIXTURES = {
    "matching_pennies": np.array([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]]),
    "coordination": coordination_game_payoffs(),
    "rps": create_rock_paper_scissors_payoffs(),
    "asymmetric": np.array([[[3., 0.], [1., 2.]], [[1., 4.], [3., 0.]]]),
    "heterogeneous": np.random.default_rng(17).random((3, 2, 1, 3)),
}
# Pure profile 0, uniform, then four Dirichlet draws with seed 42.
DISTANCE_BASELINES = {
    "matching_pennies": (
        (1.5, 0., .4244305077727705, .5288534688658404, .8689629172461633, .6913626007698301),
        (1.5, 0., .4244305077727705, .5288534688658404, .8689629172461633, .6913626007698301),
    ),
    "coordination": (
        (0., 0., .3746466963561506, .29525104358098475, .4232531000930262, .3076316431473425),
        (0., 0., .3746466963561506, .29525104358098475, .4232531000930262, .3076316431473425),
    ),
    "rps": (
        (1.7777777777777777, 0., .623097519866187, .7782787957534819, .5247105340247252, .5321218928529506),
        (1.3333333333333335, 0., .39834603544061115, .6108454745300456, .2763512864621229, .4536428902539118),
    ),
    "asymmetric": (
        (1.5, 0., .4244305077727705, .5288534688658404, .8689629172461633, .6913626007698301),
        (1.5, 0., .4244305077727705, .5288534688658404, .8689629172461633, .6913626007698301),
    ),
    "heterogeneous": (
        (2., .929890118565153, 1.252395546571471, 1.3296525076413122, 1.4976857559226513, .7526347212943629),
        (2., .929890118565153, 1.252395546571471, 1.3296525076413122, 1.4976857559226513, .7526347212943629),
    ),
}


def maximum_incentive_gain(payoffs, distribution, concept):
    """Check deviations directly, independently of the production LP matrices."""
    gains = []
    for player, n_actions in enumerate(distribution.shape):
        for deviation_action in range(n_actions):
            recommendations = (None,) if concept == "cce" else range(n_actions)
            for recommendation in recommendations:
                if recommendation == deviation_action:
                    continue
                gain = 0.
                for profile in np.ndindex(distribution.shape):
                    if recommendation is not None and profile[player] != recommendation:
                        continue
                    deviation = list(profile)
                    deviation[player] = deviation_action
                    gain += distribution[profile] * (
                        payoffs[(player, *deviation)] - payoffs[(player, *profile)]
                    )
                gains.append(gain)
    return max(gains, default=0.)


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("distribution_index", range(6))
def test_distance_baselines_and_independent_feasibility(fixture, distribution_index):
    payoffs = FIXTURES[fixture]
    shape = payoffs.shape[1:]
    size = int(np.prod(shape))
    random = np.random.default_rng(42)
    pure = np.zeros(shape)
    pure.flat[0] = 1.
    distributions = [
        pure, np.full(shape, 1 / size),
        *[random.dirichlet(np.ones(size)).reshape(shape) for _ in range(4)],
    ]
    empirical = distributions[distribution_index]
    distances = {}
    for index, concept in enumerate(("ce", "cce")):
        result = equilibrium_l1_distance(payoffs, empirical, concept)
        distances[concept] = result.distance
        assert result.distance == pytest.approx(
            DISTANCE_BASELINES[fixture][index][distribution_index], abs=1e-8, rel=1e-8
        )
        nearest = result.nearest_distribution
        assert nearest.shape == shape
        assert np.isfinite(nearest).all()
        assert nearest.min() >= -1e-9
        assert nearest.sum() == pytest.approx(1, abs=1e-9)
        assert np.abs(nearest - empirical).sum() == pytest.approx(result.distance, abs=1e-8)
        assert maximum_incentive_gain(payoffs, nearest, concept) <= EQUILIBRIUM_LP_TOLERANCE
    assert distances["cce"] <= distances["ce"] + EQUILIBRIUM_LP_TOLERANCE


@pytest.mark.parametrize("concept", ("ce", "cce"))
def test_uniform_rps_is_an_equilibrium(concept):
    empirical = np.full((3, 3), 1 / 9)
    result = equilibrium_l1_distance(FIXTURES["rps"], empirical, concept)
    assert result.distance == pytest.approx(0., abs=EQUILIBRIUM_LP_TOLERANCE)
    np.testing.assert_allclose(result.nearest_distribution, empirical, atol=EQUILIBRIUM_LP_TOLERANCE)
    assert maximum_incentive_gain(FIXTURES["rps"], empirical, concept) <= EQUILIBRIUM_LP_TOLERANCE


def test_direct_checker_distinguishes_conditional_and_unconditional_deviations():
    # A diagonal RPS mixture has no profitable fixed-action deviation, but its
    # recommendation reveals the opponent's action and is exploitable under CE.
    diagonal = np.eye(3) / 3
    assert maximum_incentive_gain(FIXTURES["rps"], diagonal, "cce") == pytest.approx(0.)
    assert maximum_incentive_gain(FIXTURES["rps"], diagonal, "ce") == pytest.approx(1 / 6)


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


def test_distance_rejects_payoffs_without_one_action_axis_per_player() -> None:
    with pytest.raises(ValueError, match="one action axis per player"):
        equilibrium_l1_distance(np.zeros((3, 2, 2)), np.full((2, 2), 0.25))


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


@pytest.mark.parametrize("shape,concept,profiles,rows,coefficients,expected_bytes", [
    ((2, 3), "ce", 6, 8, 48, 1368),
    ((2, 3), "cce", 6, 5, 30, 1056),
    ((2, 1, 3), "ce", 6, 8, 48, 1416),
    ((2, 1, 3), "cce", 6, 6, 36, 1208),
])
def test_equilibrium_resource_estimate_matches_current_dense_shapes(
    shape, concept, profiles, rows, coefficients, expected_bytes,
):
    from metrics.equilibrium_distance import estimate_equilibrium_analysis

    estimate = estimate_equilibrium_analysis(shape, concept)

    assert estimate.action_shape == shape
    assert estimate.equilibrium == concept
    assert (estimate.n_profiles, estimate.n_incentive_rows, estimate.n_dense_coefficients) == (
        profiles, rows, coefficients,
    )
    assert estimate.estimated_bytes == expected_bytes


def test_resource_estimator_never_enumerates_or_constructs_requested_matrix(monkeypatch):
    import metrics.equilibrium_distance as module

    monkeypatch.setattr(module.np, "ndindex", lambda *args: pytest.fail("enumerated joint profiles"))
    monkeypatch.setattr(module.np, "array", lambda *args, **kwargs: pytest.fail("allocated LP array"))

    estimate = module.estimate_equilibrium_analysis((100, 100, 10), "cce")

    assert estimate.n_profiles == 100_000
    assert estimate.n_dense_coefficients == 21_000_000
    assert estimate.estimated_bytes > module.EQUILIBRIUM_ANALYSIS_BUDGET_BYTES


@pytest.mark.parametrize("concept", ("ce", "cce"))
def test_shipped_games_remain_under_equilibrium_budget(concept):
    from experiments.games import PAYOFF_FACTORIES
    from metrics.equilibrium_distance import preflight_equilibrium_analysis

    for factory in PAYOFF_FACTORIES.values():
        shape = factory().shape[1:]
        assert preflight_equilibrium_analysis(shape, concept).action_shape == shape


def test_resource_preflight_allows_estimates_at_or_below_budget_only(monkeypatch):
    import metrics.equilibrium_distance as module

    estimate = module.estimate_equilibrium_analysis((2, 3), "ce")
    for budget in (estimate.estimated_bytes + 1, estimate.estimated_bytes):
        monkeypatch.setattr(module, "EQUILIBRIUM_ANALYSIS_BUDGET_BYTES", budget)
        assert module.preflight_equilibrium_analysis((2, 3), "ce") == estimate
    monkeypatch.setattr(module, "EQUILIBRIUM_ANALYSIS_BUDGET_BYTES", estimate.estimated_bytes - 1)
    with pytest.raises(module.EquilibriumAnalysisUnavailable) as captured:
        module.preflight_equilibrium_analysis((2, 3), "ce")
    assert captured.value.estimate == estimate
    assert captured.value.budget_bytes == estimate.estimated_bytes - 1


def test_oversized_ce_rejects_direct_distance_before_profile_matrix(monkeypatch):
    import metrics.equilibrium_distance as module

    monkeypatch.setattr(module.np, "ndindex", lambda *args: pytest.fail("constructed profile matrix"))
    payoffs = np.zeros((2, 100, 100))
    empirical = np.zeros((100, 100))
    empirical[0, 0] = 1.0

    with pytest.raises(module.EquilibriumAnalysisUnavailable) as captured:
        module.equilibrium_l1_distance(payoffs, empirical, "ce")

    error = captured.value
    assert error.estimate.equilibrium == "ce"
    assert error.estimate.action_shape == (100, 100)
    assert error.estimate.n_profiles == 10_000
    assert error.estimate.n_incentive_rows == 19_800
    assert error.estimate.estimated_bytes > error.budget_bytes
    assert "CE" in str(error) and "analysis budget" in str(error)
    assert module.preflight_equilibrium_analysis((100, 100), "cce").estimated_bytes < error.budget_bytes


def test_oversized_cce_rejects_direct_preparation_before_profile_matrix(monkeypatch):
    import metrics.equilibrium_distance as module

    monkeypatch.setattr(module.np, "ndindex", lambda *args: pytest.fail("constructed profile matrix"))
    payoffs = np.zeros((3, 100, 100, 10))

    with pytest.raises(module.EquilibriumAnalysisUnavailable) as captured:
        module._PreparedDistanceLP(payoffs, "cce")

    error = captured.value
    assert error.estimate.equilibrium == "cce"
    assert error.estimate.action_shape == (100, 100, 10)
    assert error.estimate.n_incentive_rows == 210
    assert error.estimate.estimated_bytes > error.budget_bytes
    assert "CCE" in str(error) and "100,000 profiles" in str(error)
