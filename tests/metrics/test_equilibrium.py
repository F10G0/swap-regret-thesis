from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog

from config import EQUILIBRIUM_LP_TOLERANCE
from metrics import (
    EquilibriumOptimizationError,
    build_cce_constraints,
    build_ce_constraints,
    equilibrium_profile_weights,
    joint_action_profiles,
    max_equilibrium_profile_weight,
)
import metrics.equilibrium as equilibrium_module
from experiments.games import (
    PAYOFF_FACTORIES,
    create_rock_paper_scissors_payoffs,
)


def asymmetric_game() -> np.ndarray:
    return np.array(
        [
            [[3.0, 0.0], [1.0, 2.0]],
            [[1.0, 4.0], [3.0, 0.0]],
        ]
    )


def coordination_game() -> np.ndarray:
    payoffs = np.array([[1.0, 0.0], [0.0, 1.0]])
    return np.stack((payoffs, payoffs))


def test_joint_action_profiles_use_numpy_c_order() -> None:
    assert list(joint_action_profiles((2, 3))) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]


def test_cce_constraints_encode_unconditional_fixed_deviations() -> None:
    constraints, bounds = build_cce_constraints(asymmetric_game())

    assert constraints.shape == (4, 4)
    assert np.array_equal(bounds, np.zeros(4))
    assert np.array_equal(constraints[1], [-2.0, 2.0, 0.0, 0.0])
    assert np.array_equal(constraints[2], [0.0, -3.0, 0.0, 3.0])


def test_ce_constraints_condition_on_the_recommended_action() -> None:
    constraints, bounds = build_ce_constraints(asymmetric_game())

    assert constraints.shape == (4, 4)
    assert np.array_equal(bounds, np.zeros(4))
    assert np.array_equal(constraints[0], [-2.0, 2.0, 0.0, 0.0])
    assert np.array_equal(constraints[1], [0.0, 0.0, 2.0, -2.0])
    assert np.array_equal(constraints[2], [3.0, 0.0, -3.0, 0.0])
    assert np.array_equal(constraints[3], [0.0, -3.0, 0.0, 3.0])


@pytest.mark.parametrize("builder", [build_ce_constraints, build_cce_constraints])
def test_constraint_system_has_a_feasible_probability_distribution(
    builder,
) -> None:
    constraints, bounds = builder(asymmetric_game())
    n_profiles = constraints.shape[1]
    result = linprog(
        np.zeros(n_profiles),
        A_ub=constraints,
        b_ub=bounds,
        A_eq=np.ones((1, n_profiles)),
        b_eq=np.ones(1),
        bounds=[(0.0, None)] * n_profiles,
        method="highs",
    )

    assert result.success
    assert np.min(result.x) >= -EQUILIBRIUM_LP_TOLERANCE
    assert np.sum(result.x) == pytest.approx(
        1.0,
        abs=EQUILIBRIUM_LP_TOLERANCE,
    )
    assert np.max(constraints @ result.x - bounds) <= (
        EQUILIBRIUM_LP_TOLERANCE
    )


@pytest.mark.parametrize("equilibrium", ["ce", "cce"])
def test_pure_nash_equilibrium_can_receive_probability_one(
    equilibrium: str,
) -> None:
    weight = max_equilibrium_profile_weight(
        coordination_game(),
        (0, 0),
        equilibrium,
    )

    assert weight == pytest.approx(1.0, abs=EQUILIBRIUM_LP_TOLERANCE)


def test_profile_weight_matrix_is_not_an_equilibrium_distribution() -> None:
    weights = equilibrium_profile_weights(coordination_game(), "ce")

    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[1, 1] == pytest.approx(1.0)
    assert np.sum(weights) > 1.0 + EQUILIBRIUM_LP_TOLERANCE


def test_solver_accepts_numpy_compatible_payoff_input() -> None:
    weights = equilibrium_profile_weights(
        coordination_game().tolist(),
        "cce",
    )

    assert weights.shape == (2, 2)
    assert weights[0, 0] == pytest.approx(1.0)


def test_solver_supports_heterogeneous_three_player_games() -> None:
    payoff_tensor = np.zeros((3, 2, 1, 2))

    ce_weights = equilibrium_profile_weights(payoff_tensor, "ce")
    cce_weights = equilibrium_profile_weights(payoff_tensor, "cce")

    assert ce_weights.shape == (2, 1, 2)
    assert cce_weights.shape == (2, 1, 2)
    assert np.allclose(ce_weights, 1.0)
    assert np.allclose(cce_weights, 1.0)


@pytest.mark.parametrize("factory", PAYOFF_FACTORIES.values())
def test_ce_profile_weights_are_bounded_by_cce_weights(factory) -> None:
    payoff_tensor = factory()
    original = payoff_tensor.copy()

    ce_weights = equilibrium_profile_weights(payoff_tensor, "ce")
    cce_weights = equilibrium_profile_weights(payoff_tensor, "cce")

    assert ce_weights.shape == payoff_tensor.shape[1:]
    assert cce_weights.shape == payoff_tensor.shape[1:]
    assert np.all(ce_weights >= -EQUILIBRIUM_LP_TOLERANCE)
    assert np.all(ce_weights <= 1.0 + EQUILIBRIUM_LP_TOLERANCE)
    assert np.all(cce_weights >= -EQUILIBRIUM_LP_TOLERANCE)
    assert np.all(cce_weights <= 1.0 + EQUILIBRIUM_LP_TOLERANCE)
    assert np.all(
        ce_weights <= cce_weights + EQUILIBRIUM_LP_TOLERANCE
    )
    assert np.array_equal(payoff_tensor, original)


def test_rps_has_a_cce_that_is_not_a_ce() -> None:
    payoff_tensor = create_rock_paper_scissors_payoffs()
    cce_constraints, cce_bounds = build_cce_constraints(payoff_tensor)
    ce_constraints, ce_bounds = build_ce_constraints(payoff_tensor)
    diagonal_distribution = np.zeros(9)
    diagonal_distribution[[0, 4, 8]] = 1.0 / 3.0

    assert np.max(
        cce_constraints @ diagonal_distribution - cce_bounds
    ) <= EQUILIBRIUM_LP_TOLERANCE
    assert np.max(
        ce_constraints @ diagonal_distribution - ce_bounds
    ) > 0.1

    ce_weights = equilibrium_profile_weights(payoff_tensor, "ce")
    cce_weights = equilibrium_profile_weights(payoff_tensor, "cce")
    assert np.any(
        cce_weights
        > ce_weights + 10.0 * EQUILIBRIUM_LP_TOLERANCE
    )


@pytest.mark.parametrize(
    "payoff_tensor, message",
    [
        ([1.0, 2.0], "must have shape"),
        (np.zeros((2, 2, 2, 2)), "number of players"),
        (np.empty((2, 0, 2)), "at least one action"),
        (np.full((2, 2, 2), np.nan), "finite"),
        (np.full((2, 2, 2), 1.0 + 1.0j), "real numeric"),
        ([["not numeric"]], "real numeric"),
    ],
)
def test_equilibrium_analysis_rejects_invalid_payoff_tensors(
    payoff_tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        equilibrium_profile_weights(payoff_tensor)


def test_equilibrium_analysis_rejects_invalid_concept_and_target() -> None:
    payoffs = coordination_game()
    with pytest.raises(ValueError, match="unknown equilibrium concept"):
        equilibrium_profile_weights(payoffs, "nash")
    with pytest.raises(ValueError, match="one action per player"):
        max_equilibrium_profile_weight(payoffs, (0,), "ce")
    with pytest.raises(ValueError, match="invalid for player 1"):
        max_equilibrium_profile_weight(payoffs, (0, 2), "cce")


def test_solver_failure_reports_concept_profile_status_and_message(
    monkeypatch,
) -> None:
    failed_result = SimpleNamespace(
        success=False,
        status=2,
        message="model is infeasible",
    )
    monkeypatch.setattr(
        equilibrium_module,
        "linprog",
        lambda *args, **kwargs: failed_result,
    )

    with pytest.raises(EquilibriumOptimizationError) as captured:
        max_equilibrium_profile_weight(
            coordination_game(),
            (1, 0),
            "cce",
        )

    message = str(captured.value)
    assert "CCE" in message
    assert "(1, 0)" in message
    assert "status 2" in message
    assert "model is infeasible" in message


def test_successful_solver_output_is_still_validated(monkeypatch) -> None:
    invalid_result = SimpleNamespace(
        success=True,
        status=0,
        message="optimal",
        x=np.zeros(4),
    )
    monkeypatch.setattr(
        equilibrium_module,
        "linprog",
        lambda *args, **kwargs: invalid_result,
    )

    with pytest.raises(
        EquilibriumOptimizationError,
        match="normalization residual",
    ):
        max_equilibrium_profile_weight(
            coordination_game(),
            (0, 0),
            "ce",
        )


@pytest.mark.parametrize(
    ("distribution", "message"),
    [
        (
            np.array(
                [
                    -2.0 * EQUILIBRIUM_LP_TOLERANCE,
                    1.0 + 2.0 * EQUILIBRIUM_LP_TOLERANCE,
                    0.0,
                    0.0,
                ]
            ),
            "probability bounds",
        ),
        (np.array([0.0, 1.0, 0.0, 0.0]), "incentive residual"),
    ],
)
def test_post_solve_validation_rejects_bound_and_incentive_violations(
    monkeypatch,
    distribution: np.ndarray,
    message: str,
) -> None:
    invalid_result = SimpleNamespace(
        success=True,
        status=0,
        message="optimal",
        x=distribution,
    )
    monkeypatch.setattr(
        equilibrium_module,
        "linprog",
        lambda *args, **kwargs: invalid_result,
    )

    with pytest.raises(EquilibriumOptimizationError, match=message):
        max_equilibrium_profile_weight(
            coordination_game(),
            (0, 1),
            "ce",
        )


def test_post_solve_validation_tolerates_small_lp_residuals(
    monkeypatch,
) -> None:
    small_error = EQUILIBRIUM_LP_TOLERANCE / 2.0
    nearly_feasible_result = SimpleNamespace(
        success=True,
        status=0,
        message="optimal",
        x=np.array([1.0 + small_error, -small_error, 0.0, 0.0]),
    )
    monkeypatch.setattr(
        equilibrium_module,
        "linprog",
        lambda *args, **kwargs: nearly_feasible_result,
    )

    weight = max_equilibrium_profile_weight(
        coordination_game(),
        (0, 0),
        "ce",
    )

    assert weight == pytest.approx(1.0 + small_error)
