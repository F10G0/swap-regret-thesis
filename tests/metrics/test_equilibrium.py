import numpy as np
import pytest

from config import EQUILIBRIUM_LP_TOLERANCE
from experiments.games import create_rock_paper_scissors_payoffs
import metrics.equilibrium as equilibrium_module
from metrics.equilibrium import equilibrium_profile_weights, max_equilibrium_profile_weight, optimize_equilibrium


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


def maximum_incentive_gain(
    payoff_tensor: np.ndarray,
    distribution: np.ndarray,
    equilibrium: str,
) -> float:
    action_shape = distribution.shape
    gains = []
    for player, n_actions in enumerate(action_shape):
        for deviation_action in range(n_actions):
            if equilibrium == "cce":
                gain = 0.0
                for profile in np.ndindex(action_shape):
                    deviation = list(profile)
                    deviation[player] = deviation_action
                    gain += distribution[profile] * (
                        payoff_tensor[(player, *deviation)]
                        - payoff_tensor[(player, *profile)]
                    )
                gains.append(gain)
                continue

            for recommended_action in range(n_actions):
                if recommended_action == deviation_action:
                    continue
                gain = 0.0
                for profile in np.ndindex(action_shape):
                    if profile[player] != recommended_action:
                        continue
                    deviation = list(profile)
                    deviation[player] = deviation_action
                    gain += distribution[profile] * (
                        payoff_tensor[(player, *deviation)]
                        - payoff_tensor[(player, *profile)]
                    )
                gains.append(gain)
    return max(gains, default=0.0)


@pytest.mark.parametrize(
    ("equilibrium", "expected_coarse"),
    [("ce", False), ("cce", True)],
)
def test_adapter_only_maps_concept_and_returns_upstream_distribution(
    monkeypatch,
    equilibrium: str,
    expected_coarse: bool,
) -> None:
    payoffs = coordination_game()
    objective = np.array([[3.0, -1.0], [0.5, 2.0]])
    upstream_distribution = np.array(
        [[1.0, 0.0], [0.0, 0.0]]
    )
    calls = []

    def fake_get_correlated_equilibrium(
        payoff_matrix,
        coarse,
        objective,
    ):
        calls.append((payoff_matrix, coarse, objective))
        return upstream_distribution

    monkeypatch.setattr(
        equilibrium_module.games_learning_equilibrium,
        "get_correlated_equilibrium",
        fake_get_correlated_equilibrium,
    )

    result = optimize_equilibrium(payoffs, equilibrium, objective)

    assert len(calls) == 1
    forwarded_payoffs, coarse, forwarded_objective = calls[0]
    assert forwarded_payoffs is payoffs
    assert coarse is expected_coarse
    assert forwarded_objective is objective
    assert result is upstream_distribution


def test_adapter_omits_upstream_objective_when_none(monkeypatch) -> None:
    upstream_distribution = np.full((2, 2), 0.25)
    calls = []

    def fake_get_correlated_equilibrium(payoff_matrix, coarse):
        calls.append((payoff_matrix, coarse))
        return upstream_distribution

    monkeypatch.setattr(
        equilibrium_module.games_learning_equilibrium,
        "get_correlated_equilibrium",
        fake_get_correlated_equilibrium,
    )

    result = optimize_equilibrium(coordination_game(), "ce")

    assert len(calls) == 1
    assert calls[0][1] is False
    assert result is upstream_distribution


@pytest.mark.parametrize("equilibrium", ["ce", "cce"])
def test_upstream_optimizer_maximizes_arbitrary_linear_objective(
    equilibrium: str,
) -> None:
    objective = np.array([[3.0, -1.0], [0.5, 2.0]])

    distribution = optimize_equilibrium(
        coordination_game(),
        equilibrium,
        objective,
    )

    assert distribution.shape == objective.shape
    assert np.sum(objective * distribution) == pytest.approx(3.0)


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


def test_profile_weights_use_one_hot_upstream_objectives(
    monkeypatch,
) -> None:
    objectives = []

    def fake_get_correlated_equilibrium(
        payoff_matrix,
        coarse,
        objective,
    ):
        objectives.append(objective.copy())
        return objective.copy()

    monkeypatch.setattr(
        equilibrium_module.games_learning_equilibrium,
        "get_correlated_equilibrium",
        fake_get_correlated_equilibrium,
    )

    weights = equilibrium_profile_weights(coordination_game(), "cce")

    assert np.array_equal(weights, np.ones((2, 2)))
    assert len(objectives) == 4
    assert all(np.count_nonzero(objective) == 1 for objective in objectives)
    assert all(np.sum(objective) == 1.0 for objective in objectives)


def test_profile_weight_matrix_is_not_an_equilibrium_distribution() -> None:
    weights = equilibrium_profile_weights(coordination_game(), "ce")

    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[1, 1] == pytest.approx(1.0)
    assert np.sum(weights) > 1.0 + EQUILIBRIUM_LP_TOLERANCE


def test_upstream_solver_supports_heterogeneous_three_player_games() -> None:
    payoff_tensor = np.zeros((3, 2, 1, 2))
    objective = np.zeros((2, 1, 2))
    objective[1, 0, 1] = 1.0

    for equilibrium in ("ce", "cce"):
        distribution = optimize_equilibrium(
            payoff_tensor,
            equilibrium,
            objective,
        )
        assert distribution.shape == (2, 1, 2)
        assert distribution[1, 0, 1] == pytest.approx(1.0)


def test_ce_profile_weights_are_bounded_by_cce_weights() -> None:
    for payoff_tensor in (
        asymmetric_game(),
        create_rock_paper_scissors_payoffs(),
    ):
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


def test_rps_diagonal_distribution_is_cce_but_not_ce() -> None:
    payoff_tensor = create_rock_paper_scissors_payoffs()
    diagonal_distribution = np.zeros((3, 3))
    np.fill_diagonal(diagonal_distribution, 1.0 / 3.0)

    assert maximum_incentive_gain(
        payoff_tensor,
        diagonal_distribution,
        "cce",
    ) <= EQUILIBRIUM_LP_TOLERANCE
    assert maximum_incentive_gain(
        payoff_tensor,
        diagonal_distribution,
        "ce",
    ) > 0.1

    ce_weights = equilibrium_profile_weights(payoff_tensor, "ce")
    cce_weights = equilibrium_profile_weights(payoff_tensor, "cce")
    assert np.any(
        cce_weights
        > ce_weights + 10.0 * EQUILIBRIUM_LP_TOLERANCE
    )


def test_unknown_equilibrium_concept_is_rejected_locally() -> None:
    with pytest.raises(ValueError, match="unknown equilibrium concept"):
        optimize_equilibrium(coordination_game(), "nash")


def test_upstream_failures_propagate_unchanged(monkeypatch) -> None:
    error = RuntimeError("CBC unavailable")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        equilibrium_module.games_learning_equilibrium,
        "get_correlated_equilibrium",
        fail,
    )

    with pytest.raises(RuntimeError) as captured:
        optimize_equilibrium(coordination_game(), "cce")

    assert captured.value is error
