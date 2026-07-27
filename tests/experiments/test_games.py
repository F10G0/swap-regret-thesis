from games_learning.game.econ_game import (
    BertrandLinear,
    BertrandLogit,
    BertrandStandard,
)
from games_learning.game.matrix_game import ExampleMatrixGames
import numpy as np
import pytest

from experiments.games import (
    PAYOFF_FACTORIES,
    create_linear_bertrand_payoffs,
    create_logit_bertrand_payoffs,
    create_matching_pennies_payoffs,
    create_rock_paper_scissors_payoffs,
    create_rock_paper_scissors_lizard_spock_payoffs,
    create_standard_bertrand_payoffs,
    normalize_payoffs,
)


def normalized_upstream_payoffs(game) -> np.ndarray:
    return np.stack(
        [
            normalize_payoffs(game.payoff_matrix[player])
            for player in range(game.n_agents)
        ]
    )


def test_benchmark_payoffs_are_valid_two_player_games() -> None:
    for factory in PAYOFF_FACTORIES.values():
        payoffs = factory()
        assert payoffs.shape[0] == 2
        assert payoffs.ndim == 3
        assert np.all(np.isfinite(payoffs))
        assert np.all((0.0 <= payoffs) & (payoffs <= 1.0))


def test_rps_is_sourced_from_games_learning() -> None:
    upstream = ExampleMatrixGames("rock_paper_scissors")

    assert np.array_equal(
        create_rock_paper_scissors_payoffs(),
        normalized_upstream_payoffs(upstream),
    )


def test_literature_benchmark_suite_has_only_role_driven_games() -> None:
    assert set(PAYOFF_FACTORIES) == {
        "rps",
        "rpsls",
        "matching_pennies",
        "bertrand_standard_o1",
        "bertrand_linear_o2",
        "bertrand_logit_o3",
        "bertrand_linear_o2_prime",
        "bertrand_logit_o3_prime",
    }


def test_rpsls_is_balanced_symmetric_zero_sum_equivalent() -> None:
    payoffs = create_rock_paper_scissors_lizard_spock_payoffs()

    assert payoffs.shape == (2, 5, 5)
    assert np.all(np.diag(payoffs[0]) == 0.5)
    assert np.all(np.sum(payoffs[0] == 1.0, axis=1) == 2)
    assert np.all(np.sum(payoffs[0] == 0.0, axis=1) == 2)
    assert np.allclose(payoffs[0], payoffs[1].T)
    assert payoffs[0, 0, 2] == 1.0
    assert payoffs[0, 3, 4] == 1.0
    assert payoffs[0, 4, 1] == 0.0


def test_matching_pennies_is_sourced_from_games_learning() -> None:
    upstream = ExampleMatrixGames("matching_pennies")
    payoffs = create_matching_pennies_payoffs()

    assert np.array_equal(payoffs, normalized_upstream_payoffs(upstream))
    assert payoffs.shape == (2, 2, 2)
    assert np.allclose(payoffs[0] + payoffs[1], 1.0)
    assert not np.allclose(payoffs[0], payoffs[1].T)


def test_normalize_payoffs_rejects_constant_values() -> None:
    with pytest.raises(ValueError, match="constant"):
        normalize_payoffs(np.ones((2, 2)))


def test_standard_bertrand_matches_games_learning() -> None:
    parameters = {
        "n_actions": 5,
        "cost": (0.1, 0.2),
        "interval": (0.1, 1.0),
        "maximum_demand": 2.0,
    }
    upstream = BertrandStandard(
        n_agents=2,
        n_discr=parameters["n_actions"],
        cost=parameters["cost"],
        interval=parameters["interval"],
        maximum_demand=parameters["maximum_demand"],
    )

    payoffs = create_standard_bertrand_payoffs(**parameters)

    assert payoffs.shape == (2, 5, 5)
    assert np.allclose(payoffs, normalized_upstream_payoffs(upstream))


def test_standard_bertrand_uses_upstream_price_sensitive_demand() -> None:
    maximum_demand = 2.0
    game = BertrandStandard(
        n_agents=2,
        n_discr=3,
        cost=(0.0, 0.0),
        interval=(0.2, 1.0),
        maximum_demand=maximum_demand,
    )
    prices = game.actions

    lower_price_profit = prices[0] * maximum_demand * (1.0 - prices[0])
    tied_price_profit = (
        prices[1]
        * maximum_demand
        * (1.0 - prices[1])
        / 2.0
    )
    assert game.payoff_matrix[0, 0, 1] == pytest.approx(
        lower_price_profit
    )
    assert game.payoff_matrix[1, 0, 1] == pytest.approx(0.0)
    assert game.payoff_matrix[0, 1, 1] == pytest.approx(
        tied_price_profit
    )
    assert game.payoff_matrix[1, 1, 1] == pytest.approx(
        tied_price_profit
    )
    assert game.payoff_matrix[0, 2, 2] == pytest.approx(0.0)
    assert game.payoff_matrix[1, 2, 2] == pytest.approx(0.0)


def test_linear_bertrand_matches_games_learning() -> None:
    parameters = {
        "n_actions": 4,
        "cost": (0.1, 0.2),
        "interval": (0.0, 1.0),
        "alpha": (0.48, 0.6),
        "beta": (0.9, 0.8),
        "gamma": 0.6,
    }
    upstream = BertrandLinear(
        n_agents=2,
        n_discr=parameters["n_actions"],
        cost=parameters["cost"],
        interval=parameters["interval"],
        alpha=parameters["alpha"],
        beta=parameters["beta"],
        gamma=parameters["gamma"],
    )

    payoffs = create_linear_bertrand_payoffs(**parameters)

    assert np.allclose(payoffs, normalized_upstream_payoffs(upstream))


def test_logit_bertrand_matches_games_learning() -> None:
    parameters = {
        "n_actions": 4,
        "cost": (0.5, 1.0),
        "interval": (0.5, 2.0),
        "alpha": (1.5, 2.0),
        "mu": (0.25, 0.3),
    }
    upstream = BertrandLogit(
        n_agents=2,
        n_discr=parameters["n_actions"],
        cost=parameters["cost"],
        interval=parameters["interval"],
        alpha=parameters["alpha"],
        mu=parameters["mu"],
    )

    payoffs = create_logit_bertrand_payoffs(**parameters)

    assert np.allclose(payoffs, normalized_upstream_payoffs(upstream))


def test_standard_bertrand_is_symmetric_with_equal_costs() -> None:
    payoffs = create_standard_bertrand_payoffs(
        n_actions=7,
        cost=(0.2, 0.2),
        interval=(0.1, 1.0),
        maximum_demand=2.0,
    )

    assert np.allclose(payoffs[0], payoffs[1].T)


@pytest.mark.parametrize(
    "factory_name",
    [
        "bertrand_standard_o1",
        "bertrand_linear_o2",
        "bertrand_logit_o3",
        "bertrand_linear_o2_prime",
        "bertrand_logit_o3_prime",
    ],
)
def test_bertrand_factories_have_21_actions_per_player(
    factory_name: str,
) -> None:
    payoffs = PAYOFF_FACTORIES[factory_name]()

    assert payoffs.shape == (2, 21, 21)


@pytest.mark.parametrize(
    "factory_name",
    [
        "bertrand_standard_o1",
        "bertrand_linear_o2",
        "bertrand_logit_o3",
    ],
)
def test_symmetric_bertrand_variants_are_symmetric(
    factory_name: str,
) -> None:
    payoffs = PAYOFF_FACTORIES[factory_name]()

    assert np.allclose(payoffs[0], payoffs[1].T)


@pytest.mark.parametrize(
    "factory_name",
    [
        "bertrand_linear_o2_prime",
        "bertrand_logit_o3_prime",
    ],
)
def test_asymmetric_bertrand_variants_are_genuinely_asymmetric(
    factory_name: str,
) -> None:
    payoffs = PAYOFF_FACTORIES[factory_name]()

    assert not np.allclose(payoffs[0], payoffs[1].T)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_actions": 1}, "greater than 1"),
        ({"n_actions": 2.5}, "integer"),
        ({"cost": (0.0,)}, "cost"),
        ({"cost": (0.0, 0.0, 0.0)}, "cost"),
        ({"cost": "00"}, "cost"),
        ({"cost": (0.0, np.nan)}, "finite numeric"),
        ({"interval": (0.0,)}, "interval"),
        ({"interval": (0.0, 1.0, 2.0)}, "interval"),
        ({"interval": (1.0, 1.0)}, "less than"),
        ({"interval": (2.0, 1.0)}, "less than"),
        ({"interval": (0.0, np.inf)}, "finite numeric"),
        ({"maximum_demand": 0.0}, "positive"),
        ({"maximum_demand": -1.0}, "positive"),
        ({"maximum_demand": np.nan}, "finite"),
    ],
)
def test_standard_bertrand_rejects_invalid_configuration(
    kwargs,
    message: str,
) -> None:
    parameters = {
        "n_actions": 3,
        "cost": (0.0, 0.0),
        "interval": (0.1, 1.0),
        "maximum_demand": 1.0,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=message):
        create_standard_bertrand_payoffs(**parameters)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": np.nan}, "alpha"),
        ({"beta": 0.0}, "beta"),
        ({"beta": (0.9, np.inf)}, "beta"),
        ({"gamma": -0.1}, "non-negative"),
        ({"gamma": 0.9}, "less than"),
        ({"gamma": np.nan}, "finite"),
    ],
)
def test_linear_bertrand_rejects_invalid_demand_parameters(
    kwargs,
    message: str,
) -> None:
    parameters = {
        "n_actions": 3,
        "cost": (0.0, 0.0),
        "interval": (0.0, 1.0),
        "alpha": 0.48,
        "beta": 0.9,
        "gamma": 0.6,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=message):
        create_linear_bertrand_payoffs(**parameters)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": (2.0, np.inf)}, "alpha"),
        ({"mu": 0.0}, "mu"),
        ({"mu": (0.25, -0.1)}, "mu"),
        ({"mu": np.nan}, "mu"),
        ({"alpha0": 1.0}, "must be zero"),
        ({"alpha0": np.nan}, "finite"),
    ],
)
def test_logit_bertrand_rejects_invalid_demand_parameters(
    kwargs,
    message: str,
) -> None:
    parameters = {
        "n_actions": 3,
        "cost": (1.0, 1.0),
        "interval": (1.0, 2.0),
        "alpha": (2.0, 2.0),
        "mu": 0.25,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=message):
        create_logit_bertrand_payoffs(**parameters)
