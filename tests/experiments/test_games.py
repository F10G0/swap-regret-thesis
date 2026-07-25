import numpy as np
import pytest

from experiments.games import (
    PAYOFF_FACTORIES,
    create_cyclic_dominance_payoffs,
    create_linear_bertrand_payoffs,
    create_logit_bertrand_payoffs,
    create_standard_bertrand_payoffs,
    normalize_payoffs,
)


def test_benchmark_payoffs_are_valid_two_player_games() -> None:
    for factory in PAYOFF_FACTORIES.values():
        payoffs = factory()
        assert payoffs.shape[0] == 2
        assert payoffs.ndim == 3
        assert np.all((0.0 <= payoffs) & (payoffs <= 1.0))


def test_cyclic_dominance_is_balanced() -> None:
    payoffs = create_cyclic_dominance_payoffs(5)
    assert np.all(np.sum(payoffs[0] == 1.0, axis=1) == 2)
    assert np.all(np.sum(payoffs[0] == 0.0, axis=1) == 2)
    assert np.allclose(payoffs[0] + payoffs[1], 1.0)


def test_normalize_payoffs_rejects_constant_values() -> None:
    with pytest.raises(ValueError, match="constant"):
        normalize_payoffs(np.ones((2, 2)))


def test_standard_bertrand_payoffs_have_expected_shape_and_range() -> None:
    payoffs = create_standard_bertrand_payoffs(
        n_actions=5,
        cost=(0.0, 0.0),
        interval=(0.1, 1.0),
    )

    assert payoffs.shape == (2, 5, 5)
    assert np.all(np.isfinite(payoffs))
    assert np.all((0.0 <= payoffs) & (payoffs <= 1.0))


def test_standard_bertrand_is_symmetric_with_equal_costs() -> None:
    payoffs = create_standard_bertrand_payoffs(
        n_actions=7,
        cost=(0.2, 0.2),
        interval=(0.1, 1.0),
        maximum_demand=2.0,
    )

    assert np.allclose(payoffs[0], payoffs[1].T)


def test_standard_bertrand_matches_raw_profit_rules_before_normalization() -> None:
    n_actions = 3
    maximum_demand = 2.0
    prices = np.linspace(0.2, 1.0, n_actions)
    expected_player_0 = np.zeros((n_actions, n_actions))
    expected_player_1 = np.zeros((n_actions, n_actions))
    for action_0, price_0 in enumerate(prices):
        for action_1, price_1 in enumerate(prices):
            if price_0 < price_1:
                demand_0, demand_1 = maximum_demand, 0.0
            elif price_1 < price_0:
                demand_0, demand_1 = 0.0, maximum_demand
            else:
                demand_0 = demand_1 = maximum_demand / 2.0
            expected_player_0[action_0, action_1] = price_0 * demand_0
            expected_player_1[action_0, action_1] = price_1 * demand_1

    payoffs = create_standard_bertrand_payoffs(
        n_actions=n_actions,
        cost=(0.0, 0.0),
        interval=(0.2, 1.0),
        maximum_demand=maximum_demand,
    )

    assert expected_player_0[0, 1] > 0.0
    assert expected_player_1[0, 1] == 0.0
    assert expected_player_0[1, 1] == pytest.approx(prices[1])
    assert expected_player_1[1, 1] == pytest.approx(prices[1])
    assert np.allclose(payoffs[0], normalize_payoffs(expected_player_0))
    assert np.allclose(payoffs[1], normalize_payoffs(expected_player_1))


def test_o1_bertrand_factory_has_21_actions_per_player() -> None:
    payoffs = PAYOFF_FACTORIES["bertrand_standard_o1"]()

    assert payoffs.shape == (2, 21, 21)


@pytest.mark.parametrize(
    "factory_name",
    [
        "bertrand_linear_o2",
        "bertrand_logit_o3",
        "bertrand_linear_o2_prime",
        "bertrand_logit_o3_prime",
    ],
)
def test_additional_bertrand_factories_have_21_actions_per_player(
    factory_name: str,
) -> None:
    payoffs = PAYOFF_FACTORIES[factory_name]()

    assert payoffs.shape == (2, 21, 21)
    assert np.all(np.isfinite(payoffs))
    assert np.all((0.0 <= payoffs) & (payoffs <= 1.0))


@pytest.mark.parametrize(
    "factory_name",
    ["bertrand_linear_o2", "bertrand_logit_o3"],
)
def test_symmetric_bertrand_variants_are_symmetric(factory_name: str) -> None:
    payoffs = PAYOFF_FACTORIES[factory_name]()

    assert np.allclose(payoffs[0], payoffs[1].T)


@pytest.mark.parametrize(
    "factory_name",
    ["bertrand_linear_o2_prime", "bertrand_logit_o3_prime"],
)
def test_asymmetric_bertrand_variants_are_genuinely_asymmetric(
    factory_name: str,
) -> None:
    payoffs = PAYOFF_FACTORIES[factory_name]()

    assert not np.allclose(payoffs[0], payoffs[1].T)


def test_linear_bertrand_matches_unclipped_demand_equation() -> None:
    n_actions = 3
    cost = (0.1, 0.2)
    alpha = (0.48, 0.6)
    beta = (0.9, 0.8)
    gamma = 0.6
    prices = np.linspace(0.0, 1.0, n_actions)
    prices_0 = prices[:, None]
    prices_1 = prices[None, :]
    demand_0 = alpha[0] - beta[0] * prices_0 + gamma * prices_1
    demand_1 = alpha[1] - beta[1] * prices_1 + gamma * prices_0
    expected_player_0 = (prices_0 - cost[0]) * demand_0
    expected_player_1 = (prices_1 - cost[1]) * demand_1

    payoffs = create_linear_bertrand_payoffs(
        n_actions=n_actions,
        cost=cost,
        interval=(0.0, 1.0),
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    assert demand_0[-1, 0] < 0.0
    assert np.allclose(payoffs[0], normalize_payoffs(expected_player_0))
    assert np.allclose(payoffs[1], normalize_payoffs(expected_player_1))


def test_logit_bertrand_matches_outside_good_demand_equation() -> None:
    n_actions = 3
    cost = (0.5, 1.0)
    alpha = (1.5, 2.0)
    mu = 0.25
    alpha0 = 0.0
    prices = np.linspace(0.5, 2.0, n_actions)
    prices_0 = prices[:, None]
    prices_1 = prices[None, :]
    exp_0 = np.exp((alpha[0] - prices_0) / mu)
    exp_1 = np.exp((alpha[1] - prices_1) / mu)
    exp_outside = np.exp(alpha0 / mu)
    denominator = exp_0 + exp_1 + exp_outside
    demand_0 = exp_0 / denominator
    demand_1 = exp_1 / denominator
    expected_player_0 = (prices_0 - cost[0]) * demand_0
    expected_player_1 = (prices_1 - cost[1]) * demand_1

    payoffs = create_logit_bertrand_payoffs(
        n_actions=n_actions,
        cost=cost,
        interval=(0.5, 2.0),
        alpha=alpha,
        mu=mu,
        alpha0=alpha0,
    )

    assert np.all(demand_0 + demand_1 < 1.0)
    assert np.allclose(payoffs[0], normalize_payoffs(expected_player_0))
    assert np.allclose(payoffs[1], normalize_payoffs(expected_player_1))


def test_logit_bertrand_softmax_is_numerically_stable() -> None:
    payoffs = create_logit_bertrand_payoffs(
        n_actions=5,
        cost=(0.0, 0.0),
        interval=(1.0, 2.0),
        alpha=(1000.0, 1000.0),
        mu=0.01,
        alpha0=999.0,
    )

    assert np.all(np.isfinite(payoffs))
    assert np.all((0.0 <= payoffs) & (payoffs <= 1.0))


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
        ({"maximum_demand": np.inf}, "finite"),
        ({"maximum_demand": 1e-300}, "constant"),
    ],
)
def test_standard_bertrand_rejects_invalid_configuration(
    kwargs: dict,
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
        ({"alpha": (0.48,)}, "alpha"),
        ({"alpha": (0.48, np.nan)}, "finite numeric"),
        ({"alpha": 0.0}, "positive"),
        ({"beta": (0.9,)}, "beta"),
        ({"beta": 0.0}, "positive"),
        ({"gamma": -0.1}, "non-negative"),
        ({"gamma": 0.9}, "less than"),
        ({"gamma": np.inf}, "finite"),
    ],
)
def test_linear_bertrand_rejects_invalid_demand_parameters(
    kwargs: dict,
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
        ({"alpha": (2.0,)}, "alpha"),
        ({"alpha": (2.0, np.nan)}, "finite numeric"),
        ({"alpha": (0.0, 2.0)}, "positive"),
        ({"mu": 0.0}, "positive"),
        ({"mu": -0.25}, "positive"),
        ({"mu": np.nan}, "finite"),
        ({"alpha0": np.inf}, "finite"),
    ],
)
def test_logit_bertrand_rejects_invalid_demand_parameters(
    kwargs: dict,
    message: str,
) -> None:
    parameters = {
        "n_actions": 3,
        "cost": (1.0, 1.0),
        "interval": (1.0, 2.0),
        "alpha": (2.0, 2.0),
        "mu": 0.25,
        "alpha0": 0.0,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=message):
        create_logit_bertrand_payoffs(**parameters)
