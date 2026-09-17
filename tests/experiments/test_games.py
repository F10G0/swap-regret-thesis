import numpy as np
import pytest

from experiments.games import (
    PAYOFF_FACTORIES,
    create_matching_pennies_payoffs,
    create_rock_paper_scissors_payoffs,
    create_rock_paper_scissors_lizard_spock_payoffs,
    normalize_payoffs,
)


def test_benchmark_payoffs_are_valid_two_player_games() -> None:
    for factory in PAYOFF_FACTORIES.values():
        payoffs = factory()
        assert payoffs.shape[0] == 2
        assert payoffs.ndim == 3
        assert np.all(np.isfinite(payoffs))
        assert np.all((0.0 <= payoffs) & (payoffs <= 1.0))


def test_rps_exact_payoff_tensor_and_digest_regression() -> None:
    from experiments.game_catalog import payoff_tensor_digest

    # Frozen from the pre-cleanup tensor, not reconstructed through the factory.
    expected = np.array([
        [[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]],
        [[0.5, 1.0, 0.0], [0.0, 0.5, 1.0], [1.0, 0.0, 0.5]],
    ])
    actual = create_rock_paper_scissors_payoffs()
    assert actual.dtype == expected.dtype == np.dtype("float64")
    np.testing.assert_array_equal(actual, expected)
    assert payoff_tensor_digest(actual) == "c5cb3962e79cda89c5a4368fd72583eff934ad574fb1d70426bd0d1cea9b5879"


def test_rps_player_and_action_ordering() -> None:
    payoffs = create_rock_paper_scissors_payoffs()
    rock, paper, scissors = range(3)
    # Entries are indexed by (player, player-0 action, player-1 action).
    for winner, loser in ((rock, scissors), (paper, rock), (scissors, paper)):
        np.testing.assert_array_equal(payoffs[:, winner, loser], [1.0, 0.0])
        np.testing.assert_array_equal(payoffs[:, loser, winner], [0.0, 1.0])
    for action in (rock, paper, scissors):
        np.testing.assert_array_equal(payoffs[:, action, action], [0.5, 0.5])


def test_rpsls_exact_payoff_tensor_and_digest_regression() -> None:
    from experiments.game_catalog import payoff_tensor_digest

    expected = np.array([
        [[0.5, 0, 1, 1, 0], [1, 0.5, 0, 0, 1], [0, 1, 0.5, 1, 0],
         [0, 1, 0, 0.5, 1], [1, 0, 1, 0, 0.5]],
        [[0.5, 1, 0, 0, 1], [0, 0.5, 1, 1, 0], [1, 0, 0.5, 0, 1],
         [1, 0, 1, 0.5, 0], [0, 1, 0, 1, 0.5]],
    ])
    actual = create_rock_paper_scissors_lizard_spock_payoffs()
    assert actual.dtype == expected.dtype == np.dtype("float64")
    np.testing.assert_array_equal(actual, expected)
    assert payoff_tensor_digest(actual) == "1831eca45bc7c7b5221f2cca85d4b722d9e299493e9f1ebd5c797d4025edf2bd"


def test_literature_benchmark_suite_has_only_role_driven_games() -> None:
    assert set(PAYOFF_FACTORIES) == {
        "matching_pennies",
        "rps",
        "rpsls",
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


def test_matching_pennies_exact_payoff_tensor_and_registry() -> None:
    from experiments.game_catalog import load_game_payoffs, payoff_tensor_digest
    from web.presentations import GAME_PRESENTATIONS

    raw_player_0 = np.array([[1.0, -1.0], [-1.0, 1.0]])
    expected = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
    payoffs = create_matching_pennies_payoffs()
    assert payoffs.dtype == expected.dtype == np.dtype("float64")
    np.testing.assert_array_equal(payoffs, expected)
    np.testing.assert_array_equal(2 * payoffs[0] - 1, raw_player_0)
    np.testing.assert_array_equal(2 * payoffs[1] - 1, -raw_player_0)
    np.testing.assert_array_equal(load_game_payoffs("matching_pennies"), expected)
    assert GAME_PRESENTATIONS["matching_pennies"]["label"] == "Matching Pennies"
    assert payoff_tensor_digest(payoffs) == "a4bd8e91cb26bb481ea53925b8c545895b027d61ea21e43c4bd55225d4e9c803"


def test_normalize_payoffs_rejects_constant_values() -> None:
    with pytest.raises(ValueError, match="constant"):
        normalize_payoffs(np.ones((2, 2)))


@pytest.mark.parametrize("game_name", [
    "bertrand_standard_o1", "bertrand_linear_o2", "bertrand_logit_o3",
    "bertrand_linear_o2_prime", "bertrand_logit_o3_prime",
])
def test_retired_game_identifiers_are_not_supported(tmp_path, game_name) -> None:
    from experiments.game_catalog import GameCatalog
    from web.presentations import GAME_PRESENTATIONS

    catalog = GameCatalog(tmp_path)
    assert set(catalog.definitions()) == {"matching_pennies", "rps", "rpsls"}
    assert game_name not in PAYOFF_FACTORIES
    assert game_name not in GAME_PRESENTATIONS
    with pytest.raises(ValueError, match="unknown game"):
        catalog.load(game_name)
