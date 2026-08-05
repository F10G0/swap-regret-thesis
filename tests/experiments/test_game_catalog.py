import json

import numpy as np
import pytest

from experiments.game_catalog import (
    CUSTOM_GAME_FORMAT_VERSION,
    CUSTOM_GAME_PREFIX,
    GameCatalog,
)
from experiments.plots.plot_regret import plot_selected_results
from experiments.scenarios.full_information_cross_play import run_full_information_cross_play_experiment
from tests.support import read_csv_rows


def test_custom_game_round_trip_is_reproducible(tmp_path) -> None:
    catalog = GameCatalog(tmp_path)

    definition = catalog.create_random("Three Player Test", 3, [2, 3, 2], 17)
    payoffs = catalog.load(definition.id)

    assert definition.id == f"{CUSTOM_GAME_PREFIX}three-player-test"
    assert definition.action_counts == (2, 3, 2)
    assert definition.payoff_structure == "general_sum"
    assert payoffs.shape == (3, 2, 3, 2)
    assert np.array_equal(payoffs, np.random.default_rng(17).random((3, 2, 3, 2)))
    assert np.all((0.0 <= payoffs) & (payoffs <= 1.0))
    assert (tmp_path / "three-player-test.npz").is_file()
    assert catalog.custom_path(definition.id) == tmp_path / "three-player-test.npz"


def test_two_player_zero_sum_game_is_reproducible_symmetric_and_constant_sum(
    tmp_path,
) -> None:
    catalog = GameCatalog(tmp_path)

    definition = catalog.create_random(
        "Zero Sum Test",
        2,
        [3, 3],
        17,
        "zero_sum",
    )
    payoffs = catalog.load(definition.id)
    sample = np.random.default_rng(17).random((3, 3))
    expected_first_payoff = 0.5 + (sample - sample.T) / 2.0

    assert definition.payoff_structure == "zero_sum"
    assert definition.action_counts == (3, 3)
    assert payoffs.shape == (2, 3, 3)
    assert np.array_equal(payoffs[0], expected_first_payoff)
    assert np.array_equal(payoffs[1], 1.0 - expected_first_payoff)
    assert np.array_equal(payoffs[0], payoffs[1].T)
    assert np.array_equal(np.diag(payoffs[0]), np.full(3, 0.5))
    assert np.allclose(payoffs.sum(axis=0), 1.0, rtol=0.0, atol=1e-12)
    assert np.allclose(payoffs[0] + payoffs[0].T, 1.0, rtol=0.0, atol=1e-12)


def test_version_one_custom_game_defaults_to_general_sum(tmp_path) -> None:
    payoff_tensor = np.random.default_rng(4).random((2, 2, 2))
    np.savez_compressed(
        tmp_path / "legacy.npz",
        format_version=np.array(1),
        name=np.array("Legacy"),
        slug=np.array("legacy"),
        seed=np.array(4),
        action_counts=np.array([2, 2]),
        payoff_tensor=payoff_tensor,
    )

    definition = GameCatalog(tmp_path).definitions()[f"{CUSTOM_GAME_PREFIX}legacy"]

    assert definition.payoff_structure == "general_sum"
    assert np.array_equal(GameCatalog(tmp_path).load(definition.id), payoff_tensor)


def test_asymmetric_zero_sum_file_is_rejected(tmp_path) -> None:
    first_payoff = np.array([[0.5, 0.8], [0.3, 0.5]])
    np.savez_compressed(
        tmp_path / "asymmetric.npz",
        format_version=np.array(CUSTOM_GAME_FORMAT_VERSION),
        name=np.array("Asymmetric"),
        slug=np.array("asymmetric"),
        seed=np.array(4),
        payoff_structure=np.array("zero_sum"),
        action_counts=np.array([2, 2]),
        payoff_tensor=np.stack((first_payoff, 1.0 - first_payoff)),
    )

    with pytest.raises(ValueError, match="symmetric and have constant sum"):
        GameCatalog(tmp_path).load(f"{CUSTOM_GAME_PREFIX}asymmetric")


@pytest.mark.parametrize(
    ("name", "n_players", "action_counts", "payoff_structure", "message"),
    [
        ("bad/name", 3, [2, 2, 2], "general_sum", "game name"),
        ("one", 1, [2], "general_sum", "between 2"),
        ("missing", 3, [2, 2], "general_sum", "one action count"),
        ("too-large", 8, [100] * 8, "general_sum", "at most"),
        ("three-player-zero-sum", 3, [2, 2, 2], "zero_sum", "two equal action sets"),
        ("unequal-zero-sum", 2, [2, 3], "zero_sum", "two equal action sets"),
        ("unknown-structure", 2, [2, 2], "cooperative", "payoff structure"),
    ],
)
def test_custom_game_validation_rejects_unsafe_configurations(
    tmp_path,
    name,
    n_players,
    action_counts,
    payoff_structure,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        GameCatalog(tmp_path).create_random(
            name,
            n_players,
            action_counts,
            0,
            payoff_structure,
        )


def test_custom_game_name_cannot_be_overwritten(tmp_path) -> None:
    catalog = GameCatalog(tmp_path)
    catalog.create_random("same", 2, [2, 2], 0)

    with pytest.raises(FileExistsError, match="already exists"):
        catalog.create_random("same", 2, [2, 2], 1)


def test_custom_game_can_be_deleted_without_affecting_other_games(tmp_path) -> None:
    catalog = GameCatalog(tmp_path)
    deleted = catalog.create_random("delete me", 2, [2, 2], 0)
    retained = catalog.create_random("keep me", 2, [2, 2], 1)

    assert catalog.delete(deleted.id) == deleted
    assert deleted.id not in catalog.definitions()
    assert retained.id in catalog.definitions()
    assert not (tmp_path / "delete-me.npz").exists()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        catalog.delete(deleted.id)
    with pytest.raises(ValueError, match="unknown custom game"):
        catalog.delete("rps")


def test_three_player_custom_game_runs_and_plots_regret_for_every_player(tmp_path) -> None:
    game_dir = tmp_path / "games"
    raw_dir = tmp_path / "raw"
    figure_dir = tmp_path / "figures"
    definition = GameCatalog(game_dir).create_random("three", 3, [2, 3, 2], 9)

    output_path = run_full_information_cross_play_experiment(
        definition.id,
        ["hedge", "hedge", "hedge"],
        horizon=2,
        output_dir=raw_dir,
        custom_game_dir=game_dir,
    )
    rows = read_csv_rows(output_path)

    assert len(rows) == 6
    assert {int(row["player"]) for row in rows} == {0, 1, 2}
    assert json.loads(rows[0]["algorithm_profile"]) == ["hedge", "hedge", "hedge"]
    assert "player_algorithm" not in rows[0]
    assert "average_expected_swap_regret" in rows[0]

    plot_selected_results(definition.id, raw_dir, figure_dir)

    assert (figure_dir / f"{definition.id}_average_expected_external_regret_player_2.png").is_file()


def test_custom_game_requires_one_algorithm_per_player(tmp_path) -> None:
    definition = GameCatalog(tmp_path / "games").create_random("three", 3, [2, 2, 2], 0)

    with pytest.raises(ValueError, match="requires 3"):
        run_full_information_cross_play_experiment(
            definition.id,
            ["hedge", "hedge"],
            horizon=1,
            output_dir=tmp_path / "raw",
            custom_game_dir=tmp_path / "games",
        )
