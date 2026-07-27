import csv
import json

import numpy as np
import pytest

from experiments.game_catalog import CUSTOM_GAME_PREFIX, GameCatalog
from experiments.plots.plot_regret import plot_selected_results
from experiments.scenarios.full_information_cross_play import run_full_information_cross_play_experiment


def test_custom_game_round_trip_is_reproducible(tmp_path) -> None:
    catalog = GameCatalog(tmp_path)

    definition = catalog.create_random("Three Player Test", 3, [2, 3, 2], 17)
    payoffs = catalog.load(definition.id)

    assert definition.id == f"{CUSTOM_GAME_PREFIX}three-player-test"
    assert definition.action_counts == (2, 3, 2)
    assert payoffs.shape == (3, 2, 3, 2)
    assert np.array_equal(payoffs, np.random.default_rng(17).random((3, 2, 3, 2)))
    assert np.all((0.0 <= payoffs) & (payoffs <= 1.0))
    assert (tmp_path / "three-player-test.npz").is_file()
    assert catalog.custom_path(definition.id) == tmp_path / "three-player-test.npz"


@pytest.mark.parametrize(
    ("name", "n_players", "action_counts", "message"),
    [
        ("bad/name", 3, [2, 2, 2], "game name"),
        ("one", 1, [2], "between 2"),
        ("missing", 3, [2, 2], "one action count"),
        ("too-large", 8, [100] * 8, "at most"),
    ],
)
def test_custom_game_validation_rejects_unsafe_configurations(
    tmp_path,
    name,
    n_players,
    action_counts,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        GameCatalog(tmp_path).create_random(name, n_players, action_counts, 0)


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
    with output_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 6
    assert {int(row["player"]) for row in rows} == {0, 1, 2}
    assert json.loads(rows[0]["algorithm_profile"]) == ["hedge", "hedge", "hedge"]
    assert {row["player_algorithm"] for row in rows} == {"hedge"}
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
