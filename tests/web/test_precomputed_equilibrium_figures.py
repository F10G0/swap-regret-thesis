from pathlib import Path
import sys

import pytest

from experiments.games import PAYOFF_FACTORIES
from web.equilibrium_figures import PRECOMPUTED_EQUILIBRIUM_DIR, equilibrium_figure_filename
from web.precompute_equilibrium_figures import precompute_equilibrium_figures


RETIRED_GAME_IDS = (
    "bertrand_standard_o1",
    "bertrand_linear_o2",
    "bertrand_logit_o3",
    "bertrand_linear_o2_prime",
    "bertrand_logit_o3_prime",
)


def test_precomputed_assets_cover_every_available_two_player_game() -> None:
    expected = {
        equilibrium_figure_filename(game_name, equilibrium)
        for game_name, factory in PAYOFF_FACTORIES.items()
        if factory().ndim == 3
        for equilibrium in ("ce", "cce")
    }
    available = {path.name for path in PRECOMPUTED_EQUILIBRIUM_DIR.glob("*.png")}

    # Historical assets may remain on disk without belonging to the catalog.
    assert expected <= available


def test_precomputation_excludes_retired_benchmarks(tmp_path, monkeypatch) -> None:
    import web.precompute_equilibrium_figures as module

    games = []

    def record(game_name, equilibrium, output_path, overwrite):
        games.append(game_name)
        return output_path

    monkeypatch.setattr(module, "_precompute_equilibrium_figure", record)
    precompute_equilibrium_figures(output_dir=tmp_path)
    assert set(games) == {"rps", "rpsls"}
    assert set(RETIRED_GAME_IDS).isdisjoint(games)


@pytest.mark.parametrize("game_name", (*RETIRED_GAME_IDS, "matching_pennies"))
def test_precomputation_rejects_retired_benchmarks_without_creating_assets(
    tmp_path, game_name
) -> None:
    output_dir = tmp_path / "assets"
    with pytest.raises(ValueError, match=f"unknown games: {game_name}"):
        precompute_equilibrium_figures([game_name], output_dir=output_dir)
    assert not output_dir.exists()


@pytest.mark.parametrize("game_name", RETIRED_GAME_IDS)
def test_precomputation_cli_rejects_retired_benchmarks(
    tmp_path, monkeypatch, capsys, game_name
) -> None:
    import web.precompute_equilibrium_figures as module

    def unexpected_precomputation(*args, **kwargs):
        pytest.fail("invalid command-line game reached asset generation")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "precompute_equilibrium_figures", unexpected_precomputation)
    monkeypatch.setattr(sys, "argv", ["precompute-equilibrium-figures", "--game", game_name])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert f"invalid choice: '{game_name}'" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_precomputation_keeps_existing_asset_without_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / equilibrium_figure_filename("rps", "ce")
    output_path.write_bytes(b"existing")

    generated = precompute_equilibrium_figures(["rps"], ("ce",), tmp_path)

    assert generated == [output_path]
    assert output_path.read_bytes() == b"existing"
    assert output_path.with_suffix(".pdf").is_file()


@pytest.mark.parametrize("workers", [0, -1, 1.5])
def test_precomputation_rejects_invalid_worker_counts(tmp_path: Path, workers) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        precompute_equilibrium_figures(["rps"], ("ce",), tmp_path, workers=workers)
