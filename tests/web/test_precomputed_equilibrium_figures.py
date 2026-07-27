from pathlib import Path

import pytest

from experiments.games import PAYOFF_FACTORIES
from web.equilibrium_figures import PRECOMPUTED_EQUILIBRIUM_DIR, equilibrium_figure_filename
from web.precompute_equilibrium_figures import precompute_equilibrium_figures


def test_precomputed_assets_cover_every_available_two_player_game() -> None:
    expected = {
        equilibrium_figure_filename(game_name, equilibrium)
        for game_name, factory in PAYOFF_FACTORIES.items()
        if factory().ndim == 3
        for equilibrium in ("ce", "cce")
    }
    available = {path.name for path in PRECOMPUTED_EQUILIBRIUM_DIR.glob("*.png")}

    assert available == expected


def test_precomputation_keeps_existing_asset_without_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / equilibrium_figure_filename("rps", "ce")
    output_path.write_bytes(b"existing")

    generated = precompute_equilibrium_figures(["rps"], ("ce",), tmp_path)

    assert generated == [output_path]
    assert output_path.read_bytes() == b"existing"


@pytest.mark.parametrize("workers", [0, -1, 1.5])
def test_precomputation_rejects_invalid_worker_counts(tmp_path: Path, workers) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        precompute_equilibrium_figures(["rps"], ("ce",), tmp_path, workers=workers)
