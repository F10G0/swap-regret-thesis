"""The retired static profile optimizer must not leak into current analysis."""

import json
from pathlib import Path

import pytest

from tests.web.support import create_test_app


@pytest.mark.parametrize("game", ("rps", "rpsls", "custom"))
def test_static_profile_routes_and_panels_are_absent(tmp_path, game):
    app, service = create_test_app(tmp_path)
    if game == "custom":
        game = service.create_custom_game("Retired panel", 2, [3, 3], 42, "zero_sum").id
        inspection = app.test_client().get(f"/custom-games/{game}")
        assert inspection.status_code == 200
        assert b"Profile Weight" not in inspection.data
        assert b"data-heatmap-source" not in inspection.data

    client = app.test_client()
    page = client.get("/")
    assert page.status_code == 200
    assert b"equilibrium-panel" not in page.data
    assert b"Profile Weight" not in page.data
    payload = page.get_data(as_text=True).split(
        '<script id="dashboard-data" type="application/json">', 1
    )[1].split("</script>", 1)[0]
    assert "equilibriumFigures" not in json.loads(payload)
    assert not any("/equilibria/" in rule.rule for rule in app.url_map.iter_rules())
    for concept in ("ce", "cce"):
        for extension in ("png", "pdf"):
            assert client.get(f"/games/{game}/equilibria/{concept}.{extension}").status_code == 404
    assert not (service.game_catalog.custom_game_dir / ".equilibria").exists()


def test_retired_static_assets_and_browser_selectors_are_absent():
    web = Path(__file__).resolve().parents[2] / "web"
    assert not (web / "static" / "equilibria").exists()
    retired = ("equilibrium-panel", "equilibrium-grid", "equilibrium-card",
               "equilibrium-explanation", "equilibriumFigures", "updateEquilibriumFigures",
               "data-heatmap-source", "Maximum CE Profile Weight", "Maximum CCE Profile Weight")
    for directory in (web / "templates", web / "static"):
        for path in directory.iterdir():
            if path.suffix in {".html", ".js", ".css"}:
                source = path.read_text()
                assert not any(token in source for token in retired), path
