"""Retired benchmark files stay archival, without becoming active UI choices."""

import csv
from io import BytesIO
import json
from pathlib import Path

from pypdf import PdfReader
import pytest

from experiments.plots import plot_regret as plotting
from experiments.results import iter_result_rows
from experiments.scenarios.full_information_cross_play import run_full_information_cross_play_experiment
from tests.web.support import create_test_app, csrf_token
from web.services import DashboardService


RETIRED_GAMES = (
    "bertrand_standard_o1", "bertrand_linear_o2", "bertrand_logit_o3",
    "bertrand_linear_o2_prime", "bertrand_logit_o3_prime", "retired_matrix_game",
)


def _run(service, game="rps"):
    return run_full_information_cross_play_experiment(
        game, ["hedge", "hedge"], horizon=3, output_dir=service.raw_dir,
        custom_game_dir=service.game_catalog.custom_game_dir,
    )


def _historical_copy(source, game, version=None):
    with source.open(newline="") as file:
        reader = csv.DictReader(file)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row.update(game=game, run_id=f"{game}_historical")
        if version is not None:
            row["implementation_version"] = str(version)
    path = source.parent / f"{game}_historical.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _snapshot(paths):
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


def _dashboard_data(response):
    return json.loads(response.get_data(as_text=True).split(
        '<script id="dashboard-data" type="application/json">', 1,
    )[1].split("</script>", 1)[0])


@pytest.mark.parametrize("game", RETIRED_GAMES)
def test_retired_results_are_downloadable_but_not_active_benchmarks(tmp_path, game):
    app, service = create_test_app(tmp_path)
    supported = _run(service)
    retired = _historical_copy(supported, game)
    originals = _snapshot([supported, retired])

    # The metadata reader is intentionally independent of the current catalog.
    assert {row["game"] for row in iter_result_rows(retired)} == {game}
    snapshot = service.result_snapshot()
    assert {summary["game"] for summary in snapshot.summaries} == {"rps"}
    assert retired.name in snapshot.filenames
    assert snapshot.warnings == [f"Skipped {retired.name}: unsupported game {game}"]

    client = app.test_client()
    page = client.get("/")
    assert page.status_code == 200
    data = _dashboard_data(page)
    assert game not in data["gameDefinitions"]
    assert game not in data["gamePresentations"]
    assert {summary["game"] for summary in data["summaries"]} == {"rps"}
    assert {context["scope"] for context in client.get(
        "/figure-builder/options?mode=fixed",
    ).json["contexts"]} == {"rps"}
    response = client.get(f"/experiments/{retired.name}")
    assert response.status_code == 200
    assert response.data == originals[retired][0]
    assert client.get(f"/games/{game}/equilibria/ce.png").status_code == 404
    assert set(plotting.collect_results(service.raw_dir)) == {"rps"}
    assert plotting.collect_results(service.raw_dir, game_name=game) == {}
    assert not list(tmp_path.rglob(f"{retired.stem}.json"))
    assert _snapshot(originals) == originals
    assert not service.jobs.recent()


@pytest.mark.parametrize("version", [0, 2, 4])
def test_old_schema_retired_results_do_not_break_strict_plot_scanning(tmp_path, version):
    app, service = create_test_app(tmp_path)
    supported = _run(service)
    retired = _historical_copy(supported, "bertrand_standard_o1", version)
    originals = _snapshot([supported, retired])
    snapshot = service.result_snapshot()
    assert {summary["game"] for summary in snapshot.summaries} == {"rps"}
    assert len(snapshot.warnings) == 1
    assert f"incompatible result implementation_version {version}" in snapshot.warnings[0]
    assert app.test_client().get("/").status_code == 200
    assert set(plotting.collect_results(service.raw_dir)) == {"rps"}
    assert not list(tmp_path.rglob(f"{retired.stem}.json"))
    assert _snapshot(originals) == originals


def test_strict_plot_scanning_still_rejects_unsupported_current_game_schema(tmp_path):
    _, service = create_test_app(tmp_path)
    supported = _run(service)
    invalid = _historical_copy(supported, "rps", 2)
    with pytest.raises(ValueError, match="incompatible result implementation_version 2"):
        plotting.collect_results(service.raw_dir)
    assert not list(tmp_path.rglob(f"{invalid.stem}.json"))


def test_plot_collection_cache_hits_still_do_not_open_current_csvs(tmp_path, monkeypatch):
    _, service = create_test_app(tmp_path)
    _run(service)
    expected = plotting.collect_results(service.raw_dir)
    original_open = Path.open

    def open_path(path, *args, **kwargs):
        if path.suffix == ".csv":
            pytest.fail("cached plot collection reopened the source CSV")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    assert plotting.collect_results(service.raw_dir) == expected


def test_supported_and_custom_visual_analysis_works_beside_retired_assets(tmp_path):
    app, service = create_test_app(tmp_path)
    supported = _run(service)
    custom = service.create_custom_game("local matrix", 2, [2, 2], 7)
    custom_result = _run(service, custom.id)
    retired = _historical_copy(supported, "bertrand_logit_o3")
    service.figure_dir.mkdir(parents=True)
    retired_assets = [service.figure_dir / f"bertrand_logit_o3_average_swap_regret_player_0.{suffix}"
                      for suffix in ("png", "pdf")]
    for path in retired_assets:
        path.write_bytes(b"protected historical figure")
    originals = _snapshot([supported, custom_result, retired, *retired_assets])
    assert service.figure_records() == []
    assert set(plotting.collect_results(service.raw_dir)) == {"rps", custom.id}

    client = app.test_client()
    token = csrf_token(client)
    assert {row["game"] for row in _dashboard_data(client.get("/"))["summaries"]} == {"rps", custom.id}
    contexts = client.get("/figure-builder/options?mode=fixed").json["contexts"]
    assert {context["scope"] for context in contexts} == {"rps", custom.id}
    context = next(context for context in contexts if context["scope"] == custom.id and context["player"] == 0)
    response = client.post("/figure-builder/collection", data={
        "_csrf_token": token, "mode": "fixed", "context_id": context["id"],
        "profiles": ["hedge_vs_hedge"],
    })
    assert response.status_code == 200
    figures = response.json["figures"]
    assert len(figures) == 6
    preview = client.get(figures[0]["url"])
    assert preview.status_code == 200 and preview.data.startswith(b"\x89PNG")
    export = client.post("/figures/download-filtered.pdf", data={
        "_csrf_token": token, "mode": "figure_builder",
        "filenames": [figures[0]["pdf_filename"], figures[5]["pdf_filename"]],
    })
    assert export.status_code == 200
    assert len(PdfReader(BytesIO(export.data)).pages) == 2
    assert client.post("/figures/download-filtered.pdf", data={
        "_csrf_token": token, "mode": "fixed", "filenames": [retired_assets[1].name],
    }).status_code == 404
    assert _snapshot(originals) == originals
    assert not service.jobs.recent()


def test_full_plot_rebuild_preserves_retired_assets_and_cleans_supported_stale_files(tmp_path, monkeypatch):
    _, service = create_test_app(tmp_path)
    supported = _run(service)
    custom = service.create_custom_game("local matrix", 2, [2, 2], 7)
    custom_result = _run(service, custom.id)
    retired = _historical_copy(supported, "bertrand_linear_o2")
    service.figure_dir.mkdir(parents=True)
    retired_assets = [service.figure_dir / f"bertrand_linear_o2_average_swap_regret_player_0.{suffix}"
                      for suffix in ("png", "pdf")]
    stale = [service.figure_dir / "rps_stale.png", service.figure_dir / f"{custom.id}_stale.pdf"]
    for path in [*retired_assets, *stale]:
        path.write_bytes(b"existing figure")
    originals = _snapshot([supported, custom_result, retired, *retired_assets])
    plotted_games = []

    def render(game, rows_by_run, output_dir):
        plotted_games.append(game)
        for suffix in ("png", "pdf"):
            (output_dir / f"{game}_average_swap_regret_player_0.{suffix}").write_bytes(b"new figure")

    monkeypatch.setattr(plotting, "plot_game_results", render)
    # The test helper disables automatic plotting; exercise the real publication path explicitly.
    DashboardService._publish_plots(service)
    assert set(plotted_games) == {"rps", custom.id}
    assert all(not path.exists() for path in stale)
    assert {record["game"] for record in service.figure_records()} == {"rps", custom.id}
    assert _snapshot(originals) == originals
