"""Retired benchmark files stay archival, without becoming active UI choices."""

import csv
import json

import pytest

from experiments.scenarios.cross_play import run_cross_play_experiment
from experiments.results import iter_result_rows
from tests.web.support import create_test_app, csrf_token
from web.validation import ExperimentForm


RETIRED_GAMES = ("bertrand_standard_o1", "retired_matrix_game")


def _run(service, game="rps"):
    return run_cross_play_experiment(
        game, ["hedge", "hedge"], horizon=3, output_dir=service.raw_dir,
        custom_game_dir=service.game_catalog.custom_game_dir,
        feedback_mode="full_information",
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
    assert {summary["game"] for summary in snapshot.summaries()} == {"rps"}
    assert retired.name in snapshot.filenames
    assert snapshot.warnings == (f"Skipped {retired.name}: unsupported game {game}",)

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
    # Both form and direct-service boundaries must refuse new runs, without altering the archive.
    values = dict(game=game, feedback_mode="full_information", algorithm_names=["hedge", "hedge"],
                  horizon="2", seed="42", replicates="1", _csrf_token=csrf_token(client))
    assert client.post("/", data=values).status_code == 400
    with pytest.raises(ValueError):
        service.submit_experiment(ExperimentForm(game, "full_information", ("hedge", "hedge"), 2, 42, 1))
    assert _snapshot(originals) == originals
    assert not service.jobs.recent()


def test_old_schema_retired_results_do_not_break_dashboard_discovery(tmp_path):
    version = 2
    app, service = create_test_app(tmp_path)
    supported = _run(service)
    retired = _historical_copy(supported, "bertrand_standard_o1", version)
    originals = _snapshot([supported, retired])
    snapshot = service.result_snapshot()
    assert {summary["game"] for summary in snapshot.summaries()} == {"rps"}
    assert len(snapshot.warnings) == 1
    assert f"incompatible result implementation_version {version}" in snapshot.warnings[0]
    assert app.test_client().get("/").status_code == 200
    assert _snapshot(originals) == originals


def test_supported_and_custom_visual_analysis_works_beside_retired_assets(tmp_path):
    app, service = create_test_app(tmp_path)
    supported = _run(service)
    custom = service.create_custom_game("local matrix", 2, [2, 2], 7)
    custom_result = _run(service, custom.id)
    retired = _historical_copy(supported, "bertrand_logit_o3")
    originals = _snapshot([supported, custom_result, retired])

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
    figure = response.json["figures"][0]
    assert client.get(figure["url"]).mimetype == "image/png"
    assert client.get(figure["pdf_url"]).mimetype == "application/pdf"
    assert _snapshot(originals) == originals
    assert not service.jobs.recent()
