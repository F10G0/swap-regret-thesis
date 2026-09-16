import json
from io import BytesIO
from pathlib import Path
import pytest
from pypdf import PdfReader

from experiments.scenarios.cross_play import run_cross_play_experiment
from experiments.spec import MAX_RUN_ID_BYTES
from tests.web.support import block_job_queue, create_test_app, csrf_token, dashboard_data, wait_for_http_response, wait_for_job
from web.validation import (
    parse_experiment_form,
    validate_leaf_filename,
)


VALID_FORM = {
    "game": "rps",
    "feedback_mode": "full_information",
    "algorithm_names": ["hedge", "hedge"],
    "horizon": "2",
    "seed": "42",
    "replicates": "1",
}


@pytest.mark.parametrize("mode,names", [("full_information", ["hedge", "hedge"]), ("bandit", ["auer_exp3", "lce_ix"])])
def test_experiment_form_accepts_feedback_and_boundary_values(mode, names):
    parsed = parse_form(VALID_FORM | {"feedback_mode": mode, "algorithm_names": names,
                                   "horizon": "100", "seed": "0", "replicates": "10"})
    assert (parsed.feedback_mode, parsed.algorithm_names) == (mode, tuple(names))
    assert (parsed.horizon, parsed.seed, parsed.replicates) == (100, 0, 10)


def parse_form(values):
    return parse_experiment_form(values, games={"rps": 2}, max_horizon=100, max_replicates=10,
        algorithms_by_feedback_mode={"full_information": ["hedge"], "bandit": ["auer_exp3", "lce_ix"]})


@pytest.mark.parametrize("field,value", [
    ("game", None), ("horizon", None), ("seed", None), ("feedback_mode", None), ("algorithm_names", None),
    ("horizon", "0"), ("horizon", "101"), ("horizon", "invalid"), ("seed", "-1"),
    ("replicates", None), ("replicates", "0"), ("replicates", "11"),
    ("game", "unknown"), ("feedback_mode", "unknown"), ("feedback_mode", "bandit"),
    ("algorithm_names", ["hedge"]), ("algorithm_names", ["unknown", "hedge"]),
])
def test_experiment_form_rejects_invalid_configuration(field, value):
    values = VALID_FORM | {field: value}
    if value is None:
        del values[field]
    with pytest.raises(ValueError):
        parse_form(values)


def test_leaf_filename_validation_rejects_paths_and_wrong_suffixes() -> None:
    with pytest.raises(ValueError, match="invalid filename"):
        validate_leaf_filename("../outside.csv", ".csv")
    with pytest.raises(ValueError, match="invalid filename"):
        validate_leaf_filename("notes.txt", ".csv")


def test_dashboard_requires_csrf_token(tmp_path: Path) -> None:
    app, _ = create_test_app(tmp_path)
    response = app.test_client().post("/", data=VALID_FORM)
    assert response.status_code == 400


def test_dashboard_returns_form_error_for_invalid_horizon(tmp_path: Path) -> None:
    app, _ = create_test_app(tmp_path)
    client = app.test_client()
    response = client.post(
        "/",
        data=VALID_FORM | {
            "_csrf_token": csrf_token(client),
            "horizon": "0",
        },
    )
    assert response.status_code == 400
    assert b"horizon must be positive" in response.data


def test_dashboard_queues_valid_experiment_and_exposes_job_status(
    tmp_path: Path,
) -> None:
    app, service = create_test_app(tmp_path)
    client = app.test_client()
    response = client.post(
        "/",
        data=VALID_FORM | {"_csrf_token": csrf_token(client)},
    )
    assert response.status_code == 302

    job = service.jobs.recent()[0]
    assert wait_for_job(service, job.id) == "succeeded"
    status_response = client.get(f"/jobs/{job.id}")
    assert status_response.status_code == 200
    assert status_response.json["status"] == "succeeded"
    assert len(list((tmp_path / "raw").glob("*.csv"))) == 1


def test_dashboard_accepts_multiple_experiments_while_queue_is_active(
    tmp_path: Path,
) -> None:
    app, service = create_test_app(tmp_path)
    client = app.test_client()
    blocker, release_blocker = block_job_queue(service.jobs)
    token = csrf_token(client)
    first_response = client.post("/", data=VALID_FORM | {"_csrf_token": token})
    second_response = client.post(
        "/",
        data=VALID_FORM | {"_csrf_token": token, "seed": "43"},
    )
    experiment_jobs = [
        job for job in service.jobs.recent()
        if job.id != blocker.id
    ]

    assert first_response.status_code == 302
    assert second_response.status_code == 302
    assert [job.status for job in experiment_jobs] == ["queued", "queued"]
    release_blocker.set()
    assert wait_for_job(service, blocker.id) == "succeeded"
    for job in experiment_jobs:
        assert wait_for_job(service, job.id) == "succeeded"
    assert len(list((tmp_path / "raw").glob("*.csv"))) == 2


def test_dashboard_group_details_downloads_and_figures(tmp_path):
    app, service = create_test_app(tmp_path)
    paths = [run_cross_play_experiment(
        "rps", ["hedge", "hedge"], horizon=2, replicate=r, output_dir=service.raw_dir,
     feedback_mode="full_information") for r in (0, 1)]
    client = app.test_client()
    summaries = dashboard_data(client.get("/"))["summaries"]
    assert len(summaries) == 2
    summary = summaries[0]
    assert summary["replicates"] == [0, 1]
    assert {run["experiment"] for run in summary["runs"]} == {p.name for p in paths}
    for run in summary["runs"]:
        assert client.get(run["download_url"]).data == (service.raw_dir / run["experiment"]).read_bytes()
    heatmap = client.get(summary["joint_actions_url"])
    heatmap_pdf = client.get(summary["joint_actions_pdf_url"])
    distance, _ = wait_for_http_response(client, summary["equilibrium_distance_pdf_url"])
    assert heatmap.status_code == heatmap_pdf.status_code == distance.status_code == 200
    assert heatmap.mimetype == "image/png" and heatmap_pdf.mimetype == distance.mimetype == "application/pdf"
    for response, figure in ((heatmap_pdf, "Joint-action distribution"),
                             (distance, "Equilibrium-distance convergence")):
        pages = PdfReader(BytesIO(response.data)).pages
        assert len(pages) == 2
        assert figure in pages[0].extract_text()
        assert "Aggregation:  Replicate mean" in pages[0].extract_text()


def test_custom_game_generator_uses_header_seed_and_zero_sum_default(tmp_path):
    app, _ = create_test_app(tmp_path)
    client = app.test_client()
    page = client.get("/custom-games").get_data(as_text=True)
    assert page.count('name="seed"') == 1
    assert 'id="custom-game-seed" name="seed" form="custom-game-form"' in page
    assert page.index('id="custom-game-seed"') < page.index('id="custom-game-form"')
    assert 'id="experiment-seed"' not in page
    assert '<option value="zero_sum" selected>Symmetric zero-sum</option>' in page
    assert '<option value="general_sum"' in page

    response = client.post("/custom-games", data={
        "_csrf_token": csrf_token(client), "name": "General", "payoff_structure": "general_sum",
        "n_players": "3", "action_counts": ["2", "2", "2"], "seed": "-1",
    })
    assert response.status_code == 400
    assert '<option value="general_sum" selected>General-sum</option>' in response.get_data(as_text=True)


@pytest.mark.parametrize("players,counts,structure", [(3, [2, 3, 2], "general_sum"), (2, [3], "zero_sum")])
def test_custom_game_creation_and_inspection(tmp_path, players, counts, structure):
    app, service = create_test_app(tmp_path)
    client = app.test_client()
    response = client.post("/custom-games", data={
        "_csrf_token": csrf_token(client), "name": "Local Game", "n_players": str(players),
        "action_counts": list(map(str, counts)), "seed": "17", "payoff_structure": structure,
    })
    assert response.status_code == 302
    definition = service.game_definitions["custom__local-game"]
    assert definition.n_players == players and definition.payoff_structure == structure
    assert definition.action_counts == (tuple(counts) if players == 3 else (3, 3))
    assert service.custom_game_file(definition.id).is_file()
    assert client.get(f"/custom-games/{definition.id}").status_code == 200
    assert b"Local Game" in client.get("/custom-games").data
    assert definition.id.encode() in client.get("/").data


@pytest.mark.parametrize("players,counts", [(3, ["2", "2", "2"]), (2, ["2", "3"])])
def test_custom_game_rejects_incompatible_zero_sum_shape(tmp_path, players, counts):
    app, service = create_test_app(tmp_path)
    client = app.test_client()
    response = client.post("/custom-games", data={
        "_csrf_token": csrf_token(client), "name": "Invalid", "payoff_structure": "zero_sum",
        "n_players": str(players), "action_counts": counts, "seed": "1",
    })
    assert response.status_code == 400
    assert b'value="zero_sum" selected' in response.data
    assert "custom__invalid" not in service.game_definitions


def test_custom_game_payoff_inspector_slice_and_download(tmp_path: Path) -> None:
    app, service = create_test_app(tmp_path)
    definition = service.create_custom_game("Inspect Me", 3, [2, 3, 2], 7)
    payoff_tensor = service.game_catalog.load(definition.id)
    client = app.test_client()

    page = client.get(f"/custom-games/{definition.id}")
    response = client.get(
        f"/custom-games/{definition.id}/payoff-slice",
        query_string=[
            ("payoff_player", "2"),
            ("row_player", "1"),
            ("column_player", "0"),
            ("fixed_action", "0"),
            ("fixed_action", "0"),
            ("fixed_action", "1"),
        ],
    )
    download = client.get(f"/custom-games/{definition.id}/download")

    assert page.status_code == 200
    assert b"Inspect Me" in page.data
    assert b'id="payoff-inspector"' in page.data
    assert "3 × 2 × 3 × 2".encode() in page.data
    assert b"Download NPZ" in page.data
    assert response.status_code == 200
    assert response.get_json()["values"] == payoff_tensor[2, :, :, 1].T.tolist()
    assert download.status_code == 200
    assert download.headers["Content-Disposition"].startswith("attachment;")
    assert download.data.startswith(b"PK")


def test_custom_game_payoff_inspector_rejects_invalid_axes_and_builtin_games(tmp_path: Path) -> None:
    app, service = create_test_app(tmp_path)
    definition = service.create_custom_game("Inspect Me", 2, [2, 2], 7)
    client = app.test_client()

    invalid_slice = client.get(
        f"/custom-games/{definition.id}/payoff-slice",
        query_string=[
            ("payoff_player", "0"),
            ("row_player", "1"),
            ("column_player", "1"),
            ("fixed_action", "0"),
            ("fixed_action", "0"),
        ],
    )

    assert invalid_slice.status_code == 400
    assert client.get("/custom-games/rps").status_code == 404
    assert client.get("/custom-games/rps/download").status_code == 404


def test_custom_game_page_deletes_game(tmp_path: Path) -> None:
    app, service = create_test_app(tmp_path)
    definition = service.create_custom_game("Delete Me", 3, [2, 2, 2], 0)
    client = app.test_client()

    response = client.post(
        "/custom-games/delete",
        data={"_csrf_token": csrf_token(client), "game_id": definition.id},
    )

    assert response.status_code == 302
    assert definition.id not in service.game_definitions
    assert not (tmp_path / "custom-games" / "delete-me.npz").exists()


def test_custom_three_player_dashboard_experiment_includes_equilibrium_convergence(tmp_path: Path) -> None:
    app, service = create_test_app(tmp_path)
    definition = service.create_custom_game("Three Players", 3, [2, 2, 2], 5)
    client = app.test_client()

    response = client.post(
        "/",
        data={
            "_csrf_token": csrf_token(client),
            "game": definition.id,
            "feedback_mode": "full_information",
            "algorithm_names": ["hedge", "hedge", "hedge"],
            "horizon": "2",
            "seed": "42",
            "replicates": "1",
        },
    )

    assert response.status_code == 302
    job = service.jobs.recent()[0]
    assert wait_for_job(service, job.id) == "succeeded"
    result_path = next(service.raw_dir.glob("*.csv"))
    page = client.get("/").get_data(as_text=True)
    payload = page.split('<script id="dashboard-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    dashboard_data = json.loads(payload)
    summary = next(summary for summary in dashboard_data["summaries"] if summary["game"] == definition.id)
    distance_response, _ = wait_for_http_response(client, summary["equilibrium_distance_url"])
    assert summary["n_players"] == 3
    assert summary["algorithm_profile"] == ["hedge", "hedge", "hedge"]
    assert summary["equilibrium_distance_url"].startswith("/experiment-groups/")
    assert summary["joint_actions_url"] is None
    assert distance_response.status_code == 200
    assert distance_response.content_type == "image/png"
    assert client.get(f"/experiments/{result_path.name}/joint-actions.png").status_code == 404

    service.clear_results()
    assert not result_path.exists()
    assert (tmp_path / "custom-games" / "three-players.npz").is_file()


def test_eight_player_srm_experiment_uses_length_safe_filename(tmp_path: Path) -> None:
    app, service = create_test_app(tmp_path)
    definition = service.create_custom_game("Eight Players", 8, [2] * 8, 5)
    client = app.test_client()

    response = client.post(
        "/",
        data={
            "_csrf_token": csrf_token(client),
            "game": definition.id,
            "feedback_mode": "full_information",
            "algorithm_names": ["stationary_regret_matching"] * 8,
            "horizon": "2",
            "seed": "42",
            "replicates": "1",
        },
    )

    assert response.status_code == 302
    job = service.jobs.recent()[0]
    assert wait_for_job(service, job.id) == "succeeded"
    result_path = next(service.raw_dir.glob("*.csv"))
    assert len(result_path.name.encode("utf-8")) <= MAX_RUN_ID_BYTES + len(".csv")
