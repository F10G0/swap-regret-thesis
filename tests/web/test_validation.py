from io import BytesIO
from pathlib import Path
import pytest
from pypdf import PdfReader

from experiments.game_catalog import MAX_CUSTOM_ACTIONS_PER_PLAYER
from experiments.scenarios.cross_play import run_cross_play_experiment
from experiments.spec import MAX_RUN_ID_BYTES
from tests.web.support import block_job_queue, browse_url, create_test_app, csrf_token, dashboard_data, wait_for_http_response, wait_for_job
from web.validation import (
    parse_adversarial_experiment_form,
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
    assert parsed.horizon_values == (100,)


def test_horizon_lists_are_sorted_deduplicated_for_both_experiment_modes():
    fixed = parse_form(VALID_FORM | {"horizon": "100, 3, 10, 3"})
    one_player = parse_adversarial_experiment_form({
        "environment": "historical_frequency_v3", "feedback_mode": "full_information",
        "algorithm_names": ["hedge"], "actions": "3", "horizon": "10 3, 10",
        "seed": "42", "replicates": "1",
    }, {"full_information": ["hedge"]}, {"historical_frequency_v3"}, 100, 100)
    assert fixed.horizon_values == (3, 10, 100)
    assert one_player.horizon_values == (3, 10)


def parse_form(values):
    return parse_experiment_form(values, games={"rps": 2}, max_horizon=100, max_replicates=10,
        algorithms_by_feedback_mode={"full_information": ["hedge"], "bandit": ["auer_exp3", "lce_ix"]})


@pytest.mark.parametrize("field,value", [
    ("game", None), ("horizon", None), ("seed", None), ("feedback_mode", None), ("algorithm_names", None),
    ("horizon", "0"), ("horizon", "101"), ("horizon", "invalid"), ("seed", "-1"),
    ("horizon", "1,0"), ("horizon", "1,invalid"), ("horizon", "1,101"),
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
    assert status_response.json["rounds_completed"] == status_response.json["rounds_total"] == 0
    assert status_response.json["eta_seconds"] is None
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
    builder_catalog = client.get("/figure-builder/options?mode=fixed").get_json()
    page = client.get(browse_url(service))
    assert "summaries" not in dashboard_data(page)
    source_summaries = service.result_snapshot().summaries(grouped=True)
    assert page.data.count(b'class="summary-row"') == len(source_summaries)
    for source in source_summaries:
        detail_url = f'/experiment-groups/fixed/{source["group_id"]}/players/{source["player"]}'
        assert detail_url.encode() in page.data
        for metric in ("external", "internal", "swap"):
            for view in ("average", "sqrt_scaling"):
                key = f"{view}_{metric}"
                expected = source[f"average_{metric}_regret"] * (
                    1 if view == "average" else source["horizon"] ** 0.5)
                assert f'{expected:.6f}'.encode() in page.data
    assert len(source_summaries) == 2
    response = client.get(f'/experiment-groups/fixed/{source_summaries[0]["group_id"]}/players/0')
    assert response.status_code == 200
    assert response.cache_control.no_store
    summary = response.get_json()
    source = source_summaries[0]
    assert {key: summary[key] for key in (
        "group_id", "player", "game", "feedback_mode", "horizon", "seed",
        "replicate_label", "replicate_count", "stationary_method", "algorithm_profile",
    )} == {key: source[key] for key in (
        "group_id", "player", "game", "feedback_mode", "horizon", "seed",
        "replicate_label", "replicate_count", "stationary_method", "algorithm_profile",
    )}
    for metric in ("external", "internal", "swap"):
        for view in ("average", "sqrt_scaling"):
            expected = source[f"average_{metric}_regret"] * (
                1 if view == "average" else source["horizon"] ** 0.5)
            assert summary["display_regrets"][f"{view}_{metric}"] == expected
    assert {run["experiment"] for run in summary["runs"]} == {p.name for p in paths}
    assert [run["replicate"] for run in summary["runs"]] == [0, 1]
    assert all(set(run) == {"experiment", "replicate", "download_url"} for run in summary["runs"])
    assert client.get("/figure-builder/options?mode=fixed").get_json() == builder_catalog
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
        information = " ".join(pages[0].extract_text().split())
        assert figure in information
        assert "Aggregation: Replicate mean" in information


def test_fixed_detail_endpoint_rejects_invalid_and_stale_requests(tmp_path):
    app, service = create_test_app(tmp_path)
    run_cross_play_experiment(
        "rps", ["hedge", "hedge"], horizon=2, output_dir=service.raw_dir,
        feedback_mode="full_information",
    )
    client = app.test_client()
    group_id = service.result_snapshot().groups("dashboard")[0].records[0].group_id
    base = f"/experiment-groups/fixed/{group_id}/players"
    for url in (f"/experiment-groups/fixed/not-a-group/players/0",
                f"{base}/invalid", f"{base}/-1", f"{base}/2"):
        response = client.get(url)
        assert response.status_code == 400
        assert "error" in response.get_json()
    absent = client.get("/experiment-groups/fixed/0000000000000000/players/0")
    assert absent.status_code == 404
    assert "Refresh results" in absent.get_json()["error"]
    assert service.delete_result_group("fixed", group_id) == 1
    stale = client.get(f"{base}/0")
    assert stale.status_code == 404
    assert "Refresh results" in stale.get_json()["error"]


def test_dashboard_run_projection_preserves_duplicate_membership(tmp_path):
    app, service = create_test_app(tmp_path)
    paths = [run_cross_play_experiment(
        "rps", ["hedge", "hedge"], horizon=2, replicate=replicate, output_dir=service.raw_dir,
        feedback_mode="full_information") for replicate in (0, 1)]
    duplicate = service.raw_dir / "zz_duplicate.csv"
    duplicate.write_bytes(paths[0].read_bytes())
    snapshot = service.result_snapshot()
    group = snapshot.groups("dashboard")[0]
    group_id = group.records[0].group_id
    assert group.paths == paths
    assert set(snapshot.detail_paths(group_id)) == {*paths, duplicate}
    client = app.test_client()
    builder_catalog = client.get("/figure-builder/options?mode=fixed").get_json()
    page = client.get(browse_url(service))
    assert "summaries" not in dashboard_data(page)
    assert page.data.count(b'class="summary-row"') == 2
    for player in (0, 1):
        row = client.get(f"/experiment-groups/fixed/{group_id}/players/{player}").get_json()
        assert [run["experiment"] for run in row["runs"]] == [path.name for path in paths]
        assert [run["replicate"] for run in row["runs"]] == [0, 1]
    assert client.get("/figure-builder/options?mode=fixed").get_json() == builder_catalog
    assert service.delete_result_group("fixed", group_id) == 3
    assert not any(path.is_file() for path in (*paths, duplicate))


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


def test_corrupt_custom_archive_warns_on_page_without_hiding_valid_game(tmp_path):
    app, service = create_test_app(tmp_path)
    valid = service.create_custom_game("Usable Game", 2, [2, 2], 7)
    (service.game_catalog.custom_game_dir / "broken.npz").write_bytes(b"PK\x03\x04truncated")
    client = app.test_client()

    page = client.get("/custom-games")
    dashboard = client.get("/")

    assert page.status_code == dashboard.status_code == 200
    assert b"Game file warnings" in page.data
    assert b"Skipped broken.npz:" in page.data
    assert valid.id.encode() in page.data
    assert valid.id.encode() in dashboard.data


def test_custom_game_action_limit_is_rendered_from_backend_and_enforced(tmp_path):
    app, service = create_test_app(tmp_path)
    client = app.test_client()
    page = client.get("/custom-games").get_data(as_text=True)
    assert f'data-max-actions="{MAX_CUSTOM_ACTIONS_PER_PLAYER}"' in page
    token = csrf_token(client)
    form = {"_csrf_token": token, "payoff_structure": "general_sum",
            "n_players": "2", "seed": "7"}

    accepted = client.post("/custom-games", data=form | {
        "name": "Limit 100", "action_counts": ["100", "1"],
    })
    rejected = client.post("/custom-games", data=form | {
        "name": "Limit 101", "action_counts": ["101", "1"],
    })

    assert accepted.status_code == 302
    assert rejected.status_code == 400
    assert b"must not exceed 100" in rejected.data
    assert service.game_definitions["custom__limit-100"].action_counts == (100, 1)
    assert "custom__limit-101" not in service.game_definitions


@pytest.mark.parametrize("counts,algorithms,bad_player", [
    ([1, 2], ["bm_optimistic_hedge", "regret_matching"], 0),
    ([2, 1], ["regret_matching", "bm_optimistic_hedge"], 1),
])
def test_bm_optimistic_hedge_rejects_one_action_player_before_queueing(
    tmp_path, counts, algorithms, bad_player,
):
    app, service = create_test_app(tmp_path)
    definition = service.create_custom_game("Mixed Game", 2, counts, 7)
    client = app.test_client()
    response = client.post("/", headers={"Accept": "application/json"}, data=VALID_FORM | {
        "_csrf_token": csrf_token(client), "game": definition.id, "algorithm_names": algorithms,
    })

    assert response.status_code == 400
    assert "BM-OptHedge" in response.json["error"]
    assert "at least 2 actions" in response.json["error"]
    assert f"player {bad_player}" in response.json["error"]
    assert service.jobs.recent() == []
    assert list(service.raw_dir.glob("*.csv")) == []


@pytest.mark.parametrize("counts,algorithms", [
    ([1, 2], ["regret_matching", "bm_optimistic_hedge"]),
    ([2, 2], ["bm_optimistic_hedge", "bm_optimistic_hedge"]),
])
def test_compatible_one_action_or_k2_bm_experiment_still_runs(tmp_path, counts, algorithms):
    app, service = create_test_app(tmp_path)
    definition = service.create_custom_game("Allowed Game", 2, counts, 7)
    client = app.test_client()
    response = client.post("/", data=VALID_FORM | {
        "_csrf_token": csrf_token(client), "game": definition.id, "algorithm_names": algorithms,
    })

    assert response.status_code == 302
    assert wait_for_job(service, service.jobs.recent()[0].id) == "succeeded"


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
    page = client.get("/")
    assert "summaries" not in dashboard_data(page)
    group_id = next(row["group_id"] for row in service.result_snapshot().summaries(grouped=True)
                    if row["game"] == definition.id)
    summary = client.get(f"/experiment-groups/fixed/{group_id}/players/0").get_json()
    distance_response, _ = wait_for_http_response(client, summary["equilibrium_distance_url"])
    assert len(summary["algorithm_profile"]) == 3
    assert summary["algorithm_profile"] == ["hedge", "hedge", "hedge"]
    assert summary["equilibrium_distance_url"].startswith("/experiment-groups/")
    assert summary["equilibrium_distance_unavailable"] is None
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


def test_large_custom_game_keeps_results_but_equilibrium_analysis_is_unavailable(tmp_path: Path) -> None:
    app, service = create_test_app(tmp_path)
    definition = service.create_custom_game("Large Analysis", 2, [100, 100], 7, "zero_sum")
    result_path = run_cross_play_experiment(
        definition.id, ["hedge", "hedge"], horizon=2, seed=42, replicate=0,
        output_dir=service.raw_dir, custom_game_dir=service.game_catalog.custom_game_dir,
        feedback_mode="full_information",
    )
    client = app.test_client()

    page = client.get("/")
    assert "summaries" not in dashboard_data(page)
    group_id = next(row["group_id"] for row in service.result_snapshot().summaries(grouped=True)
                    if row["game"] == definition.id)
    summary = client.get(f"/experiment-groups/fixed/{group_id}/players/0").get_json()
    reason = summary["equilibrium_distance_unavailable"]

    assert page.status_code == 200
    assert definition.id in service.game_definitions
    assert definition.id.encode() in client.get("/custom-games").data
    assert result_path.is_file()
    assert summary["equilibrium_distance_url"] is None
    assert summary["equilibrium_distance_pdf_url"] is None
    assert "CE equilibrium-distance analysis is unavailable" in reason
    assert "analysis budget" in reason
    assert "game and regret results remain available" in reason
    assert summary["joint_actions_url"] is not None
    assert client.get(summary["joint_actions_url"]).status_code == 200
    assert client.get(summary["runs"][0]["download_url"]).status_code == 200

    for url in (
        f"/experiments/{result_path.name}/equilibrium-distance.png",
        f"/experiment-groups/{summary['group_id']}/equilibrium-distance.png",
    ):
        response = client.get(url)
        assert response.status_code == 422
        assert response.json["status"] == "unavailable"
        assert "analysis budget" in response.json["error"]
    assert service._convergence_futures == {}
