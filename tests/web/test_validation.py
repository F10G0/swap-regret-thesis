import json
from pathlib import Path
from threading import Event
import time

import numpy as np
import pytest

from experiments.scenarios.full_information_cross_play import run_full_information_cross_play_experiment
from experiments.spec import MAX_RUN_ID_BYTES
from web.presentations import GAME_PRESENTATIONS
from tests.web.support import create_test_app, csrf_token, wait_for_http_response, wait_for_job
from web.validation import (
    parse_experiment_form,
    parse_positive_integer,
    validate_leaf_filename,
)
from experimental.equilibrium_trajectory.settings import (
    parse_final_interval_segments,
    parse_focus_final_interval,
    parse_trajectory_comparison_view,
)


VALID_FORM = {
    "game": "rps",
    "feedback_mode": "full_information",
    "algorithm_names": ["hedge", "hedge"],
    "horizon": "2",
    "seed": "42",
    "replicates": "1",
    "regret_evaluation": "expected",
}


def wait_for_trajectory_comparison(client, url: str):
    return wait_for_http_response(client, url, headers={"Accept": "application/json"}, timeout=10)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_integer_validation(value: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        parse_positive_integer(value, "horizon")


def test_positive_integer_enforces_maximum() -> None:
    with pytest.raises(ValueError, match="must not exceed 100"):
        parse_positive_integer("101", "horizon", maximum=100)


@pytest.mark.parametrize(("value", "expected"), [(None, 10), ("1", 1), ("50", 50)])
def test_final_interval_segment_validation_accepts_supported_values(value: str | None, expected: int) -> None:
    assert parse_final_interval_segments(value) == expected


@pytest.mark.parametrize("value", ["0", "51", "invalid"])
def test_final_interval_segment_validation_rejects_unsupported_values(value: str) -> None:
    with pytest.raises(ValueError, match="final interval segments"):
        parse_final_interval_segments(value)


@pytest.mark.parametrize(("value", "expected"), [(None, False), ("0", False), ("1", True)])
def test_focus_final_interval_validation_accepts_boolean_query_values(value: str | None, expected: bool) -> None:
    assert parse_focus_final_interval(value) is expected


def test_focus_final_interval_validation_rejects_unknown_query_value() -> None:
    with pytest.raises(ValueError, match="focus_final_interval"):
        parse_focus_final_interval("true")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "geometry"), ("geometry", "geometry"), ("unified", "unified")],
)
def test_trajectory_comparison_view_validation(value, expected) -> None:
    assert parse_trajectory_comparison_view(value) == expected


def test_trajectory_comparison_view_validation_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="comparison_view"):
        parse_trajectory_comparison_view("unknown")


def test_experiment_form_rejects_algorithm_from_wrong_feedback_mode() -> None:
    values = VALID_FORM | {
        "feedback_mode": "bandit",
        "algorithm_names": ["hedge", "hedge"],
    }
    with pytest.raises(ValueError, match="not available for bandit"):
        parse_experiment_form(
            values,
            games={"rps"},
            algorithms_by_feedback_mode={
                "full_information": ["hedge"],
                "bandit": ["exp3"],
            },
            max_horizon=100,
        )


def test_bandit_form_accepts_replicate_batch() -> None:
    form = parse_experiment_form(
        VALID_FORM | {"feedback_mode": "bandit", "algorithm_names": ["exp3", "lce_ix"], "replicates": "20"},
        games={"rps"},
        algorithms_by_feedback_mode={"full_information": ["hedge"], "bandit": ["exp3", "lce_ix"]},
        max_horizon=100,
    )

    assert form.replicates == 20


def test_full_information_form_uses_one_replicate() -> None:
    values = {
        key: value
        for key, value in VALID_FORM.items()
        if key != "replicates"
    }
    form = parse_experiment_form(
        values,
        games={"rps"},
        algorithms_by_feedback_mode={"full_information": ["hedge"]},
        max_horizon=100,
    )

    assert form.replicates == 1

    stale_values = VALID_FORM | {"replicate": "9", "replicates": "20"}
    stale_form = parse_experiment_form(
        stale_values,
        games={"rps"},
        algorithms_by_feedback_mode={"full_information": ["hedge"]},
        max_horizon=100,
    )
    assert stale_form.replicates == 1


@pytest.mark.parametrize("feedback_mode", ["full_information", "bandit"])
@pytest.mark.parametrize("regret_evaluation", ["expected", "realized", "both"])
def test_experiment_form_accepts_every_regret_evaluation_for_each_feedback_mode(
    feedback_mode: str,
    regret_evaluation: str,
) -> None:
    algorithm = "hedge" if feedback_mode == "full_information" else "exp3"
    form = parse_experiment_form(
        VALID_FORM | {
            "feedback_mode": feedback_mode,
            "algorithm_names": [algorithm, algorithm],
            "regret_evaluation": regret_evaluation,
        },
        games={"rps"},
        algorithms_by_feedback_mode={"full_information": ["hedge"], "bandit": ["exp3"]},
        max_horizon=100,
    )

    assert form.regret_evaluation == regret_evaluation


def test_experiment_form_rejects_unknown_regret_evaluation() -> None:
    with pytest.raises(ValueError, match="unknown regret evaluation"):
        parse_experiment_form(
            VALID_FORM | {"regret_evaluation": "unknown"},
            games={"rps"},
            algorithms_by_feedback_mode={"full_information": ["hedge"]},
            max_horizon=100,
        )


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
    assert job.reload_page is True
    assert wait_for_job(service, job.id) == "succeeded"
    status_response = client.get(f"/jobs/{job.id}")
    assert status_response.status_code == 200
    assert status_response.json["status"] == "succeeded"
    assert len(list((tmp_path / "raw").glob("*.csv"))) == 1


def test_dashboard_exposes_regret_evaluation_control(tmp_path: Path) -> None:
    app, _ = create_test_app(tmp_path)

    page = app.test_client().get("/").get_data(as_text=True)

    assert 'id="regret-evaluation"' in page
    assert '<option value="expected"' in page
    assert '<option value="realized"' in page
    assert '<option value="both"' in page
    assert "Choose whether to record expected regret, realized regret, or both without changing learner feedback." in page


def test_dashboard_accepts_multiple_experiments_while_queue_is_active(
    tmp_path: Path,
) -> None:
    app, service = create_test_app(tmp_path)
    client = app.test_client()
    blocker_started = Event()
    release_blocker = Event()

    def block_queue(job) -> None:
        blocker_started.set()
        assert release_blocker.wait(timeout=2)

    blocker = service.jobs.submit("blocker", block_queue)
    assert blocker_started.wait(timeout=1)
    token = csrf_token(client)
    first_response = client.post("/", data=VALID_FORM | {"_csrf_token": token})
    second_response = client.post(
        "/",
        data=VALID_FORM | {"_csrf_token": token, "seed": "43"},
    )
    page = client.get("/").get_data(as_text=True)
    experiment_jobs = [
        job for job in service.jobs.recent()
        if job.id != blocker.id
    ]

    assert first_response.status_code == 302
    assert second_response.status_code == 302
    assert [job.status for job in experiment_jobs] == ["queued", "queued"]
    assert '<button id="queue-experiment" class="button-primary" type="submit">' in page
    assert 'id="jobs-heading">Job status</h2>' in page
    release_blocker.set()
    assert wait_for_job(service, blocker.id) == "succeeded"
    for job in experiment_jobs:
        assert wait_for_job(service, job.id) == "succeeded"
    assert len(list((tmp_path / "raw").glob("*.csv"))) == 2


def test_plot_rebuild_returns_json_without_navigation(tmp_path: Path) -> None:
    app, service = create_test_app(tmp_path)
    client = app.test_client()
    response = client.post("/plots/rebuild", data={"_csrf_token": csrf_token(client)}, headers={"Accept": "application/json"})

    assert response.status_code == 202
    assert response.json["reload_page"] is False
    assert response.json["url"].endswith(f"/jobs/{response.json['id']}")
    assert wait_for_job(service, response.json["id"]) == "succeeded"


def test_figure_inventory_endpoint_returns_json(tmp_path: Path) -> None:
    app, service = create_test_app(tmp_path)
    service.figure_dir.mkdir(parents=True)
    filename = "rps_average_expected_external_regret_player_0.png"
    (service.figure_dir / filename).write_bytes(b"png")
    pdf_filename = Path(filename).with_suffix(".pdf").name
    (service.figure_dir / pdf_filename).write_bytes(b"pdf")

    response = app.test_client().get("/figures")

    assert response.status_code == 200
    assert response.json == [{
        "filename": filename,
        "pdf_filename": pdf_filename,
        "game": "rps",
        "player": 0,
        "regret": "external",
        "source": "expected",
        "url": f"/figures/{filename}",
        "pdf_url": f"/figures/{pdf_filename}",
        "view": "average",
    }]


def test_figure_source_filter_has_only_separate_sources(tmp_path: Path) -> None:
    app, service = create_test_app(tmp_path)
    service.figure_dir.mkdir(parents=True)
    for source in ("expected", "realized"):
        (service.figure_dir / f"rps_average_{source}_external_regret_player_0.png").write_bytes(b"png")

    page = app.test_client().get("/").get_data(as_text=True)
    source_filter = page.split('<select id="filter-source">', 1)[1].split("</select>", 1)[0]

    assert '<option value="expected">Expected</option>' in source_filter
    assert '<option value="realized">Realized</option>' in source_filter
    assert 'value="all"' not in source_filter
    assert "Expected and realized" not in source_filter


def test_dashboard_uses_compact_management_and_has_no_all_pairs_action(
    tmp_path: Path,
) -> None:
    app, _ = create_test_app(tmp_path)
    client = app.test_client()

    page = client.get("/").get_data(as_text=True)
    removed_route = client.post(
        "/run-all-pairs",
        data={"_csrf_token": csrf_token(client)},
    )

    assert '<details class="data-management">' in page
    assert "Raw experiment files" in page
    assert 'id="queue-all-pairs"' not in page
    assert "Queue missing algorithm pairs" not in page
    assert removed_route.status_code == 404


def test_comparison_view_change_only_marks_pending_state() -> None:
    script = Path("web/static/experimental_trajectory.js").read_text(
        encoding="utf-8"
    )
    handler = script.split(
        'element("trajectory-comparison-view")?.addEventListener("change",',
        1,
    )[1].split("});", 1)[0]

    assert "saveTrajectoryComparisonView" in handler
    assert "updateTrajectoryComparisonDirtyState" in handler
    assert "generateTrajectoryComparison" not in handler


def test_dashboard_renders_result_details_and_serves_joint_action_heatmap(tmp_path: Path) -> None:
    app, service = create_test_app(tmp_path)
    run_full_information_cross_play_experiment("rps", ["hedge", "hedge"], horizon=2, output_dir=service.raw_dir)

    client = app.test_client()
    dashboard_response = client.get("/")
    page = dashboard_response.get_data(as_text=True)
    payload = page.split('<script id="dashboard-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    dashboard_data = json.loads(payload)
    summary = dashboard_data["summaries"][0]
    assert "trajectoryComparisonCandidates" not in dashboard_data
    assert service._experimental_trajectory_dashboard is None
    workspace_response = client.get(
        "/experimental/trajectory-comparisons"
    )
    workspace_page = workspace_response.get_data(as_text=True)
    workspace_payload = workspace_page.split(
        '<script id="experimental-trajectory-data" type="application/json">',
        1,
    )[1].split("</script>", 1)[0]
    workspace_data = json.loads(workspace_payload)
    candidate = workspace_data["trajectoryComparisonCandidates"][0]
    assert not list(
        service.detail_figure_dir.glob("trajectory_comparison_*.png")
    )
    heatmap_response = client.get(summary["joint_actions_url"])
    distance_response, distance_statuses = wait_for_http_response(client, summary["equilibrium_distance_url"])
    comparison_url = (
        f"{workspace_data['trajectoryComparisonUrl']}?member={candidate['group_id']}"
        "&final_interval_segments=6&focus_final_interval=0"
    )
    comparison_response, comparison_statuses = wait_for_trajectory_comparison(
        client,
        comparison_url,
    )
    comparison_payload = comparison_response.get_json()
    trajectory_response = client.get(comparison_payload["image_url"])
    unified_response, unified_statuses = wait_for_trajectory_comparison(
        client,
        f"{comparison_url}&comparison_view=unified",
    )
    unified_payload = unified_response.get_json()
    unified_image_response = client.get(unified_payload["image_url"])

    assert dashboard_response.status_code == 200
    assert b"Reuse parameters" in dashboard_response.data
    assert b'id="detail-downloads"' in dashboard_response.data
    assert b"Equilibrium Convergence" in dashboard_response.data
    assert b">Trajectories</a>" in dashboard_response.data
    assert b'class="panel results-controls"' in dashboard_response.data
    assert b'id="filter-summary-source"' in dashboard_response.data
    assert b'data-regret-source="expected"' in dashboard_response.data
    assert b'data-regret-source="realized"' in dashboard_response.data
    assert b'class="panel disclosure-panel equilibrium-panel"' in dashboard_response.data
    assert b'id="trajectory-comparison-view"' not in dashboard_response.data
    assert workspace_response.status_code == 200
    assert b"Equilibrium Trajectory Comparison" in workspace_response.data
    assert b'id="trajectory-comparison-view"' in workspace_response.data
    assert b'Unified equilibrium-relative' in workspace_response.data
    assert b'id="detail-equilibrium-trajectory-card"' not in dashboard_response.data
    assert b'id="final-interval-segments"' in workspace_response.data
    assert b'id="focus-final-interval"' in workspace_response.data
    assert b"Focus final log interval" in workspace_response.data
    assert b'min="1"' in workspace_response.data
    assert b'max="50"' in workspace_response.data
    assert "equilibrium_trajectory_url" not in summary
    assert dashboard_response.data.count(b"Loading heatmap") == 3
    assert heatmap_response.status_code == 200
    assert heatmap_response.content_type == "image/png"
    assert 202 in distance_statuses + comparison_statuses + unified_statuses
    assert distance_response.status_code == 200
    assert distance_response.content_type == "image/png"
    assert trajectory_response.status_code == 200
    assert trajectory_response.content_type == "image/png"
    assert comparison_payload["members"][0]["color"] == candidate["color"]
    assert comparison_payload["comparison_view"] == "geometry"
    assert unified_payload["comparison_view"] == "unified"
    assert unified_payload["artifact_id"] != comparison_payload["artifact_id"]
    assert unified_image_response.status_code == 200
    assert unified_image_response.content_type == "image/png"
    assert client.get(
        f"{workspace_data['trajectoryComparisonUrl']}?member={candidate['group_id']}"
        "&final_interval_segments=0"
    ).status_code == 400
    comparison_endpoint = workspace_data["trajectoryComparisonUrl"]
    assert client.get(
        f"{comparison_endpoint}?member={candidate['group_id']}"
        "&focus_final_interval=true"
    ).status_code == 400
    assert client.get(
        f"{comparison_endpoint}?final_interval_segments=6"
    ).status_code == 400
    assert client.get(
        f"{comparison_endpoint}?member={candidate['group_id']}"
        "&comparison_view=unknown"
    ).status_code == 400


def test_dashboard_combines_matching_replicates_and_retains_raw_downloads(tmp_path: Path) -> None:
    app, service = create_test_app(tmp_path)
    for replicate in range(2):
        run_full_information_cross_play_experiment(
            "rps", ["hedge", "hedge"], horizon=3, seed=42, replicate=replicate, output_dir=service.raw_dir
        )

    response = app.test_client().get("/")
    page = response.get_data(as_text=True)
    payload = page.split('<script id="dashboard-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    summaries = json.loads(payload)["summaries"]
    raw_player_zero = [row for row in service.result_snapshot().summaries if row["player"] == 0]
    expected = sum(row["average_expected_external_regret"] for row in raw_player_zero) / 2

    assert response.status_code == 200
    assert len(summaries) == 2
    assert all(summary["replicates"] == [0, 1] for summary in summaries)
    assert all(summary["replicate_count"] == 2 for summary in summaries)
    assert all(len(summary["runs"]) == 2 for summary in summaries)
    assert summaries[0]["average_expected_external_regret"] == pytest.approx(expected)
    assert summaries[0]["joint_actions_url"].startswith("/experiment-groups/")
    assert summaries[0]["equilibrium_distance_url"].startswith("/experiment-groups/")
    assert {run["experiment"] for run in summaries[0]["runs"]} == set(service.result_snapshot().filenames)


def test_dashboard_renders_and_serves_theoretical_equilibrium_heatmaps(
    tmp_path: Path,
) -> None:
    app, _ = create_test_app(tmp_path)
    client = app.test_client()

    dashboard_response = client.get("/")
    ce_response, ce_statuses = wait_for_http_response(
        client,
        "/games/rps/equilibria/ce.png",
    )
    cce_response, cce_statuses = wait_for_http_response(
        client,
        "/games/rps/equilibria/cce.png",
    )

    assert dashboard_response.status_code == 200
    assert b"Maximum CE Profile Weight" in dashboard_response.data
    assert b"Maximum CCE Profile Weight" in dashboard_response.data
    assert b"Each cell is optimized independently" in dashboard_response.data
    assert b"not itself an" in dashboard_response.data
    assert dashboard_response.data.count(b"Loading heatmap") == 2
    assert ce_statuses == [200]
    assert cce_statuses == [200]
    assert ce_response.status_code == 200
    assert ce_response.content_type == "image/png"
    assert cce_response.status_code == 200
    assert cce_response.content_type == "image/png"


def test_equilibrium_heatmap_request_serves_precomputed_asset_immediately(tmp_path: Path) -> None:
    app, _ = create_test_app(tmp_path)
    client = app.test_client()

    started_at = time.monotonic()
    first_response = client.get("/games/rps/equilibria/ce.png")
    elapsed = time.monotonic() - started_at

    assert first_response.status_code == 200
    assert first_response.content_type == "image/png"
    assert elapsed < 0.5


def test_dashboard_renders_balanced_top_controls_and_theme_selector(
    tmp_path: Path,
) -> None:
    app, _ = create_test_app(tmp_path)

    page = app.test_client().get("/").get_data(as_text=True)

    assert 'class="top-control-grid"' in page
    assert 'class="control-card control-card-game"' in page
    assert 'aria-describedby="game-description"' in page
    assert 'aria-describedby="feedback-description"' in page
    assert 'aria-describedby="regret-evaluation-description"' in page
    assert 'class="field-grid field-grid-two horizon-seed-grid"' in page
    assert 'id="replicate-fields" class="field" hidden' in page
    horizon_seed = page.split('class="field-grid field-grid-two horizon-seed-grid"', 1)[1].split("</div>", 3)
    assert any('id="horizon"' in fragment for fragment in horizon_seed)
    assert any('id="seed"' in fragment for fragment in horizon_seed)
    assert 'id="replicate"' not in page
    assert 'id="replicates"' in page
    assert "Number of rounds run by each experiment; the dashboard maximum is 100." in page
    assert "Base seed used to derive reproducible random streams for every player and bandit replicate." in page
    assert 'class="theme-control" for="primary-theme"' in page
    assert 'id="primary-theme" aria-label="Primary color theme"' in page
    assert page.index('id="primary-theme"') < page.index('id="experiment-form"')
    assert 'id="equilibrium-palette"' not in page
    assert "common.js" in page
    assert "dashboard.js" in page
    for theme in ("green", "blue", "purple", "orange", "red"):
        assert f'<option value="{theme}">' in page
    assert 'swap-regret-primary-theme' in page


def test_dashboard_renders_readable_bertrand_names_and_parameters(
    tmp_path: Path,
) -> None:
    app, _ = create_test_app(tmp_path)

    response = app.test_client().get("/")
    page = response.get_data(as_text=True)
    dashboard_payload = page.split(
        '<script id="dashboard-data" type="application/json">',
        maxsplit=1,
    )[1].split("</script>", maxsplit=1)[0]
    dashboard_data = json.loads(dashboard_payload)

    assert response.status_code == 200
    for game_name in (
        "bertrand_linear_o2",
        "bertrand_logit_o3",
        "bertrand_linear_o2_prime",
        "bertrand_logit_o3_prime",
    ):
        presentation = GAME_PRESENTATIONS[game_name]
        assert f'value="{game_name}"' in page
        assert presentation["label"] in page
        assert dashboard_data["gamePresentations"][game_name] == presentation


@pytest.mark.parametrize(
    "game_name",
    [
        "bertrand_linear_o2",
        "bertrand_logit_o3",
        "bertrand_linear_o2_prime",
        "bertrand_logit_o3_prime",
    ],
)
def test_dashboard_serves_bertrand_equilibrium_heatmaps_with_readable_titles(
    tmp_path: Path,
    game_name: str,
) -> None:
    app, _ = create_test_app(tmp_path)

    response, statuses = wait_for_http_response(
        app.test_client(),
        f"/games/{game_name}/equilibria/ce.png",
    )

    assert statuses == [200]
    assert response.status_code == 200
    assert response.content_type == "image/png"


@pytest.mark.parametrize(
    "url",
    [
        "/games/unknown/equilibria/ce.png",
        "/games/rps/equilibria/nash.png",
    ],
)
def test_equilibrium_heatmap_route_rejects_unknown_parameters(
    tmp_path: Path,
    url: str,
) -> None:
    app, _ = create_test_app(tmp_path)

    assert app.test_client().get(url).status_code == 404


def test_custom_game_page_creates_and_lists_game(tmp_path: Path) -> None:
    app, service = create_test_app(tmp_path)
    client = app.test_client()
    response = client.post(
        "/custom-games",
        data={
            "_csrf_token": csrf_token(client),
            "name": "Three Player Test",
            "n_players": "3",
            "action_counts": ["2", "3", "2"],
            "seed": "17",
        },
    )

    assert response.status_code == 302
    definition = service.game_definitions["custom__three-player-test"]
    assert definition.n_players == 3
    assert definition.action_counts == (2, 3, 2)
    assert (tmp_path / "custom-games" / "three-player-test.npz").is_file()

    library_page = client.get("/custom-games")
    dashboard_page = client.get("/")
    assert b"Three Player Test" in library_page.data
    assert b"Other games" in dashboard_page.data
    assert b"custom__three-player-test" in dashboard_page.data
    assert b"View payoffs" in library_page.data
    assert b"common.js" in library_page.data
    assert b"custom_games.js" in library_page.data
    assert b"dashboard.js" not in library_page.data


def test_custom_game_page_creates_symmetric_zero_sum_game(
    tmp_path: Path,
) -> None:
    app, service = create_test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/custom-games",
        data={
            "_csrf_token": csrf_token(client),
            "name": "Random Zero Sum",
            "payoff_structure": "zero_sum",
            "n_players": "2",
            "action_counts": ["3"],
            "seed": "23",
        },
    )

    definition = service.game_definitions["custom__random-zero-sum"]
    payoffs = service.game_catalog.load(definition.id)
    library_page = client.get("/custom-games")
    inspector_page = client.get(f"/custom-games/{definition.id}")
    dashboard_page = client.get("/")
    ce_response = client.get(
        f"/games/{definition.id}/equilibria/ce.png"
    )
    cce_response = client.get(
        f"/games/{definition.id}/equilibria/cce.png"
    )

    assert response.status_code == 302
    assert definition.payoff_structure == "zero_sum"
    assert definition.action_counts == (3, 3)
    assert np.allclose(payoffs[0] + payoffs[1], 1.0)
    assert np.allclose(payoffs[0], payoffs[1].T)
    assert np.allclose(payoffs[0] + payoffs[0].T, 1.0)
    assert b"Symmetric zero-sum" in library_page.data
    assert b'id="custom-payoff-structure"' in library_page.data
    assert b'value="zero_sum"' in library_page.data
    assert inspector_page.data.count(b"Equilibrium profile weights") == 1
    assert b"Maximum CE Profile Weight" in inspector_page.data
    assert b"Maximum CCE Profile Weight" in inspector_page.data
    equilibrium_url = (
        f"/games/{definition.id}/equilibria/ce.png".encode()
    )
    assert equilibrium_url in inspector_page.data
    assert equilibrium_url in dashboard_page.data
    assert ce_response.status_code == 200
    assert ce_response.content_type == "image/png"
    assert cce_response.status_code == 200
    assert cce_response.content_type == "image/png"
    assert len(list((tmp_path / "custom-games" / ".equilibria").glob("*.png"))) == 2

    script = client.get("/static/custom_games.js").get_data(as_text=True)
    assert 'fieldCount = symmetricZeroSum ? 1 : playerCount' in script
    assert '"Actions per player"' in script


def test_custom_game_page_rejects_zero_sum_with_more_than_two_players(
    tmp_path: Path,
) -> None:
    app, _ = create_test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/custom-games",
        data={
            "_csrf_token": csrf_token(client),
            "name": "Invalid Zero Sum",
            "payoff_structure": "zero_sum",
            "n_players": "3",
            "action_counts": ["2", "2", "2"],
            "seed": "1",
        },
    )

    assert response.status_code == 400
    assert b"symmetric zero-sum games require two equal action sets" in response.data
    assert b'value="zero_sum" selected' in response.data


def test_custom_game_page_rejects_zero_sum_with_unequal_action_counts(
    tmp_path: Path,
) -> None:
    app, _ = create_test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/custom-games",
        data={
            "_csrf_token": csrf_token(client),
            "name": "Asymmetric Zero Sum",
            "payoff_structure": "zero_sum",
            "n_players": "2",
            "action_counts": ["2", "3"],
            "seed": "1",
        },
    )

    assert response.status_code == 400
    assert b"symmetric zero-sum games require two equal action sets" in response.data


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


def test_dashboard_exposes_player_synchronization_control(tmp_path: Path) -> None:
    app, _ = create_test_app(tmp_path)

    page = app.test_client().get("/")

    assert b'id="synchronize-players"' in page.data
    assert "Player 0 → all".encode() in page.data
    assert b'id="swap-players"' not in page.data


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
    workspace_page = client.get(
        "/experimental/trajectory-comparisons"
    ).get_data(as_text=True)
    workspace_payload = workspace_page.split(
        '<script id="experimental-trajectory-data" type="application/json">',
        1,
    )[1].split("</script>", 1)[0]
    workspace_data = json.loads(workspace_payload)
    candidate = next(
        candidate
        for candidate in workspace_data["trajectoryComparisonCandidates"]
        if candidate["game"] == definition.id
    )
    distance_response, _ = wait_for_http_response(client, summary["equilibrium_distance_url"])
    comparison_response, _ = wait_for_trajectory_comparison(
        client,
        (
            f"{workspace_data['trajectoryComparisonUrl']}?member={candidate['group_id']}"
            "&final_interval_segments=2&focus_final_interval=1"
        ),
    )
    trajectory_response = client.get(comparison_response.get_json()["image_url"])

    assert summary["n_players"] == 3
    assert summary["algorithm_profile"] == ["hedge", "hedge", "hedge"]
    assert summary["equilibrium_distance_url"].startswith("/experiment-groups/")
    assert "equilibrium_trajectory_url" not in summary
    assert summary["joint_actions_url"] is None
    assert distance_response.status_code == 200
    assert distance_response.content_type == "image/png"
    assert trajectory_response.status_code == 200
    assert trajectory_response.content_type == "image/png"
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
