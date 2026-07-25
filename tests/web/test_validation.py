import json
from pathlib import Path
from threading import Event
import time

import pytest

from experiments.scenarios.full_information_cross_play import run_full_information_cross_play_experiment
from web import create_app
from web.services import DashboardService, GAME_PRESENTATIONS
from web.validation import (
    parse_experiment_form,
    parse_positive_integer,
    validate_leaf_filename,
)


VALID_FORM = {
    "game": "rps",
    "feedback_mode": "full_information",
    "algorithm_player_0": "hedge",
    "algorithm_player_1": "hedge",
    "horizon": "2",
    "seed": "42",
    "replicate": "0",
    "replicates": "1",
}


def create_test_app(tmp_path: Path) -> tuple:
    service = DashboardService(
        results_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        figure_dir=tmp_path / "figures",
    )
    service._publish_plots = lambda game_name=None: None
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "MAX_HORIZON": 100,
        },
        service=service,
    )
    return app, service


def csrf_token(client) -> str:
    client.get("/")
    with client.session_transaction() as session:
        return session["_csrf_token"]


def wait_for_job(service: DashboardService, job_id: str) -> str:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = service.jobs.get(job_id)
        if job is not None and job.status in {"succeeded", "failed"}:
            return job.status
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_integer_validation(value: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        parse_positive_integer(value, "horizon")


def test_positive_integer_enforces_maximum() -> None:
    with pytest.raises(ValueError, match="must not exceed 100"):
        parse_positive_integer("101", "horizon", maximum=100)


def test_experiment_form_rejects_algorithm_from_wrong_feedback_mode() -> None:
    values = VALID_FORM | {
        "feedback_mode": "bandit",
        "algorithm_player_0": "hedge",
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
        VALID_FORM | {"feedback_mode": "bandit", "algorithm_player_0": "exp3", "algorithm_player_1": "lce_ix", "replicates": "20"},
        games={"rps"},
        algorithms_by_feedback_mode={"full_information": ["hedge"], "bandit": ["exp3", "lce_ix"]},
        max_horizon=100,
    )

    assert form.replicates == 20


def test_full_information_form_uses_one_replicate() -> None:
    form = parse_experiment_form(VALID_FORM | {"replicates": "20"}, games={"rps"}, algorithms_by_feedback_mode={"full_information": ["hedge"]}, max_horizon=100)

    assert form.replicates == 1


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
    assert "You can queue more while one is running." in page
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

    response = app.test_client().get("/figures")

    assert response.status_code == 200
    assert response.json == [{"filename": filename, "game": "rps", "player": 0, "regret": "external", "source": "expected", "url": f"/figures/{filename}", "view": "average"}]


def test_dashboard_renders_result_details_and_serves_joint_action_heatmap(tmp_path: Path) -> None:
    app, service = create_test_app(tmp_path)
    result_path = run_full_information_cross_play_experiment("rps", ["hedge", "hedge"], horizon=2, output_dir=service.raw_dir)

    client = app.test_client()
    dashboard_response = client.get("/")
    heatmap_response = client.get(f"/experiments/{result_path.name}/joint-actions.png")

    assert dashboard_response.status_code == 200
    assert b"Reuse parameters" in dashboard_response.data
    assert b"Download raw CSV" in dashboard_response.data
    assert heatmap_response.status_code == 200
    assert heatmap_response.content_type == "image/png"


def test_dashboard_renders_and_serves_theoretical_equilibrium_heatmaps(
    tmp_path: Path,
) -> None:
    app, _ = create_test_app(tmp_path)
    client = app.test_client()

    dashboard_response = client.get("/")
    ce_response = client.get("/games/rps/equilibria/ce.png")
    cce_response = client.get("/games/rps/equilibria/cce.png")

    assert dashboard_response.status_code == 200
    assert b"Maximum CE Profile Weight" in dashboard_response.data
    assert b"Maximum CCE Profile Weight" in dashboard_response.data
    assert b"Each cell is optimized independently" in dashboard_response.data
    assert b"not itself an" in dashboard_response.data
    assert ce_response.status_code == 200
    assert ce_response.content_type == "image/png"
    assert cce_response.status_code == 200
    assert cce_response.content_type == "image/png"


def test_dashboard_renders_balanced_top_controls_and_theme_selector(
    tmp_path: Path,
) -> None:
    app, _ = create_test_app(tmp_path)

    page = app.test_client().get("/").get_data(as_text=True)

    assert 'class="top-control-grid"' in page
    assert 'class="control-card control-card-game"' in page
    assert 'aria-describedby="game-description"' in page
    assert 'aria-describedby="feedback-description"' in page
    assert 'id="primary-theme" class="sidebar-theme"' in page
    assert page.index('id="primary-theme"') < page.index('id="experiment-form"')
    assert 'id="equilibrium-palette"' not in page
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
    monkeypatch,
    game_name: str,
) -> None:
    from experiments.plots import plot_equilibrium_weights

    rendered_game_names = []

    def fake_plot(
        payoff_tensor,
        equilibrium,
        output_path,
        game_name=None,
    ) -> None:
        rendered_game_names.append(game_name)
        Path(output_path).write_bytes(b"png")

    monkeypatch.setattr(
        plot_equilibrium_weights,
        "plot_equilibrium_profile_weights",
        fake_plot,
    )
    app, _ = create_test_app(tmp_path)

    response = app.test_client().get(
        f"/games/{game_name}/equilibria/ce.png"
    )

    assert response.status_code == 200
    assert response.content_type == "image/png"
    assert rendered_game_names == [GAME_PRESENTATIONS[game_name]["label"]]


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
