from pathlib import Path
import time

import pytest

from experiments.scenarios.full_information_cross_play import run_full_information_cross_play_experiment
from web import create_app
from web.services import DashboardService
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
