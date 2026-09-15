from experiments.scenarios.cross_play import run_cross_play_experiment
import json
import shutil
import subprocess

import pytest
from pathlib import Path
from threading import Event
import time

from web import create_app
from web.services import DashboardService


ADVERSARIAL_FORM = {
    "experiment_type": "adversarial", "environment": "historical_frequency_v3",
    "feedback_mode": "full_information", "algorithm_names": ["hedge"],
    "actions": "3", "horizon": "4", "seed": "7", "replicates": "1",
}


def record_fixed_runs(service, profile, *, mode="full_information", game="rps", horizon=30, seed=42, replicates=(0, 1)):
    return [run_cross_play_experiment(game, profile.split("_vs_"), feedback_mode=mode, horizon=horizon, seed=seed, replicate=replicate,
                   output_dir=service.raw_dir, max_recorded_points=10) for replicate in replicates]


def create_service(tmp_path: Path) -> DashboardService:
    return DashboardService(
        results_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        figure_dir=tmp_path / "figures",
        custom_game_dir=tmp_path / "custom-games",
    )


def create_test_app(
    tmp_path: Path,
    *,
    max_replicates: int = 100,
):
    service = create_service(tmp_path)
    config = {
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "MAX_HORIZON": 100,
        "MAX_REPLICATES": max_replicates,
    }
    return create_app(config, service=service), service


def block_job_queue(manager):
    started = Event()
    release = Event()

    def block(_job) -> None:
        started.set()
        assert release.wait(timeout=2)

    blocker = manager.submit("blocker", block)
    assert started.wait(timeout=1)
    return blocker, release


def csrf_token(client) -> str:
    client.get("/")
    with client.session_transaction() as session:
        return session["_csrf_token"]


def wait_for_job(owner, job_id: str) -> str:
    manager = getattr(owner, "jobs", owner)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job is not None and job.status in {"succeeded", "failed", "cancelled"}:
            return job.status
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def wait_for_http_response(client, url: str, *, headers: dict | None = None, timeout: int = 5):
    deadline = time.monotonic() + timeout
    statuses = []
    while time.monotonic() < deadline:
        response = client.get(url, headers=headers)
        statuses.append(response.status_code)
        if response.status_code != 202:
            return response, statuses
        time.sleep(0.01)
    raise AssertionError(f"request {url} did not finish")


def wait_for_async_result(request_result, timeout: int = 5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result, error = request_result()
        if result is not None or error is not None:
            return result, error
        time.sleep(0.01)
    raise AssertionError("asynchronous operation did not finish")


def dashboard_data(response):
    assert response.status_code == 200
    return json.loads(response.get_data(as_text=True).split(
        '<script id="dashboard-data" type="application/json">', 1)[1].split('</script>', 1)[0])


def submit_and_wait(client, service, values, url="/"):
    response = client.post(url, data=values | {"_csrf_token": csrf_token(client)})
    assert response.status_code == 302
    job = service.jobs.recent()[0]
    assert wait_for_job(service, job.id) == "succeeded"
    return job


def run_node(script, *args, payload=None, jsdom=False):
    node = shutil.which("node")
    if node is None or (jsdom and subprocess.run([node, "-e", "require('jsdom')"], capture_output=True).returncode):
        pytest.skip("Node.js" + (" with jsdom" if jsdom else "") + " is unavailable")
    completed = subprocess.run([node, "-e", script, *map(str, args)],
        input=json.dumps(payload) if payload is not None else None, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def run_ui(app, service, script, mode="fixed", **data):
    static = Path(__file__).parents[2] / "web/static"
    payload = {"page": app.test_client().get("/", query_string={"mode": mode}).get_data(as_text=True),
               "catalog": service.figure_builder.catalog(mode),
               "script": "\n".join((static / name).read_text() for name in ("common.js", "dashboard.js", "figure_builder.js"))} | data
    run_node(script, payload=payload, jsdom=True)
