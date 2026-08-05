from pathlib import Path
import time

from web import create_app
from web.services import DashboardService


def create_service(tmp_path: Path) -> DashboardService:
    service = DashboardService(
        results_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        figure_dir=tmp_path / "figures",
        custom_game_dir=tmp_path / "custom-games",
    )
    service._publish_plots = lambda game_name=None: None
    return service


def create_test_app(
    tmp_path: Path,
    *,
    experimental: bool = True,
    max_replicates: int = 100,
    disable_adversarial_plots: bool = False,
):
    service = create_service(tmp_path)
    if disable_adversarial_plots:
        service._publish_adversarial_plots = lambda: None
    config = {
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "MAX_HORIZON": 100,
        "MAX_REPLICATES": max_replicates,
    }
    if experimental:
        config["TEST_ENABLE_EXPERIMENTAL_TRAJECTORIES"] = True
    return create_app(config, service=service), service


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
