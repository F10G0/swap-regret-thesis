from pathlib import Path
import time

import pytest

from web import create_app
from web.services import DashboardService
from web.validation import parse_adversarial_experiment_form


VALID_FORM = {
    "algorithm_name": "hedge",
    "n_actions": "3",
    "memory_window": "0",
    "horizon": "4",
    "seed": "7",
}


def _app(tmp_path: Path):
    service = DashboardService(
        results_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        figure_dir=tmp_path / "figures",
        custom_game_dir=tmp_path / "custom-games",
    )
    service._publish_adversarial_plots = lambda: None
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "MAX_HORIZON": 100,
            "MAX_REPLICATES": 10,
        },
        service=service,
    )
    return app, service


def _csrf_token(client) -> str:
    client.get("/adversarial")
    with client.session_transaction() as session:
        return session["_csrf_token"]


def _wait_for_job(service: DashboardService, job_id: str) -> str:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = service.jobs.get(job_id)
        if job is not None and job.status in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            return job.status
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def test_adversarial_form_validation() -> None:
    form = parse_adversarial_experiment_form(
        VALID_FORM,
        algorithms=["hedge"],
        max_actions=100,
        max_horizon=100,
    )

    assert form.algorithm_name == "hedge"
    assert form.n_actions == 3
    assert form.memory_window == 0

    custom = parse_adversarial_experiment_form(
        VALID_FORM | {"memory_window": "37"},
        algorithms=["hedge"],
        max_actions=100,
        max_horizon=100,
    )
    assert custom.memory_window == 37


@pytest.mark.parametrize("memory_window", ["-1", "101", "invalid"])
def test_adversarial_memory_window_validation(memory_window: str) -> None:
    with pytest.raises(ValueError, match="memory window"):
        parse_adversarial_experiment_form(
            VALID_FORM | {"memory_window": memory_window},
            algorithms=["hedge"],
            max_actions=100,
            max_horizon=100,
        )


def test_adversarial_page_is_separate_from_fixed_game_controls(tmp_path) -> None:
    app, _ = _app(tmp_path)

    client = app.test_client()
    page = client.get("/adversarial").get_data(as_text=True)
    script = client.get("/static/adversarial.js").get_data(as_text=True)

    assert "Punish the historical leader" in page
    assert 'name="algorithm_name"' in page
    assert 'name="n_actions"' in page
    assert 'name="memory_window"' in page
    assert "0 means full history" in page
    assert 'name="replicates"' not in page
    assert 'name="replicate"' not in page
    assert 'name="game"' not in page
    assert "CE/CCE" not in page
    assert "Clear results" in page
    assert "swap-regret-adversarial-form" in script
    assert "adversarialForm.elements" in script


def test_adversarial_page_queues_one_run_and_renders_results(tmp_path) -> None:
    app, service = _app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/adversarial",
        data=VALID_FORM
        | {"memory_window": "37", "_csrf_token": _csrf_token(client)},
    )

    assert response.status_code == 302
    job = service.jobs.recent()[0]
    assert _wait_for_job(service, job.id) == "succeeded"
    assert len(list(service.adversarial_raw_dir.glob("*.csv"))) == 1

    page = client.get("/adversarial").get_data(as_text=True)
    assert "Final target regret" in page
    assert page.count("Download</a>") == 1
    assert ">Expected</th>" in page
    assert ">Realized</th>" in page
    assert "Last 37 rounds" in page


def test_adversarial_download_and_figure_routes_are_scoped(tmp_path) -> None:
    app, service = _app(tmp_path)
    client = app.test_client()
    client.post(
        "/adversarial",
        data=VALID_FORM | {"_csrf_token": _csrf_token(client)},
    )
    job = service.jobs.recent()[0]
    assert _wait_for_job(service, job.id) == "succeeded"
    filename = next(service.adversarial_raw_dir.glob("*.csv")).name

    download = client.get(f"/adversarial/experiments/{filename}")
    traversal = client.get("/adversarial/experiments/../outside.csv")

    assert download.status_code == 200
    assert download.headers["Content-Disposition"].startswith("attachment")
    assert traversal.status_code == 404

    service.adversarial_figure_dir.mkdir(parents=True, exist_ok=True)
    figure_names = (
        "historical_frequency_3_actions_average_expected_external_regret.png",
        "historical_frequency_3_actions_expected_external_regret_over_sqrt_t.png",
    )
    for figure_name in figure_names:
        pdf_name = Path(figure_name).with_suffix(".pdf").name
        (service.adversarial_figure_dir / figure_name).write_bytes(b"png")
        (service.adversarial_figure_dir / pdf_name).write_bytes(b"pdf")
        assert client.get(f"/adversarial/figures/{figure_name}").status_code == 200
        assert (
            client.get(f"/adversarial/figures/{pdf_name}").content_type
            == "application/pdf"
        )

    for diagnostic in ("action", "punished_action"):
        behavior_name = f"{Path(filename).stem}_{diagnostic}_frequency.png"
        (service.adversarial_figure_dir / behavior_name).write_bytes(b"png")
        (service.adversarial_figure_dir / Path(behavior_name).with_suffix(".pdf")).write_bytes(b"pdf")
        assert (
            client.get(f"/adversarial/figures/{behavior_name}").status_code
            == 200
        )
    page = client.get("/adversarial").get_data(as_text=True)
    assert "Learner and adversary behavior" in page
    assert "Learner action frequency" in page
    assert "Punished-action frequency" in page
    assert "Average external" in page
    assert "External / sqrt(t)" in page
    assert "Download PDF" in page


def test_clear_adversarial_results_deletes_csvs_and_figures(tmp_path) -> None:
    app, service = _app(tmp_path)
    client = app.test_client()
    service.adversarial_raw_dir.mkdir(parents=True)
    service.adversarial_figure_dir.mkdir(parents=True)
    csv_path = service.adversarial_raw_dir / "result.csv"
    csv_path.write_text("data", encoding="utf-8")
    for index in range(2):
        (service.adversarial_figure_dir / f"figure-{index}.png").write_bytes(
            b"png"
        )

    response = client.post(
        "/adversarial/results/clear",
        data={"_csrf_token": _csrf_token(client)},
    )

    assert response.status_code == 302
    assert list(service.adversarial_figure_dir.glob("*.png")) == []
    assert not csv_path.exists()
