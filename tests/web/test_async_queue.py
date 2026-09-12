import pytest

from tests.web.support import create_test_app, csrf_token
from web.jobs import Job, ServiceBusyError


FIXED = {
    "game": "rps", "feedback_mode": "full_information",
    "algorithm_names": ["hedge", "hedge"], "horizon": "10", "seed": "42", "replicates": "1",
}
ADVERSARIAL = FIXED | {
    "experiment_type": "adversarial", "algorithm_names": ["hedge"],
    "environment": "lazy_random_walk_v1",
    "n_actions": "9", "environment_seed": "42", "scaling_action_counts": "3,9", "scaling_replicates": "1",
}
CASES = [
    ("/", FIXED, "submit_experiment", "fixed"),
    ("/", ADVERSARIAL, "submit_adversarial_experiment", "adversarial"),
    ("/adversarial/action-scaling", ADVERSARIAL, "submit_adversarial_scaling_experiment", "adversarial"),
]


@pytest.mark.parametrize("url,form,method,mode", CASES)
def test_queue_returns_job_without_redirect_for_background_submission(tmp_path, monkeypatch, url, form, method, mode):
    app, service = create_test_app(tmp_path)
    client = app.test_client()
    token = csrf_token(client)
    calls = []

    def submit(parsed_form):
        calls.append(parsed_form)
        return Job("job123", "Experiment <script>unsafe</script>", "queued", "Waiting to start", "now")

    monkeypatch.setattr(service, method, submit)
    response = client.post(url, data=form | {"_csrf_token": token}, headers={"Accept": "application/json"})

    assert response.status_code == 202
    assert "Location" not in response.headers
    assert len(calls) == 1
    assert response.json["job"]["url"] == "/jobs/job123"
    assert response.json["job"]["status"] == "queued"
    html = response.json["job_html"]
    assert 'data-job-id="job123"' in html
    assert 'action="/jobs/job123/cancel"' in html
    assert f'value="{token}"' in html
    assert f'value="{mode}"' in html
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in html
    with client.session_transaction() as session:
        assert not session.get("_flashes")


@pytest.mark.parametrize("url,form,method,mode", CASES)
def test_background_queue_returns_validation_errors_in_place(tmp_path, url, form, method, mode):
    app, service = create_test_app(tmp_path)
    client = app.test_client()
    response = client.post(url, data=form | {"horizon": "0", "_csrf_token": csrf_token(client)},
                           headers={"Accept": "application/json"})
    assert response.status_code == 400
    assert "horizon" in response.json["error"].lower()
    assert service.jobs.recent() == []


@pytest.mark.parametrize("error", [FileExistsError("already queued"), ServiceBusyError("busy"), ValueError("invalid")])
def test_background_queue_preserves_duplicate_and_busy_errors(tmp_path, monkeypatch, error):
    app, service = create_test_app(tmp_path)
    client = app.test_client()

    def submit(_form):
        raise error

    monkeypatch.setattr(service, "submit_experiment", submit)
    response = client.post("/", data=FIXED | {"_csrf_token": csrf_token(client)}, headers={"Accept": "application/json"})
    assert response.status_code == 400
    assert response.json["error"] == str(error)


def test_background_queue_still_requires_csrf(tmp_path):
    app, service = create_test_app(tmp_path)
    response = app.test_client().post("/", data=FIXED, headers={"Accept": "application/json"})
    assert response.status_code == 400
    assert service.jobs.recent() == []


@pytest.mark.parametrize("url,form,method,mode", CASES)
def test_queue_retains_plain_html_form_fallback(tmp_path, monkeypatch, url, form, method, mode):
    app, service = create_test_app(tmp_path)
    client = app.test_client()
    monkeypatch.setattr(service, method, lambda _form: Job("job123", "Experiment", "queued", "Waiting", "now"))
    response = client.post(url, data=form | {"_csrf_token": csrf_token(client)})
    assert response.status_code == 302
    assert response.headers["Location"] == ("/" if mode == "fixed" else "/?mode=adversarial")
