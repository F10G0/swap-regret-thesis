from pathlib import Path
import csv

import pytest

from tests.web.support import ADVERSARIAL_FORM as VALID_FORM, create_test_app, csrf_token as _csrf_token, dashboard_data, submit_and_wait
from web.validation import (
    parse_action_counts,
    parse_adversarial_experiment_form,
    parse_adversarial_scaling_form,
)
from experiments.scenarios.adversarial import (
    HISTORICAL_FREQUENCY_ENVIRONMENT,
    RANDOM_WALK_ENVIRONMENT,
)
from experiments.seeding import (
    ENVIRONMENT_SEED_DOMAIN,
    LEARNER_SEED_DOMAIN,
    domain_separated_seed,
)


ENVIRONMENTS = {HISTORICAL_FREQUENCY_ENVIRONMENT, RANDOM_WALK_ENVIRONMENT}


def _app(tmp_path: Path):
    return create_test_app(tmp_path, max_replicates=10)


def _opening_tag(page: str, element_id: str) -> str:
    before, after = page.split(f'id="{element_id}"', 1)
    return before.rsplit("<", 1)[1] + f'id="{element_id}"' + after.split(">", 1)[0]


def test_adversarial_form_validation() -> None:
    form = parse_adversarial_experiment_form(
        VALID_FORM,
        algorithms_by_feedback_mode={
            "full_information": ["hedge"],
            "bandit": ["exp3_ix"],
        },
        environments=ENVIRONMENTS,
        max_actions=100,
        max_horizon=100,
    )

    assert form.algorithm_name == "hedge"
    assert form.feedback_mode == "full_information"
    assert form.n_actions == 3
    assert form.replicates == 1

    with pytest.raises(ValueError, match="replicates must not exceed 10"):
        parse_adversarial_experiment_form(
            VALID_FORM | {"replicates": "11"},
            algorithms_by_feedback_mode={"full_information": ["hedge"]},
            environments=ENVIRONMENTS,
            max_actions=100,
            max_horizon=100,
            max_replicates=10,
        )


def test_adversarial_scaling_form_validation() -> None:
    form = parse_adversarial_scaling_form(
        VALID_FORM | {
            "scaling_action_counts": "10, 2 5",
            "scaling_replicates": "4",
        },
        algorithms_by_feedback_mode={"full_information": ["hedge"]},
        environments=ENVIRONMENTS,
        max_actions=100,
        max_horizon=100,
        max_replicates=10,
    )

    assert form.action_counts == (2, 5, 10)
    assert form.replicates == 4
    assert form.horizon == 4


@pytest.mark.parametrize("value", ["2", "1, 2", "2, 2", "2, invalid"])
def test_action_count_list_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_action_counts(value, max_actions=100)


def test_adversarial_algorithm_must_match_feedback_mode() -> None:
    with pytest.raises(ValueError, match="not available for bandit"):
        parse_adversarial_experiment_form(
            VALID_FORM | {"feedback_mode": "bandit"},
            algorithms_by_feedback_mode={
                "full_information": ["hedge"],
                "bandit": ["exp3_ix"],
            },
            environments=ENVIRONMENTS,
            max_actions=100,
            max_horizon=100,
        )


def test_experiments_page_switches_to_one_player_controls(tmp_path):
    app, _ = _app(tmp_path)
    client = app.test_client()
    fixed = client.get("/")
    adversarial = client.get("/?mode=adversarial")
    assert dashboard_data(fixed)["mode"] == "fixed"
    assert dashboard_data(adversarial)["mode"] == "adversarial"
    assert b'name="game"' in fixed.data and b'name="game"' not in adversarial.data
    for name in ("environment", "algorithm_names", "n_actions", "scaling_action_counts", "scaling_replicates"):
        assert f'name="{name}"'.encode() in adversarial.data


def test_adversarial_form_controls_match_the_rendered_environment(tmp_path) -> None:
    app, _ = _app(tmp_path)
    client = app.test_client()

    historical_page = client.get("/?mode=adversarial").get_data(as_text=True)
    assert "hidden" in _opening_tag(historical_page, "adversarial-environment-seed-field")
    assert "hidden" in _opening_tag(historical_page, "environment-panel")
    assert "hidden" in _opening_tag(historical_page, "historical-frequency-rule")
    assert "hidden" in _opening_tag(historical_page, "random-walk-rule")

    response = client.post(
        "/",
        data=VALID_FORM | {
            "environment": RANDOM_WALK_ENVIRONMENT,
            "replicates": "0",
            "_csrf_token": _csrf_token(client),
        },
    )
    assert response.status_code == 400
    random_walk_page = response.get_data(as_text=True)
    assert "hidden" not in _opening_tag(random_walk_page, "adversarial-environment-seed-field")
    assert "hidden" in _opening_tag(random_walk_page, "environment-panel")
    assert "hidden" in _opening_tag(random_walk_page, "historical-frequency-rule")
    assert "hidden" in _opening_tag(random_walk_page, "random-walk-rule")


@pytest.mark.parametrize("mode", ["fixed", "adversarial"])
def test_algorithm_options_follow_feedback_mode(tmp_path, mode) -> None:
    algorithm = "auer_exp3"
    app, _ = _app(tmp_path)
    client = app.test_client()
    labels = dashboard_data(client.get("/", query_string={"mode": mode}))["algorithmLabels"]
    assert {name: labels[name] for name in ("lce_ix", "auer_exp3", "regret_matching", "stationary_regret_matching")} == {
        "lce_ix": "LCE-IX", "auer_exp3": "AuerExp3", "regret_matching": "RM", "stationary_regret_matching": "SRM",
    }
    response = client.post(
        "/",
        data=VALID_FORM | {
            "experiment_type": mode,
            "game": "rps",
            "feedback_mode": "bandit",
            "algorithm_names": [algorithm] * (2 if mode == "fixed" else 1),
            "replicates": "0",
            "_csrf_token": _csrf_token(client),
        },
    )

    assert response.status_code == 400
    page = response.get_data(as_text=True)
    options = page.split('<select id="algorithm_player_0"', 1)[1].split("</select>", 1)[0]
    assert f'<option value="{algorithm}" selected' in options
    assert '<option value="auer_exp3"' in options
    assert "AuerExp3" in options
    assert '<option value="exp3"' not in options
    assert '<option value="lce_ix"' in options
    assert '<option value="hedge"' not in options


def test_adversarial_page_runs_action_space_scaling_batch(tmp_path):
    app, service = create_test_app(tmp_path, max_replicates=10)
    client = app.test_client()
    submit_and_wait(client, service, VALID_FORM | {"scaling_action_counts": "2, 4",
        "scaling_replicates": "2"}, "/adversarial/action-scaling")
    path = next(service.adversarial_scaling_raw_dir.glob("*.csv"))
    assert len(path.read_text().splitlines()) == 5
    figures = service.adversarial_scaling_figure_records()
    assert len(figures) == 1
    assert b"Regret by action-space size" in client.get("/?mode=adversarial").data
    assert client.get(f"/adversarial/action-scaling/experiments/{path.name}").data == path.read_bytes()
    for key, mimetype in (("filename", "image/png"), ("pdf_filename", "application/pdf")):
        response = client.get(f"/adversarial/action-scaling/figures/{figures[0][key]}")
        assert response.status_code == 200 and response.mimetype == mimetype
    response = client.post("/adversarial/action-scaling/delete-experiment", data={
        "filename": path.name, "_csrf_token": _csrf_token(client)})
    assert response.status_code == 302 and not path.exists()


@pytest.mark.parametrize("workers", [1, 2])
def test_adversarial_page_queues_replicates_with_common_seed_schedule(tmp_path, workers):
    app, service = _app(tmp_path)
    service.replicate_workers = workers
    job = submit_and_wait(app.test_client(), service, VALID_FORM | {
        "environment": RANDOM_WALK_ENVIRONMENT, "replicates": "3"})
    completed = service.jobs.get(job.id)
    assert (completed.completed, completed.total) == (3, 3)
    assert not list(tmp_path.rglob("*.png")) and not list(tmp_path.rglob("*.pdf"))
    results = service.result_snapshot("adversarial")
    summaries, warnings = results.summaries(), list(results.warnings)
    assert warnings == [] and {row["replicate"] for row in summaries} == {0, 1, 2}
    assert {row["base_environment_seed"] for row in summaries} == {11}
    assert {row["base_learner_seed"] for row in summaries} == {7}
    for row in summaries:
        assert row["environment_seed"] == domain_separated_seed(11, row["replicate"], ENVIRONMENT_SEED_DOMAIN)
        assert row["learner_seed"] == domain_separated_seed(7, row["replicate"], LEARNER_SEED_DOMAIN)


@pytest.mark.parametrize("environment,feedback,algorithm", [
    (HISTORICAL_FREQUENCY_ENVIRONMENT, "full_information", "hedge"),
    (RANDOM_WALK_ENVIRONMENT, "bandit", "auer_exp3"),
])
def test_adversarial_submission_results_and_deletion(tmp_path, environment, feedback, algorithm):
    app, service = _app(tmp_path)
    client = app.test_client()
    submit_and_wait(client, service, VALID_FORM | {"environment": environment,
        "feedback_mode": feedback, "algorithm_names": [algorithm]})
    path = next(service.adversarial_raw_dir.glob("*.csv"))
    with path.open(newline="") as file:
        row = next(csv.DictReader(file))
    assert (row["environment"], row["feedback_mode"], row["algorithm"]) == (environment, feedback, algorithm)
    results = service.result_snapshot("adversarial")
    summaries, warnings = results.summaries(), list(results.warnings)
    assert not warnings and len(summaries) == 1
    assert summaries[0]["average_regret"] is not None
    assert b"Final regret summary" in client.get("/?mode=adversarial").data
    assert client.get(f"/adversarial/experiments/{path.name}").data == path.read_bytes()
    response = client.post("/adversarial/delete-experiment", data={
        "filename": path.name, "_csrf_token": _csrf_token(client)})
    assert response.status_code == 302 and not path.exists()


def test_adversarial_csv_download_is_scoped(tmp_path):
    app, service = _app(tmp_path)
    client = app.test_client()
    service.adversarial_raw_dir.mkdir(parents=True)
    (service.adversarial_raw_dir / "recorded.csv").write_bytes(b"recorded")
    download = client.get("/adversarial/experiments/recorded.csv")
    assert download.status_code == 200 and download.data == b"recorded"
    assert download.headers["Content-Disposition"].startswith("attachment")
    assert client.get("/adversarial/experiments/../outside.csv").status_code == 404


def test_clear_adversarial_results_deletes_csvs_and_figures(tmp_path) -> None:
    app, service = _app(tmp_path)
    client = app.test_client()
    service.adversarial_raw_dir.mkdir(parents=True)
    service.adversarial_scaling_raw_dir.mkdir(parents=True)
    service.adversarial_scaling_figure_dir.mkdir(parents=True)
    csv_path = service.adversarial_raw_dir / "result.csv"
    csv_path.write_text("data", encoding="utf-8")
    scaling_csv = service.adversarial_scaling_raw_dir / "scaling.csv"
    scaling_csv.write_text("data", encoding="utf-8")
    scaling_figure = service.adversarial_scaling_figure_dir / "scaling.png"
    scaling_figure.write_bytes(b"png")

    response = client.post(
        "/adversarial/results/clear",
        data={"_csrf_token": _csrf_token(client)},
    )

    assert response.status_code == 302
    assert not csv_path.exists()
    assert not scaling_csv.exists()
    assert not scaling_figure.exists()
