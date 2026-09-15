from pathlib import Path
import csv

import pytest

from tests.web.support import ADVERSARIAL_FORM as VALID_FORM, create_test_app, csrf_token as _csrf_token, dashboard_data, submit_and_wait
from web.validation import (
    parse_action_counts,
    parse_adversarial_experiment_form,
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
    assert form.action_counts == (3,)
    assert form.seed == 7
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


@pytest.mark.parametrize(("value", "expected"), [
    ("9", (9,)), ("3,5,10", (3, 5, 10)), ("3 5 10", (3, 5, 10)), ("10, 2 5", (10, 2, 5)),
])
def test_action_count_list_accepts_supported_separators_and_preserves_order(value, expected) -> None:
    assert parse_action_counts(value, max_actions=100) == expected


@pytest.mark.parametrize("value", ["", "1", "101", "2, 2", "2, invalid"])
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
    for response, card_class, label, select_id, hint_id in (
        (fixed, "control-card-game", "Game", "game", "game-description"),
        (adversarial, "control-card-environment", "Environment", "adversarial-environment", "environment-description"),
    ):
        card = response.get_data(as_text=True).split(f'<div class="control-card {card_class}">', 1)[1].split("</div>", 1)[0]
        assert '<div class="field">' not in card
        assert card.index(f'<label for="{select_id}">{label}</label>') < card.index(f'<select id="{select_id}"') < card.index(f'id="{hint_id}" class="hint"')
    for name in ("environment", "algorithm_names", "actions"):
        assert f'name="{name}"'.encode() in adversarial.data
    page = adversarial.get_data(as_text=True)
    assert page.index('id="feedback-mode"') < page.index('id="actions"') < page.index('id="horizon"')
    assert "One value runs a standard experiment; multiple values run action-space scaling." in page
    for removed in ("Action-space scaling", "Replicates per K", "Queue scaling experiment", "Data management", "Raw experiment files", "Adversarial CSV files", "Action-scaling CSV files"):
        assert removed not in page
    for response in (fixed, adversarial):
        rendered = response.get_data(as_text=True)
        assert rendered.count('name="seed"') == 1
        assert 'id="experiment-seed" name="seed" form="experiment-form"' in rendered
        assert rendered.index('class="brand section-heading sidebar-heading"') < rendered.index('id="experiment-seed"') < rendered.index('id="experiment-mode"')
        assert 'class="field-grid field-grid-two execution-size-grid"' in rendered
        assert rendered.index('id="horizon"') < rendered.index('id="replicates"') < rendered.index('id="players"')
        assert rendered.count("Reset all experiment results") == 1
        assert rendered.count('action="/reset"') == 1
        assert rendered.count("Queue experiment") == 1
        for obsolete_copy in ("Choose whether learners observe", "Number of rounds run", "Run replicates 0 through", "Shared by figures", "Figures for the current result selection", "Stored experiment results"):
            assert obsolete_copy not in rendered
    assert b'id="primary-theme"' not in fixed.data and b'>Theme<' not in fixed.data
    assert b"Queue experiment" in fixed.data and b"Queue experiment" in adversarial.data
    css = (Path(__file__).parents[2] / "web" / "static" / "dashboard.css").read_text()
    common = (Path(__file__).parents[2] / "web" / "static" / "common.js").read_text()
    assert "--accent: #176b4d;" in css and ':root[data-theme=' not in css
    assert "primary-theme" not in common and "themeStorageKey" not in common
    assert "input:focus-visible,\nselect:focus-visible," in css
    assert "outline: 2px solid var(--accent-focus);" in css


def test_one_player_form_has_one_seed_and_sidebar_environment_description(tmp_path) -> None:
    app, _ = _app(tmp_path)
    client = app.test_client()

    historical_page = client.get("/?mode=adversarial").get_data(as_text=True)
    assert historical_page.count('name="seed"') == 1
    assert "Environment seed" not in historical_page and 'name="environment_seed"' not in historical_page
    assert 'id="environment-description"' in historical_page
    assert "The most frequent half of actions (rounded up)" in historical_page
    assert 'aria-describedby="environment-description"' in historical_page
    assert 'id="environment-description" class="hint"' in historical_page
    assert 'id="environment-panel"' not in historical_page

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
    assert random_walk_page.count('name="seed"') == 1
    assert "Environment seed" not in random_walk_page and 'name="environment_seed"' not in random_walk_page
    assert "Independent lazy random walks derived reproducibly from the experiment seed." in random_walk_page
    assert 'id="environment-panel"' not in random_walk_page


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
    submit_and_wait(client, service, VALID_FORM | {"actions": "2, 4", "replicates": "2"})
    path = next(service.adversarial_scaling_raw_dir.glob("*.csv"))
    assert len(path.read_text().splitlines()) == 5
    figures = service.adversarial_scaling_figure_records()
    assert len(figures) == 1
    catalog = client.get("/figure-builder/options?mode=adversarial").get_json()
    assert catalog["contexts"] == [] and "scaling" not in catalog
    rejected = client.post("/figure-builder/collection", data={
        "_csrf_token": _csrf_token(client), "mode": "adversarial", "context_id": "a" * 24,
        "comparison_mode": "profiles", "metric": "external", "view": "average", "profiles": "hedge",
    })
    assert rejected.status_code == 400
    page = client.get("/?mode=adversarial").get_data(as_text=True)
    assert page.index('id="summary-heading"') < page.index('id="scaling-results"')
    assert '<p class="eyebrow">Scaling</p>' in page
    assert '<h2 id="scaling-results-heading">Action-space scaling</h2>' in page
    assert "Final regret across action-space sizes." in page
    assert '<p class="eyebrow">Optional</p>' not in page and "Regret by action-space size" not in page
    assert "data-result-card" not in page and "data-result-section" not in page
    assert client.get(f"/adversarial/action-scaling/experiments/{path.name}").data == path.read_bytes()
    for key, mimetype in (("filename", "image/png"), ("pdf_filename", "application/pdf")):
        response = client.get(f"/adversarial/action-scaling/figures/{figures[0][key]}")
        assert response.status_code == 200 and response.mimetype == mimetype
    assert path.exists()


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
    assert {row["base_environment_seed"] for row in summaries} == {7}
    assert {row["base_learner_seed"] for row in summaries} == {7}
    for row in summaries:
        assert row["environment_seed"] == domain_separated_seed(7, row["replicate"], ENVIRONMENT_SEED_DOMAIN)
        assert row["learner_seed"] == domain_separated_seed(7, row["replicate"], LEARNER_SEED_DOMAIN)
    grouped = results.summaries(grouped=True)
    assert len(grouped) == 1 and grouped[0]["replicate_label"] == "0–2" and grouped[0]["replicate_count"] == 3
    for field in ("average_external_regret", "average_internal_regret", "average_swap_regret"):
        assert grouped[0][field] == pytest.approx(sum(row[field] for row in summaries) / 3)
    page = app.test_client().get("/?mode=adversarial")
    assert len(dashboard_data(page)["summaries"]) == 1
    assert b"1 replicate group" in page.data
    assert b"Seed / replicates" in page.data and b"<th>Replicate</th>" not in page.data
    submit_and_wait(app.test_client(), service, VALID_FORM | {
        "environment": RANDOM_WALK_ENVIRONMENT, "replicates": "1", "seed": "8"})
    assert len(dashboard_data(app.test_client().get("/?mode=adversarial"))["summaries"]) == 2


@pytest.mark.parametrize("environment,feedback,algorithm", [
    (HISTORICAL_FREQUENCY_ENVIRONMENT, "full_information", "hedge"),
    (RANDOM_WALK_ENVIRONMENT, "bandit", "auer_exp3"),
])
def test_adversarial_submission_persists_results(tmp_path, environment, feedback, algorithm):
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
    assert b"Recorded output" in client.get("/?mode=adversarial").data
    assert client.get(f"/adversarial/experiments/{path.name}").data == path.read_bytes()
    assert path.exists()


def test_adversarial_csv_download_is_scoped(tmp_path):
    app, service = _app(tmp_path)
    client = app.test_client()
    service.adversarial_raw_dir.mkdir(parents=True)
    (service.adversarial_raw_dir / "recorded.csv").write_bytes(b"recorded")
    download = client.get("/adversarial/experiments/recorded.csv")
    assert download.status_code == 200 and download.data == b"recorded"
    assert download.headers["Content-Disposition"].startswith("attachment")
    assert client.get("/adversarial/experiments/../outside.csv").status_code == 404


def test_global_reset_from_one_player_deletes_all_experiment_artifacts(tmp_path) -> None:
    app, service = _app(tmp_path)
    client = app.test_client()
    service.adversarial_raw_dir.mkdir(parents=True)
    service.adversarial_scaling_raw_dir.mkdir(parents=True)
    service.adversarial_scaling_figure_dir.mkdir(parents=True)
    service.raw_dir.mkdir(parents=True)
    service.figure_builder.output_dir.mkdir(parents=True)
    fixed_csv = service.raw_dir / "fixed.csv"
    fixed_csv.write_text("data", encoding="utf-8")
    csv_path = service.adversarial_raw_dir / "result.csv"
    csv_path.write_text("data", encoding="utf-8")
    scaling_csv = service.adversarial_scaling_raw_dir / "scaling.csv"
    scaling_csv.write_text("data", encoding="utf-8")
    scaling_figure = service.adversarial_scaling_figure_dir / "scaling.png"
    scaling_figure.write_bytes(b"png")
    cached_figure = service.figure_builder.output_dir / "cached.png"
    cached_figure.write_bytes(b"png")

    response = client.post(
        "/reset",
        data={"_csrf_token": _csrf_token(client), "confirmation": "reset-results", "return_to": "adversarial"},
    )

    assert response.status_code == 302
    assert response.headers["Location"] == "/?mode=adversarial"
    assert not any(path.exists() for path in (fixed_csv, csv_path, scaling_csv, scaling_figure, cached_figure))
