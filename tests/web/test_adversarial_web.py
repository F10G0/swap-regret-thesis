from pathlib import Path
import csv
import json
import shutil
import subprocess

import pytest

from tests.web.support import create_test_app, csrf_token as _csrf_token, wait_for_job as _wait_for_job
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


VALID_FORM = {
    "experiment_type": "adversarial",
    "environment": HISTORICAL_FREQUENCY_ENVIRONMENT,
    "feedback_mode": "full_information",
    "algorithm_names": ["hedge"],
    "n_actions": "3",
    "horizon": "4",
    "environment_seed": "11",
    "seed": "7",
    "replicates": "1",
}
ENVIRONMENTS = {HISTORICAL_FREQUENCY_ENVIRONMENT, RANDOM_WALK_ENVIRONMENT}


def _app(tmp_path: Path):
    return create_test_app(tmp_path, max_replicates=10, disable_adversarial_plots=True)


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
    assert not hasattr(form, "regret_evaluation")

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
    assert not hasattr(form, "regret_evaluation")
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


def test_experiments_page_switches_to_one_player_controls(tmp_path) -> None:
    app, _ = _app(tmp_path)

    client = app.test_client()
    fixed_page = client.get("/").get_data(as_text=True)
    page = client.get("/?mode=adversarial").get_data(as_text=True)
    script = client.get("/static/dashboard.js").get_data(as_text=True)
    payload = page.split('<script id="dashboard-data" type="application/json">', 1)[1].split("</script>", 1)[0]

    assert "Experiment type" in page
    assert 'id="experiment-mode"' in page
    assert '<option value="fixed"' in page
    assert '<option value="adversarial" selected' in page
    assert ">Experiments</a>" in page
    assert ">Adversarial</a>" not in page
    assert "dashboard.js" in page
    assert "adversarial.js" not in page
    assert "Punish the historical leaders" in page
    assert 'name="algorithm_names"' in page
    for algorithm in ("hedge", "bm", "ito", "regret_matching", "stationary_regret_matching"):
        assert f'<option value="{algorithm}"' in page
    assert 'name="environment"' in page
    assert 'class="control-card control-card-environment"' in page
    assert 'name="initialization_mode"' not in page
    assert 'name="feedback_mode"' in page
    assert 'name="regret_evaluation"' not in page
    assert '<option value="expected"' not in page
    assert '<option value="realized"' not in page
    assert '<option value="both"' not in page
    assert 'name="n_actions"' in page
    assert 'name="memory_window"' not in page
    assert 'name="environment_seed"' in page
    assert 'name="seed"' in page
    assert 'name="scaling_action_counts"' in page
    assert 'name="scaling_replicates"' in page
    assert "Queue scaling experiment" in page
    assert "complete history" in page
    assert "exp3_ix" in page
    assert "Independent lazy random walk" in page
    assert "Independent lazy reward walks" in page
    assert 'name="replicates"' in page
    assert 'name="replicate"' not in page
    assert 'name="game"' not in page
    assert "CE/CCE" not in page
    assert "Clear results" in page
    assert "swap-regret-adversarial-form" in script
    assert "onePlayerMode" in script
    assert "updateOnePlayerEnvironment" in script
    assert json.loads(payload)["mode"] == "adversarial"
    for name in (
        "feedback_mode",
        "horizon",
        "seed",
        "algorithm_names",
        "replicates",
    ):
        assert f'name="{name}"' in fixed_page
        assert f'name="{name}"' in page
    assert 'action="/"' in fixed_page
    assert 'action="/"' in page


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


@pytest.mark.parametrize("algorithm", ["auer_exp3", "exp3_ix"])
def test_adversarial_algorithm_options_follow_feedback_mode(tmp_path, algorithm) -> None:
    app, _ = _app(tmp_path)
    client = app.test_client()
    response = client.post(
        "/",
        data=VALID_FORM | {
            "feedback_mode": "bandit",
            "algorithm_names": [algorithm],
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


def test_adversarial_page_runs_action_space_scaling_batch(tmp_path) -> None:
    app, service = create_test_app(
        tmp_path,
        max_replicates=10,
    )
    client = app.test_client()

    response = client.post(
        "/adversarial/action-scaling",
        data=VALID_FORM | {
            "scaling_action_counts": "2, 4",
            "scaling_replicates": "2",
            "_csrf_token": _csrf_token(client),
        },
    )

    assert response.status_code == 302
    job = service.jobs.recent()[0]
    assert _wait_for_job(service, job.id) == "succeeded"
    result = next(service.adversarial_scaling_raw_dir.glob("*.csv"))
    assert len(result.read_text(encoding="utf-8").splitlines()) == 5
    figures = service.adversarial_scaling_figure_records()
    assert len(figures) == 1
    assert "source" not in figures[0]

    page = client.get("/?mode=adversarial").get_data(as_text=True)
    assert "Regret by action-space size" in page
    assert "K=2, 4 · 2 replicates" in page
    assert "confidence-toggle" not in page
    assert "data-result-card" in page
    assert "data-result-section" in page
    assert page.count('class="figure-open"') == len(figures)
    assert 'id="close-figure-dialog"' in page
    assert client.get(
        f"/adversarial/action-scaling/experiments/{result.name}"
    ).status_code == 200
    for figure in figures:
        assert client.get(
            f"/adversarial/action-scaling/figures/{figure['filename']}"
        ).status_code == 200
        assert client.get(
            f"/adversarial/action-scaling/figures/{figure['pdf_filename']}"
        ).status_code == 200


def test_adversarial_page_queues_one_run_and_renders_results(tmp_path) -> None:
    app, service = _app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/",
        data=VALID_FORM
        | {
            "experiment_type": "adversarial",
            "_csrf_token": _csrf_token(client),
        },
    )

    assert response.status_code == 302
    job = service.jobs.recent()[0]
    assert _wait_for_job(service, job.id) == "succeeded"
    assert len(list(service.adversarial_raw_dir.glob("*.csv"))) == 1

    page = client.get("/?mode=adversarial").get_data(as_text=True)
    assert "Final regret summary" in page
    assert page.count("Download</a>") == 1
    assert "External · R/T</th>" in page
    assert "Full history · top half punished" in page


def test_adversarial_filters_update_the_rendered_page_immediately(tmp_path) -> None:
    node = shutil.which("node")
    if node is None or subprocess.run([node, "-e", "require('jsdom')"], capture_output=True).returncode:
        pytest.skip("Node.js with jsdom is unavailable")
    app, service = _app(tmp_path)
    client = app.test_client()
    token = _csrf_token(client)
    for form in (
        VALID_FORM,
        VALID_FORM | {
            "environment": RANDOM_WALK_ENVIRONMENT,
            "feedback_mode": "bandit",
            "algorithm_names": ["exp3_ix"],
            "horizon": "5",
            "seed": "9",
        },
    ):
        client.post("/", data=form | {"_csrf_token": token})
        assert _wait_for_job(service, service.jobs.recent()[0].id) == "succeeded"

    static_dir = Path(__file__).parents[2] / "web" / "static"
    payload = {
        "page": client.get("/?mode=adversarial").get_data(as_text=True),
        "common": (static_dir / "common.js").read_text(encoding="utf-8"),
        "dashboard": (static_dir / "dashboard.js").read_text(encoding="utf-8"),
        "builder": (static_dir / "figure_builder.js").read_text(),
        "catalog": service.figure_builder.catalog("adversarial"),
    }
    script = r'''const fs = require("fs");
const {JSDOM} = require("jsdom");
const payload = JSON.parse(fs.readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/?mode=adversarial", runScripts: "outside-only"});
const window = dom.window;
window.fetch = async () => ({ok: true, json: async () => payload.catalog});
window.HTMLElement.prototype.scrollIntoView = () => {};
window.eval(payload.common + "\n" + payload.dashboard + "\n" + payload.builder);
const document = window.document;
const visible = (selector) => [...document.querySelectorAll(selector)].filter((node) => !node.hidden);
const filteredTo = (selector, key, value) => {
    const nodes = visible(selector);
    return nodes.length > 0 && nodes.length < document.querySelectorAll(selector).length
        && nodes.every((node) => node.dataset[key] === value);
};
const select = (id, value) => {
    const control = document.getElementById(id);
    control.value = value;
    control.dispatchEvent(new window.Event("change", {bubbles: true}));
};
(async () => {
await new Promise(resolve => setImmediate(resolve));
select("filter-scope", "lazy_random_walk_v1");
document.getElementById("filter-select-all").click();
if (!filteredTo(".summary-row", "scope", "lazy_random_walk_v1")) process.exit(2);
if (document.getElementById("environment-panel").hidden) process.exit(3);
if (document.getElementById("random-walk-rule").hidden) process.exit(4);
if (!document.getElementById("historical-frequency-rule").hidden) process.exit(5);
if (document.getElementById("filter-feedback").value !== "bandit") process.exit(7);
if (!filteredTo(".summary-row", "profile", "exp3_ix")) process.exit(11);
select("filter-metric", "internal");
select("filter-view", "sqrt_scaling");
const row = visible(".summary-row")[0];
const cells = [...row.querySelectorAll("[data-regret]")].filter(cell => !cell.hidden);
if (cells.length !== 1 || cells[0].dataset.metric !== "sqrt_scaling_internal") process.exit(12);
document.getElementById("filter-clear-all").click();
if (visible(".summary-row").length) process.exit(13);
for (const id of ["filter-horizon", "filter-seed", "filter-player-algorithm", "filter-secondary"]) {
    if (document.getElementById(id)) process.exit(14);
}
dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});'''
    result = subprocess.run([node, "-e", script], input=json.dumps(payload), capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_adversarial_page_records_and_summarizes_canonical_regret(tmp_path) -> None:
    app, service = _app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/",
        data=VALID_FORM
        | {
            "_csrf_token": _csrf_token(client),
        },
    )

    assert response.status_code == 302
    job = service.jobs.recent()[0]
    assert _wait_for_job(service, job.id) == "succeeded"
    result = next(service.adversarial_raw_dir.glob("*.csv"))
    header = result.read_text(encoding="utf-8").splitlines()[0]
    assert "external_regret" in header
    assert "expected_" not in header and "realized_" not in header
    summary = service.adversarial_result_summaries()[0][0]
    assert "regret_evaluation" not in summary
    assert summary["average_regret"] is not None


@pytest.mark.parametrize("workers", [1, 2])
def test_adversarial_page_queues_replicates_with_common_seed_schedule(
    tmp_path, workers,
) -> None:
    app, service = _app(tmp_path)
    service.replicate_workers = workers
    client = app.test_client()

    response = client.post(
        "/",
        data=VALID_FORM
        | {
            "environment": RANDOM_WALK_ENVIRONMENT,
            "replicates": "3",
            "_csrf_token": _csrf_token(client),
        },
    )

    assert response.status_code == 302
    job = service.jobs.recent()[0]
    assert _wait_for_job(service, job.id) == "succeeded"
    completed = service.jobs.get(job.id)
    assert (completed.completed, completed.total) == (3, 3)
    assert len(list(service.adversarial_raw_dir.glob("*.csv"))) == 3

    summaries, warnings = service.adversarial_result_summaries()
    assert warnings == []
    assert {row["replicate"] for row in summaries} == {0, 1, 2}
    assert {row["base_environment_seed"] for row in summaries} == {11}
    assert {row["base_learner_seed"] for row in summaries} == {7}
    assert {row["environment_seed"] for row in summaries} == {
        domain_separated_seed(11, replicate, ENVIRONMENT_SEED_DOMAIN)
        for replicate in range(3)
    }
    assert {row["learner_seed"] for row in summaries} == {
        domain_separated_seed(7, replicate, LEARNER_SEED_DOMAIN)
        for replicate in range(3)
    }


def test_adversarial_page_queues_bandit_run(tmp_path) -> None:
    app, service = _app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/",
        data=VALID_FORM
        | {
            "feedback_mode": "bandit",
            "algorithm_names": ["exp3_ix"],
            "_csrf_token": _csrf_token(client),
        },
    )

    assert response.status_code == 302
    job = service.jobs.recent()[0]
    assert _wait_for_job(service, job.id) == "succeeded"
    result = next(service.adversarial_raw_dir.glob("*.csv"))
    assert ",bandit," in result.read_text(encoding="utf-8").splitlines()[1]
    assert "Bandit feedback" in client.get("/?mode=adversarial").get_data(as_text=True)


def test_adversarial_page_queues_random_walk_run(tmp_path) -> None:
    app, service = _app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/",
        data=VALID_FORM
        | {
            "environment": RANDOM_WALK_ENVIRONMENT,
            "environment_seed": "23",
            "seed": "29",
            "_csrf_token": _csrf_token(client),
        },
    )

    assert response.status_code == 302
    job = service.jobs.recent()[0]
    assert _wait_for_job(service, job.id) == "succeeded"
    result = next(service.adversarial_raw_dir.glob("*.csv"))
    with result.open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["environment"] == RANDOM_WALK_ENVIRONMENT
    assert "initialization_mode" not in row
    assert row["base_environment_seed"] == "23"
    assert row["base_learner_seed"] == "29"
    assert row["environment_seed"] == str(
        domain_separated_seed(23, 0, ENVIRONMENT_SEED_DOMAIN)
    )
    assert row["learner_seed"] == str(
        domain_separated_seed(29, 0, LEARNER_SEED_DOMAIN)
    )
    page = client.get("/?mode=adversarial").get_data(as_text=True)
    assert "Centered at 0.5" in page
    assert "Independent lazy random walk" in page


def test_adversarial_download_and_figure_routes_are_scoped(tmp_path) -> None:
    app, service = _app(tmp_path)
    client = app.test_client()
    client.post(
        "/",
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
        f"adversarial_{HISTORICAL_FREQUENCY_ENVIRONMENT}_full_information_3_actions_average_external_regret.png",
        f"adversarial_{HISTORICAL_FREQUENCY_ENVIRONMENT}_full_information_3_actions_external_regret_over_sqrt_t.png",
        f"adversarial_{RANDOM_WALK_ENVIRONMENT}_bandit_3_actions_average_external_regret.png",
        f"adversarial_{RANDOM_WALK_ENVIRONMENT}_bandit_3_actions_external_regret_over_sqrt_t.png",
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

    page = client.get("/?mode=adversarial").get_data(as_text=True)
    assert "Figure builder" in page
    assert "Algorithm profiles" in page
    assert 'id="results-controls-heading"' in page
    assert "Filter results" in page
    assert "Recorded output" not in page
    assert 'id="filter-metric"' in page
    assert 'id="builder-figure"' in page
    assert page.count('class="figure-open"') == 0
    assert 'id="figure-dialog"' in page
    assert 'id="close-figure-dialog"' in page
    assert 'id="summary-table"' in page
    assert 'id="filter-scope"' in page
    assert 'id="filter-feedback"' in page
    assert 'id="result-filters"' in page
    assert 'id="filter-player"' in page
    assert f'data-scope="{HISTORICAL_FREQUENCY_ENVIRONMENT}"' in page
    assert 'data-feedback="full_information"' in page
    assert "Download PDF" in page


def test_clear_adversarial_results_deletes_csvs_and_figures(tmp_path) -> None:
    app, service = _app(tmp_path)
    client = app.test_client()
    service.adversarial_raw_dir.mkdir(parents=True)
    service.adversarial_figure_dir.mkdir(parents=True)
    service.adversarial_scaling_raw_dir.mkdir(parents=True)
    service.adversarial_scaling_figure_dir.mkdir(parents=True)
    csv_path = service.adversarial_raw_dir / "result.csv"
    csv_path.write_text("data", encoding="utf-8")
    for index in range(2):
        (service.adversarial_figure_dir / f"figure-{index}.png").write_bytes(
            b"png"
        )
    scaling_csv = service.adversarial_scaling_raw_dir / "scaling.csv"
    scaling_csv.write_text("data", encoding="utf-8")
    scaling_figure = service.adversarial_scaling_figure_dir / "scaling.png"
    scaling_figure.write_bytes(b"png")

    response = client.post(
        "/adversarial/results/clear",
        data={"_csrf_token": _csrf_token(client)},
    )

    assert response.status_code == 302
    assert list(service.adversarial_figure_dir.glob("*.png")) == []
    assert not csv_path.exists()
    assert not scaling_csv.exists()
    assert not scaling_figure.exists()
