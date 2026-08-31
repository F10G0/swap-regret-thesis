from pathlib import Path
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


VALID_FORM = {
    "experiment_type": "adversarial",
    "environment": HISTORICAL_FREQUENCY_ENVIRONMENT,
    "initialization_mode": "centered",
    "feedback_mode": "full_information",
    "regret_evaluation": "both",
    "algorithm_names": ["hedge"],
    "n_actions": "3",
    "horizon": "4",
    "environment_seed": "11",
    "seed": "7",
    "replicates": "1",
}
ENVIRONMENTS = {HISTORICAL_FREQUENCY_ENVIRONMENT, RANDOM_WALK_ENVIRONMENT}
INITIALIZATION_MODES = {"centered", "uniform_grid"}


def _app(tmp_path: Path):
    return create_test_app(tmp_path, experimental=False, max_replicates=10, disable_adversarial_plots=True)


def _opening_tag(page: str, element_id: str) -> str:
    before, after = page.split(f'id="{element_id}"', 1)
    return before.rsplit("<", 1)[1] + f'id="{element_id}"' + after.split(">", 1)[0]


def test_adversarial_form_validation() -> None:
    form = parse_adversarial_experiment_form(
        VALID_FORM,
        algorithms_by_feedback_mode={
            "full_information": ["hedge"],
            "bandit": ["exp3"],
        },
        environments=ENVIRONMENTS,
        initialization_modes=INITIALIZATION_MODES,
        max_actions=100,
        max_horizon=100,
    )

    assert form.algorithm_name == "hedge"
    assert form.feedback_mode == "full_information"
    assert form.n_actions == 3
    assert form.replicates == 1
    assert form.regret_evaluation == "both"

    with pytest.raises(ValueError, match="replicates must not exceed 10"):
        parse_adversarial_experiment_form(
            VALID_FORM | {"replicates": "11"},
            algorithms_by_feedback_mode={"full_information": ["hedge"]},
            environments=ENVIRONMENTS,
            initialization_modes=INITIALIZATION_MODES,
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
        initialization_modes=INITIALIZATION_MODES,
        max_actions=100,
        max_horizon=100,
        max_replicates=10,
    )

    assert form.action_counts == (2, 5, 10)
    assert form.replicates == 4
    assert form.regret_evaluation == "both"
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
                "bandit": ["exp3"],
            },
            environments=ENVIRONMENTS,
            initialization_modes=INITIALIZATION_MODES,
            max_actions=100,
            max_horizon=100,
        )


def test_adversarial_form_rejects_unknown_regret_evaluation() -> None:
    with pytest.raises(ValueError, match="unknown regret evaluation"):
        parse_adversarial_experiment_form(
            VALID_FORM | {"regret_evaluation": "unknown"},
            algorithms_by_feedback_mode={"full_information": ["hedge"]},
            environments=ENVIRONMENTS,
            initialization_modes=INITIALIZATION_MODES,
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
    assert 'name="initialization_mode"' in page
    assert 'name="feedback_mode"' in page
    assert 'name="regret_evaluation"' in page
    assert '<option value="expected"' in page
    assert '<option value="realized"' in page
    assert '<option value="both"' in page
    assert 'name="n_actions"' in page
    assert 'name="memory_window"' not in page
    assert 'name="environment_seed"' in page
    assert 'name="seed"' in page
    assert 'name="scaling_action_counts"' in page
    assert 'name="scaling_replicates"' in page
    assert "Queue scaling experiment" in page
    assert "complete history" in page
    assert "exp3" in page
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
        "regret_evaluation",
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
    assert "hidden" in _opening_tag(historical_page, "adversarial-initialization-field")
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
    assert "hidden" not in _opening_tag(random_walk_page, "adversarial-initialization-field")
    assert "hidden" not in _opening_tag(random_walk_page, "adversarial-environment-seed-field")
    assert "hidden" in _opening_tag(random_walk_page, "environment-panel")
    assert "hidden" in _opening_tag(random_walk_page, "historical-frequency-rule")
    assert "hidden" in _opening_tag(random_walk_page, "random-walk-rule")


def test_adversarial_algorithm_options_follow_feedback_mode(tmp_path) -> None:
    app, _ = _app(tmp_path)
    client = app.test_client()
    response = client.post(
        "/",
        data=VALID_FORM | {
            "feedback_mode": "bandit",
            "algorithm_names": ["exp3"],
            "replicates": "0",
            "_csrf_token": _csrf_token(client),
        },
    )

    assert response.status_code == 400
    page = response.get_data(as_text=True)
    options = page.split('<select id="algorithm_player_0"', 1)[1].split("</select>", 1)[0]
    assert '<option value="exp3" selected' in options
    assert '<option value="exp3_ix"' in options
    assert '<option value="lce_ix"' in options
    assert '<option value="hedge"' not in options


def test_adversarial_page_runs_action_space_scaling_batch(tmp_path) -> None:
    app, service = create_test_app(
        tmp_path,
        experimental=False,
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
    assert {figure["source"] for figure in figures} == {"expected", "realized"}

    page = client.get("/?mode=adversarial").get_data(as_text=True)
    assert "Regret by action-space size" in page
    assert "K=2, 4 · 2 replicates" in page
    assert page.count('class="confidence-toggle"') == len(figures)
    assert 'id="confidence-intervals"' not in page
    assert "Hide 95% CI" in page
    assert "data-without-confidence-src" in page
    assert "data-without-confidence-download" in page
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
            f"/adversarial/action-scaling/figures/{figure['confidence_free_filename']}"
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
    assert ">Expected</th>" in page
    assert ">Realized</th>" in page
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
            "regret_evaluation": "realized",
            "algorithm_names": ["exp3"],
            "horizon": "5",
            "seed": "9",
        },
    ):
        client.post("/", data=form | {"_csrf_token": token})
        assert _wait_for_job(service, service.jobs.recent()[0].id) == "succeeded"

    service.adversarial_figure_dir.mkdir(parents=True, exist_ok=True)
    for environment, feedback in (
        (HISTORICAL_FREQUENCY_ENVIRONMENT, "full_information"),
        (RANDOM_WALK_ENVIRONMENT, "bandit"),
    ):
        for source in ("expected", "realized"):
            for regret in ("external", "internal"):
                for suffix in (f"average_{source}_{regret}_regret", f"{source}_{regret}_regret_over_sqrt_t"):
                    name = f"adversarial_{environment}_{feedback}_3_actions_{suffix}.png"
                    (service.adversarial_figure_dir / name).write_bytes(b"png")

    static_dir = Path(__file__).parents[2] / "web" / "static"
    payload = {
        "page": client.get("/?mode=adversarial").get_data(as_text=True),
        "common": (static_dir / "common.js").read_text(encoding="utf-8"),
        "dashboard": (static_dir / "dashboard.js").read_text(encoding="utf-8"),
    }
    script = r'''const fs = require("fs");
const {JSDOM} = require("jsdom");
const payload = JSON.parse(fs.readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/?mode=adversarial", runScripts: "outside-only"});
const window = dom.window;
window.fetch = async () => ({ok: true, json: async () => ({})});
window.HTMLElement.prototype.scrollIntoView = () => {};
window.eval(payload.common + "\n" + payload.dashboard);
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
const input = (id, value) => {
    const control = document.getElementById(id);
    control.value = value;
    control.dispatchEvent(new window.Event("input", {bubbles: true}));
};
if (!document.getElementById("environment-panel").hidden) process.exit(1);
select("filter-scope", "lazy_random_walk_v1");
if (!filteredTo(".summary-row", "scope", "lazy_random_walk_v1")) process.exit(2);
if (document.getElementById("environment-panel").hidden) process.exit(3);
if (document.getElementById("random-walk-rule").hidden) process.exit(4);
if (!document.getElementById("historical-frequency-rule").hidden) process.exit(5);
select("filter-scope", "all");
if (!document.getElementById("environment-panel").hidden) process.exit(6);
select("filter-secondary", "bandit");
if (!filteredTo("#figure-grid .figure-card", "secondary", "bandit")) process.exit(7);
select("filter-secondary", "all");
select("filter-source", "realized");
if (!filteredTo("#figure-grid .figure-card", "source", "realized")) process.exit(8);
select("filter-source", "expected");
select("filter-regret", "internal");
if (!filteredTo("#figure-grid .figure-card", "regret", "internal")) process.exit(9);
select("filter-regret", "all");
select("filter-view", "sqrt_scaling");
if (!filteredTo("#figure-grid .figure-card", "view", "sqrt_scaling")) process.exit(10);
select("filter-player-algorithm", "exp3");
if (!filteredTo(".summary-row", "playerAlgorithm", "exp3")) process.exit(11);
select("filter-player-algorithm", "all");
input("filter-horizon", "5");
if (!filteredTo(".summary-row", "horizon", "5")) process.exit(12);
input("filter-horizon", "");
input("filter-seed", "9");
if (!filteredTo(".summary-row", "seed", "9")) process.exit(13);
input("filter-seed", "");
select("filter-regret-evaluation", "realized");
if (!filteredTo(".summary-row", "regretEvaluation", "realized")) process.exit(14);
select("filter-summary-source", "realized");
if (visible('[data-regret-source="realized"]').length === 0) process.exit(15);
if (visible('[data-regret-source="expected"]').length !== 0) process.exit(16);'''
    result = subprocess.run([node, "-e", script], input=json.dumps(payload), capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_adversarial_page_records_only_selected_regret_source(tmp_path) -> None:
    app, service = _app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/",
        data=VALID_FORM
        | {
            "regret_evaluation": "expected",
            "_csrf_token": _csrf_token(client),
        },
    )

    assert response.status_code == 302
    job = service.jobs.recent()[0]
    assert _wait_for_job(service, job.id) == "succeeded"
    result = next(service.adversarial_raw_dir.glob("*.csv"))
    header = result.read_text(encoding="utf-8").splitlines()[0]
    assert "expected_external_regret" in header
    assert "realized_external_regret" not in header
    summary = service.adversarial_result_summaries()[0][0]
    assert summary["regret_evaluation"] == "expected"
    assert summary["expected_regret"] is not None
    assert summary["realized_regret"] is None


def test_adversarial_page_queues_replicates_with_common_seed_schedule(
    tmp_path,
) -> None:
    app, service = _app(tmp_path)
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
    assert {row["environment_seed"] for row in summaries} == {11, 12, 13}
    assert {row["learner_seed"] for row in summaries} == {7, 8, 9}


def test_adversarial_page_queues_bandit_run(tmp_path) -> None:
    app, service = _app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/",
        data=VALID_FORM
        | {
            "feedback_mode": "bandit",
            "algorithm_names": ["exp3"],
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
            "initialization_mode": "uniform_grid",
            "environment_seed": "23",
            "seed": "29",
            "_csrf_token": _csrf_token(client),
        },
    )

    assert response.status_code == 302
    job = service.jobs.recent()[0]
    assert _wait_for_job(service, job.id) == "succeeded"
    result = next(service.adversarial_raw_dir.glob("*.csv"))
    row = result.read_text(encoding="utf-8").splitlines()[1]
    assert ",lazy_random_walk_v1,uniform_grid,0.1,23,29," in row
    page = client.get("/?mode=adversarial").get_data(as_text=True)
    assert "Uniform over the reward grid" in page
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
        f"adversarial_{HISTORICAL_FREQUENCY_ENVIRONMENT}_full_information_3_actions_average_expected_external_regret.png",
        f"adversarial_{HISTORICAL_FREQUENCY_ENVIRONMENT}_full_information_3_actions_expected_external_regret_over_sqrt_t.png",
        f"adversarial_{RANDOM_WALK_ENVIRONMENT}_bandit_3_actions_average_expected_external_regret.png",
        f"adversarial_{RANDOM_WALK_ENVIRONMENT}_bandit_3_actions_expected_external_regret_over_sqrt_t.png",
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
    assert "Average External" in page
    assert "External / sqrt(t)" in page
    assert 'id="results-controls-heading"' in page
    assert "Filter results" in page
    assert "Recorded output" not in page
    assert 'id="figure-filters"' in page
    assert 'id="figure-grid"' in page
    assert page.count('class="figure-open"') == len(service.adversarial_figure_records())
    assert 'id="figure-dialog"' in page
    assert 'id="close-figure-dialog"' in page
    assert 'id="summary-table"' in page
    assert 'id="filter-scope"' in page
    assert 'id="filter-secondary"' in page
    assert 'data-result-filter="scope"' in page
    assert 'data-result-filter="secondary"' in page
    assert f'data-scope="{HISTORICAL_FREQUENCY_ENVIRONMENT}"' in page
    assert 'data-secondary="full_information"' in page
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
