from pathlib import Path
import shutil
import subprocess

import pytest

from experiments.scenarios.adversarial import RANDOM_WALK_ENVIRONMENT
from experiments.scenarios.adversarial_scaling import AdversarialScalingSpec, run_adversarial_scaling_experiment
from tests.web.support import (
    ADVERSARIAL_FORM as VALID_FORM, create_test_app, csrf_token, record_fixed_runs as result,
    run_ui, submit_and_wait,
)
from web.jobs import Job


@pytest.mark.parametrize("mode,terminal", [("fixed", "succeeded"), ("fixed", "failed"), ("adversarial", "cancelled")])
def test_browser_submission_and_polling_preserve_jobs_without_navigation(tmp_path, monkeypatch, mode, terminal):
    app, service = create_test_app(tmp_path)
    monkeypatch.setattr(service.jobs, "recent", lambda: [Job("old", "Existing", "running", "Running", "now")])
    method = "submit_adversarial_scaling_experiment" if mode == "adversarial" else "submit_experiment"
    endpoint = "/adversarial/action-scaling" if mode == "adversarial" else "/"
    monkeypatch.setattr(service, method, lambda form: Job("new", "New job", "queued", "Waiting", "now"))
    values = VALID_FORM if mode == "adversarial" else {
        "game": "rps", "feedback_mode": "full_information", "algorithm_names": ["hedge", "bm"],
        "horizon": "4", "seed": "7", "replicates": "1",
    }
    client = app.test_client()
    response = client.post(endpoint, headers={"Accept": "application/json"}, data=values | {
        "_csrf_token": csrf_token(client), "scaling_action_counts": "3,9", "scaling_replicates": "1",
    })
    assert response.status_code == 202
    script = r'''
const assert = require("assert").strict;
const {JSDOM, VirtualConsole} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const virtualConsole = new VirtualConsole();
virtualConsole.on("jsdomError", error => {throw error;}); // Includes unexpected navigation/reload.
const dom = new JSDOM(payload.page, {url: "http://localhost/", runScripts: "outside-only", virtualConsole});
const w = dom.window, d = w.document;
const form = d.getElementById("experiment-form"), status = d.getElementById("experiment-submit-status");
const buttons = [...form.querySelectorAll('button[type="submit"]')];
const submitter = payload.endpoint === "/" ? buttons[0] : form.querySelector(`[formaction="${payload.endpoint}"]`);
assert(submitter, "The rendered page must expose the requested submit button");
// Older jsdom lacks SubmitEvent; supply its button at the DOM event boundary.
if (!w.SubmitEvent) form.addEventListener("submit", event => {event.submitter = submitter;}, true);
const timers = new Map(), posts = [], polls = [];
let nextTimer = 0, rejectSubmission = null;
w.setTimeout = fn => {timers.set(++nextTimer, fn); return nextTimer;};
w.clearTimeout = id => timers.delete(id);
w.fetch = (url, options) => {
    if (options && options.method === "POST") {
        assert.equal(new URL(url, w.location).pathname, payload.endpoint);
        if (rejectSubmission) return rejectSubmission();
        return new Promise(resolve => posts.push({options, resolve}));
    }
    if (url.startsWith("/jobs/")) return new Promise(resolve => polls.push({url, resolve}));
    return Promise.resolve({ok: true, json: async () => payload.catalog});
};
const tick = () => new Promise(resolve => setImmediate(resolve));
const submit = () => {
    const event = w.SubmitEvent ? new w.SubmitEvent("submit", {cancelable: true, submitter})
                               : new w.Event("submit", {cancelable: true});
    assert.equal(form.dispatchEvent(event), false);
};
const finish = (request, json) => request.resolve({ok: true, json: async () => json});
w.eval(payload.script);
(async () => {
    await tick();
    assert.deepEqual(polls.map(p => p.url), ["/jobs/old"]);
    const expected = new w.FormData(form);
    submit(); submit();
    assert.equal(posts.length, 1, status.textContent);
    assert.equal(buttons.every((button) => button.disabled), true);
    assert.equal(posts[0].options.headers.Accept, "application/json");
    assert.equal(posts[0].options.body.get("_csrf_token"), expected.get("_csrf_token"));
    assert.deepEqual([...posts[0].options.body.getAll("algorithm_names")], [...expected.getAll("algorithm_names")]);
    finish(posts[0], payload.queued); await tick();
    const added = d.querySelector('[data-job-id="new"]');
    assert.equal(added.dataset.status, "queued");
    assert.equal(added.querySelector("form").getAttribute("action"), "/jobs/new/cancel");
    assert.equal(d.querySelector(".jobs-panel").hidden, false);
    assert.equal(status.textContent, payload.queued.message);
    assert.equal(buttons.some((button) => button.disabled), false);
    assert.equal(polls.length, 1); // Queuing during the pending poll must not overlap it.
    finish(polls[0], {id: "old", status: "succeeded", message: "Done"}); await tick();
    assert.equal(d.getElementById("busy-indicator").hidden, false);
    assert.equal(d.getElementById("refresh-results-notice").hidden, false);
    assert.equal(timers.size, 1);
    const [id, pollAgain] = [...timers][0]; timers.delete(id); pollAgain();
    assert.deepEqual(polls.map(p => p.url), ["/jobs/old", "/jobs/new"]);
    finish(polls[1], {id: "new", status: payload.terminal, message: "Finished"}); await tick();
    assert.equal(added.dataset.status, payload.terminal);
    assert(added.classList.contains("job-" + payload.terminal));
    assert.equal(added.querySelector("[data-job-status]").textContent, payload.terminal);
    assert.equal(added.querySelector("[data-job-message]").textContent, "Finished");
    assert.equal(added.querySelector("form"), null);
    assert.equal(d.getElementById("busy-indicator").hidden, true);
    assert.equal(timers.size, 0);
    for (const message of ["already queued", "Network unavailable"]) {
        rejectSubmission = () => message === "already queued"
            ? Promise.resolve({ok: false, json: async () => ({error: message})})
            : Promise.reject(new Error(message));
        submit(); await tick();
        assert.equal(status.textContent, message);
        assert.equal(d.querySelectorAll('[data-job-id="new"]').length, 1);
        assert.equal(buttons.some(button => button.disabled), false);
    }
    assert.equal(d.querySelector('[data-job-id="old"]').dataset.status, "succeeded");
    assert.equal(w.location.href, "http://localhost/");
    dom.window.close();
})().catch((error) => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, script, mode, endpoint=endpoint, terminal=terminal, queued=response.get_json())


@pytest.mark.parametrize("filename", ["common.js", "dashboard.js", "custom_games.js", "figure_builder.js"])
def test_web_javascript_parses(filename: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    path = Path(__file__).parents[2] / "web" / "static" / filename
    result = subprocess.run([node, "--check", path], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_shared_filters_six_figures_summary_export_and_stale_responses(tmp_path):
    app, service = create_test_app(tmp_path)
    for profile in ("hedge_vs_hedge", "ito_vs_ito", "hedge_vs_ito"):
        result(service, profile)
    result(service, "auer_exp3_vs_bm", mode="bandit")
    result(service, "bm_vs_bm", game="rpsls")
    script = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/", runScripts: "outside-only"});
const w = dom.window, d = w.document;
const f = name => d.getElementById("filter-" + name), b = name => d.getElementById("builder-" + name);
const tick = () => new Promise(resolve => setImmediate(resolve));
const change = (name, value) => {f(name).value = value; f(name).dispatchEvent(new w.Event("change"));};
const submit = () => d.getElementById("figure-builder").dispatchEvent(new w.Event("submit", {cancelable: true}));
const visibleRows = () => [...d.querySelectorAll(".summary-row")].filter(row => !row.hidden);
const visibleCards = () => [...b("figure").children].filter(card => !card.hidden);
const allFigures = ["external", "internal", "swap"].flatMap(metric => ["average", "sqrt_scaling"].map(view => ({
    metric, view, title: metric + " " + view, filename: metric + "_" + view + ".png",
    pdf_filename: metric + "_" + view + ".pdf", url: "/" + metric + "_" + view + ".png",
    pdf_url: "/" + metric + "_" + view + ".pdf"
})));
let requests = [], exports = [], finish, finishExport, downloads = 0;
w.HTMLElement.prototype.scrollIntoView = () => {};
w.HTMLAnchorElement.prototype.click = () => downloads++;
w.URL.createObjectURL = () => "blob:test";
w.URL.revokeObjectURL = () => {};
w.fetch = async (url, options) => {
    if (!options) return {ok: true, json: async () => payload.catalog};
    if (options.body.get("mode") === "figure_builder") {
        exports.push(options.body.getAll("filenames"));
        return new Promise(resolve => finishExport = () => resolve({ok: true, blob: async () => new w.Blob(["pdf"])}));
    }
    requests.push(options.body);
    return new Promise(resolve => finish = () => resolve({ok: true, json: async () => ({
        profiles: options.body.getAll("profiles"), figures: allFigures
    })}));
};
w.eval(payload.script);
(async () => {
    await tick(); await tick();
    assert.equal(b("generate").disabled, true);
    assert.equal(visibleRows().length, 0);
    assert.equal(d.querySelectorAll("#figure-builder select, .summary-panel select, .summary-panel input").length, 0);
    change("feedback", "full_information");
    f("select-all").click();
    assert.equal(visibleRows().length, 3);
    assert(visibleRows().every(row => row.dataset.player === "0"));
    submit();
    assert.equal(requests.length, 1);
    assert.deepEqual(requests[0].getAll("profiles"), ["hedge_vs_hedge", "hedge_vs_ito", "ito_vs_ito"]);
    assert.equal(requests[0].has("metric"), false);
    change("metric", "internal"); change("view", "sqrt_scaling");
    finish(); await tick(); await tick();
    assert.equal(b("figure").children.length, 6);
    assert.deepEqual(visibleCards().map(card => card.dataset.filename), ["internal_sqrt_scaling.png"]);
    assert.equal(requests.length, 1); // Display filters never regenerate the collection.
    for (const row of visibleRows()) {
        const visible = [...row.querySelectorAll("[data-regret]")].filter(cell => !cell.hidden);
        assert.equal(visible.length, 1);
        assert.equal(visible[0].dataset.metric, "sqrt_scaling_internal");
        const average = row.querySelector('[data-metric="average_internal"]');
        assert(Math.abs(Number(visible[0].dataset.value) - Number(average.dataset.value) * Math.sqrt(Number(row.dataset.horizon))) < 1e-12);
    }
    change("metric", "all"); change("view", "average");
    b("download").click();
    assert.deepEqual(exports[0], ["external_average.pdf", "internal_average.pdf", "swap_average.pdf"]);
    finishExport(); await tick(); await tick(); assert.equal(downloads, 1);
    b("download").click(); change("metric", "swap"); finishExport(); await tick(); await tick();
    assert.equal(downloads, 1); // Do not download an obsolete selection after filters change.
    f("clear-all").click();
    assert.equal(visibleRows().length, 0); assert.equal(b("figure").children.length, 0);
    assert.equal(b("download").disabled, true); submit(); assert.equal(requests.length, 1);
    f("profiles").value = "hedge_vs_ito";
    f("profiles").dispatchEvent(new w.Event("change"));
    assert.deepEqual(visibleRows().map(row => row.dataset.profile), ["hedge_vs_ito"]);
    submit(); f("clear-all").click(); finish(); await tick(); await tick();
    assert.equal(b("figure").children.length, 0);
    change("view", "all"); assert.equal(f("profiles").selectedOptions.length, 0);
    change("feedback", "bandit"); f("select-all").click(); change("player", "1"); f("select-all").click();
    assert.equal(visibleRows().length, 1);
    assert.equal(visibleRows()[0].dataset.profile, "auer_exp3_vs_bm");
    assert.equal(visibleRows()[0].dataset.player, "1");
    change("scope", "rpsls"); f("select-all").click();
    assert.equal(visibleRows().length, 1); assert.equal(visibleRows()[0].dataset.scope, "rpsls");
    f("clear-all").click();
    const persisted = JSON.parse(w.localStorage.getItem("swap-regret-shared-filters-fixed"));
    assert.deepEqual(persisted.selections[f("context").value], []);
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, script)


def test_scaling_only_results_remain_filterable(tmp_path):
    app, service = create_test_app(tmp_path)
    spec = AdversarialScalingSpec(environment=RANDOM_WALK_ENVIRONMENT, feedback_mode="bandit",
        algorithm_name="auer_exp3", action_counts=(3, 6), replicates=2, horizon=10,
        environment_seed=7, learner_seed=42)
    run_adversarial_scaling_experiment(spec, service.adversarial_scaling_raw_dir, workers=1)
    service.adversarial_scaling_figure_dir.mkdir(parents=True)
    (service.adversarial_scaling_figure_dir / f"{spec.run_id}_regret_by_actions.png").write_bytes(b"preview")
    script = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/", runScripts: "outside-only"});
const w = dom.window, d = w.document;
w.fetch = async () => ({ok: true, json: async () => payload.catalog});
w.eval(payload.script);
(async () => {
    await new Promise(resolve => setImmediate(resolve));
    d.getElementById("filter-select-all").click();
    assert.equal(d.querySelector("[data-result-card]").hidden, false);
    assert.equal(d.getElementById("builder-generate").disabled, true);
    const metric = d.getElementById("filter-metric");
    metric.value = "swap"; metric.dispatchEvent(new w.Event("change"));
    assert.equal(d.querySelector("[data-result-card]").hidden, true);
    metric.value = "external"; metric.dispatchEvent(new w.Event("change"));
    assert.equal(d.querySelector("[data-result-card]").hidden, false);
    d.getElementById("filter-clear-all").click();
    assert.equal(d.querySelector("[data-result-card]").hidden, true);
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, script, "adversarial")


def test_adversarial_filters_update_the_rendered_page_immediately(tmp_path) -> None:
    app, service = create_test_app(tmp_path)
    client = app.test_client()
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
        submit_and_wait(client, service, form)

    script = r'''const fs = require("fs");
const {JSDOM} = require("jsdom");
const payload = JSON.parse(fs.readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/?mode=adversarial", runScripts: "outside-only"});
const window = dom.window;
window.fetch = async () => ({ok: true, json: async () => payload.catalog});
window.HTMLElement.prototype.scrollIntoView = () => {};
window.eval(payload.script);
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
    run_ui(app, service, script, "adversarial")
