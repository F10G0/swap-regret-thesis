from pathlib import Path
import shutil
import subprocess

import pytest

from experiments.scenarios.adversarial import RANDOM_WALK_ENVIRONMENT, run_adversarial_experiment
from tests.web.support import (
    ADVERSARIAL_FORM as VALID_FORM, browse_url, create_test_app, csrf_token, production_ui_page,
    record_fixed_runs as result, run_node, run_ui, submit_and_wait,
)
from web.jobs import Job


def fixed_details(app, service):
    client = app.test_client()
    return {
        (url := f'/experiment-groups/fixed/{row["group_id"]}/players/{row["player"]}'):
        client.get(url).get_json()
        for row in service.result_snapshot().summaries(grouped=True)
    }


@pytest.mark.parametrize("mode", ["fixed", "adversarial"])
def test_job_status_scroll_area_keeps_all_active_jobs(tmp_path, monkeypatch, mode):
    app, service = create_test_app(tmp_path)
    statuses = ["succeeded"] * 5 + ["queued", "running"] * 4
    jobs = [
        Job(f"job-{index}", f"Job {index}", status, status, "now",
            rounds_total=100 if status in {"queued", "running"} else 0)
        for index, status in enumerate(statuses)
    ]
    monkeypatch.setattr(service.jobs, "recent", lambda: jobs)
    monkeypatch.setattr(service.jobs, "is_busy", lambda: True)
    page = production_ui_page(app, service, mode)
    monkeypatch.setattr(service.jobs, "recent", lambda: [jobs[5]])
    short_page = production_ui_page(app, service, mode)
    css = (Path(__file__).parents[2] / "web/static/dashboard.css").read_text()
    script = r'''
const assert = require("assert").strict;
const {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/"});
const d = dom.window.document;
const stylesheet = d.createElement("style");
stylesheet.textContent = payload.css;
d.head.append(stylesheet);
const rules = [...stylesheet.sheet.cssRules];
const rule = selector => rules.find(item => item.selectorText === selector);
const scroll = rule(".job-list").style;
assert.equal(scroll.getPropertyValue("max-height"), "20rem");
assert.equal(scroll.getPropertyValue("overflow-y"), "auto");
assert.equal(scroll.getPropertyValue("padding-right"), "8px");
assert.equal(scroll.getPropertyValue("height"), "");
assert.equal(scroll.getPropertyValue("min-height"), "");
for (const selector of [".jobs-panel", ".content"]) {
    assert.equal(rule(selector).style.getPropertyValue("overflow-y"), "");
}
const panel = d.querySelector(".jobs-panel");
const list = panel.querySelector(".job-list");
assert.equal(list.tagName, "OL");
assert.equal(list.tabIndex, 0);
assert.equal(list.getAttribute("aria-label"), "Job list");
assert.deepEqual([...list.children].map(job => job.dataset.jobId), payload.expected);
assert.equal(list.querySelectorAll('[data-status="succeeded"]').length, 5);
assert.equal(list.querySelectorAll('[data-status="queued"], [data-status="running"]').length, 8);
assert.equal(list.querySelectorAll("[data-status-url]").length, payload.expected.length);
assert.equal(list.querySelectorAll("[data-job-round-progress]").length, 8);
assert.equal(list.querySelectorAll('form[action$="/cancel"]').length, 8);
assert.equal(list.contains(d.getElementById("jobs-heading")), false);
assert.equal(list.contains(d.getElementById("busy-indicator")), false);
assert.equal(d.getElementById("busy-indicator").hidden, false);
for (const section of [d.getElementById("result-filters"), d.getElementById("figure-results"),
                       d.querySelector(".summary-panel")]) {
    assert(panel.compareDocumentPosition(section) & dom.window.Node.DOCUMENT_POSITION_FOLLOWING);
}
dom.window.close();
const short = new JSDOM(payload.shortPage, {url: "http://localhost/"});
assert.deepEqual(
    [...short.window.document.querySelector(".job-list").children].map(job => job.dataset.jobId),
    ["job-5"],
);
short.window.close();
'''
    run_node(script, payload={"page": page, "shortPage": short_page, "css": css,
                              "expected": [job.id for job in jobs]}, jsdom=True)


@pytest.mark.parametrize("mode,terminal", [("fixed", "succeeded"), ("fixed", "failed"), ("adversarial", "cancelled")])
def test_browser_submission_and_polling_preserve_jobs_without_navigation(tmp_path, monkeypatch, mode, terminal):
    app, service = create_test_app(tmp_path)
    monkeypatch.setattr(service.jobs, "recent", lambda: [Job("old", "Existing", "running", "Running", "now")])
    method = "submit_adversarial_experiment" if mode == "adversarial" else "submit_experiment"
    endpoint = "/"
    monkeypatch.setattr(service, method, lambda form: Job("new", "New job", "queued", "Waiting", "now", rounds_total=100))
    values = VALID_FORM | {"actions": "3,9"} if mode == "adversarial" else {
        "game": "rps", "feedback_mode": "full_information", "algorithm_names": ["hedge", "bm_hedge"],
        "horizon": "4", "seed": "7", "replicates": "1",
    }
    client = app.test_client()
    response = client.post(endpoint, headers={"Accept": "application/json"}, data=values | {
        "_csrf_token": csrf_token(client),
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
    assert.equal(d.querySelectorAll('input[name="seed"]').length, 1);
    assert.equal(d.getElementById("experiment-seed").form, form);
    assert.equal(expected.get("seed"), d.getElementById("experiment-seed").value);
    submit(); submit();
    assert.equal(posts.length, 1, status.textContent);
    assert.equal(buttons.every((button) => button.disabled), true);
    assert.equal(posts[0].options.headers.Accept, "application/json");
    assert.equal(posts[0].options.body.get("_csrf_token"), expected.get("_csrf_token"));
    assert.equal(posts[0].options.body.get("seed"), expected.get("seed"));
    assert.deepEqual([...posts[0].options.body.getAll("algorithm_names")], [...expected.getAll("algorithm_names")]);
    finish(posts[0], payload.queued); await tick();
    const added = d.querySelector('[data-job-id="new"]');
    assert.equal(added.dataset.status, "queued");
    assert.equal(added.querySelector("form").getAttribute("action"), "/jobs/new/cancel");
    assert.equal(added.querySelector("progress").max, 100);
    assert.equal(added.querySelector("progress").value, 0);
    assert.equal(added.querySelector("[data-job-eta]").textContent, "Estimating…");
    assert.equal(d.querySelector('[data-job-id="old"] progress'), null);
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
    finish(polls[1], {id: "new", status: "running", message: "2 / 4 runs completed",
        rounds_completed: 67, rounds_total: 100, eta_seconds: 240}); await tick();
    assert.equal(added.querySelector("progress").value, 67);
    assert.equal(added.querySelector("[data-job-round-percent]").textContent, "67%");
    assert.equal(added.querySelector("[data-job-eta]").textContent, "~4 min");
    assert.equal(timers.size, 1);
    const [nextId, pollTerminal] = [...timers][0]; timers.delete(nextId); pollTerminal();
    assert.deepEqual(polls.map(p => p.url), ["/jobs/old", "/jobs/new", "/jobs/new"]);
    const succeeded = payload.terminal === "succeeded";
    finish(polls[2], {id: "new", status: payload.terminal, message: "Finished",
        rounds_completed: succeeded ? 100 : 67, rounds_total: 100, eta_seconds: succeeded ? 0 : 240}); await tick();
    assert.equal(added.dataset.status, payload.terminal);
    assert(added.classList.contains("job-" + payload.terminal));
    assert.equal(added.querySelector("[data-job-status]").textContent, payload.terminal);
    assert.equal(added.querySelector("[data-job-message]").textContent, "Finished");
    assert.equal(added.querySelector("[data-job-eta]").textContent, succeeded ? "Complete" : "~4 min");
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


def test_job_polling_failure_is_visible_retried_and_cleared(tmp_path, monkeypatch):
    app, service = create_test_app(tmp_path)
    monkeypatch.setattr(service.jobs, "recent", lambda: [Job("old", "Existing", "running", "Running", "now")])
    script = r'''
const assert = require("assert").strict;
const {JSDOM, VirtualConsole} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const virtualConsole = new VirtualConsole();
virtualConsole.on("jsdomError", error => {throw error;});
const dom = new JSDOM(payload.page, {url: "http://localhost/", runScripts: "outside-only", virtualConsole});
const w = dom.window, d = w.document;
const timers = [];
let polls = 0;
w.setTimeout = (callback, delay) => {timers.push({callback, delay}); return timers.length;};
w.clearTimeout = () => {};
w.fetch = url => {
    if (url === "/jobs/old") {
        polls += 1;
        if (polls <= 2) return Promise.reject(new Error("Network unavailable"));
        return Promise.resolve({ok: true, json: async () => ({id: "old", status: "running", message: "Still running"})});
    }
    return Promise.resolve({ok: true, json: async () => payload.catalog});
};
const tick = () => new Promise(resolve => setImmediate(resolve));
const retry = () => {
    assert.equal(timers.length, 1);
    const timer = timers.shift();
    assert.equal(timer.delay, 1200);
    timer.callback();
};
w.eval(payload.script);
(async () => {
    const status = d.getElementById("job-poll-status");
    await tick();
    assert.equal(polls, 1);
    assert.equal(status.hidden, false);
    assert.match(status.textContent, /temporarily unavailable/i);
    assert.equal(d.querySelector('[data-job-id="old"]').dataset.status, "running");
    const warning = status.textContent;
    retry(); await tick();
    assert.equal(polls, 2);
    assert.equal(d.querySelectorAll("#job-poll-status").length, 1);
    assert.equal(status.textContent, warning);
    assert.equal(status.hidden, false);
    retry(); await tick();
    assert.equal(polls, 3);
    assert.equal(status.hidden, true);
    assert.equal(status.textContent, "");
    assert.equal(d.querySelector('[data-job-id="old"]').dataset.status, "running");
    assert.equal(timers.length, 1);
    assert.equal(timers[0].delay, 1200);
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, script)


@pytest.mark.parametrize("filename", ["common.js", "dashboard.js", "custom_games.js", "figure_builder.js"])
def test_web_javascript_parses(filename: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    path = Path(__file__).parents[2] / "web" / "static" / filename
    result = subprocess.run([node, "--check", path], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_success_flash_is_removed_after_timeout_but_error_remains():
    script = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dom = new JSDOM(`<div class="flash-stack"><div class="notice notice-success">Saved</div>
    <div class="notice notice-error">Failed</div></div>`, {runScripts: "outside-only"});
const timers = [];
dom.window.setTimeout = (callback, delay) => timers.push({callback, delay});
dom.window.eval(payload.script);
assert.equal(timers.length, 1);
assert.equal(timers[0].delay, 4000);
assert.equal(dom.window.document.querySelector(".notice-error").textContent, "Failed");
timers[0].callback();
assert.equal(dom.window.document.querySelector(".notice-success"), null);
assert.equal(dom.window.document.querySelector(".notice-error").textContent, "Failed");
dom.window.close();
'''
    common = Path(__file__).parents[2] / "web/static/common.js"
    run_node(script, payload={"script": common.read_text()}, jsdom=True)


def test_custom_game_form_state_persists_and_validation_state_takes_precedence(tmp_path):
    app, service = create_test_app(tmp_path)
    definition = service.create_custom_game("Inspect Me", 2, [2, 2], 7, "zero_sum")
    client = app.test_client()
    error_page = client.post("/custom-games", data={
        "_csrf_token": csrf_token(client), "name": "Submitted Game", "payoff_structure": "general_sum",
        "n_players": "3", "action_counts": ["4", "5", "6"], "seed": "-1",
    }).get_data(as_text=True)
    static = Path(__file__).parents[2] / "web" / "static"
    payload = {
        "page": client.get("/custom-games").get_data(as_text=True),
        "inspectionPage": client.get(f"/custom-games/{definition.id}").get_data(as_text=True),
        "errorPage": error_page,
        "script": "\n".join((static / name).read_text() for name in ("common.js", "custom_games.js")),
    }
    script = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const key = "swap-regret-custom-game-form";
const render = (page, stored = null) => {
    const dom = new JSDOM(page, {url: "http://localhost/custom-games", runScripts: "outside-only"});
    if (stored !== null) dom.window.localStorage.setItem(key, stored);
    dom.window.eval(payload.script);
    return dom;
};
const values = dom => {
    const d = dom.window.document;
    return {
        name: d.getElementById("custom-game-name").value,
        seed: d.getElementById("custom-game-seed").value,
        payoffStructure: d.getElementById("custom-payoff-structure").value,
        playerCount: d.getElementById("custom-player-count").value,
        actionCounts: [...d.querySelectorAll("#custom-action-counts input")].map(input => input.value),
    };
};

const first = render(payload.page), w = first.window, d = w.document;
assert.equal(d.getElementById("custom-action-counts").dataset.maxActions, "100");
assert([...d.querySelectorAll("#custom-action-counts input")].every(input => input.max === "100"));
const change = (id, value, eventName) => {
    const input = d.getElementById(id); input.value = value;
    input.dispatchEvent(new w.Event(eventName, {bubbles: true}));
};
change("custom-game-name", "Stateful Game", "input");
change("custom-game-seed", "91", "input");
change("custom-payoff-structure", "general_sum", "change");
change("custom-player-count", "3", "input");
["2", "3", "4"].forEach((value, index) => {
    const input = d.querySelectorAll("#custom-action-counts input")[index]; input.value = value;
    input.dispatchEvent(new w.Event("input", {bubbles: true}));
});
const expected = values(first);
const form = d.getElementById("custom-game-form");
assert.equal(form.dispatchEvent(new w.Event("submit", {cancelable: true})), true);
const stored = w.localStorage.getItem(key);
assert.deepEqual(JSON.parse(stored), expected);
first.window.close();

const inspection = render(payload.inspectionPage, stored);
assert.deepEqual(values(inspection), expected);
inspection.window.close();

const boundary = render(payload.page, JSON.stringify({...expected, actionCounts: ["100", "3", "4"]}));
assert.equal(values(boundary).actionCounts[0], "100");
assert.equal(boundary.window.document.querySelector("#custom-action-counts input").max, "100");
boundary.window.close();

const overflow = render(payload.page, JSON.stringify({...expected, actionCounts: ["101", "3", "4"]}));
assert.equal(values(overflow).actionCounts[0], "2");
overflow.window.close();

const failed = render(payload.errorPage, stored);
const submitted = {name: "Submitted Game", seed: "-1", payoffStructure: "general_sum",
    playerCount: "3", actionCounts: ["4", "5", "6"]};
assert.deepEqual(values(failed), submitted);
assert.deepEqual(JSON.parse(failed.window.localStorage.getItem(key)), submitted);
failed.window.close();
'''
    run_node(script, payload=payload, jsdom=True)


def test_sidebar_seed_is_shared_while_other_form_state_remains_mode_specific(tmp_path):
    app, service = create_test_app(tmp_path)
    script = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const render = (page, stored) => {
    const dom = new JSDOM(page, {url: "http://localhost/", runScripts: "outside-only"});
    const w = dom.window;
    Object.entries(stored).forEach(([key, value]) => w.localStorage.setItem(key, value));
    w.fetch = async () => ({ok: true, json: async () => payload.catalog});
    w.eval(payload.script);
    return dom;
};
const contents = storage => Object.fromEntries([...Array(storage.length)].map((_, index) => {
    const key = storage.key(index); return [key, storage.getItem(key)];
}));
const fixedKey = "swap-regret-experiment-form", onePlayerKey = "swap-regret-adversarial-form";
const seedKey = "swap-regret-experiment-seed";
const first = render(payload.fixedPage, {
    [fixedKey]: JSON.stringify({horizon: "17", seed: "11"}),
    [onePlayerKey]: JSON.stringify({actions: "9", seed: "22"}),
});
const firstSeed = first.window.document.getElementById("experiment-seed");
assert.equal(firstSeed.value, "42");
assert.equal(first.window.document.getElementById("horizon").value, "17");
assert.equal(JSON.parse(first.window.localStorage.getItem(fixedKey)).seed, undefined);
firstSeed.value = "73";
firstSeed.dispatchEvent(new first.window.Event("input", {bubbles: true}));
const fixedHorizon = first.window.document.getElementById("horizon");
fixedHorizon.value = "18"; fixedHorizon.dispatchEvent(new first.window.Event("input", {bubbles: true}));
let saved = contents(first.window.localStorage);
assert.equal(saved[seedKey], "73");
assert.equal(JSON.parse(saved[fixedKey]).seed, undefined);
assert.equal(JSON.parse(saved[fixedKey]).horizon, "18");
assert.equal(JSON.parse(saved[onePlayerKey]).actions, "9");
first.window.close();

const onePlayer = render(payload.onePlayerPage, saved);
const onePlayerSeed = onePlayer.window.document.getElementById("experiment-seed");
assert.equal(onePlayerSeed.value, "73");
assert.equal(onePlayer.window.document.getElementById("actions").value, "9");
assert.equal(JSON.parse(onePlayer.window.localStorage.getItem(onePlayerKey)).seed, undefined);
onePlayerSeed.value = "84";
onePlayerSeed.dispatchEvent(new onePlayer.window.Event("change", {bubbles: true}));
const actions = onePlayer.window.document.getElementById("actions");
actions.value = "7"; actions.dispatchEvent(new onePlayer.window.Event("input", {bubbles: true}));
saved = contents(onePlayer.window.localStorage);
assert.equal(saved[seedKey], "84");
assert.equal(JSON.parse(saved[onePlayerKey]).seed, undefined);
assert.equal(JSON.parse(saved[onePlayerKey]).actions, "7");
onePlayer.window.close();

const returned = render(payload.fixedPage, saved);
assert.equal(returned.window.document.getElementById("experiment-seed").value, "84");
assert.equal(returned.window.document.getElementById("horizon").value, "18");
returned.window.close();
'''
    static = Path(__file__).parents[2] / "web/static"
    payload = {
        "fixedPage": production_ui_page(app, service),
        "onePlayerPage": production_ui_page(app, service, "adversarial"),
        "catalog": service.figure_builder.catalog("fixed"),
        "script": "\n".join((static / name).read_text() for name in ("common.js", "dashboard.js", "figure_builder.js")),
    }
    run_node(script, payload=payload, jsdom=True)


def test_result_filter_modes_cache_restoration_races_and_export(tmp_path):
    app, service = create_test_app(tmp_path)
    for profile in ("hedge_vs_hedge", "ito_hedge_vs_ito_hedge", "hedge_vs_ito_hedge"):
        result(service, profile)
    result(service, "auer_exp3_vs_bm_exp3", mode="bandit")
    result(service, "bm_hedge_vs_bm_hedge", game="rpsls")
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
const profileValue = value => [...f("profiles").options].find(option =>
    option.value === value || option.value.endsWith(`__${value}`)).value;
const selectProfiles = values => {
    const selected = values.map(profileValue);
    [...f("profiles").options].forEach(option => option.selected = selected.includes(option.value));
    f("profiles").dispatchEvent(new w.Event("change"));
};
const figuresFor = (body, tag) => {
    const comparison = body.get("comparison_mode");
    const metrics = comparison === "regrets" ? ["all"] : ["external", "internal", "swap"];
    const views = ["average", "sqrt_scaling"];
    const figures = metrics.flatMap(metric => views.map(view => ({
        metric, view, title: metric + " " + view, filename: `${tag}_${metric}_${view}.png`,
        pdf_filename: `${tag}_${metric}_${view}.pdf`, url: `/${tag}_${metric}_${view}.png`,
        pdf_url: `/${tag}_${metric}_${view}.pdf`
    })));
    return figures;
};
const cacheKey = body => [body.get("comparison_mode"), body.get("action") || "", body.get("horizon"), ...body.getAll("profiles")].join("/");
let cacheRequests = [], generations = [], exports = [], cached = new Map(), downloads = 0;
w.HTMLElement.prototype.scrollIntoView = () => {};
w.HTMLAnchorElement.prototype.click = () => downloads++;
w.URL.createObjectURL = () => "blob:test";
w.URL.revokeObjectURL = () => {};
w.fetch = async (url, options) => {
    if (!options) return {ok: true, json: async () => payload.catalog};
    if (options.body.get("mode") === "figure_builder") {
        return new Promise(resolve => exports.push({filenames: options.body.getAll("filenames"), resolve}));
    }
    if (new URL(url, w.location).pathname === "/figure-builder/cache") {
        return new Promise(resolve => cacheRequests.push({body: options.body, resolve}));
    }
    return new Promise(resolve => generations.push({body: options.body, resolve}));
};
const finishCache = (index, hit = cached.has(cacheKey(cacheRequests[index].body)), items = null) => {
    const request = cacheRequests[index], figures = items || cached.get(cacheKey(request.body)) || [];
    request.resolve({ok: true, json: async () => ({cached: hit, figures: hit ? figures : []})});
};
const finishGeneration = (index, tag) => {
    const request = generations[index], figures = figuresFor(request.body, tag);
    cached.set(cacheKey(request.body), figures);
    request.resolve({ok: true, json: async () => ({
        profiles: request.body.getAll("profiles"), figures
    })});
};
const finishExport = index => exports[index].resolve({ok: true, blob: async () => new w.Blob(["pdf"])});
w.eval(payload.script);
    for (const name of ["scope", "feedback", "player", "horizon", "metric", "view", "context", "profiles", "compare-profiles", "compare-regrets", "compare-horizons"]) {
    assert.equal(f(name).disabled, true, name + " must be disabled while loading");
}
(async () => {
    await tick(); await tick();
    const game = d.getElementById("game"), gameDescription = d.getElementById("game-description");
    const gameDescriptionBefore = gameDescription.getBoundingClientRect();
    game.value = "rpsls"; game.dispatchEvent(new w.Event("change", {bubbles: true}));
    const gameDescriptionAfter = gameDescription.getBoundingClientRect();
    assert.equal(d.getElementById("game-description"), gameDescription);
    assert.deepEqual([gameDescriptionAfter.top, gameDescriptionAfter.bottom],
        [gameDescriptionBefore.top, gameDescriptionBefore.bottom]);
    const following = w.Node.DOCUMENT_POSITION_FOLLOWING;
    assert.equal(d.getElementById("results-controls-heading").textContent, "Result filters");
    assert.equal(d.getElementById("figure-results-heading").textContent, "Generated figures");
    assert.equal(d.getElementById("summary-heading").textContent, "Recorded output");
    const headings = [...d.querySelectorAll("h2")].map(heading => heading.textContent);
    assert.equal(headings.includes("Filter results"), false); assert.equal(headings.includes("Result filter"), false);
    assert(d.getElementById("figure-results").classList.contains("panel"));
    assert.deepEqual([d.getElementById("result-filters"), d.getElementById("figure-results"),
        d.querySelector(".summary-panel")].map(panel => [panel.querySelector(".eyebrow").textContent,
            panel.querySelector(".panel-heading h2").textContent]),
        [["Analysis", "Result filters"], ["Visualization", "Generated figures"], ["Data", "Recorded output"]]);
    assert.equal(d.querySelector(".figures-panel"), null);
    assert.equal(f("select-all"), null); assert.equal(f("clear-all"), null);
    assert(b("generate").closest("#result-filters"));
    assert(b("download").closest("#figure-results"));
    assert(b("status").closest("#figure-results"));
    assert.deepEqual([...d.querySelectorAll(".segmented-control span")].map(node => node.textContent),
        ["Regret notions", "Algorithm profiles", "Horizons"]);
    assert.equal(f("compare-regrets").checked, true); assert.equal(f("compare-profiles").checked, false);
    assert(d.querySelector(".shared-result-filters").compareDocumentPosition(d.querySelector(".comparison-control")) & following);
    assert(d.querySelector(".comparison-control").compareDocumentPosition(f("profiles")) & following);
    assert.equal(f("profiles").size, 1); assert.equal(f("profiles").multiple, false);
    assert.equal(f("profiles-label").textContent, "Algorithm profile");
    assert.equal(f("profiles").selectedOptions.length, 1);
    assert.equal(f("profiles").options[0].value, "all");
    assert.equal(f("profiles").options[0].textContent, "All algorithm profiles");
    assert.equal(f("scope").disabled, false); assert.equal(f("feedback").disabled, false);
    assert.deepEqual([...f("feedback").options].map(option => [option.value, option.textContent]),
        [["full_information", "Full information"], ["bandit", "Bandit feedback"], ["both", "Both"]]);
    assert.equal(f("player").disabled, false); assert.equal(f("metric").disabled, true);
    assert.equal(f("metric").value, "all");
    assert.deepEqual([...f("view").options].map(option => [option.value, option.textContent]), [
        ["all", "Both views"], ["average", "Average regret (R / T)"],
        ["sqrt_scaling", "Scaling (R / sqrt(T))"]]);
    assert.equal(f("view").disabled, false); assert.equal(f("profiles").disabled, false);
    assert.equal(f("compare-profiles").disabled, false); assert.equal(f("compare-regrets").disabled, false);
    assert.equal(f("compare-horizons").disabled, true);
    assert.equal(b("generate").textContent, "Generate figures");
    assert.equal(b("generate").classList.contains("button-primary"), true);
    assert.equal(b("download").classList.contains("button-primary"), false);
    const detailGrid = d.getElementById("detail-joint-actions").parentElement;
    assert(detailGrid.classList.contains("detail-figure-grid"));
    assert.equal(d.getElementById("detail-convergence").parentElement, detailGrid);
    assert(d.getElementById("detail-joint-actions").classList.contains("detail-figure-card"));
    assert(d.getElementById("detail-convergence").classList.contains("detail-figure-card"));
    for (const id of ["detail-heatmap", "detail-heatmap-download", "detail-equilibrium-distance",
            "detail-equilibrium-distance-download", "detail-equilibrium-distance-card"]) assert(d.getElementById(id));
    assert.equal(b("generate").disabled, true);
    assert(visibleRows().length > 0);
    assert.equal(visibleRows()[0].children[1].textContent, "Full information");
    assert.deepEqual([...d.querySelectorAll('#figure-builder select, .summary-panel select')].map(select => select.id), ["result-page-size"]);
    assert.equal(cacheRequests.length, 1); assert.equal(generations.length, 0);
    assert.deepEqual(cacheRequests[0].body.getAll("profiles"), ["hedge_vs_hedge"]);
    assert.equal(cacheRequests[0].body.get("comparison_mode"), "regrets");
    finishCache(0, false); await tick(); await tick();
    assert.equal(b("status").textContent, "No generated figures for this selection.");
    assert.equal(b("generate").disabled, false);
    f("compare-profiles").checked = true;
    f("compare-profiles").dispatchEvent(new w.Event("change"));
    assert.equal(f("profiles").multiple, true); assert.equal(f("profiles").size, 3);
    assert.equal(f("profiles-label").textContent, "Algorithm profiles");
    assert.equal([...f("profiles").options].some(option => option.value === "all"), false);
    assert.equal(f("profiles").selectedOptions.length, 1);
    assert.equal(f("metric").disabled, false);
    assert.deepEqual([...f("metric").options].map(option => option.value), ["all", "external", "internal", "swap"]);
    assert.deepEqual([...f("view").options].map(option => option.value), ["all", "average", "sqrt_scaling"]);
    change("metric", "internal"); change("view", "average");
    assert.equal(cacheRequests.length, 2); assert.equal(generations.length, 0);
    assert.equal(b("figure").children.length, 0); assert.equal(b("generate").disabled, true);
    finishCache(1, false); await tick(); await tick();
    assert.equal(b("status").textContent, "No generated figures for this selection.");
    assert.equal(b("generate").disabled, false);
    submit();
    assert.equal(generations.length, 1);
    finishGeneration(0, "profile-a"); await tick(); await tick();
    assert.equal(b("figure").children.length, 6); assert.equal(visibleCards().length, 1);
    assert.deepEqual([visibleCards()[0].dataset.metric, visibleCards()[0].dataset.view], ["internal", "average"]);
    assert.equal(b("generate").disabled, true);
    change("metric", "all");
    assert.equal(cacheRequests.length, 2); assert.equal(visibleCards().length, 3);
    change("view", "all");
    assert.equal(cacheRequests.length, 2); assert.equal(visibleCards().length, 6);
    change("metric", "internal"); change("view", "average");
    assert.equal(cacheRequests.length, 2); assert.equal(visibleCards().length, 1);
    selectProfiles(["hedge_vs_hedge", "hedge_vs_ito_hedge"]);
    assert.equal(cacheRequests.length, 3); assert.equal(b("figure").children.length, 0);
    finishCache(2, false); await tick(); await tick();
    assert.equal(b("generate").disabled, false);
    selectProfiles(["hedge_vs_hedge"]);
    finishCache(3); await tick(); await tick();
    assert.equal(b("status").textContent, "Cached figures loaded.");
    assert.equal(b("figure").firstChild.dataset.filename, "profile-a_external_average.png");
    assert.equal(b("figure").children.length, 6); assert.equal(visibleCards().length, 1);
    assert.equal(b("generate").disabled, true); assert.equal(b("download").disabled, false);

    selectProfiles(["hedge_vs_hedge", "hedge_vs_ito_hedge"]);
    finishCache(4, false); await tick(); await tick();
    f("compare-regrets").checked = true;
    f("compare-regrets").dispatchEvent(new w.Event("change"));
    assert.equal(f("profiles").multiple, false);
    assert.equal(f("profiles").size, 1);
    assert.equal(f("profiles-label").textContent, "Algorithm profile");
    assert.deepEqual([...f("profiles").selectedOptions].map(option => option.value), ["hedge_vs_hedge"]);
    assert.equal(f("metric").value, "all");
    assert.equal(f("metric").disabled, true);
    assert.equal(f("view").disabled, false);
    assert.equal(cacheRequests[5].body.get("comparison_mode"), "regrets");
    assert.deepEqual(cacheRequests[5].body.getAll("profiles"), ["hedge_vs_hedge"]);
    assert.equal(cacheRequests[5].body.has("metric"), false);
    const stored = JSON.parse(w.localStorage.getItem("swap-regret-profile-batch-filters-fixed"));
    assert.equal(stored.horizon, "30");
    assert(Object.values(stored.selections).some(selection => selection.includes("hedge_vs_hedge")));
    finishCache(5, false); await tick(); await tick();
    submit(); finishGeneration(1, "regret-a"); await tick(); await tick();
    assert.equal(b("figure").children.length, 2); assert.equal(visibleCards().length, 1);
    change("view", "all");
    assert.equal(visibleCards().length, 2); assert.equal(f("metric").value, "all");
    change("view", "sqrt_scaling");
    assert.equal(f("metric").value, "all"); assert.equal(f("metric").disabled, true);
    assert.equal(visibleCards().length, 1);
    assert.equal(visibleCards()[0].dataset.view, "sqrt_scaling");
    assert.equal(cacheRequests.length, 6); assert.equal(generations.length, 2);
    change("view", "average");
    assert.equal(f("metric").value, "all"); assert.equal(f("metric").disabled, true);
    assert.equal(visibleCards().length, 1);
    f("profiles").value = profileValue("ito_hedge_vs_ito_hedge"); f("profiles").dispatchEvent(new w.Event("change"));
    assert.equal(cacheRequests.length, 7); assert.equal(b("figure").children.length, 0);
    f("profiles").value = profileValue("hedge_vs_hedge"); f("profiles").dispatchEvent(new w.Event("change"));
    finishCache(7); await tick(); await tick();
    assert.equal(b("figure").firstChild.dataset.filename, "regret-a_all_average.png");
    finishCache(6, true, figuresFor(cacheRequests[6].body, "stale-b")); await tick(); await tick();
    assert.equal(b("figure").firstChild.dataset.filename, "regret-a_all_average.png");
    b("download").click();
    assert.deepEqual(exports[0].filenames, ["regret-a_all_average.pdf"]);
    finishExport(0); await tick(); await tick();
    assert.equal(downloads, 1);
    f("compare-profiles").checked = true;
    f("compare-profiles").dispatchEvent(new w.Event("change"));
    assert.equal(f("profiles").multiple, true);
    assert.deepEqual([...f("view").options].map(option => option.value), ["all", "average", "sqrt_scaling"]);
    assert.equal(f("view").value, "average");
    assert.equal(f("metric").disabled, false);
    assert.equal(f("metric").value, "internal");
    assert.deepEqual([...f("profiles").selectedOptions].map(option => option.value).sort(),
        ["hedge_vs_hedge", "hedge_vs_ito_hedge"].sort());
    finishCache(8); await tick(); await tick();

    change("feedback", "both");
    assert.deepEqual([...f("profiles").options].map(option => option.textContent).sort(),
        ["EXP3 vs BM-EXP3", "Hedge vs Hedge", "Hedge vs Ito-Hedge", "Ito-Hedge vs Ito-Hedge"].sort());
    selectProfiles(["hedge_vs_hedge", "auer_exp3_vs_bm_exp3"]);
    assert.equal(visibleRows().length, 1); // The current server page is not re-filtered in place.
    f("compare-regrets").checked = true;
    f("compare-regrets").dispatchEvent(new w.Event("change"));
    assert.equal(f("profiles").multiple, false); assert.equal(f("profiles").selectedOptions.length, 1);
    assert.equal(f("profiles").options[0].value, "all");
    f("profiles").value = "all"; f("profiles").dispatchEvent(new w.Event("change"));
    assert.equal(visibleRows().length, 1);
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, script, url=browse_url(service, scope="rps",
        comparison_mode="regrets", feedback="full_information", horizon="30",
        player="0", profiles=["hedge_vs_hedge"]))


def test_result_filter_sizes_profiles_and_disables_empty_dependencies(tmp_path):
    app, service = create_test_app(tmp_path)
    contexts = []
    for index, count in enumerate((1, 2, 3, 8, 9), start=1):
        contexts.append({
            "id": f"{index:024x}", "mode": "fixed", "scope": f"game_{count}",
            "scope_label": f"Game {count}", "feedback_modes": ["full_information"], "player": 0,
            "batch_label": f"{count} profiles", "result_keys": [], "comparison_modes": ["regrets", "profiles"],
            "horizons": [30],
            "profiles": [{"id": f"profile_{profile}",
                          "feedback_mode": "full_information", "label": f"Profile {profile}",
                          "metrics": ["external", "internal", "swap"], "horizons": [30]} for profile in range(count)],
        })
    catalog = {"contexts": contexts,
               "metrics": [{"id": metric, "label": metric.title()} for metric in ("external", "internal", "swap")],
               "views": [{"id": "average", "label": "Average"}, {"id": "sqrt_scaling", "label": "Scaling"}]}
    script = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/", runScripts: "outside-only"});
const w = dom.window, d = w.document, f = name => d.getElementById("filter-" + name);
w.fetch = async () => ({ok: true, json: async () => payload.catalog});
w.eval(payload.script);
for (const select of d.querySelectorAll("#result-filters select")) assert.equal(select.disabled, true);
(async () => {
    await new Promise(resolve => setImmediate(resolve));
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(f("compare-regrets").checked, true);
    assert.equal(f("profiles").multiple, false); assert.equal(f("profiles").size, 1);
    f("compare-profiles").checked = true;
    f("compare-profiles").dispatchEvent(new w.Event("change"));
    for (const [scope, size] of [["game_1", 2], ["game_2", 2], ["game_3", 3], ["game_8", 8], ["game_9", 8]]) {
        f("scope").value = scope;
        f("scope").dispatchEvent(new w.Event("change"));
        assert.equal(f("profiles").size, size);
        assert.equal(f("profiles").disabled, false);
    }
    assert([...d.querySelectorAll("#result-filters select")].every(select => select.disabled || select.options.length));
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, script, catalog=catalog)

    empty_script = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/", runScripts: "outside-only"});
const w = dom.window, d = w.document;
w.fetch = async () => ({ok: true, json: async () => payload.catalog});
w.eval(payload.script);
(async () => {
    await new Promise(resolve => setImmediate(resolve));
    await new Promise(resolve => setImmediate(resolve));
    for (const control of d.querySelectorAll("#result-filters select, .comparison-control input")) {
        assert.equal(control.disabled, true);
    }
    const profiles = d.getElementById("filter-profiles");
    assert.equal(d.getElementById("filter-compare-regrets").checked, true);
    assert.equal(profiles.multiple, false); assert.equal(profiles.size, 1);
    assert.equal(d.getElementById("builder-generate").disabled, true);
    assert.equal(d.getElementById("builder-status").textContent, "No figures to display.");
    assert.equal(d.getElementById("empty-results-heading").textContent, "Recorded output");
    assert.equal(d.querySelector(".summary-panel .eyebrow").textContent, "Data");
    assert.equal(d.querySelector(".summary-panel .empty-state strong").textContent, "Loading saved result view…");
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, empty_script)

    css = (Path(__file__).parents[2] / "web" / "static" / "dashboard.css").read_text()
    assert all(selector in css for selector in ("button:disabled", "select:disabled", ".segmented-control input:disabled + span", ".result-panel", ".panel-heading", ".empty-state"))
    assert "background: var(--surface-muted);" in css


def test_horizon_filter_and_comparison_use_one_combined_artifact(tmp_path):
    app, service = create_test_app(tmp_path)
    for horizon in (20, 40, 80):
        result(service, "hedge_vs_hedge", horizon=horizon)
    script = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/", runScripts: "outside-only"});
const w = dom.window, d = w.document, f = name => d.getElementById("filter-" + name);
const tick = () => new Promise(resolve => setImmediate(resolve));
const visibleCards = () => [...d.getElementById("builder-figure").children].filter(card => !card.hidden);
let generations = 0;
w.fetch = async (url, options) => {
    if (!options) return {ok: true, json: async () => payload.catalog};
    if (new URL(String(url), w.location).pathname === "/figure-builder/cache") {
        return {ok: true, json: async () => ({cached: false, figures: []})};
    }
    generations += 1;
    return {ok: true, json: async () => ({figures: [{
        metric: "all", view: "horizon_scaling", title: "Horizon scaling", filename: "horizons.png",
        pdf_filename: "horizons.pdf", url: "/horizons.png", pdf_url: "/horizons.pdf"
    }]})};
};
w.eval(payload.script);
(async () => {
    await tick(); await tick();
    assert.deepEqual([...f("horizon").options].map(option => option.value), ["20", "40", "80"]);
    assert.equal(f("horizon").value, "20"); assert.equal(f("horizon").disabled, false);
    f("horizon").value = "40"; f("horizon").dispatchEvent(new w.Event("change"));
    assert([...d.querySelectorAll(".summary-row")].every(row => !row.hidden));
    assert.equal(f("compare-horizons").disabled, false);
    f("compare-horizons").checked = true;
    f("compare-horizons").dispatchEvent(new w.Event("change"));
    await tick(); await tick();
    assert.equal(f("profiles").multiple, false);
    assert.deepEqual([...f("horizon").options].map(option => option.value), ["all"]);
    assert.equal(f("horizon").disabled, true);
    assert.deepEqual([...f("profiles").options].map(option => option.value), ["all", "hedge_vs_hedge"]);
    f("profiles").value = "all"; f("profiles").dispatchEvent(new w.Event("change"));
    await tick(); await tick();
    assert.equal(f("profiles").value, "all");
    assert.deepEqual([...f("metric").options].map(option => option.value), ["all"]);
    assert.equal(f("metric").disabled, true);
    assert.deepEqual([...f("view").options].map(option => option.value), ["horizon_scaling"]);
    assert.equal(f("view").disabled, true);
    d.getElementById("figure-builder").dispatchEvent(new w.Event("submit", {cancelable: true}));
    await tick(); await tick();
    assert.equal(generations, 1);
    assert.deepEqual(visibleCards().map(card => [card.dataset.metric, card.dataset.view]), [["all", "horizon_scaling"]]);
    assert.equal(generations, 1);
    f("compare-profiles").checked = true;
    f("compare-profiles").dispatchEvent(new w.Event("change"));
    assert.equal([...f("profiles").options].some(option => option.value === "all"), false);
    assert.deepEqual([...f("horizon").options].map(option => option.value), ["20", "40", "80"]);
    assert.equal(f("horizon").value, "40"); assert.equal(f("horizon").disabled, false);
    assert.deepEqual([...f("view").options].map(option => option.value), ["all", "average", "sqrt_scaling"]);
    f("compare-horizons").checked = true;
    f("compare-horizons").dispatchEvent(new w.Event("change"));
    assert.equal(f("profiles").value, "all");
    const stored = JSON.parse(w.localStorage.getItem("swap-regret-profile-batch-filters-fixed"));
    assert(Object.values(stored.selections).some(selection => selection.length === 1 && selection[0] === "all"));
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, script)


def test_filtered_deletion_does_not_collect_dom_ids_and_fixed_row_interactions_still_work(tmp_path):
    app, service = create_test_app(tmp_path)
    result(service, "hedge_vs_hedge")
    result(service, "ito_hedge_vs_ito_hedge")
    script = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/", runScripts: "outside-only"});
const w = dom.window, d = w.document, f = name => d.getElementById("filter-" + name);
const tick = () => new Promise(resolve => setImmediate(resolve));
w.fetch = async (url, options) => String(url).startsWith("/experiment-groups/fixed/")
    ? {ok: true, json: async () => payload.details[String(url)]}
    : options
        ? {ok: true, json: async () => ({cached: false, figures: []})}
        : {ok: true, json: async () => payload.catalog};
w.confirm = () => false;
w.HTMLElement.prototype.scrollIntoView = () => {};
w.eval(payload.script);
const form = d.getElementById("delete-filtered-experiments");
const button = d.getElementById("delete-filtered-experiments-button");
const ids = () => [...form.querySelectorAll('input[name="group_id"]')].map(input => input.value);
const visibleRows = () => [...d.querySelectorAll(".summary-row")].filter(row => !row.hidden);
(async () => {
    await tick(); await tick();
    assert(d.querySelector("#summary-table thead .sticky-actions"));
    assert([...d.querySelectorAll("#summary-table tbody .sticky-actions")].every(cell => cell.querySelector("button")));
    assert.equal(d.querySelectorAll(".summary-row").length, 1);
    assert.equal(d.querySelectorAll(".summary-row button").length, 1);
    assert.equal(visibleRows().length, 1);
    assert.deepEqual(ids(), []);
    assert.equal(button.textContent.trim(), "Delete filtered experiments");
    assert.equal(button.disabled, false);

    const row = visibleRows()[0], detail = d.getElementById("experiment-detail");
    row.querySelector("button").click();
    assert.equal(detail.hidden, true);
    row.click();
    assert.equal(detail.hidden, false);
    assert.equal(d.getElementById("detail-status").textContent, "Loading result details…");
    await tick(); await tick();
    const selected = payload.details[row.dataset.detailUrl];
    assert.equal(d.getElementById("detail-content").hidden, false);
    assert.deepEqual([...d.querySelectorAll("#detail-metadata dt")].map(item => item.textContent),
        ["Feedback", "Profile", "Horizon", "Seed", "Replicates", "Stationary solver"]);
    assert.equal(d.querySelectorAll("#detail-regrets strong").length, Object.keys(selected.display_regrets).length);
    const downloadLinks = [...d.querySelectorAll("#detail-downloads a")];
    assert.deepEqual(downloadLinks.map(link => [link.getAttribute("href"), link.getAttribute("download")]),
        selected.runs.map(run => [run.download_url, run.experiment]));

    assert.deepEqual(ids(), []);
    assert.equal(visibleRows().length, 1);
    assert.equal(button.textContent.trim(), "Delete filtered experiments");
    assert.equal(button.disabled, false);
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, script, details=fixed_details(app, service))

@pytest.mark.parametrize("mode", ["fixed", "adversarial"])
def test_filtered_delete_browser_previews_server_count_and_reconfirms_on_stale_response(
    tmp_path, mode,
):
    app, service = create_test_app(tmp_path)
    if mode == "fixed":
        result(service, "hedge_vs_hedge")
    else:
        run_adversarial_experiment(
            "hedge", environment="historical_frequency_v3", feedback_mode="full_information",
            n_actions=2, horizon=2, seed=42, output_dir=service.adversarial_raw_dir,
        )
    script = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/", runScripts: "outside-only"});
const w = dom.window, d = w.document;
const tick = () => new Promise(resolve => setImmediate(resolve));
const previews = [], deletes = [], confirmations = [];
const digest = "a".repeat(64);
let previewMode = "success", pendingPreview = null;
w.fetch = (url, options = {}) => {
    const path = String(url);
    if (path.endsWith("/delete-filtered/preview")) {
        previews.push({path, options});
        if (previewMode === "pending") return new Promise(resolve => {pendingPreview = resolve;});
        const count = previewMode === "zero" ? 0 : 7;
        return Promise.resolve(previewMode === "failure"
            ? {ok: false, status: 400, json: async () => ({error: "Preview unavailable"})}
            : {ok: true, status: 200, json: async () => ({count, digest})});
    }
    if (path.endsWith("/delete-filtered")) {
        return new Promise(resolve => deletes.push({path, options, resolve}));
    }
    if (options.method === "POST") {
        return Promise.resolve({ok: true, json: async () => ({cached: false, figures: []})});
    }
    return Promise.resolve({ok: true, json: async () => payload.catalog});
};
w.confirm = message => { confirmations.push(message); return true; };
w.HTMLElement.prototype.scrollIntoView = () => {};
w.eval(payload.script);
const form = d.getElementById("delete-filtered-experiments");
const button = d.getElementById("delete-filtered-experiments-button");
const submit = () => form.dispatchEvent(new w.Event("submit", {bubbles: true, cancelable: true}));
(async () => {
    await tick(); await tick(); await tick();
    assert(d.getElementById("filter-context").value);
    assert.equal(button.disabled, false);
    assert.equal(submit(), false);
    assert.equal(previews.length, 1);
    assert.equal(deletes.length, 0);
    await tick(); await tick();
    assert.equal(confirmations.length, 1);
    assert(confirmations[0].includes("7 filtered experiments"));
    assert.equal(deletes.length, 1);
    const previewBody = previews[0].options.body, deleteBody = deletes[0].options.body;
    assert.equal(previewBody.get("_csrf_token"), deleteBody.get("_csrf_token"));
    assert.equal(previewBody.get("mode"), payload.mode_label);
    assert.equal(deleteBody.get("context"), d.getElementById("filter-context").value);
    assert.equal(deleteBody.get("digest"), digest);
    assert.deepEqual(deleteBody.getAll("group_id"), []);
    assert.equal(deleteBody.get("resultKeys"), null);
    assert.equal(d.querySelectorAll(".summary-row:not([hidden])").length, 1);

    deletes[0].resolve({ok: false, status: 409, json: async () =>
        ({error: "Results changed; please review and confirm again.", reconfirmation_required: true})});
    await tick(); await tick();
    assert(d.getElementById("filtered-deletion-status").textContent.includes("review and confirm again"));
    assert.equal(deletes.length, 1);
    assert.equal(button.disabled, false);

    previewMode = "failure";
    assert.equal(submit(), false);
    await tick(); await tick();
    assert.equal(previews.length, 2);
    assert.equal(deletes.length, 1);
    assert.equal(confirmations.length, 1);
    assert(d.getElementById("filtered-deletion-status").textContent.includes("Preview unavailable"));

    previewMode = "zero";
    assert.equal(submit(), false);
    await tick(); await tick();
    assert.equal(previews.length, 3);
    assert.equal(deletes.length, 1);
    assert.equal(confirmations.length, 1);
    assert(d.getElementById("filtered-deletion-status").textContent.includes("No filtered experiments"));

    previewMode = "pending";
    assert.equal(submit(), false);
    assert.equal(submit(), false);
    assert.equal(previews.length, 4); // No duplicate request while preview is in flight.
    assert.equal(button.disabled, true);
    const canonical = JSON.parse(d.getElementById("dashboard-data").textContent).browsingState;
    d.dispatchEvent(new w.CustomEvent("results-filter-change", {detail: {
        ...canonical, profiles: ["missing"],
    }}));
    pendingPreview({ok: true, status: 200, json: async () => ({count: 7, digest})});
    await tick(); await tick();
    assert(d.getElementById("filtered-deletion-status").textContent.includes("Filters changed"));
    assert.equal(confirmations.length, 1);
    assert.equal(deletes.length, 1);
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, script, mode=mode, mode_label=mode)


def test_adversarial_builder_controls_leave_server_rows_unchanged(tmp_path) -> None:
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
    for form in (
        VALID_FORM | {"actions": "2,4"},
        VALID_FORM | {"environment": RANDOM_WALK_ENVIRONMENT, "algorithm_names": ["hedge"],
                      "actions": "2,4", "horizon": "5", "seed": "9"},
        VALID_FORM | {"environment": RANDOM_WALK_ENVIRONMENT, "feedback_mode": "bandit",
                      "algorithm_names": ["exp3_ix"], "actions": "2,4", "horizon": "5", "seed": "9"},
        VALID_FORM | {"environment": RANDOM_WALK_ENVIRONMENT, "feedback_mode": "bandit",
                      "algorithm_names": ["exp3_ix"], "actions": "3,4", "horizon": "6", "seed": "9"},
        VALID_FORM | {"environment": RANDOM_WALK_ENVIRONMENT, "feedback_mode": "bandit",
                      "algorithm_names": ["exp3_ix"], "actions": "5,6", "horizon": "5", "seed": "10"},
    ):
        submit_and_wait(client, service, form)

    script = r'''const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/?mode=adversarial", runScripts: "outside-only"});
const w = dom.window, d = w.document, f = name => d.getElementById("filter-" + name);
const initialData = JSON.parse(d.getElementById("dashboard-data").textContent);
assert.equal(Object.prototype.hasOwnProperty.call(initialData, "summaries"), false);
const select = (name, value) => {f(name).value = value; f(name).dispatchEvent(new w.Event("change", {bubbles: true}));};
const tick = () => new Promise(resolve => setImmediate(resolve));
const generationUrls = [];
w.fetch = async (url, options) => {
    if (!options) return {ok: true, json: async () => payload.catalog};
    if (new URL(String(url), w.location).pathname === "/figure-builder/cache") {
        return {ok: true, json: async () => ({cached: false, figures: []})};
    }
    generationUrls.push(String(url));
    return {ok: true, json: async () => ({figures: []})};
};
w.HTMLElement.prototype.scrollIntoView = () => {};
w.eval(payload.script);
(async () => {
    await tick(); await tick();
    const table = d.getElementById("summary-table");
    const initialRows = [...table.tBodies[0].rows].map(row => row.dataset.resultKey);
    assert.equal(d.querySelector('label[for="filter-action"]').textContent, "Action");
    assert.equal(f("player"), null);
    assert.deepEqual([...d.querySelectorAll(".segmented-control span")].map(node => node.textContent),
        ["Regret notions", "Algorithm profiles", "Action spaces", "Horizons"]);

    const description = d.getElementById("environment-description");
    const before = description.getBoundingClientRect();
    const environment = d.getElementById("adversarial-environment");
    environment.value = "lazy_random_walk_v1";
    environment.dispatchEvent(new w.Event("change", {bubbles: true}));
    assert.equal(d.getElementById("environment-description"), description);
    const after = description.getBoundingClientRect();
    assert.deepEqual([after.top, after.bottom], [before.top, before.bottom]);
    assert(description.textContent.includes("Independent lazy random walks"));

    select("scope", "lazy_random_walk_v1");
    assert.deepEqual([...f("feedback").options].map(option => option.value), ["full_information", "bandit", "both"]);
    select("feedback", "bandit");
    const target = payload.catalog.contexts.find(context => context.scope === "lazy_random_walk_v1"
        && context.horizons.includes(5) && context.base_seed === 9);
    select("context", target.id);
    assert.deepEqual([...f("horizon").options].map(option => option.value), ["5"]);
    select("horizon", "5");
    assert.deepEqual([...f("action").options].map(option => option.value), ["2", "3", "4"]);
    assert(![...f("action").options].some(option => option.value === "all"));
    assert.deepEqual([...f("metric").options].map(option => option.value), ["all"]);
    assert.deepEqual([...f("view").options].map(option => option.value), ["all", "average", "sqrt_scaling"]);
    assert.equal(f("metric").disabled, true); assert.equal(f("profiles").multiple, false);
    select("action", "3");
    assert.deepEqual([...f("horizon").options].map(option => option.value), ["5", "6"]);
    assert.deepEqual([...table.tBodies[0].rows].map(row => row.dataset.resultKey), initialRows);

    f("compare-profiles").checked = true;
    f("compare-profiles").dispatchEvent(new w.Event("change"));
    assert.equal(f("action").disabled, false);
    assert.equal(f("profiles").multiple, true);
    assert.equal(f("metric").disabled, false);
    assert.deepEqual([...f("metric").options].map(option => option.value), ["all", "external", "internal", "swap"]);
    assert(![...f("view").options].some(option => option.value === "log_log_fit"));
    select("metric", "internal");
    select("view", "sqrt_scaling");
    assert.deepEqual([...table.tBodies[0].rows].map(row => row.dataset.resultKey), initialRows);

    f("compare-actions").checked = true;
    f("compare-actions").dispatchEvent(new w.Event("change"));
    assert.deepEqual([...f("action").options].map(option => [option.value, option.textContent]), [["all", "All actions"]]);
    assert.equal(f("action").disabled, true);
    assert.equal(f("metric").value, "internal"); assert.equal(f("metric").disabled, false);
    assert.deepEqual([...f("metric").options].map(option => option.value), ["all", "external", "internal", "swap"]);
    assert(![...f("view").options].some(option => option.value === "log_log_fit"));
    assert.equal(f("profiles").multiple, false); assert.equal(f("profiles").selectedOptions.length, 1);
    assert.equal(f("horizon").value, "5"); assert.equal(f("horizon").disabled, false);
    assert.deepEqual([...table.tBodies[0].rows].map(row => row.dataset.resultKey), initialRows);

    select("feedback", "both");
    assert.deepEqual([...f("profiles").options].map(option => [option.value, option.textContent]), [
        ["exp3_ix", "EXP3-IX"], ["hedge", "Hedge"]]);
    f("profiles").value = "hedge";
    f("profiles").dispatchEvent(new w.Event("change", {bubbles: true}));
    assert.deepEqual([...table.tBodies[0].rows].map(row => row.dataset.resultKey), initialRows);
    assert([...table.tBodies[0].rows].every(row => !row.hidden));

    f("compare-horizons").checked = true;
    f("compare-horizons").dispatchEvent(new w.Event("change"));
    assert.equal(f("action").value, "3"); assert.equal(f("action").disabled, false);
    assert.deepEqual([...f("horizon").options].map(option => option.value), ["all"]);
    assert.equal(f("horizon").disabled, true);
    assert.equal(f("profiles").multiple, false); assert.equal(f("profiles").selectedOptions.length, 1);

    f("compare-regrets").checked = true;
    f("compare-regrets").dispatchEvent(new w.Event("change"));
    assert.equal(f("action").value, "3"); assert.equal(f("action").disabled, false);
    assert(![...f("action").options].some(option => option.value === "all"));
    assert.equal(f("horizon").value, "5"); assert.equal(f("horizon").disabled, false);
    assert.equal(f("metric").value, "all"); assert.equal(f("metric").disabled, true);
    for (const id of ["filter-seed", "filter-player-algorithm", "filter-secondary"]) {
        assert.equal(d.getElementById(id), null);
    }
    await tick(); await tick();
    const builderForm = d.getElementById("figure-builder");
    Object.defineProperty(builderForm, "action", {value: f("action")});
    assert.equal(builderForm.dispatchEvent(new w.Event("submit", {cancelable: true})), false);
    await tick();
    assert.deepEqual(generationUrls, ["/figure-builder/collection"]);
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});'''
    run_ui(app, service, script, "adversarial")
    css = (Path(__file__).parents[2] / "web" / "static" / "dashboard.css").read_text()
    assert ".summary-row[data-summary-index] {" in css
    assert "\n.summary-row {\n" not in css


def test_fixed_detail_keyboard_races_and_failures_are_recoverable(tmp_path):
    app, service = create_test_app(tmp_path)
    result(service, "hedge_vs_hedge", replicates=(0,))
    script = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/", runScripts: "outside-only"});
const w = dom.window, d = w.document;
const tick = () => new Promise(resolve => setImmediate(resolve));
const pending = [];
w.HTMLElement.prototype.scrollIntoView = () => {};
w.fetch = (url, options) => String(url).startsWith("/experiment-groups/fixed/")
    ? new Promise((resolve, reject) => pending.push({url, resolve, reject}))
    : Promise.resolve(options && options.cache === "no-store"
        ? {ok: false, status: 503}
        : {ok: true, json: async () => payload.catalog});
w.eval(payload.script);
const rows = [...d.querySelectorAll(".summary-row")];
assert.equal(rows.length, 2);
const [a, b] = rows;
const finish = (request, row) => request.resolve({
    ok: true, json: async () => payload.details[row.dataset.detailUrl],
});
(async () => {
    await tick(); await tick();
    assert.equal(a.hidden, false);
    a.dispatchEvent(new w.KeyboardEvent("keydown", {key: "Enter", bubbles: true, cancelable: true}));
    assert.equal(pending.length, 1);
    assert.equal(d.getElementById("experiment-detail").getAttribute("aria-busy"), "true");
    assert.match(d.getElementById("detail-status").textContent, /Loading/);
    assert.equal(d.getElementById("detail-content").hidden, true);

    assert.equal(b.hidden, false);
    b.dispatchEvent(new w.KeyboardEvent("keydown", {key: " ", bubbles: true, cancelable: true}));
    assert.equal(pending.length, 2);
    finish(pending[1], b);
    await tick(); await tick();
    assert.equal(d.getElementById("detail-content").hidden, false);
    assert.match(d.getElementById("detail-title").textContent, /player 1/);
    finish(pending[0], a);
    await tick(); await tick();
    assert.match(d.getElementById("detail-title").textContent, /player 1/);
    assert.deepEqual([...d.querySelectorAll("#detail-downloads a")].map(link => link.getAttribute("href")),
        payload.details[b.dataset.detailUrl].runs.map(run => run.download_url));

    a.click();
    assert.equal(pending.length, 3);
    pending[2].resolve({ok: false, status: 404, json: async () => ({error: "missing"})});
    await tick(); await tick();
    assert.match(d.getElementById("detail-status").textContent, /Refresh results/);
    assert.equal(d.getElementById("detail-content").hidden, true);

    b.click();
    assert.equal(pending.length, 4);
    pending[3].reject(new Error("offline"));
    await tick(); await tick();
    assert.match(d.getElementById("detail-status").textContent, /Could not load result details/);
    assert.equal(d.getElementById("detail-content").hidden, true);

    a.click();
    assert.equal(pending.length, 5);
    b.click();
    assert.equal(pending.length, 6);
    finish(pending[4], a);
    await tick(); await tick();
    assert.match(d.getElementById("detail-title").textContent, /player 1/);
    finish(pending[5], b);
    await tick(); await tick();
    assert.equal(d.getElementById("detail-content").hidden, false);
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, script, url=browse_url(service, player="all",
        comparison_mode="regrets", horizon="30", profiles=["hedge_vs_hedge"]),
        details=fixed_details(app, service))


def test_equilibrium_unavailable_notice_preserves_other_result_controls(tmp_path):
    from experiments.scenarios.cross_play import run_cross_play_experiment

    app, service = create_test_app(tmp_path)
    definition = service.create_custom_game("Large Analysis", 2, [100, 100], 7, "zero_sum")
    run_cross_play_experiment(
        definition.id, ["hedge", "hedge"], horizon=2, output_dir=service.raw_dir,
        custom_game_dir=service.game_catalog.custom_game_dir, feedback_mode="full_information",
    )
    result(service, "hedge_vs_hedge", horizon=2, replicates=(0,))
    script = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const tick = () => new Promise(resolve => setImmediate(resolve));
const render = page => {
    const dom = new JSDOM(page, {url: "http://localhost/", runScripts: "outside-only"});
    const w = dom.window, d = w.document;
    w.Element.prototype.scrollIntoView = () => {};
    w.fetch = async (url, options) => String(url).startsWith("/experiment-groups/fixed/")
        ? {ok: true, json: async () => payload.details[String(url)]}
        : options && options.cache === "no-store"
            ? {ok: false, status: 503}
            : {ok: true, json: async () => payload.catalog};
    w.eval(payload.script);
    return {dom, d};
};
(async () => {
    const large = render(payload.page);
    const largeRow = large.d.querySelector(".summary-row");
    assert(largeRow);
    assert.equal(payload.details[largeRow.dataset.detailUrl].equilibrium_distance_url, null);
    largeRow.click();
    await tick(); await tick();
    const notice = large.d.getElementById("detail-equilibrium-distance-unavailable");
    assert.equal(large.d.getElementById("experiment-detail").hidden, false);
    assert.equal(large.d.getElementById("detail-convergence").hidden, false);
    assert.equal(large.d.getElementById("detail-equilibrium-distance-card").hidden, true);
    assert.equal(notice.hidden, false);
    assert.match(notice.textContent, /analysis budget/);
    assert.match(notice.textContent, /regret results remain available/);
    assert.equal(large.d.getElementById("detail-joint-actions").hidden, false);
    assert(large.d.getElementById("detail-regrets").children.length > 0);
    assert(large.d.querySelector("#detail-downloads a"));
    large.dom.window.close();

    const normal = render(payload.normalPage);
    const normalRow = normal.d.querySelector(".summary-row");
    assert(normalRow);
    assert(payload.details[normalRow.dataset.detailUrl].equilibrium_distance_url);
    normalRow.click();
    await tick(); await tick();
    const normalNotice = normal.d.getElementById("detail-equilibrium-distance-unavailable");
    assert.equal(normalNotice.hidden, true);
    assert.equal(normalNotice.textContent, "");
    assert.equal(normal.d.getElementById("detail-equilibrium-distance-card").hidden, false);
    assert(normal.d.getElementById("detail-equilibrium-distance-download").href.endsWith(".pdf"));
    normal.dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, script, url=browse_url(service, scope=definition.id, player="0",
        comparison_mode="regrets", horizon="2", profiles=["hedge_vs_hedge"]),
        normalPage=production_ui_page(app, service, url=browse_url(service, scope="rps",
            player="0", comparison_mode="regrets", horizon="2", profiles=["hedge_vs_hedge"])),
        details=fixed_details(app, service))
