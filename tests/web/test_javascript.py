from pathlib import Path
import shutil
import subprocess

import pytest

from experiments.scenarios.adversarial import RANDOM_WALK_ENVIRONMENT
from tests.web.support import (
    ADVERSARIAL_FORM as VALID_FORM, create_test_app, csrf_token, record_fixed_runs as result,
    run_node, run_ui, submit_and_wait,
)
from web.jobs import Job


@pytest.mark.parametrize("mode,terminal", [("fixed", "succeeded"), ("fixed", "failed"), ("adversarial", "cancelled")])
def test_browser_submission_and_polling_preserve_jobs_without_navigation(tmp_path, monkeypatch, mode, terminal):
    app, service = create_test_app(tmp_path)
    monkeypatch.setattr(service.jobs, "recent", lambda: [Job("old", "Existing", "running", "Running", "now")])
    method = "submit_adversarial_experiment" if mode == "adversarial" else "submit_experiment"
    endpoint = "/"
    monkeypatch.setattr(service, method, lambda form: Job("new", "New job", "queued", "Waiting", "now"))
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
        "fixedPage": app.test_client().get("/").get_data(as_text=True),
        "onePlayerPage": app.test_client().get("/?mode=adversarial").get_data(as_text=True),
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
    return metrics.flatMap(metric => views.map(view => ({
        metric, view, title: metric + " " + view, filename: `${tag}_${metric}_${view}.png`,
        pdf_filename: `${tag}_${metric}_${view}.pdf`, url: `/${tag}_${metric}_${view}.png`,
        pdf_url: `/${tag}_${metric}_${view}.pdf`
    })));
};
const cacheKey = body => [body.get("comparison_mode"), body.get("action") || "", ...body.getAll("profiles")].join("/");
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
for (const name of ["scope", "feedback", "player", "metric", "view", "context", "profiles", "compare-profiles", "compare-regrets"]) {
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
        ["Regret notions", "Algorithm profiles"]);
    assert.equal(f("compare-regrets").checked, true); assert.equal(f("compare-profiles").checked, false);
    assert(d.querySelector(".shared-result-filters").compareDocumentPosition(d.querySelector(".comparison-control")) & following);
    assert(d.querySelector(".comparison-control").compareDocumentPosition(f("profiles")) & following);
    assert.equal(f("profiles").size, 1); assert.equal(f("profiles").multiple, false);
    assert.equal(f("profiles-label").textContent, "Algorithm profile");
    assert.equal(f("profiles").selectedOptions.length, 1);
    assert.equal(f("scope").disabled, false); assert.equal(f("feedback").disabled, false);
    assert.deepEqual([...f("feedback").options].map(option => [option.value, option.textContent]),
        [["full_information", "Full information"], ["bandit", "Bandit feedback"], ["both", "Both"]]);
    assert.equal(f("player").disabled, false); assert.equal(f("metric").disabled, true);
    assert.equal(f("view").disabled, false); assert.equal(f("profiles").disabled, false);
    assert.equal(f("compare-profiles").disabled, false); assert.equal(f("compare-regrets").disabled, false);
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
    assert.equal(d.querySelectorAll("#figure-builder select, .summary-panel select, .summary-panel input").length, 0);
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
    assert.deepEqual([...f("profiles").selectedOptions].map(option => option.value), ["hedge_vs_hedge"]);
    assert.equal(f("metric").disabled, false);
    assert.deepEqual([...f("metric").options].map(option => option.value), ["all", "external", "internal", "swap"]);
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
    const stored = JSON.parse(w.localStorage.getItem("swap-regret-shared-filters-fixed"));
    assert.deepEqual(stored.selections[`${f("context").value}:full_information`], ["hedge_vs_hedge"]);
    finishCache(5, false); await tick(); await tick();
    submit(); finishGeneration(1, "regret-a"); await tick(); await tick();
    assert.equal(b("figure").children.length, 2); assert.equal(visibleCards().length, 1);
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
    assert.equal(f("metric").disabled, false);
    assert.equal(f("metric").value, "internal");
    assert.deepEqual([...f("profiles").selectedOptions].map(option => option.value), ["hedge_vs_hedge"]);
    finishCache(8); await tick(); await tick();

    change("feedback", "both");
    assert.deepEqual([...f("profiles").options].map(option => option.textContent).sort(),
        ["EXP3 vs BM-EXP3", "Hedge vs Hedge", "Hedge vs Ito-Hedge", "Ito-Hedge vs Ito-Hedge"].sort());
    selectProfiles(["hedge_vs_hedge", "auer_exp3_vs_bm_exp3"]);
    assert.deepEqual(new Set(visibleRows().map(row => row.dataset.feedback)), new Set(["full_information", "bandit"]));
    f("compare-regrets").checked = true;
    f("compare-regrets").dispatchEvent(new w.Event("change"));
    assert.equal(f("profiles").multiple, false); assert.equal(f("profiles").selectedOptions.length, 1);
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, script)


def test_result_filter_sizes_profiles_and_disables_empty_dependencies(tmp_path):
    app, service = create_test_app(tmp_path)
    contexts = []
    for index, count in enumerate((1, 2, 3, 8, 9), start=1):
        contexts.append({
            "id": f"{index:024x}", "mode": "fixed", "scope": f"game_{count}",
            "scope_label": f"Game {count}", "feedback_modes": ["full_information"], "player": 0,
            "batch_label": f"{count} profiles", "result_keys": [],
            "profiles": [{"id": f"profile_{profile}",
                          "feedback_mode": "full_information", "label": f"Profile {profile}",
                          "metrics": ["external", "internal", "swap"]} for profile in range(count)],
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
    assert.equal(d.querySelector(".summary-panel .empty-state strong").textContent, "No results yet.");
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, empty_script)

    css = (Path(__file__).parents[2] / "web" / "static" / "dashboard.css").read_text()
    assert all(selector in css for selector in ("button:disabled", "select:disabled", ".segmented-control input:disabled + span", ".result-panel", ".panel-heading", ".empty-state"))
    assert "background: var(--surface-muted);" in css


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
const visible = selector => [...d.querySelectorAll(selector)].filter(node => !node.hidden);
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
    assert.equal(d.querySelector('label[for="filter-action"]').textContent, "Action");
    assert.equal(f("player"), null);
    assert.deepEqual([...d.querySelectorAll(".segmented-control span")].map(node => node.textContent),
        ["Regret notions", "Algorithm profiles", "Action spaces"]);

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
        && context.horizon === 5 && context.base_seed === 9);
    select("context", target.id);
    assert.deepEqual([...f("action").options].map(option => option.value), ["2", "3", "4"]);
    assert(![...f("action").options].some(option => option.value === "all"));
    assert.deepEqual([...f("metric").options].map(option => option.value), ["all"]);
    assert.equal(f("metric").disabled, true); assert.equal(f("profiles").multiple, false);
    select("action", "3");
    assert(visible(".summary-row").every(row => row.dataset.scope === "lazy_random_walk_v1"
        && row.dataset.action === "3" && row.dataset.profile === "exp3_ix"));

    f("compare-profiles").checked = true;
    f("compare-profiles").dispatchEvent(new w.Event("change"));
    assert.equal(f("action").disabled, false);
    assert.equal(f("profiles").multiple, true);
    assert.equal(f("metric").disabled, false);
    assert.deepEqual([...f("metric").options].map(option => option.value), ["all", "external", "internal", "swap"]);
    select("metric", "internal");
    select("view", "sqrt_scaling");
    const cells = [...visible(".summary-row")[0].querySelectorAll("[data-regret]")].filter(cell => !cell.hidden);
    assert.equal(cells.length, 1); assert.equal(cells[0].dataset.metric, "sqrt_scaling_internal");

    f("compare-actions").checked = true;
    f("compare-actions").dispatchEvent(new w.Event("change"));
    assert.deepEqual([...f("action").options].map(option => [option.value, option.textContent]), [["all", "All actions"]]);
    assert.equal(f("action").disabled, true);
    assert.equal(f("metric").value, "internal"); assert.equal(f("metric").disabled, false);
    assert.deepEqual([...f("metric").options].map(option => option.value), ["all", "external", "internal", "swap"]);
    assert.equal(f("profiles").multiple, false); assert.equal(f("profiles").selectedOptions.length, 1);
    assert.deepEqual(visible(".summary-row").map(row => row.dataset.action).sort(), ["2", "3", "4"]);

    select("feedback", "both");
    assert.deepEqual([...f("profiles").options].map(option => [option.value, option.textContent]), [
        ["exp3_ix", "EXP3-IX"], ["hedge", "Hedge"]]);
    f("profiles").value = "hedge";
    f("profiles").dispatchEvent(new w.Event("change", {bubbles: true}));
    assert.deepEqual(new Set(visible(".summary-row").map(row => row.dataset.feedback)), new Set(["full_information"]));
    assert.deepEqual(visible(".summary-row").map(row => row.dataset.action).sort(), ["2", "4"]);

    f("compare-regrets").checked = true;
    f("compare-regrets").dispatchEvent(new w.Event("change"));
    assert.equal(f("action").value, "3"); assert.equal(f("action").disabled, false);
    assert(![...f("action").options].some(option => option.value === "all"));
    assert.equal(f("metric").value, "all"); assert.equal(f("metric").disabled, true);
    for (const id of ["filter-horizon", "filter-seed", "filter-player-algorithm", "filter-secondary"]) {
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
