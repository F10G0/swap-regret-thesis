from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("override", ["", "/adversarial/action-scaling"])
def test_queue_experiment_submits_in_place_and_prevents_double_clicks(override) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    path = Path(__file__).parents[2] / "web" / "static" / "dashboard.js"
    script = r'''
const assert = require("assert").strict;
const source = require("fs").readFileSync(process.argv[1], "utf8");
const start = source.indexOf("async function queueExperiment");
const end = source.indexOf("\nfunction formFields", start);
const status = {};
const buttons = [{disabled: false}, {disabled: false}];
const form = {dataset: {}, action: "/", querySelectorAll: () => buttons};
const panel = {hidden: true, querySelector: () => ({insertAdjacentHTML: (position, html) => {
    assert.equal(position, "afterbegin"); assert.equal(html, "<li>queued</li>");
}})};
let polled = 0;
let saved = 0;
let calls = 0;
let complete;
let submitted;
global.dashboardData = {jobs: []};
global.element = () => status;
global.document = {querySelector: () => panel};
global.FormData = class {constructor() {return [["_csrf_token", "token"], ["algorithm_names", "hedge"], ["algorithm_names", "bm"]];}};
global.saveFormState = () => saved++;
global.setBusy = () => {};
global.pollActiveJobs = () => polled++;
global.fetch = (url, options) => {
    calls++; submitted = options;
    assert.equal(url, process.argv[2] || "/");
    return new Promise((resolve) => complete = resolve);
};
global.window = {location: {reload() {throw new Error("Unexpected navigation");}}};
let prevented = 0;
const event = {currentTarget: form, submitter: {getAttribute: () => process.argv[2]}, preventDefault: () => prevented++};
eval(source.slice(start, end));
(async () => {
    const first = queueExperiment(event);
    assert.equal(buttons.every((button) => button.disabled), true);
    await queueExperiment(event);
    assert.equal(calls, 1);
    assert.equal(prevented, 2);
    complete({ok: true, json: async () => ({job: {id: "new", url: "/jobs/new"}, job_html: "<li>queued</li>", message: "Queued"})});
    await first;
    assert.equal(submitted.method, "POST");
    assert.equal(submitted.headers.Accept, "application/json");
    assert.equal(submitted.body.get("_csrf_token"), "token");
    assert.deepEqual(submitted.body.getAll("algorithm_names"), ["hedge", "bm"]);
    assert.equal(dashboardData.jobs[0].id, "new");
    assert.equal(panel.hidden, false);
    assert.equal(polled, 1);
    assert.equal(saved, 1);
    assert.equal(status.textContent, "Queued");
    assert.equal(buttons.some((button) => button.disabled), false);
    global.fetch = async () => ({ok: false, json: async () => ({error: "already queued"})});
    await queueExperiment(event);
    assert.equal(status.textContent, "already queued");
    assert.equal(dashboardData.jobs.length, 1);
    assert.equal(form.dataset.submitting, "false");
    global.fetch = async () => {throw new Error("Network unavailable");};
    await queueExperiment(event);
    assert.equal(status.textContent, "Network unavailable");
    assert.equal(buttons.some((button) => button.disabled), false);
})().catch((error) => {console.error(error); process.exit(1);});
'''
    result = subprocess.run([node, "-e", script, path, override], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_job_polling_does_not_overlap_and_keeps_new_jobs_running() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    path = Path(__file__).parents[2] / "web" / "static" / "dashboard.js"
    script = r'''
const assert = require("assert").strict;
const source = require("fs").readFileSync(process.argv[1], "utf8");
const start = source.indexOf("async function pollActiveJobs");
const end = source.indexOf("\nfunction updateJob", start);
let jobPollInFlight = false;
let jobPollTimer = null;
let calls = 0;
let complete;
let timers = 0;
let busy = false;
const notice = {hidden: true};
global.dashboardData = {jobs: [{id: "old", status: "running", url: "/jobs/old"}]};
global.element = () => notice;
global.setBusy = (value) => busy = value;
global.updateJob = (job) => Object.assign(dashboardData.jobs.find((stored) => stored.id === job.id), job);
global.fetch = () => {calls++; return new Promise((resolve) => complete = resolve);};
global.window = {clearTimeout() {}, setTimeout() {timers++; return timers;}, location: {reload() {throw new Error("Unexpected reload");}}};
eval(source.slice(start, end));
(async () => {
    const first = pollActiveJobs();
    dashboardData.jobs.push({id: "new", status: "queued", url: "/jobs/new"});
    await pollActiveJobs();
    assert.equal(calls, 1);
    complete({ok: true, json: async () => ({id: "old", status: "succeeded"})});
    await first;
    assert.equal(busy, true);
    assert.equal(timers, 1);
    assert.equal(jobPollInFlight, false);
    assert.equal(notice.hidden, false);
    global.fetch = async (url) => {
        assert.equal(url, "/jobs/new");
        return {ok: true, json: async () => ({id: "new", status: "succeeded"})};
    };
    await pollActiveJobs();
    assert.equal(busy, false);
    assert.equal(timers, 1);
})().catch((error) => {console.error(error); process.exit(1);});
'''
    result = subprocess.run([node, "-e", script, path], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("filename", ["common.js", "dashboard.js", "custom_games.js", "experimental_trajectory.js", "figure_builder.js"])
def test_web_javascript_parses(filename: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    path = Path(__file__).parents[2] / "web" / "static" / filename
    result = subprocess.run([node, "--check", path], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_dashboard_filter_matching() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    path = Path(__file__).parents[2] / "web" / "static" / "dashboard.js"
    script = r'''const assert = require("assert").strict;
const source = require("fs").readFileSync(process.argv[1], "utf8");
const start = source.indexOf("function matchesResultFilters");
const end = source.indexOf("\n}\n", start) + 3;
eval(source.slice(start, end));
const record = {dataset: {scope: "rps", feedback: "bandit", player: "0", profile: "auer_exp3_vs_bm"}};
const state = {scope: "rps", feedback: "bandit", player: "0", profiles: ["auer_exp3_vs_bm"]};
assert(matchesResultFilters(record, state));
for (const change of [{scope: "rpsls"}, {feedback: "full_information"}, {player: "1"},
                      {profiles: []}, {profiles: ["bm_vs_auer_exp3"]}]) {
    assert(!matchesResultFilters(record, {...state, ...change}));
}
assert(!matchesResultFilters(record, null));'''
    result = subprocess.run([node, "-e", script, path], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_game_analysis_follows_the_global_game_filter() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    path = Path(__file__).parents[2] / "web" / "static" / "dashboard.js"
    script = r'''const source = require("fs").readFileSync(process.argv[1], "utf8");
const extract = (name) => {
    const start = source.indexOf(`function ${name}`);
    const body = source.indexOf("{", start);
    let depth = 0;
    for (let index = body; index < source.length; index += 1) {
        if (source[index] === "{") depth += 1;
        if (source[index] === "}") depth -= 1;
        if (depth === 0) return source.slice(start, index + 1);
    }
};
const elements = {
    "filter-scope": {value: "all"},
    "equilibrium-panel": {hidden: false, open: false},
    "equilibrium-game": {textContent: ""},
    "equilibrium-grid": {hidden: true},
    "equilibrium-explanation": {hidden: true},
    "equilibrium-unavailable": {hidden: false},
};
global.element = (id) => elements[id] || null;
global.dashboardData = {
    equilibriumFigures: {rps: {ce: {}, cce: {}}},
    gamePresentations: {rps: {label: "Rock–Paper–Scissors", description: "RPS"}},
};
global.setHeatmapSource = () => {};
eval(extract("gamePresentation"));
eval(extract("selectedResultScope"));
eval(extract("updateEquilibriumFigures"));
updateEquilibriumFigures();
if (!elements["equilibrium-panel"].hidden) process.exit(1);
elements["filter-scope"].value = "rps";
updateEquilibriumFigures();
if (elements["equilibrium-panel"].hidden) process.exit(2);
if (elements["equilibrium-game"].textContent !== "Rock–Paper–Scissors") process.exit(3);
if (elements["equilibrium-grid"].hidden || elements["equilibrium-explanation"].hidden) process.exit(4);
elements["filter-scope"].value = "all";
updateEquilibriumFigures();
if (!elements["equilibrium-panel"].hidden) process.exit(5);'''
    result = subprocess.run([node, "-e", script, path], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("terminal_status", ["succeeded", "failed", "cancelled"])
def test_completed_job_offers_refresh_without_reloading_dashboard(terminal_status) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    path = Path(__file__).parents[2] / "web" / "static" / "dashboard.js"
    script = r'''const source = require("fs").readFileSync(process.argv[1], "utf8");
const start = source.indexOf("async function pollActiveJobs");
const end = source.indexOf("\nfunction updateJob", start);
let reloaded = false;
let jobPollInFlight = false;
let jobPollTimer = null;
const notice = {hidden: true};
let busy = true;
global.dashboardData = {jobs: [{status: "running", url: "/jobs/1"}]};
global.element = () => notice;
global.setBusy = (value) => busy = value;
global.updateJob = (job) => Object.assign(dashboardData.jobs[0], job);
global.fetch = async () => ({ok: true, json: async () => ({status: process.argv[2]})});
global.window = {location: {reload: () => reloaded = true}, clearTimeout() {}, setTimeout: () => {throw new Error("All jobs finished");}};
eval(source.slice(start, end));
pollActiveJobs().then(() => {
    if (notice.hidden || reloaded || busy || jobPollInFlight) process.exit(1);
});'''
    result = subprocess.run([node, "-e", script, path, terminal_status], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
