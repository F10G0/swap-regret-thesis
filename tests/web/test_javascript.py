from pathlib import Path
import shutil
import subprocess

import pytest


def test_filtered_pdf_download_uses_visible_order_and_active_ci_links() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    path = Path(__file__).parents[2] / "web" / "static" / "dashboard.js"
    script = r'''
const assert = require("assert").strict;
const source = require("fs").readFileSync(process.argv[1], "utf8");
const start = source.indexOf("function visibleFigureDownloads");
const end = source.indexOf("\nfunction selectAvailableFigureSource", start);
const card = (filename, hidden = false) => ({hidden, filename, querySelector() {
    return {getAttribute: () => this.filename};
}});
const cards = [card("third_without_ci.pdf"), card("hidden.pdf", true), card("first.pdf")];
const button = {dataset: {}, disabled: false};
const status = {hidden: true, textContent: ""};
global.element = (id) => id === "download-filtered-figures" ? button : status;
let clicked = false;
global.document = {
    querySelectorAll: () => cards,
    createElement: () => ({click() {clicked = true;}, remove() {}}),
    body: {append() {}},
};
global.FormData = class {constructor() {return [["mode", "fixed"], ["_csrf_token", "token"]];}};
global.window = {setTimeout() {}};
URL.createObjectURL = () => "blob:export";
let sent;
global.fetch = async (url, options) => {
    sent = options.body;
    assert.equal(url, "/figures/download-filtered.pdf");
    assert.equal(options.method, "POST");
    assert.equal(button.disabled, true);
    return {ok: true, blob: async () => ({})};
};
eval(source.slice(start, end));
const event = {preventDefault() {}, currentTarget: {action: "/figures/download-filtered.pdf"}};
(async () => {
    assert.deepEqual(visibleFigureDownloads(), ["third_without_ci.pdf", "first.pdf"]);
    await downloadFilteredFigures(event);
    assert.deepEqual(sent.getAll("filenames"), ["third_without_ci.pdf", "first.pdf"]);
    assert.equal(sent.get("_csrf_token"), "token");
    assert.equal(sent.get("mode"), "fixed");
    assert.equal(clicked, true);
    assert.equal(button.disabled, false);
    // Changing the displayed variant or order is reflected immediately.
    cards[0].filename = "third.pdf";
    cards.reverse();
    assert.deepEqual(visibleFigureDownloads(), ["first.pdf", "third.pdf"]);
    global.fetch = async () => ({ok: false, json: async () => ({error: "Figure missing"})});
    await downloadFilteredFigures(event);
    assert.equal(status.textContent, "Figure missing");
    assert.equal(button.disabled, false);
    cards.forEach((card) => card.hidden = true);
    global.fetch = () => {throw new Error("Empty selections must not be submitted");};
    await downloadFilteredFigures(event);
})().catch((error) => {console.error(error); process.exit(1);});
'''
    result = subprocess.run([node, "-e", script, path], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_filtered_pdf_button_tracks_empty_filters_and_active_download() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    path = Path(__file__).parents[2] / "web" / "static" / "dashboard.js"
    script = r'''
const assert = require("assert").strict;
const source = require("fs").readFileSync(process.argv[1], "utf8");
const start = source.indexOf("function applyFilters");
const end = source.indexOf("\nfunction visibleFigureDownloads", start);
const button = {dataset: {}};
const counter = {};
let matches = false;
global.element = (id) => ({"download-filtered-figures": button, "figure-counter": counter}[id]);
global.document = {querySelectorAll: (selector) => selector === "#figure-grid .figure-card" ? [{}] : []};
global.matchesFilters = () => matches;
global.updateSummarySourceColumns = global.updateSummaryRows = () => {};
eval(source.slice(start, end));
applyFilters();
assert.equal(button.disabled, true);
assert.equal(counter.textContent, "0 figures");
matches = true;
applyFilters();
assert.equal(button.disabled, false);
button.dataset.exporting = "true";
applyFilters();
assert.equal(button.disabled, true);
'''
    result = subprocess.run([node, "-e", script, path], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("filename", ["common.js", "dashboard.js", "custom_games.js", "experimental_trajectory.js"])
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
    script = r'''const source = require("fs").readFileSync(process.argv[1], "utf8");
const start = source.indexOf("function matchesFilters");
const end = source.indexOf("\n}\n", start) + 3;
const controls = {
    result: [{value: "rps", dataset: {resultFilter: "scope"}, hasAttribute: () => false}],
    tokens: [{value: "exp3_ix", dataset: {summaryFilter: "algorithms"}, hasAttribute: () => true}],
};
global.document = {querySelectorAll: (selector) => controls[selector]};
eval(source.slice(start, end));
const record = {dataset: {scope: "rps", algorithms: "hedge exp3_ix"}};
if (!matchesFilters(record, "result", "resultFilter")) process.exit(1);
controls.result[0].value = "matching_pennies";
if (matchesFilters(record, "result", "resultFilter")) process.exit(2);
if (!matchesFilters(record, "tokens", "summaryFilter")) process.exit(3);'''
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


def test_dashboard_filters_apply_immediately() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    path = Path(__file__).parents[2] / "web" / "static" / "dashboard.js"
    script = r'''const source = require("fs").readFileSync(process.argv[1], "utf8");
const start = source.indexOf("function installFilterPersistence");
const end = source.indexOf("\n}\n", start) + 3;
const listeners = {};
global.resultFilterControls = () => [
    {dataset: {}, matches: (selector) => selector === "input", addEventListener: (name, handler) => listeners.input = handler},
    {dataset: {}, matches: () => false, addEventListener: (name, handler) => listeners.change = handler},
];
global.saveFilterState = () => listeners.saved = true;
global.applyFilters = () => listeners.applied = true;
eval(source.slice(start, end));
installFilterPersistence();
if (!listeners.input || !listeners.change) process.exit(1);
listeners.input();
if (!listeners.saved || !listeners.applied) process.exit(2);
listeners.saved = false;
listeners.applied = false;
listeners.change();
if (!listeners.saved || !listeners.applied) process.exit(3);'''
    result = subprocess.run([node, "-e", script, path], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_completed_job_refreshes_dashboard() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    path = Path(__file__).parents[2] / "web" / "static" / "dashboard.js"
    script = r'''const source = require("fs").readFileSync(process.argv[1], "utf8");
const start = source.indexOf("async function pollActiveJobs");
const end = source.indexOf("\nfunction updateJob", start);
let reloaded = false;
let saved = false;
global.dashboardData = {jobs: [{status: "running", url: "/jobs/1"}]};
global.setBusy = () => {};
global.updateJob = () => {};
global.saveFormState = () => saved = true;
global.fetch = async () => ({ok: true, json: async () => ({status: "succeeded"})});
global.window = {location: {reload: () => reloaded = true}, setTimeout: () => {}};
eval(source.slice(start, end));
pollActiveJobs().then(() => {
    if (!saved || !reloaded) process.exit(1);
});'''
    result = subprocess.run([node, "-e", script, path], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
