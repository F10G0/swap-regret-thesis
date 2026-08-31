from pathlib import Path
import shutil
import subprocess

import pytest


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
    tokens: [{value: "exp3", dataset: {summaryFilter: "algorithms"}, hasAttribute: () => true}],
};
global.document = {querySelectorAll: (selector) => controls[selector]};
eval(source.slice(start, end));
const record = {dataset: {scope: "rps", algorithms: "hedge exp3"}};
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
