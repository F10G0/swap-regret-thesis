from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from config import FIGURE_DIR, RAW_DIR, RESULTS_DIR
from experiments.algorithm_labels import algorithm_profile_label
from experiments.games import PAYOFF_FACTORIES
from experiments.plots.plot_equilibrium_convergence import (
    plot_result_equilibrium_distance,
    plot_result_equilibrium_trajectory,
)
from experiments.plots.plot_joint_actions import plot_joint_actions
from web.equilibrium_figures import PRECOMPUTED_EQUILIBRIUM_DIR
from web.presentations import GAME_PRESENTATIONS
from web.result_groups import aggregate_result_summaries, result_group_filenames
from web.result_index import ResultIndex, SUMMARY_REGRET_FIELDS


AVERAGE_RE = re.compile(
    r"^(.+)_average_(expected|realized)_"
    r"(external|internal|swap)_regret_player_(\d+)\.png$"
)
SCALING_RE = re.compile(
    r"^(.+)_(expected|realized)_"
    r"(external|internal|swap)_regret_over_sqrt_t_player_(\d+)\.png$"
)
EQUILIBRIUM_RE = re.compile(
    r"^(.+)_(ce|cce)_blue_lower_origin_maximum_profile_weight\.png$"
)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Swap Regret Experiment Report</title>
<style>
:root {
    --accent: #16845b;
    --bg: #f5f7f6;
    --card: #ffffff;
    --soft: #eef2f0;
    --text: #17201c;
    --muted: #66736c;
    --line: #d9e1dd;
}
:root[data-theme="blue"] { --accent: #2563eb; }
:root[data-theme="purple"] { --accent: #7c3aed; }
:root[data-theme="orange"] { --accent: #d97706; }
:root[data-theme="red"] { --accent: #dc2626; }
* { box-sizing: border-box; }
body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 14px/1.45 system-ui, sans-serif;
}
.shell {
    max-width: 1500px;
    margin: 0 auto;
    padding: 24px;
}
.head {
    display: flex;
    justify-content: space-between;
    align-items: start;
    gap: 20px;
    margin-bottom: 18px;
}
h1 { margin: 0; }
.muted { color: var(--muted); }
.toolbar,
.panel {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
}
.toolbar {
    display: grid;
    grid-template-columns: repeat(5, minmax(130px, 1fr));
    gap: 10px;
    padding: 14px;
    margin-bottom: 16px;
}
.field {
    display: grid;
    gap: 5px;
}
.field label {
    font-size: 12px;
    font-weight: 700;
    color: var(--muted);
}
select {
    padding: 8px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--soft);
    color: var(--text);
}
.panel {
    margin-bottom: 16px;
    overflow: hidden;
}
.panel-head {
    display: flex;
    justify-content: space-between;
    padding: 14px 16px;
    border-bottom: 1px solid var(--line);
}
.tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 12px 16px 0;
}
.tab {
    border: 1px solid var(--line);
    border-radius: 999px;
    background: var(--soft);
    color: var(--text);
    padding: 7px 11px;
    cursor: pointer;
}
.tab.active {
    background: var(--accent);
    border-color: var(--accent);
    color: #ffffff;
}
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 14px;
    padding: 16px;
}
.card {
    border: 1px solid var(--line);
    border-radius: 10px;
    overflow: hidden;
    background: var(--soft);
}
.card img {
    display: block;
    width: 100%;
    max-height: 620px;
    object-fit: contain;
    background: #ffffff;
}
.meta { padding: 10px; }
.meta small {
    display: block;
    color: var(--muted);
    margin-top: 3px;
}
.table { overflow: auto; }
table {
    width: 100%;
    border-collapse: collapse;
    min-width: 1100px;
}
th,
td {
    padding: 9px 10px;
    border-bottom: 1px solid var(--line);
    text-align: left;
    white-space: nowrap;
}
th {
    background: var(--soft);
    font-size: 11px;
    text-transform: uppercase;
}
#summary tr {
    cursor: pointer;
}
#summary tr:hover {
    background: var(--soft);
}
#summary tr.selected {
    background: var(--soft);
    box-shadow: inset 3px 0 0 var(--accent);
}
.summary-detail {
    border-top: 1px solid var(--line);
}
.summary-detail-head {
    padding: 14px 16px 0;
}
.trajectory-toggle {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-top: 8px;
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
}
.trajectory-toggle input {
    margin: 0;
}
.empty,
.warning { padding: 18px; }
.warning {
    margin-bottom: 16px;
    border-radius: 10px;
    background: #fff4cc;
    color: #5e4a00;
}
[hidden] { display: none !important; }
@media (max-width: 850px) {
    .toolbar { grid-template-columns: repeat(2, 1fr); }
    .head { flex-direction: column; }
}
@media (max-width: 520px) {
    .toolbar { grid-template-columns: 1fr; }
    .shell { padding: 12px; }
    .grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="shell">
    <div class="head">
        <div>
            <h1>Swap Regret Experiment Report</h1>
            <div class="muted">Static interactive snapshot — no backend required.</div>
        </div>
        <div class="field">
            <label for="theme">Accent</label>
            <select id="theme">
                <option value="green">Green</option>
                <option value="blue">Blue</option>
                <option value="purple">Purple</option>
                <option value="orange">Orange</option>
                <option value="red">Red</option>
            </select>
        </div>
    </div>

    <div id="warning" class="warning" hidden></div>

    <div class="toolbar">
        <div class="field">
            <label for="game">Game</label>
            <select id="game"></select>
        </div>
        <div class="field regret-only">
            <label for="source">Source</label>
            <select id="source">
                <option value="all">All</option>
                <option value="expected">Expected</option>
                <option value="realized">Realized</option>
            </select>
        </div>
        <div class="field regret-only">
            <label for="regret">Regret</label>
            <select id="regret">
                <option value="all">All</option>
                <option value="external">External</option>
                <option value="internal">Internal</option>
                <option value="swap">Swap</option>
            </select>
        </div>
        <div class="field regret-only">
            <label for="view">View</label>
            <select id="view">
                <option value="all">All</option>
                <option value="average">Average</option>
                <option value="sqrt_scaling">Regret / sqrt(t)</option>
            </select>
        </div>
        <div class="field regret-only">
            <label for="player">Player</label>
            <select id="player">
                <option value="all">All</option>
            </select>
        </div>
    </div>

    <section class="panel">
        <div class="panel-head">
            <b>Figures</b>
            <span id="figure-count" class="muted"></span>
        </div>
        <div class="tabs">
            <button class="tab active" data-kind="regret" type="button">Regret</button>
            <button class="tab" data-kind="detail" type="button">Result details</button>
            <button class="tab" data-kind="equilibrium" type="button">Theoretical CE / CCE</button>
        </div>
        <div id="figures" class="grid"></div>
        <div id="figure-empty" class="empty" hidden>No matching figures.</div>
    </section>

    <section class="panel">
        <div class="panel-head">
            <b>Final regret summary</b>
            <span id="summary-count" class="muted"></span>
        </div>
        <div class="table">
            <table>
                <thead>
                    <tr>
                        <th>Game</th>
                        <th>Feedback</th>
                        <th>Eval.</th>
                        <th>Algorithms</th>
                        <th>Horizon</th>
                        <th>Seed</th>
                        <th>Player</th>
                        <th>Replicates</th>
                        <th>Exp ext</th>
                        <th>Exp int</th>
                        <th>Exp swap</th>
                        <th>Real ext</th>
                        <th>Real int</th>
                        <th>Real swap</th>
                    </tr>
                </thead>
                <tbody id="summary"></tbody>
            </table>
        </div>
        <div id="summary-empty" class="empty" hidden>No matching summaries.</div>
        <div id="summary-detail" class="summary-detail" hidden>
            <div class="summary-detail-head">
                <b id="summary-detail-title">Selected replicate group</b>
                <div class="muted">Saved group-level empirical and equilibrium analysis.</div>
            </div>
            <div id="summary-detail-grid" class="grid"></div>
            <div id="summary-detail-empty" class="empty" hidden>No saved detail figures for this group.</div>
        </div>
    </section>
</div>

<script id="report-data" type="application/json">__REPORT_DATA__</script>
<script>
"use strict";

const reportData = JSON.parse(document.getElementById("report-data").textContent);
const byId = (id) => document.getElementById(id);
const storageKey = "swap-regret-static-report";
const metricFields = [
    "average_expected_external_regret",
    "average_expected_internal_regret",
    "average_expected_swap_regret",
    "average_realized_external_regret",
    "average_realized_internal_regret",
    "average_realized_swap_regret",
];
let selectedKind = "regret";

function loadState() {
    try {
        return JSON.parse(window.localStorage.getItem(storageKey)) || {};
    } catch (_error) {
        return {};
    }
}

function saveState() {
    try {
        window.localStorage.setItem(storageKey, JSON.stringify({
            game: byId("game").value,
            source: byId("source").value,
            regret: byId("regret").value,
            view: byId("view").value,
            player: byId("player").value,
            theme: byId("theme").value,
            kind: selectedKind,
        }));
    } catch (_error) {
        // localStorage may be unavailable for local files.
    }
}

function createOption(value, text) {
    const option = document.createElement("option");
    option.value = String(value);
    option.textContent = text;
    return option;
}

function restoreSelect(id, value) {
    const select = byId(id);
    if (value === undefined) {
        return;
    }
    if ([...select.options].some((option) => option.value === String(value))) {
        select.value = String(value);
    }
}

function gameLabel(game) {
    return reportData.labels[game] || game;
}

function figureMatchesFilters(figure) {
    if (figure.kind !== selectedKind) {
        return false;
    }

    const game = byId("game").value;
    if (game !== "all" && figure.game && figure.game !== game) {
        return false;
    }

    if (selectedKind !== "regret") {
        return true;
    }

    return (
        (byId("source").value === "all" || figure.source === byId("source").value)
        && (byId("regret").value === "all" || figure.regret === byId("regret").value)
        && (byId("view").value === "all" || figure.view === byId("view").value)
        && (byId("player").value === "all" || String(figure.player) === byId("player").value)
    );
}

function renderFigures() {
    const container = byId("figures");
    container.replaceChildren();

    const figures = reportData.figures.filter(figureMatchesFilters);
    for (const figure of figures) {
        const card = document.createElement("article");
        const link = document.createElement("a");
        const image = document.createElement("img");
        const meta = document.createElement("div");
        const title = document.createElement("b");
        const subtitle = document.createElement("small");

        card.className = "card";
        meta.className = "meta";
        link.href = figure.path;
        link.target = "_blank";
        link.rel = "noopener";
        image.src = figure.path;
        image.alt = figure.title;
        image.loading = "lazy";
        title.textContent = figure.title;
        subtitle.textContent = [
            figure.game ? gameLabel(figure.game) : "",
            figure.profile || "",
        ].filter(Boolean).join(" · ");

        link.append(image);
        meta.append(title, subtitle);
        card.append(link, meta);
        container.append(card);
    }

    byId("figure-count").textContent = `${figures.length} figure${figures.length === 1 ? "" : "s"}`;
    byId("figure-empty").hidden = figures.length !== 0;

    document.querySelectorAll(".regret-only").forEach((element) => {
        element.hidden = selectedKind !== "regret";
    });
}

function makeCell(value) {
    const cell = document.createElement("td");
    cell.textContent = value;
    return cell;
}

function metricText(row, field) {
    if (!(field in row)) {
        return "—";
    }

    const confidence = row.confidence_intervals?.[field] || 0;
    const value = Number(row[field]).toPrecision(4);
    if (confidence <= 0) {
        return value;
    }
    return `${value} ± ${Number(confidence).toPrecision(2)}`;
}

function renderSummaryDetail(row, selectedRow) {
    document.querySelectorAll("#summary tr").forEach((item) => {
        item.classList.toggle("selected", item === selectedRow);
    });

    const panel = byId("summary-detail");
    const grid = byId("summary-detail-grid");
    grid.replaceChildren();

    const details = row.details || {};
    const figureSpecs = [
        ["joint_actions", "Mean empirical joint-action distribution"],
        ["distance", "Mean CE / CCE distance"],
    ];

    let availableCount = 0;

    for (const [key, titleText] of figureSpecs) {
        if (!details[key]) {
            continue;
        }

        availableCount += 1;

        const card = document.createElement("article");
        const link = document.createElement("a");
        const image = document.createElement("img");
        const meta = document.createElement("div");
        const title = document.createElement("b");
        const subtitle = document.createElement("small");

        card.className = "card";
        meta.className = "meta";
        link.href = details[key];
        link.target = "_blank";
        link.rel = "noopener";
        image.src = details[key];
        image.alt = titleText;
        image.loading = "lazy";
        title.textContent = titleText;
        subtitle.textContent = `${gameLabel(row.game)} · ${row.profile}`;

        link.append(image);
        meta.append(title, subtitle);
        card.append(link, meta);
        grid.append(card);
    }

    const trajectories = details.trajectories || {};
    const defaultTrajectory = trajectories.show || trajectories.hide;
    if (defaultTrajectory) {
        availableCount += 1;

        const card = document.createElement("article");
        const link = document.createElement("a");
        const image = document.createElement("img");
        const meta = document.createElement("div");
        const title = document.createElement("b");
        const subtitle = document.createElement("small");
        const toggle = document.createElement("label");
        const checkbox = document.createElement("input");
        const toggleText = document.createElement("span");

        card.className = "card";
        meta.className = "meta";
        toggle.className = "trajectory-toggle";
        checkbox.type = "checkbox";
        checkbox.disabled = !(trajectories.show && trajectories.hide);
        toggleText.textContent = "Hide first";

        const setTrajectory = () => {
            const path = checkbox.checked
                ? (trajectories.hide || trajectories.show)
                : (trajectories.show || trajectories.hide);
            link.href = path;
            image.src = path;
        };

        image.alt = "Mean projected joint-distribution trajectory";
        image.loading = "lazy";
        title.textContent = "Mean projected joint-distribution trajectory";
        subtitle.textContent = `${gameLabel(row.game)} · ${row.profile} · 10 nodes`;

        checkbox.addEventListener("change", setTrajectory);
        setTrajectory();

        toggle.append(checkbox, toggleText);
        link.target = "_blank";
        link.rel = "noopener";
        link.append(image);
        meta.append(title, subtitle, toggle);
        card.append(link, meta);
        grid.append(card);
    }

    byId("summary-detail-title").textContent =
        `${gameLabel(row.game)} · ${row.profile} · horizon ${row.horizon} · seed ${row.seed}`;
    byId("summary-detail-empty").hidden = availableCount !== 0;
    panel.hidden = false;
}

function renderSummary() {
    const game = byId("game").value;
    const rows = reportData.summaries.filter((row) => game === "all" || row.game === game);
    const body = byId("summary");
    body.replaceChildren();
    byId("summary-detail").hidden = true;

    for (const row of rows) {
        const tr = document.createElement("tr");
        const values = [
            gameLabel(row.game),
            row.feedback_mode.replaceAll("_", " "),
            row.regret_evaluation,
            row.profile,
            row.horizon,
            row.seed,
            row.player,
            row.replicate_label,
            ...metricFields.map((field) => metricText(row, field)),
        ];
        values.forEach((value) => tr.append(makeCell(value)));
        tr.addEventListener("click", () => renderSummaryDetail(row, tr));
        body.append(tr);
    }

    byId("summary-count").textContent = `${rows.length} row${rows.length === 1 ? "" : "s"}`;
    byId("summary-empty").hidden = rows.length !== 0;
}

function render() {
    renderFigures();
    renderSummary();
    saveState();
}

function setup() {
    const saved = loadState();

    byId("game").append(createOption("all", "All games"));
    reportData.games.forEach((game) => {
        byId("game").append(createOption(game, gameLabel(game)));
    });

    const players = [...new Set(
        reportData.figures
            .filter((figure) => figure.kind === "regret")
            .map((figure) => figure.player)
    )].sort((a, b) => a - b);

    players.forEach((player) => {
        byId("player").append(createOption(player, `Player ${player}`));
    });

    restoreSelect("game", saved.game);
    restoreSelect("source", saved.source);
    restoreSelect("regret", saved.regret);
    restoreSelect("view", saved.view);
    restoreSelect("player", saved.player);
    restoreSelect("theme", saved.theme);

    if (["regret", "detail", "equilibrium"].includes(saved.kind)) {
        selectedKind = saved.kind;
    }

    document.documentElement.dataset.theme = byId("theme").value;

    ["game", "source", "regret", "view", "player"].forEach((id) => {
        byId(id).addEventListener("change", render);
    });

    byId("theme").addEventListener("change", () => {
        document.documentElement.dataset.theme = byId("theme").value;
        saveState();
    });

    document.querySelectorAll(".tab").forEach((button) => {
        button.classList.toggle("active", button.dataset.kind === selectedKind);
        button.addEventListener("click", () => {
            selectedKind = button.dataset.kind;
            document.querySelectorAll(".tab").forEach((item) => {
                item.classList.toggle("active", item === button);
            });
            render();
        });
    });
}

setup();
if (reportData.warnings.length) {
    byId("warning").hidden = false;
    byId("warning").textContent = `Warnings: ${reportData.warnings.join(" | ")}`;
}
render();
</script>
</body>
</html>
"""


def _relative_path(path: Path, results_dir: Path) -> str:
    return Path(os.path.relpath(path, results_dir)).as_posix()


def _result_prefix_metadata(summaries: list[dict], aggregated: list[dict]) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}

    for row in summaries:
        metadata[Path(row["experiment"]).stem] = {
            "game": row["game"],
            "profile": algorithm_profile_label(row["algorithm_profile"]),
        }

    for row in aggregated:
        metadata[row["group_id"]] = {
            "game": row["game"],
            "profile": algorithm_profile_label(row["algorithm_profile"]),
        }

    return metadata


def _detail_metadata(stem: str, prefixes: dict[str, dict[str, str]]) -> dict[str, str]:
    for prefix in sorted(prefixes, key=len, reverse=True):
        if stem.startswith(prefix):
            return prefixes[prefix]
    return {}


def _detail_title(stem: str) -> str:
    if "joint_actions" in stem:
        return "Empirical joint-action distribution"
    if "equilibrium_distance" in stem:
        return "CE / CCE distance"
    if "equilibrium_trajectory" in stem:
        return "Projected equilibrium trajectory"
    return stem.replace("_", " ")


def _collect_generated_figures(
    figure_dir: Path,
    results_dir: Path,
    prefixes: dict[str, dict[str, str]],
) -> list[dict]:
    if not figure_dir.exists():
        return []

    report_dir = figure_dir / "report"
    figures: list[dict] = []

    for path in sorted(figure_dir.rglob("*.png")):
        if report_dir in path.parents:
            continue

        average_match = AVERAGE_RE.fullmatch(path.name)
        scaling_match = SCALING_RE.fullmatch(path.name)
        regret_match = average_match or scaling_match

        if regret_match is not None:
            source = regret_match.group(2)
            regret = regret_match.group(3)
            player = int(regret_match.group(4))
            view = "average" if average_match is not None else "sqrt_scaling"
            figures.append(
                {
                    "kind": "regret",
                    "path": _relative_path(path, results_dir),
                    "game": regret_match.group(1),
                    "source": source,
                    "regret": regret,
                    "player": player,
                    "view": view,
                    "title": (
                        f"{source.title()} {regret} regret · Player {player} · "
                        f"{'average' if view == 'average' else 'regret / sqrt(t)'}"
                    ),
                }
            )
            continue

        metadata = _detail_metadata(path.stem, prefixes)
        figures.append(
            {
                "kind": "detail",
                "path": _relative_path(path, results_dir),
                "game": metadata.get("game"),
                "profile": metadata.get("profile", ""),
                "title": _detail_title(path.stem),
            }
        )

    return figures


def _copy_equilibrium_figures(results_dir: Path, source_dir: Path) -> list[dict]:
    target_dir = results_dir / "figures" / "report" / "equilibria"
    target_dir.mkdir(parents=True, exist_ok=True)

    for path in target_dir.glob("*.png"):
        path.unlink()

    if not source_dir.exists():
        return []

    figures: list[dict] = []
    for source in sorted(source_dir.glob("*.png")):
        match = EQUILIBRIUM_RE.fullmatch(source.name)
        if match is None:
            continue

        target = target_dir / source.name
        shutil.copy2(source, target)
        figures.append(
            {
                "kind": "equilibrium",
                "path": _relative_path(target, results_dir),
                "game": match.group(1),
                "title": f"Maximum {match.group(2).upper()} profile weight",
            }
        )

    return figures


def _group_cache_stem(group_id: str, filenames: list[str]) -> str:
    membership = "\n".join(filenames)
    digest = sha256(membership.encode("utf-8")).hexdigest()[:8]
    return f"{group_id}_{digest}"


def _is_current(output_path: Path, input_paths: list[Path]) -> bool:
    if not output_path.is_file():
        return False
    input_mtime = max(path.stat().st_mtime_ns for path in input_paths)
    return output_path.stat().st_mtime_ns >= input_mtime


def _ensure_group_detail_figures(
    snapshot_summaries: list[dict],
    aggregated: list[dict],
    raw_dir: Path,
    figure_dir: Path,
    results_dir: Path,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    detail_dir = figure_dir / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    by_group: dict[str, dict] = {}
    for row in aggregated:
        by_group.setdefault(row["group_id"], row)

    details: dict[str, dict[str, object]] = {}
    warnings: list[str] = []

    for group_id, row in by_group.items():
        filenames = result_group_filenames(snapshot_summaries, group_id)
        input_paths = [raw_dir / filename for filename in filenames]
        if any(not path.is_file() for path in input_paths):
            warnings.append(f"Skipped detail analysis for {group_id}: missing result file")
            continue

        cache_stem = _group_cache_stem(group_id, filenames)
        group_details: dict[str, object] = {}

        if row["game"] in PAYOFF_FACTORIES and row["n_players"] == 2:
            joint_path = detail_dir / f"{cache_stem}_replicate_mean_joint_actions_blue_lower_origin.png"
            try:
                if not _is_current(joint_path, input_paths):
                    with tempfile.TemporaryDirectory(prefix=".report-joint-actions-", dir=detail_dir) as temporary_directory:
                        temporary_path = Path(temporary_directory) / joint_path.name
                        plot_joint_actions(input_paths, temporary_path)
                        os.replace(temporary_path, joint_path)
                group_details["joint_actions"] = _relative_path(joint_path, results_dir)
            except Exception as error:
                warnings.append(f"Skipped joint-action figure for {group_id}: {type(error).__name__}: {error}")

        distance_path = detail_dir / f"{cache_stem}_replicate_mean_equilibrium_distance.png"
        trajectory_show_path = detail_dir / f"{cache_stem}_p10_from_round_1_replicate_mean_equilibrium_trajectory.png"
        trajectory_hide_path = detail_dir / f"{cache_stem}_p10_hide_round_1_replicate_mean_equilibrium_trajectory.png"

        try:
            if not _is_current(distance_path, input_paths):
                with tempfile.TemporaryDirectory(prefix=".report-equilibrium-distance-", dir=detail_dir) as temporary_directory:
                    temporary_path = Path(temporary_directory) / distance_path.name
                    plot_result_equilibrium_distance(input_paths, temporary_path)
                    os.replace(temporary_path, distance_path)
            group_details["distance"] = _relative_path(distance_path, results_dir)
        except Exception as error:
            warnings.append(f"Skipped equilibrium distance for {group_id}: {type(error).__name__}: {error}")
            if _is_current(distance_path, input_paths):
                group_details["distance"] = _relative_path(distance_path, results_dir)

        trajectory_paths: dict[str, str] = {}
        for key, hide_first, output_path in (
            ("show", False, trajectory_show_path),
            ("hide", True, trajectory_hide_path),
        ):
            try:
                if not _is_current(output_path, input_paths):
                    with tempfile.TemporaryDirectory(prefix=f".report-equilibrium-trajectory-{key}-", dir=detail_dir) as temporary_directory:
                        temporary_path = Path(temporary_directory) / output_path.name
                        plot_result_equilibrium_trajectory(
                            input_paths,
                            temporary_path,
                            trajectory_points=10,
                            hide_first=hide_first,
                        )
                        os.replace(temporary_path, output_path)
                trajectory_paths[key] = _relative_path(output_path, results_dir)
            except Exception as error:
                warnings.append(
                    f"Skipped equilibrium trajectory ({key}) for {group_id}: "
                    f"{type(error).__name__}: {error}"
                )
                if _is_current(output_path, input_paths):
                    trajectory_paths[key] = _relative_path(output_path, results_dir)

        if trajectory_paths:
            group_details["trajectories"] = trajectory_paths

        details[group_id] = group_details

    return details, warnings


def _public_summary(row: dict, details_by_group: dict[str, dict[str, object]]) -> dict:
    result = {
        "game": row["game"],
        "feedback_mode": row["feedback_mode"],
        "regret_evaluation": row["regret_evaluation"],
        "profile": algorithm_profile_label(row["algorithm_profile"]),
        "horizon": row["horizon"],
        "seed": row["seed"],
        "player": row["player"],
        "replicate_label": row["replicate_label"],
        "confidence_intervals": row["confidence_intervals"],
        "group_id": row["group_id"],
        "details": details_by_group.get(row["group_id"], {}),
    }

    for field in SUMMARY_REGRET_FIELDS:
        if field in row:
            result[field] = row[field]

    return result


def build_report(
    figure_dir: str | Path = FIGURE_DIR,
    results_dir: str | Path = RESULTS_DIR,
    raw_dir: str | Path = RAW_DIR,
    equilibrium_dir: str | Path = PRECOMPUTED_EQUILIBRIUM_DIR,
) -> Path:
    figure_dir = Path(figure_dir)
    results_dir = Path(results_dir)
    raw_dir = Path(raw_dir)
    equilibrium_dir = Path(equilibrium_dir)

    snapshot = ResultIndex(raw_dir).snapshot()
    aggregated = aggregate_result_summaries(snapshot.summaries)
    prefixes = _result_prefix_metadata(snapshot.summaries, aggregated)
    details_by_group, detail_warnings = _ensure_group_detail_figures(
        snapshot.summaries,
        aggregated,
        raw_dir,
        figure_dir,
        results_dir,
    )

    figures = _collect_generated_figures(figure_dir, results_dir, prefixes)
    figures.extend(_copy_equilibrium_figures(results_dir, equilibrium_dir))

    summaries = [_public_summary(row, details_by_group) for row in aggregated]
    summaries.sort(
        key=lambda row: (
            row["game"],
            row["feedback_mode"],
            row["profile"],
            row["horizon"],
            row["seed"],
            row["player"],
        )
    )

    games: set[str] = {str(row["game"]) for row in summaries}
    for figure in figures:
        game = figure.get("game")
        if isinstance(game, str) and game:
            games.add(game)

    labels: dict[str, str] = {
        game: GAME_PRESENTATIONS.get(game, {}).get(
            "label",
            game.replace("_", " ").title(),
        )
        for game in games
    }

    payload = json.dumps(
        {
            "games": sorted(games, key=lambda game: labels[game]),
            "labels": labels,
            "figures": figures,
            "summaries": summaries,
            "warnings": [*snapshot.warnings, *detail_warnings],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")

    output_path = results_dir / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        HTML_TEMPLATE.replace("__REPORT_DATA__", payload),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    output_path = build_report()
    print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
