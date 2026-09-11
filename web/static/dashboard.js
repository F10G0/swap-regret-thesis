"use strict";

const dashboardDataElement = document.getElementById("dashboard-data");
const dashboardData = dashboardDataElement
    ? JSON.parse(dashboardDataElement.textContent)
    : {mode: "fixed", equilibriumFigures: {}, gameDefinitions: {}, gamePresentations: {}, jobs: [], summaries: [], algorithms: {}, algorithmLabels: {}};
const onePlayerMode = dashboardData.mode === "adversarial";
const formStorageKey = onePlayerMode ? "swap-regret-adversarial-form" : "swap-regret-experiment-form";
let resultFilters = null;
let jobPollInFlight = false;
let jobPollTimer = null;

async function queueExperiment(event) {
    event.preventDefault();
    const form = event.currentTarget;
    if (form.dataset.submitting === "true") return;
    const body = new URLSearchParams(new FormData(form));
    const action = (event.submitter && event.submitter.getAttribute("formaction")) || form.action;
    const buttons = [...form.querySelectorAll('button[type="submit"]')];
    const disabled = buttons.map((button) => button.disabled);
    const status = element("experiment-submit-status");
    form.dataset.submitting = "true";
    buttons.forEach((button) => button.disabled = true);
    status.hidden = false;
    status.className = "notice";
    status.textContent = "Queueing experiment…";
    saveFormState();
    try {
        const response = await fetch(action, {method: "POST", headers: {Accept: "application/json"}, body});
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.job) {
            throw new Error(data.error || "Could not queue the experiment. Refresh the page and try again.");
        }
        dashboardData.jobs.unshift(data.job);
        const panel = document.querySelector(".jobs-panel");
        panel.querySelector(".job-list").insertAdjacentHTML("afterbegin", data.job_html);
        panel.hidden = false;
        status.className = "notice notice-success";
        status.textContent = data.message;
        setBusy(true);
        pollActiveJobs();
    } catch (error) {
        status.className = "notice notice-error";
        status.textContent = error.message;
    } finally {
        form.dataset.submitting = "false";
        buttons.forEach((button, index) => button.disabled = disabled[index]);
    }
}

function formFields() {
    const form = element("experiment-form");
    return form ? [...form.elements].filter((field) => field.name && field.type !== "hidden" && field.name !== "algorithm_names") : [];
}

function gamePresentation(game) {
    return dashboardData.gamePresentations[game] || {
        label: game,
        description: "",
    };
}

function saveFormState() {
    const state = Object.fromEntries(formFields().map((field) => [field.name, field.value]));
    state.algorithmNames = playerAlgorithmSelects().map((select) => select.value);
    saveLocalJson(formStorageKey, state, "experiment parameters");
}

function restoreFormState() {
    const state = restoreLocalJson(formStorageKey, "experiment parameters");
    if (!state) {
        return;
    }

    for (const control of formFields()) {
        const key = control.name in state ? control.name : control.id;
        let value = state[key];
        if (value === undefined && control.name === "seed") value = state.learner_seed;
        if (value === undefined) {
            continue;
        }
        if (control instanceof HTMLSelectElement && ![...control.options].some((option) => option.value === value)) {
            continue;
        }
        const fallback = control.value;
        control.value = value;
        if (!control.checkValidity()) {
            control.value = fallback;
        }
    }
    updatePlayerControls(state.algorithmNames || []);
}

function installFormPersistence() {
    const form = element("experiment-form");
    if (form) {
        form.addEventListener("input", saveFormState);
        form.addEventListener("change", saveFormState);
        form.addEventListener("submit", saveFormState);
    }
}

function updateAlgorithmSelect(select, algorithms) {
    replaceSelectOptions(select, algorithms, dashboardData.algorithmLabels);
}

function playerAlgorithmSelects() {
    return [...document.querySelectorAll(".player-algorithm")];
}

function updatePlayerControls(preferredValues = null) {
    const feedback = element("feedback-mode");
    const algorithms = dashboardData.algorithms[feedback ? feedback.value : ""] || [];
    const existingValues = preferredValues || playerAlgorithmSelects().map((select) => select.value);
    const gameSelect = element("game");
    const game = gameSelect ? gameSelect.value : "";
    const definition = (dashboardData.gameDefinitions || {})[game] || {n_players: 1};
    const container = element("players");
    if (!container) {
        return;
    }
    const fields = [];
    for (let player = 0; player < definition.n_players; player += 1) {
        const fieldset = document.createElement("fieldset");
        const legend = document.createElement("legend");
        const field = document.createElement("div");
        const label = document.createElement("label");
        const select = document.createElement("select");
        const id = `algorithm_player_${player}`;
        legend.textContent = `Player ${player}`;
        field.className = "field";
        label.htmlFor = id;
        label.textContent = "Algorithm";
        select.id = id;
        select.name = "algorithm_names";
        select.className = "player-algorithm";
        select.required = true;
        field.append(label, select);
        if (definition.n_players > 1) {
            fieldset.append(legend, field);
            fields.push(fieldset);
        } else {
            fields.push(field);
        }
        updateAlgorithmSelect(select, algorithms);
        if (algorithms.includes(existingValues[player])) {
            select.value = existingValues[player];
        }
    }
    container.replaceChildren(...fields);
}

function updateAlgorithmsForFeedbackMode() {
    const feedbackSelect = element("feedback-mode");
    if (!feedbackSelect) {
        return;
    }

    const algorithms = dashboardData.algorithms[feedbackSelect.value] || [];
    playerAlgorithmSelects().forEach((select) => updateAlgorithmSelect(select, algorithms));
}

function selectedResultScope() {
    const scope = element("filter-scope");
    return scope && scope.value !== "all" ? scope.value : "";
}

function updateEquilibriumFigures() {
    const game = selectedResultScope();
    const panel = element("equilibrium-panel");
    if (panel) {
        panel.hidden = !game;
    }
    if (!game) {
        return;
    }
    const urls = (dashboardData.equilibriumFigures || {})[game];

    const presentation = gamePresentation(game);
    const gameLabel = element("equilibrium-game");
    if (gameLabel) {
        gameLabel.textContent = presentation.label;
    }
    const grid = element("equilibrium-grid");
    const explanation = element("equilibrium-explanation");
    const unavailable = element("equilibrium-unavailable");
    if (grid) {
        grid.hidden = !urls;
    }
    if (explanation) {
        explanation.hidden = !urls;
    }
    if (unavailable) {
        unavailable.hidden = Boolean(urls);
    }
    if (!urls || !panel || !panel.open) {
        return;
    }
    for (const equilibrium of ["ce", "cce"]) {
        const image = element(`${equilibrium}-equilibrium-image`);
        const download = element(`${equilibrium}-equilibrium-download`);
        if (download) {
            download.removeAttribute("href");
            download.setAttribute("aria-disabled", "true");
            download.classList.add("is-disabled");
        }
        if (image) {
            setHeatmapSource(image, urls[equilibrium].png, () => {
                if (download) {
                    download.href = urls[equilibrium].pdf;
                    download.removeAttribute("aria-disabled");
                    download.classList.remove("is-disabled");
                }
            });
            image.alt = `Maximum ${equilibrium.toUpperCase()} profile weight for ${presentation.label}`;
        }
    }
}

function updateDashboardForGame(preferredAlgorithms = null) {
    updatePlayerControls(preferredAlgorithms);
    const game = element("game");
    const description = element("game-description");
    if (game && description) {
        description.textContent = gamePresentation(game.value).description;
    }
}

function updateOnePlayerEnvironment() {
    const environmentSelect = element("adversarial-environment");
    const environment = environmentSelect ? environmentSelect.value : "";
    if (!environment) {
        return;
    }
    const randomWalk = environment === dashboardData.randomWalkEnvironment;
    for (const [fieldId, enabled] of [
        ["adversarial-environment-seed-field", randomWalk],
    ]) {
        const field = element(fieldId);
        if (!field) {
            continue;
        }
        field.hidden = !enabled;
        field.querySelectorAll("input, select").forEach((input) => {
            input.disabled = !enabled;
        });
    }
}

function updateEnvironmentAnalysis() {
    const environment = selectedResultScope();
    const panel = element("environment-panel");
    const historicalRule = element("historical-frequency-rule");
    const randomWalkRule = element("random-walk-rule");
    const randomWalk = environment === dashboardData.randomWalkEnvironment;
    if (panel) {
        panel.hidden = !environment;
    }
    if (historicalRule) {
        historicalRule.hidden = !environment || randomWalk;
    }
    if (randomWalkRule) {
        randomWalkRule.hidden = !environment || !randomWalk;
    }
}

function updateFilteredAnalysis() {
    if (onePlayerMode) {
        updateEnvironmentAnalysis();
    } else {
        updateEquilibriumFigures();
    }
}

function synchronizePlayerValues() {
    const [playerZero, ...otherPlayers] = playerAlgorithmSelects();
    if (playerZero) {
        otherPlayers.forEach((select) => {
            select.value = playerZero.value;
        });
    }
}

function matchesResultFilters(record, state = resultFilters) {
    if (!state) return false;
    return record.dataset.scope === state.scope
        && record.dataset.feedback === state.feedback
        && record.dataset.player === state.player
        && state.profiles.includes(record.dataset.profile);
}

function updateSummaryRows() {
    document.querySelectorAll(".summary-row").forEach((row) => {
        row.hidden = !matchesResultFilters(row) || !resultFilters.resultKeys.includes(row.dataset.resultKey);
    });
    document.querySelectorAll("#summary-table [data-regret]").forEach((cell) => {
        cell.hidden = !resultFilters
            || (resultFilters.metric !== "all" && cell.dataset.regret !== resultFilters.metric)
            || (resultFilters.view !== "all" && cell.dataset.view !== resultFilters.view);
    });
    const detail = element("experiment-detail");
    if (detail && selectedSummary) {
        const row = [...document.querySelectorAll(".summary-row")].find((row) =>
            dashboardData.summaries[Number(row.dataset.summaryIndex)] === selectedSummary);
        if (!row || row.hidden) detail.hidden = true;
    }
    document.querySelectorAll("#detail-regrets [data-regret]").forEach((cell) => {
        cell.hidden = !resultFilters
            || (resultFilters.metric !== "all" && cell.dataset.regret !== resultFilters.metric)
            || (resultFilters.view !== "all" && cell.dataset.view !== resultFilters.view);
    });
    highlightBestValues();
}

function highlightBestValues() {
    document.querySelectorAll("[data-metric]").forEach((cell) => cell.classList.remove("best-value"));
    const groups = new Map();
    document.querySelectorAll("[data-metric][data-value]").forEach((cell) => {
        if (cell.hidden) {
            return;
        }
        const row = cell.closest("tr");
        if (row.hidden) {
            return;
        }
        const keyParts = [
            cell.dataset.metric, row.dataset.scope, row.dataset.player, row.dataset.feedback, row.dataset.horizon,
            row.dataset.seed, row.dataset.stationaryMethod, row.dataset.target, row.dataset.configuration,
        ];
        const key = keyParts.join("|");
        groups.set(key, [...(groups.get(key) || []), cell]);
    });
    groups.forEach((cells) => {
        const minimum = Math.min(...cells.map((cell) => Number(cell.dataset.value)));
        cells.filter((cell) => Number(cell.dataset.value) === minimum).forEach((cell) => cell.classList.add("best-value"));
    });
}

function installTableSorting() {
    const table = element("summary-table");
    if (!table) {
        return;
    }
    table.querySelectorAll("th").forEach((header, column) => {
        header.tabIndex = 0;
        header.title = "Sort column";
        const sort = () => {
            const rows = [...table.tBodies[0].rows];
            const ascending = header.dataset.direction !== "ascending";
            table.querySelectorAll("th").forEach((cell) => delete cell.dataset.direction);
            header.dataset.direction = ascending ? "ascending" : "descending";
            const value = (row) => row.cells[column].dataset.value === undefined ? row.cells[column].textContent.trim() : row.cells[column].dataset.value;
            const values = rows.map(value);
            const numeric = values.every((value) => value !== "" && Number.isFinite(Number(value)));
            rows.sort((left, right) => {
                const leftValue = value(left);
                const rightValue = value(right);
                const comparison = numeric ? Number(leftValue) - Number(rightValue) : leftValue.localeCompare(rightValue);
                return ascending ? comparison : -comparison;
            });
            rows.forEach((row) => table.tBodies[0].append(row));
        };
        header.addEventListener("click", sort);
        header.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                sort();
            }
        });
    });
}

function applyFilters() {
    document.querySelectorAll("[data-result-card]").forEach((card) => {
        card.hidden = !matchesResultFilters(card)
            || (resultFilters.metric !== "all" && card.dataset.metric !== resultFilters.metric);
    });
    document.querySelectorAll("[data-result-section]").forEach((section) => {
        section.hidden = ![...section.querySelectorAll("[data-result-card]")].some((card) => !card.hidden);
    });
    updateSummaryRows();
}

function openFigure(card) {
    const dialog = element("figure-dialog");
    if (!card || !dialog) {
        return;
    }

    const image = element("dialog-figure-image");
    const title = element("dialog-figure-title");
    const download = element("dialog-figure-download");
    const preview = card.querySelector("img");
    const sourceDownload = card.querySelector("a[download]");
    image.src = preview.src;
    image.alt = preview.alt;
    title.textContent = card.querySelector(".figure-open span").textContent;
    download.href = sourceDownload.href;
    download.download = sourceDownload.download;
    download.textContent = sourceDownload.textContent;
    dialog.showModal();
}

function addDetail(metadata, label, value) {
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value;
    metadata.append(term, description);
}

let selectedSummary = null;

function showExperimentDetail(index) {
    const summary = dashboardData.summaries[index];
    const panel = element("experiment-detail");
    if (!summary || !panel) {
        return;
    }

    selectedSummary = summary;
    panel.hidden = false;
    const gameLabel = gamePresentation(summary.game).label;
    element("detail-title").textContent = `${gameLabel} · player ${summary.player}`;
    const metadata = element("detail-metadata");
    metadata.replaceChildren();
    addDetail(metadata, "Feedback", summary.feedback_mode);
    addDetail(metadata, "Profile", summary.profile_label);
    addDetail(metadata, "Horizon", summary.horizon);
    addDetail(metadata, "Seed", summary.seed);
    addDetail(metadata, "Replicates", `${summary.replicate_label} (n=${summary.replicate_count})`);
    addDetail(metadata, "Stationary solver", summary.stationary_method);
    addDetail(metadata, "Implementation", summary.implementation_version || "legacy");

    const regrets = element("detail-regrets");
    regrets.replaceChildren();
    Object.entries(summary.display_regrets).forEach(([name, value]) => {
        const kind = name.split("_").pop();
        const view = name.startsWith("average_") ? "average" : "sqrt_scaling";
        const metric = document.createElement("div");
        metric.dataset.regret = kind;
        metric.dataset.view = view;
        const label = document.createElement("span");
        const number = document.createElement("strong");
        label.textContent = kind + (view === "average" ? " R/T" : " R/√T");
        number.textContent = Number(value).toFixed(6);
        metric.append(label, number);
        regrets.append(metric);
    });
    updateSummaryRows();

    const downloads = element("detail-downloads");
    downloads.replaceChildren(...summary.runs.map((run) => {
        const link = document.createElement("a");
        link.href = run.download_url;
        link.download = run.experiment;
        link.textContent = `Download replicate ${run.replicate} CSV`;
        return link;
    }));
    const heatmap = element("detail-heatmap");
    const jointActions = element("detail-joint-actions");
    jointActions.hidden = !summary.joint_actions_url;
    const heatmapDownload = element("detail-heatmap-download");
    if (summary.joint_actions_url) {
        setHeatmapSource(heatmap, summary.joint_actions_url);
        heatmap.alt = `Mean empirical joint-action distribution for ${gameLabel} across ${summary.replicate_count} replicate${summary.replicate_count === 1 ? "" : "s"}`;
        heatmapDownload.href = summary.joint_actions_pdf_url;
        heatmapDownload.download = `${summary.group_id}_mean_joint_actions.pdf`;
    }
    const distanceImage = element("detail-equilibrium-distance");
    const convergence = element("detail-convergence");
    const distanceAvailable = Boolean(summary.equilibrium_distance_url);
    convergence.hidden = !distanceAvailable;
    element("detail-equilibrium-distance-card").hidden = !distanceAvailable;
    const distanceDownload = element("detail-equilibrium-distance-download");
    if (distanceAvailable) {
        setHeatmapSource(distanceImage, summary.equilibrium_distance_url, null, "Computing equilibrium distances…");
        distanceImage.alt = `Mean CE and CCE L1 distance by horizon for ${gameLabel}`;
        distanceDownload.href = summary.equilibrium_distance_pdf_url;
        distanceDownload.download = `${summary.group_id}_mean_equilibrium_distance.pdf`;
    }
    panel.scrollIntoView({behavior: "smooth", block: "nearest"});
}

function reuseSelectedExperiment() {
    if (!selectedSummary) {
        return;
    }
    element("game").value = selectedSummary.game;
    element("feedback-mode").value = selectedSummary.feedback_mode;
    updateDashboardForGame(selectedSummary.algorithm_profile);
    element("horizon").value = selectedSummary.horizon;
    element("seed").value = selectedSummary.seed;
    element("replicates").value = selectedSummary.replicate_count;
    saveFormState();
    element("experiment-form").scrollIntoView({behavior: "smooth"});
}

function setBusy(busy) {
    if (element("busy-indicator")) {
        element("busy-indicator").hidden = !busy;
    }
    document.querySelectorAll("[data-busy-control]").forEach((control) => {
        control.disabled = busy;
    });
}

async function pollActiveJobs() {
    if (jobPollInFlight) return;
    window.clearTimeout(jobPollTimer);
    jobPollTimer = null;
    const activeJobs = dashboardData.jobs.filter((job) => (
        job.status === "queued" || job.status === "running"
    ));
    if (activeJobs.length === 0) {
        setBusy(false);
        return;
    }

    jobPollInFlight = true;
    try {
        const responses = await Promise.all(activeJobs.map((job) => fetch(job.url)));
        if (responses.some((response) => !response.ok)) {
            throw new Error("job status request failed");
        }
        const jobs = await Promise.all(responses.map((response) => response.json()));
        jobs.forEach(updateJob);

        const terminalJobs = jobs.filter((job) => ["succeeded", "failed", "cancelled"].includes(job.status));
        if (terminalJobs.length > 0) {
            element("refresh-results-notice").hidden = false;
        }
    } catch (error) {
        console.warn("Could not refresh job status", error);
    } finally {
        jobPollInFlight = false;
        const busy = dashboardData.jobs.some((job) => ["queued", "running"].includes(job.status));
        setBusy(busy);
        if (busy) jobPollTimer = window.setTimeout(pollActiveJobs, 1200);
    }
}

function updateJob(job) {
    const storedJob = dashboardData.jobs.find((candidate) => candidate.id === job.id);
    if (storedJob) {
        Object.assign(storedJob, job);
    }
    const item = document.querySelector(`[data-job-id="${job.id}"]`);
    if (!item) {
        return;
    }

    updateJobElement(item, job);
}

listen("feedback-mode", "change", updateAlgorithmsForFeedbackMode);
listen("game", "change", () => {
    updateDashboardForGame();
});
listen("adversarial-environment", "change", () => {
    updateOnePlayerEnvironment();
});
listen("equilibrium-panel", "toggle", (event) => {
    if (event.currentTarget.open) {
        updateEquilibriumFigures();
    }
});
listen("synchronize-players", "click", () => {
    synchronizePlayerValues();
    saveFormState();
});
document.addEventListener("click", (event) => {
    const figureButton = event.target.closest(".figure-open");
    if (figureButton) {
        openFigure(figureButton.closest(".figure-card"));
    }
});
listen("close-figure-dialog", "click", () => element("figure-dialog").close());
document.querySelectorAll(".summary-row").forEach((row) => {
    if (row.dataset.summaryIndex === undefined) {
        return;
    }
    const showDetail = () => showExperimentDetail(Number(row.dataset.summaryIndex));
    row.addEventListener("click", showDetail);
    row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            showDetail();
        }
    });
});
listen("reuse-experiment", "click", reuseSelectedExperiment);
listen("experiment-form", "submit", queueExperiment);
listen("refresh-results", "click", () => {
    saveFormState();
    window.location.reload();
});
restoreFormState();
installFormPersistence();
updateDashboardForGame(playerAlgorithmSelects().map((select) => select.value));
if (onePlayerMode) {
    updateOnePlayerEnvironment();
}
installTableSorting();
document.addEventListener("results-filter-change", (event) => {
    resultFilters = event.detail;
    applyFilters();
    updateFilteredAnalysis();
});
applyFilters();
updateFilteredAnalysis();
pollActiveJobs();
