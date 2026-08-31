"use strict";

const dashboardDataElement = document.getElementById("dashboard-data");
const dashboardData = dashboardDataElement
    ? JSON.parse(dashboardDataElement.textContent)
    : {mode: "fixed", figures: [], equilibriumFigures: {}, gameDefinitions: {}, gamePresentations: {}, jobs: [], summaries: [], algorithms: {}, algorithmLabels: {}};
const onePlayerMode = dashboardData.mode === "adversarial";
const formStorageKey = onePlayerMode ? "swap-regret-adversarial-form" : "swap-regret-experiment-form";
const filterStorageKey = "swap-regret-result-filters";

function formFields() {
    const form = element("experiment-form");
    return form ? [...form.elements].filter((field) => field.name && field.type !== "hidden" && field.name !== "algorithm_names") : [];
}

function resultFilterControls() {
    return [...document.querySelectorAll('[id^="filter-"]')];
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

function alignedRegretEvaluation(feedbackMode) {
    return feedbackMode === "bandit" ? "realized" : "expected";
}

function updateRegretEvaluationForFeedback(previousFeedbackMode) {
    const feedbackSelect = element("feedback-mode");
    const feedback = feedbackSelect ? feedbackSelect.value : "";
    const evaluation = element("regret-evaluation");
    if (evaluation && evaluation.value === alignedRegretEvaluation(previousFeedbackMode)) {
        evaluation.value = alignedRegretEvaluation(feedback);
    }
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
        ["adversarial-initialization-field", randomWalk],
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

function matchesFilters(record, selector, key) {
    return [...document.querySelectorAll(selector)].every((control) => {
        const value = control.value;
        if (!value || value === "all") return true;
        const actual = record.dataset[control.dataset[key]] || "";
        return control.hasAttribute("data-token-filter") ? actual.split(" ").includes(value) : actual === value;
    });
}

function saveFilterState() {
    const state = Object.fromEntries(resultFilterControls().map((control) => [control.id, control.value]));
    saveLocalJson(filterStorageKey, state, "result filters");
}

function restoreFilterState() {
    const state = restoreLocalJson(filterStorageKey, "result filters");
    if (!state) {
        return;
    }

    resultFilterControls().forEach((control) => {
        const value = state[control.id];
        if (value === undefined) {
            return;
        }
        if (control instanceof HTMLSelectElement && ![...control.options].some((option) => option.value === value)) {
            return;
        }
        control.value = value;
    });
}

function installFilterPersistence() {
    resultFilterControls().forEach((control) => {
        const update = () => {
            saveFilterState();
            applyFilters();
            if (control.dataset.resultFilter === "scope") {
                updateFilteredAnalysis();
            }
        };
        control.addEventListener(control.matches("input") ? "input" : "change", update);
    });
}

function updateSummaryRows() {
    document.querySelectorAll(".summary-row").forEach((row) => {
        row.hidden = !matchesFilters(row, "[data-result-filter]", "resultFilter") || !matchesFilters(row, "[data-summary-filter]", "summaryFilter");
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
            cell.dataset.metric, row.dataset.scope, row.dataset.secondary, row.dataset.feedback, row.dataset.horizon,
            row.dataset.seed, row.dataset.stationaryMethod, row.dataset.regretEvaluation, row.dataset.target, row.dataset.configuration,
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
    let visible = 0;
    document.querySelectorAll("#figure-grid .figure-card").forEach((card) => {
        const matches = matchesFilters(card, "[data-result-filter]", "resultFilter") && matchesFilters(card, "[data-figure-filter]", "figureFilter");
        card.hidden = !matches;
        visible += Number(matches);
    });

    const counter = element("figure-counter");
    if (counter) {
        counter.textContent = `${visible} figure${visible === 1 ? "" : "s"}`;
    }
    if (element("figure-empty")) {
        element("figure-empty").hidden = visible > 0;
    }
    document.querySelectorAll("[data-result-card]").forEach((card) => {
        card.hidden = !matchesFilters(card, "[data-result-filter]", "resultFilter");
    });
    document.querySelectorAll("[data-result-section]").forEach((section) => {
        section.hidden = ![...section.querySelectorAll("[data-result-card]")].some((card) => !card.hidden);
    });
    updateSummarySourceColumns();
    updateSummaryRows();
}

function selectAvailableFigureSource(figures) {
    const source = element("filter-source");
    if (!source) {
        return;
    }
    const availableSources = new Set(figures.map((figure) => figure.source));
    if (availableSources.size === 0 || availableSources.has(source.value)) {
        return;
    }
    const availableOption = [...source.options].find((option) => availableSources.has(option.value));
    if (availableOption) {
        source.value = availableOption.value;
    }
}

function selectAvailableSummarySource() {
    const source = element("filter-summary-source");
    if (!source) {
        return;
    }
    const availableSources = new Set();
    dashboardData.summaries.forEach((summary) => {
        summary.regret_sources.forEach((source) => availableSources.add(source));
    });
    [...source.options].forEach((option) => {
        option.disabled = !availableSources.has(option.value);
    });
    if (!availableSources.has(source.value)) {
        const available = [...source.options].find((option) => !option.disabled);
        source.value = available ? available.value : "expected";
    }
}

function updateSummarySourceColumns() {
    const sourceSelect = element("filter-summary-source");
    const source = sourceSelect ? sourceSelect.value : "expected";
    document.querySelectorAll("[data-regret-source]").forEach((cell) => {
        cell.hidden = cell.dataset.regretSource !== source;
    });
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

function toggleFigureConfidence(button) {
    const card = button.closest(".figure-card");
    const show = button.getAttribute("aria-pressed") !== "true";
    const prefix = show ? "withConfidence" : "withoutConfidence";
    card.querySelectorAll("[data-with-confidence-src]").forEach((image) => {
        image.src = image.dataset[`${prefix}Src`];
    });
    card.querySelectorAll("[data-with-confidence-href]").forEach((link) => {
        link.href = link.dataset[`${prefix}Href`];
        link.download = link.dataset[`${prefix}Download`] || link.download;
    });
    button.setAttribute("aria-pressed", show);
    button.textContent = show ? "Hide 95% CI" : "Show 95% CI";
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
    addDetail(metadata, "Regret evaluation", summary.regret_evaluation);
    addDetail(metadata, "Profile", summary.profile_label);
    addDetail(metadata, "Horizon", summary.horizon);
    addDetail(metadata, "Seed", summary.seed);
    addDetail(metadata, "Replicates", `${summary.replicate_label} (n=${summary.replicate_count})`);
    addDetail(metadata, "Stationary solver", summary.stationary_method);
    addDetail(metadata, "Implementation", summary.implementation_version || "legacy");

    const regrets = element("detail-regrets");
    regrets.replaceChildren();
    Object.entries(summary).filter(([name]) => name.startsWith("average_") && name.endsWith("_regret")).forEach(([name, value]) => {
        const metric = document.createElement("div");
        const label = document.createElement("span");
        const number = document.createElement("strong");
        label.textContent = name.replace(/_/g, " ");
        const confidence = summary.confidence_intervals[name] || 0;
        number.textContent = summary.replicate_count > 1
            ? `${Number(value).toFixed(6)} ± ${Number(confidence).toFixed(6)}`
            : Number(value).toFixed(6);
        metric.append(label, number);
        regrets.append(metric);
    });

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
    element("feedback-mode").dataset.previousValue = selectedSummary.feedback_mode;
    element("regret-evaluation").value = selectedSummary.regret_evaluation;
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
    const activeJobs = dashboardData.jobs.filter((job) => (
        job.status === "queued" || job.status === "running"
    ));
    if (activeJobs.length === 0) {
        setBusy(false);
        return;
    }

    try {
        const responses = await Promise.all(activeJobs.map((job) => fetch(job.url)));
        if (responses.some((response) => !response.ok)) {
            throw new Error("job status request failed");
        }
        const jobs = await Promise.all(responses.map((response) => response.json()));
        jobs.forEach(updateJob);

        const terminalJobs = jobs.filter((job) => ["succeeded", "failed", "cancelled"].includes(job.status));
        if (terminalJobs.length > 0) {
            saveFormState();
            window.location.reload();
            return;
        }
    } catch (error) {
        console.warn("Could not refresh job status", error);
    }
    window.setTimeout(pollActiveJobs, 1200);
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

listen("feedback-mode", "change", (event) => {
    updateRegretEvaluationForFeedback(event.currentTarget.dataset.previousValue || event.currentTarget.value);
    event.currentTarget.dataset.previousValue = event.currentTarget.value;
    updateAlgorithmsForFeedbackMode();
});
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
    const confidenceToggle = event.target.closest(".confidence-toggle");
    if (confidenceToggle) {
        toggleFigureConfidence(confidenceToggle);
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
restoreFormState();
if (element("feedback-mode")) {
    element("feedback-mode").dataset.previousValue = element("feedback-mode").value;
}
installFormPersistence();
updateDashboardForGame(playerAlgorithmSelects().map((select) => select.value));
if (onePlayerMode) {
    updateOnePlayerEnvironment();
}
installTableSorting();
restoreFilterState();
selectAvailableFigureSource(dashboardData.figures);
selectAvailableSummarySource();
installFilterPersistence();
applyFilters();
updateFilteredAnalysis();
pollActiveJobs();
