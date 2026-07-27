"use strict";

const dashboardDataElement = document.getElementById("dashboard-data");
const dashboardData = dashboardDataElement
    ? JSON.parse(dashboardDataElement.textContent)
    : {figures: [], figuresUrl: "", equilibriumFigures: {}, gameDefinitions: {}, gamePresentations: {}, jobs: [], summaries: [], algorithms: {}, algorithmLabels: {}};
const formStorageKey = "swap-regret-experiment-form";
const filterStorageKey = "swap-regret-result-filters";
const trajectoryPointsStorageKey = "swap-regret-trajectory-points";
const trajectoryHideFirstStorageKey = "swap-regret-trajectory-hide-first";
const formFieldIds = ["game", "feedback-mode", "regret-evaluation", "horizon", "seed", "replicate", "replicates"];
const filterFieldIds = [
    "filter-game",
    "filter-source",
    "filter-regret",
    "filter-player",
    "filter-view",
    "filter-feedback",
    "filter-regret-evaluation",
    "filter-player-algorithm",
    "filter-co-player-algorithm",
    "filter-horizon",
    "filter-seed",
];

function element(id) {
    return document.getElementById(id);
}

function gamePresentation(game) {
    return dashboardData.gamePresentations[game] || {
        label: game,
        description: "",
    };
}

function saveFormState() {
    const state = Object.fromEntries(formFieldIds.map((id) => [id, element(id)?.value ?? ""]));
    state.algorithmNames = playerAlgorithmSelects().map((select) => select.value);
    try {
        window.localStorage.setItem(formStorageKey, JSON.stringify(state));
    } catch (error) {
        console.warn("Could not save experiment parameters", error);
    }
}

function restoreFormState() {
    let state;
    try {
        state = JSON.parse(window.localStorage.getItem(formStorageKey));
    } catch (error) {
        console.warn("Could not restore experiment parameters", error);
        return;
    }
    if (!state) {
        return;
    }

    for (const id of ["game", "feedback-mode", "regret-evaluation"]) {
        const select = element(id);
        if (select && [...select.options].some((option) => option.value === state[id])) {
            select.value = state[id];
        }
    }
    const legacyAlgorithms = [state.algorithm_player_0, state.algorithm_player_1].filter(Boolean);
    updatePlayerControls(state.algorithmNames || legacyAlgorithms);
    for (const id of ["horizon", "seed", "replicate", "replicates"]) {
        const input = element(id);
        if (input && state[id] !== undefined) {
            input.value = state[id];
        }
    }
    updateReplicateVisibility();
}

function installFormPersistence() {
    const form = element("experiment-form");
    form?.addEventListener("input", saveFormState);
    form?.addEventListener("change", saveFormState);
    form?.addEventListener("submit", saveFormState);
}

function updateAlgorithmSelect(select, algorithms) {
    if (!select) {
        return;
    }

    const previousValue = select.value;
    select.replaceChildren(...algorithms.map((algorithm) => {
        const option = document.createElement("option");
        option.value = algorithm;
        option.textContent = dashboardData.algorithmLabels[algorithm] || algorithm;
        return option;
    }));

    select.value = algorithms.includes(previousValue)
        ? previousValue
        : algorithms[0] || "";
}

function playerAlgorithmSelects() {
    return [...document.querySelectorAll(".player-algorithm")];
}

function updatePlayerControls(preferredValues = null) {
    const game = element("game")?.value;
    const definition = dashboardData.gameDefinitions[game];
    const container = element("players");
    if (!definition || !container) {
        return;
    }
    const existingValues = preferredValues || playerAlgorithmSelects().map((select) => select.value);
    const algorithms = dashboardData.algorithms[element("feedback-mode")?.value] || [];
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
        fieldset.append(legend, field);
        fields.push(fieldset);
        updateAlgorithmSelect(select, algorithms);
        if (algorithms.includes(existingValues[player])) {
            select.value = existingValues[player];
        }
    }
    container.replaceChildren(...fields);
    const allPairsButton = element("queue-all-pairs");
    if (allPairsButton) {
        allPairsButton.hidden = definition.source !== "builtin" || definition.n_players !== 2;
    }
}

function updateAlgorithmsForFeedbackMode() {
    const feedbackSelect = element("feedback-mode");
    if (!feedbackSelect) {
        return;
    }

    const algorithms = dashboardData.algorithms[feedbackSelect.value] || [];
    playerAlgorithmSelects().forEach((select) => updateAlgorithmSelect(select, algorithms));
}

function updateReplicateVisibility() {
    const field = element("replicates-field");
    if (field) {
        const hidden = element("feedback-mode")?.value !== "bandit";
        field.hidden = hidden;
        field.closest(".replicate-grid")?.classList.toggle("replicate-grid-single", hidden);
    }
}

function alignedRegretEvaluation(feedbackMode) {
    return feedbackMode === "bandit" ? "realized" : "expected";
}

function updateRegretEvaluationForFeedback(previousFeedbackMode) {
    const feedback = element("feedback-mode")?.value;
    const evaluation = element("regret-evaluation");
    if (evaluation && evaluation.value === alignedRegretEvaluation(previousFeedbackMode)) {
        evaluation.value = alignedRegretEvaluation(feedback);
    }
}

function setHeatmapSource(image, source, onReady = null, loadingMessage = "Loading heatmap…") {
    if (!image || !source) {
        return;
    }
    const frame = image.closest(".heatmap-frame");
    if (!frame) {
        image.src = source;
        return;
    }

    const requestId = String(Number(image.dataset.loadRequest || 0) + 1);
    const status = frame.querySelector(".heatmap-loading");
    const previousObjectUrl = image.dataset.objectUrl;
    if (previousObjectUrl) {
        URL.revokeObjectURL(previousObjectUrl);
        delete image.dataset.objectUrl;
    }
    image.removeAttribute("src");
    image.dataset.loadRequest = requestId;
    frame.classList.add("is-loading");
    frame.classList.remove("has-error");
    frame.setAttribute("aria-busy", "true");
    if (status) {
        status.textContent = loadingMessage;
    }

    const finish = (failed) => {
        if (image.dataset.loadRequest !== requestId) {
            return;
        }
        frame.classList.remove("is-loading");
        frame.classList.toggle("has-error", failed);
        frame.removeAttribute("aria-busy");
        if (failed && status) {
            status.textContent = "Heatmap unavailable";
        }
    };
    const load = async () => {
        try {
            const response = await fetch(source, {cache: "no-store"});
            if (image.dataset.loadRequest !== requestId) {
                return;
            }
            if (response.status === 202) {
                const payload = await response.json();
                if (status) {
                    status.textContent = payload.message || "Computing equilibrium heatmap…";
                }
                const retryAfter = Number(response.headers.get("Retry-After")) || 2;
                window.setTimeout(load, Math.max(1, retryAfter) * 1000);
                return;
            }
            if (!response.ok) {
                throw new Error(`Heatmap request failed (${response.status})`);
            }

            const objectUrl = URL.createObjectURL(await response.blob());
            if (image.dataset.loadRequest !== requestId) {
                URL.revokeObjectURL(objectUrl);
                return;
            }
            image.onload = () => {
                if (image.dataset.loadRequest !== requestId) {
                    URL.revokeObjectURL(objectUrl);
                    return;
                }
                image.dataset.objectUrl = objectUrl;
                finish(false);
                onReady?.();
            };
            image.onerror = () => {
                URL.revokeObjectURL(objectUrl);
                finish(true);
            };
            image.src = objectUrl;
        } catch (error) {
            console.warn("Could not load heatmap", error);
            finish(true);
        }
    };
    load();
}

function updateEquilibriumFigures() {
    const game = element("game")?.value;
    const urls = dashboardData.equilibriumFigures[game];
    if (!game) {
        return;
    }

    const presentation = gamePresentation(game);
    const gameLabel = element("equilibrium-game");
    if (gameLabel) {
        gameLabel.textContent = presentation.label;
    }
    const gameDescription = element("game-description");
    if (gameDescription) {
        gameDescription.textContent = presentation.description;
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
    if (!urls) {
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
            setHeatmapSource(image, urls[equilibrium], () => {
                if (download) {
                    download.href = urls[equilibrium];
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
    updateEquilibriumFigures();
}

function synchronizePlayerValues() {
    const [playerZero, ...otherPlayers] = playerAlgorithmSelects();
    if (playerZero) {
        otherPlayers.forEach((select) => {
            select.value = playerZero.value;
        });
    }
}

function selectedFilter(id) {
    return element(id)?.value || "all";
}

function saveFilterState() {
    const state = Object.fromEntries(filterFieldIds.map((id) => [id, element(id)?.value ?? ""]));
    try {
        window.localStorage.setItem(filterStorageKey, JSON.stringify(state));
    } catch (error) {
        console.warn("Could not save result filters", error);
    }
}

function restoreFilterState() {
    let state;
    try {
        state = JSON.parse(window.localStorage.getItem(filterStorageKey));
    } catch (error) {
        console.warn("Could not restore result filters", error);
        return;
    }
    if (!state) {
        return;
    }

    filterFieldIds.forEach((id) => {
        const control = element(id);
        const value = state[id];
        if (!control || value === undefined) {
            return;
        }
        if (control instanceof HTMLSelectElement && ![...control.options].some((option) => option.value === value)) {
            return;
        }
        control.value = value;
    });
}

function installFilterPersistence() {
    filterFieldIds.forEach((id) => {
        const control = element(id);
        if (!control) {
            return;
        }
        const eventName = control instanceof HTMLSelectElement ? "change" : "input";
        control.addEventListener(eventName, () => {
            saveFilterState();
            applyFilters();
        });
    });
}

function updateSummaryRows() {
    const game = selectedFilter("filter-game");
    const player = selectedFilter("filter-player");
    const feedback = selectedFilter("filter-feedback");
    const regretEvaluation = selectedFilter("filter-regret-evaluation");
    const playerAlgorithm = selectedFilter("filter-player-algorithm");
    const coPlayerAlgorithm = selectedFilter("filter-co-player-algorithm");
    const horizon = element("filter-horizon")?.value || "";
    const seed = element("filter-seed")?.value || "";

    document.querySelectorAll(".summary-row").forEach((row) => {
        const gameMatches = game === "all" || row.dataset.game === game;
        const playerMatches = player === "all" || row.dataset.player === player;
        const feedbackMatches = feedback === "all" || row.dataset.feedback === feedback;
        const evaluationMatches = regretEvaluation === "all" || row.dataset.regretEvaluation === regretEvaluation;
        const playerAlgorithmMatches = playerAlgorithm === "all" || row.dataset.playerAlgorithm === playerAlgorithm;
        const coPlayerAlgorithmMatches = coPlayerAlgorithm === "all" || row.dataset.coPlayerAlgorithms.split(" ").includes(coPlayerAlgorithm);
        const horizonMatches = !horizon || row.dataset.horizon === horizon;
        const seedMatches = !seed || row.dataset.seed === seed;
        row.hidden = !(gameMatches && playerMatches && feedbackMatches && evaluationMatches && playerAlgorithmMatches && coPlayerAlgorithmMatches && horizonMatches && seedMatches);
    });
    highlightBestValues();
}

function highlightBestValues() {
    document.querySelectorAll("[data-metric]").forEach((cell) => cell.classList.remove("best-value"));
    const groups = new Map();
    document.querySelectorAll("[data-metric][data-value]").forEach((cell) => {
        const row = cell.closest("tr");
        if (row.hidden) {
            return;
        }
        const keyParts = [
            cell.dataset.metric, row.dataset.game, row.dataset.feedback, row.dataset.player, row.dataset.horizon,
            row.dataset.seed, row.dataset.stationaryMethod, row.dataset.regretEvaluation,
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
    table?.querySelectorAll("th").forEach((header, column) => {
        header.tabIndex = 0;
        header.title = "Sort column";
        const sort = () => {
            const rows = [...table.tBodies[0].rows];
            const ascending = header.dataset.direction !== "ascending";
            table.querySelectorAll("th").forEach((cell) => delete cell.dataset.direction);
            header.dataset.direction = ascending ? "ascending" : "descending";
            const values = rows.map((row) => row.cells[column].dataset.value ?? row.cells[column].textContent.trim());
            const numeric = values.every((value) => value !== "" && Number.isFinite(Number(value)));
            rows.sort((left, right) => {
                const leftValue = left.cells[column].dataset.value ?? left.cells[column].textContent.trim();
                const rightValue = right.cells[column].dataset.value ?? right.cells[column].textContent.trim();
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
    const filters = {
        game: selectedFilter("filter-game"),
        source: selectedFilter("filter-source"),
        regret: selectedFilter("filter-regret"),
        player: selectedFilter("filter-player"),
        view: selectedFilter("filter-view"),
    };
    let visible = 0;
    document.querySelectorAll(".figure-card").forEach((card) => {
        const matches = Object.entries(filters).every(([name, value]) => value === "all" || card.dataset[name] === value);
        card.hidden = !matches;
        visible += Number(matches);
    });

    const counter = element("figure-counter");
    if (counter) {
        counter.textContent = `${visible} figure${visible === 1 ? "" : "s"}`;
    }
    updateSummaryRows();
}

function openFigure(index) {
    const figure = dashboardData.figures[index];
    const dialog = element("figure-dialog");
    if (!figure || !dialog) {
        return;
    }

    const image = element("dialog-figure-image");
    const title = element("dialog-figure-title");
    const download = element("dialog-figure-download");
    image.src = figure.url;
    const gameLabel = gamePresentation(figure.game).label;
    image.alt = `${gameLabel}, ${figure.source} ${figure.regret} regret, player ${figure.player}`;
    title.textContent = `${gameLabel} · ${figure.source} ${figure.regret} · player ${figure.player}`;
    download.href = figure.url;
    download.download = figure.filename;
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

function trajectoryPoints() {
    const input = element("trajectory-points");
    if (!input) {
        return 10;
    }
    const minimum = Number(input.min) || 2;
    const maximum = Number(input.max) || 50;
    const fallback = Number(input.defaultValue) || 10;
    const parsed = Number(input.value);
    const value = input.value !== "" && Number.isFinite(parsed) ? Math.trunc(parsed) : fallback;
    const normalized = Math.min(maximum, Math.max(minimum, value));
    input.value = normalized;
    return normalized;
}

function saveTrajectoryPoints() {
    try {
        window.localStorage.setItem(trajectoryPointsStorageKey, String(trajectoryPoints()));
    } catch (error) {
        console.warn("Could not save trajectory points", error);
    }
}

function restoreTrajectoryPoints() {
    const input = element("trajectory-points");
    if (!input) {
        return;
    }
    try {
        const stored = window.localStorage.getItem(trajectoryPointsStorageKey);
        if (stored !== null) {
            input.value = stored;
        }
    } catch (error) {
        console.warn("Could not restore trajectory points", error);
    }
    trajectoryPoints();
}

function trajectoryHideFirst() {
    return element("trajectory-hide-first")?.checked ?? false;
}

function saveTrajectoryHideFirst() {
    try {
        window.localStorage.setItem(trajectoryHideFirstStorageKey, trajectoryHideFirst() ? "1" : "0");
    } catch (error) {
        console.warn("Could not save hide-first preference", error);
    }
}

function restoreTrajectoryHideFirst() {
    const input = element("trajectory-hide-first");
    if (!input) {
        return;
    }
    try {
        input.checked = window.localStorage.getItem(trajectoryHideFirstStorageKey) === "1";
    } catch (error) {
        console.warn("Could not restore hide-first preference", error);
    }
}

function loadEquilibriumTrajectory(summary) {
    const image = element("detail-equilibrium-trajectory");
    const download = element("detail-equilibrium-trajectory-download");
    const points = trajectoryPoints();
    const hideFirst = trajectoryHideFirst();
    const source = new URL(summary.equilibrium_trajectory_url, window.location.href);
    source.searchParams.set("points", points);
    source.searchParams.set("hide_first", hideFirst ? "1" : "0");
    setHeatmapSource(image, source.href, null, "Computing projected equilibrium regions…");
    image.alt = `Projected mean joint-distribution trajectory for ${gamePresentation(summary.game).label} using ${points} points${hideFirst ? ", excluding round 1" : ""}`;
    download.href = source.href;
    download.download = `${summary.group_id}_mean_equilibrium_trajectory_p${points}${hideFirst ? "_hide_first" : ""}.png`;
}

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

    const regrets = element("detail-regrets");
    regrets.replaceChildren();
    Object.entries(summary).filter(([name]) => name.startsWith("average_") && name.endsWith("_regret")).forEach(([name, value]) => {
        const metric = document.createElement("div");
        const label = document.createElement("span");
        const number = document.createElement("strong");
        label.textContent = name.replaceAll("_", " ");
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
        heatmapDownload.href = summary.joint_actions_url;
        heatmapDownload.download = `${summary.group_id}_mean_joint_actions.png`;
    }
    const distanceImage = element("detail-equilibrium-distance");
    const convergence = element("detail-convergence");
    const distanceAvailable = Boolean(summary.equilibrium_distance_url);
    const trajectoryAvailable = Boolean(summary.equilibrium_trajectory_url);
    convergence.hidden = !distanceAvailable;
    element("detail-equilibrium-distance-card").hidden = !distanceAvailable;
    element("detail-equilibrium-trajectory-card").hidden = !trajectoryAvailable;
    const distanceDownload = element("detail-equilibrium-distance-download");
    if (distanceAvailable) {
        setHeatmapSource(distanceImage, summary.equilibrium_distance_url, null, "Computing equilibrium distances…");
        distanceImage.alt = `Mean CE and CCE L1 distance by horizon for ${gameLabel}`;
        distanceDownload.href = summary.equilibrium_distance_url;
        distanceDownload.download = `${summary.group_id}_mean_equilibrium_distance.png`;
    }
    if (trajectoryAvailable) {
        loadEquilibriumTrajectory(summary);
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
    updateReplicateVisibility();
    element("horizon").value = selectedSummary.horizon;
    element("seed").value = selectedSummary.seed;
    element("replicate").value = selectedSummary.replicate;
    element("replicates").value = selectedSummary.replicate_count;
    saveFormState();
    element("experiment-form").scrollIntoView({behavior: "smooth"});
}

function createFigureCard(figure, index) {
    const card = document.createElement("article");
    card.className = "figure-card";
    Object.assign(card.dataset, {
        figureIndex: index,
        game: figure.game,
        source: figure.source,
        regret: figure.regret,
        player: figure.player,
        view: figure.view,
    });

    const button = document.createElement("button");
    button.className = "figure-open";
    button.type = "button";
    const label = document.createElement("span");
    const gameLabel = gamePresentation(figure.game).label;
    label.textContent = `${gameLabel} · ${figure.source} · Player ${figure.player} · ${figure.regret}`;
    const image = document.createElement("img");
    image.src = figure.url;
    image.alt = `${gameLabel}, ${figure.source} ${figure.regret} regret for player ${figure.player}`;
    image.loading = "lazy";
    button.append(label, image);

    const download = document.createElement("a");
    download.href = figure.url;
    download.download = figure.filename;
    download.textContent = "Download PNG";
    card.append(button, download);
    return card;
}

function renderFigures(figures) {
    dashboardData.figures = figures;
    element("figure-grid")?.replaceChildren(...figures.map(createFigureCard));
    if (element("figure-filters")) {
        element("figure-filters").hidden = figures.length === 0;
    }
    if (element("figure-empty")) {
        element("figure-empty").hidden = figures.length > 0;
    }
    applyFilters();
}

async function refreshFigures() {
    const response = await fetch(dashboardData.figuresUrl, {headers: {"Accept": "application/json"}});
    if (!response.ok) {
        throw new Error("figure inventory request failed");
    }
    const figures = await response.json();
    const version = Date.now();
    renderFigures(figures.map((figure) => ({...figure, url: `${figure.url.split("?")[0]}?v=${version}`})));
}

function setBusy(busy) {
    if (element("busy-indicator")) {
        element("busy-indicator").hidden = !busy;
    }
    document.querySelectorAll("[data-busy-control]").forEach((control) => {
        control.disabled = busy;
    });
}

function showNotice(message, category) {
    const notice = document.createElement("div");
    notice.className = `notice notice-${category}`;
    notice.textContent = message;
    element("notice-stack")?.append(notice);
    window.setTimeout(() => notice.remove(), 6000);
}

function addJob(job) {
    dashboardData.jobs.unshift(job);
    const panel = document.querySelector(".jobs-panel");
    let list = panel?.querySelector(".job-list");
    if (!list && panel) {
        panel.querySelector(".empty")?.remove();
        list = document.createElement("ol");
        list.className = "job-list";
        panel.append(list);
    }
    if (!list) {
        return;
    }

    const item = document.createElement("li");
    item.className = `job job-${job.status}`;
    item.dataset.jobId = job.id;
    const description = document.createElement("div");
    const title = document.createElement("strong");
    const message = document.createElement("p");
    title.textContent = job.description;
    message.textContent = job.message;
    description.append(title, message);
    const actions = document.createElement("div");
    actions.className = "job-actions";
    const status = document.createElement("span");
    status.className = "status";
    status.textContent = job.status;
    actions.append(status);
    item.append(description, actions);
    list.prepend(item);
}

async function submitPlotRebuild(event) {
    event.preventDefault();
    const form = event.currentTarget;
    setBusy(true);

    try {
        const response = await fetch(form.action, {method: "POST", body: new FormData(form), headers: {"Accept": "application/json"}});
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || "Could not queue figure rebuild");
        }
        addJob(payload);
        showNotice(`Queued plot job ${payload.id.slice(0, 8)}.`, "success");
        pollActiveJobs();
    } catch (error) {
        setBusy(false);
        showNotice(error.message, "error");
    }
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
        if (terminalJobs.some((job) => job.reload_page)) {
            saveFormState();
            window.location.reload();
            return;
        }
        if (terminalJobs.length === jobs.length) {
            if (terminalJobs.some((job) => job.status === "succeeded")) {
                await refreshFigures();
            }
            setBusy(false);
            return;
        }
    } catch (error) {
        console.warn("Could not refresh job status", error);
        if (!dashboardData.jobs.some((job) => job.status === "queued" || job.status === "running")) {
            setBusy(false);
            showNotice(error.message, "error");
            return;
        }
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

    item.classList.remove("job-queued", "job-running", "job-succeeded", "job-failed", "job-cancelled");
    item.classList.add(`job-${job.status}`);
    const message = item.querySelector("p");
    const status = item.querySelector(".status");
    if (message) {
        message.textContent = job.message;
    }
    if (status) {
        status.textContent = job.status;
    }
    if (["succeeded", "failed", "cancelled"].includes(job.status)) {
        item.querySelector(".job-actions form")?.remove();
    }
}

element("feedback-mode")?.addEventListener("change", (event) => {
    updateRegretEvaluationForFeedback(event.currentTarget.dataset.previousValue || event.currentTarget.value);
    event.currentTarget.dataset.previousValue = event.currentTarget.value;
    updateAlgorithmsForFeedbackMode();
    updateReplicateVisibility();
});
element("game")?.addEventListener("change", () => {
    updateDashboardForGame();
    saveFormState();
});
element("synchronize-players")?.addEventListener("click", () => {
    synchronizePlayerValues();
    saveFormState();
});
element("figure-grid")?.addEventListener("click", (event) => {
    const button = event.target.closest(".figure-open");
    if (button) {
        openFigure(Number(button.closest(".figure-card").dataset.figureIndex));
    }
});
element("close-figure-dialog")?.addEventListener("click", () => element("figure-dialog").close());
document.querySelectorAll(".summary-row").forEach((row) => {
    const showDetail = () => showExperimentDetail(Number(row.dataset.summaryIndex));
    row.addEventListener("click", showDetail);
    row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            showDetail();
        }
    });
});
element("reuse-experiment")?.addEventListener("click", reuseSelectedExperiment);
element("rebuild-plots-form")?.addEventListener("submit", submitPlotRebuild);
element("trajectory-points")?.addEventListener("change", () => {
    saveTrajectoryPoints();
    if (selectedSummary?.equilibrium_trajectory_url) {
        loadEquilibriumTrajectory(selectedSummary);
    }
});
element("trajectory-hide-first")?.addEventListener("change", () => {
    saveTrajectoryHideFirst();
    if (selectedSummary?.equilibrium_trajectory_url) {
        loadEquilibriumTrajectory(selectedSummary);
    }
});

restoreFormState();
if (element("feedback-mode")) {
    element("feedback-mode").dataset.previousValue = element("feedback-mode").value;
}
restoreTrajectoryPoints();
restoreTrajectoryHideFirst();
updateDashboardForGame(playerAlgorithmSelects().map((select) => select.value));
installFormPersistence();
installTableSorting();
restoreFilterState();
installFilterPersistence();
applyFilters();
pollActiveJobs();
