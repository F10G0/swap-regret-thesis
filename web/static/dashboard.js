"use strict";

const dashboardDataElement = document.getElementById("dashboard-data");
const dashboardData = dashboardDataElement
    ? JSON.parse(dashboardDataElement.textContent)
    : {figures: [], figuresUrl: "", jobs: [], summaries: [], algorithms: {}, algorithmLabels: {}};
const formStorageKey = "swap-regret-experiment-form";
const formFieldIds = ["game", "feedback-mode", "algorithm_player_0", "algorithm_player_1", "horizon", "seed", "replicate", "replicates"];

function element(id) {
    return document.getElementById(id);
}

function saveFormState() {
    const state = Object.fromEntries(formFieldIds.map((id) => [id, element(id)?.value ?? ""]));
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

    for (const id of ["game", "feedback-mode"]) {
        const select = element(id);
        if (select && [...select.options].some((option) => option.value === state[id])) {
            select.value = state[id];
        }
    }
    updateAlgorithmsForFeedbackMode();

    for (const id of ["algorithm_player_0", "algorithm_player_1"]) {
        const select = element(id);
        if (select && [...select.options].some((option) => option.value === state[id])) {
            select.value = state[id];
        }
    }
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

function updateAlgorithmsForFeedbackMode() {
    const feedbackSelect = element("feedback-mode");
    if (!feedbackSelect) {
        return;
    }

    const algorithms = dashboardData.algorithms[feedbackSelect.value] || [];
    updateAlgorithmSelect(element("algorithm_player_0"), algorithms);
    updateAlgorithmSelect(element("algorithm_player_1"), algorithms);
}

function updateReplicateVisibility() {
    const field = element("replicates-field");
    if (field) {
        field.hidden = element("feedback-mode")?.value !== "bandit";
    }
}

function swapPlayerValues() {
    const playerZero = element("algorithm_player_0");
    const playerOne = element("algorithm_player_1");
    if (playerZero && playerOne) {
        [playerZero.value, playerOne.value] = [playerOne.value, playerZero.value];
    }
}

function selectedFilter(id) {
    return element(id)?.value || "all";
}

function updateSummaryRows() {
    const game = selectedFilter("filter-game");
    const player = selectedFilter("filter-player");
    const feedback = selectedFilter("filter-feedback");
    const playerAlgorithm = selectedFilter("filter-player-algorithm");
    const opponentAlgorithm = selectedFilter("filter-opponent-algorithm");
    const horizon = element("filter-horizon")?.value || "";
    const seed = element("filter-seed")?.value || "";

    document.querySelectorAll(".summary-row").forEach((row) => {
        const gameMatches = game === "all" || row.dataset.game === game;
        const playerMatches = player === "all" || row.dataset.player === player;
        const feedbackMatches = feedback === "all" || row.dataset.feedback === feedback;
        const playerAlgorithmMatches = playerAlgorithm === "all" || row.dataset.playerAlgorithm === playerAlgorithm;
        const opponentAlgorithmMatches = opponentAlgorithm === "all" || row.dataset.opponentAlgorithm === opponentAlgorithm;
        const horizonMatches = !horizon || row.dataset.horizon === horizon;
        const seedMatches = !seed || row.dataset.seed === seed;
        row.hidden = !(gameMatches && playerMatches && feedbackMatches && playerAlgorithmMatches && opponentAlgorithmMatches && horizonMatches && seedMatches);
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
            row.dataset.seed, row.dataset.replicate, row.dataset.stationaryMethod,
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
    image.alt = `${figure.game}, ${figure.source} ${figure.regret} regret, player ${figure.player}`;
    title.textContent = `${figure.game} · ${figure.source} ${figure.regret} · player ${figure.player}`;
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

function showExperimentDetail(index) {
    const summary = dashboardData.summaries[index];
    const panel = element("experiment-detail");
    if (!summary || !panel) {
        return;
    }

    selectedSummary = summary;
    panel.hidden = false;
    element("detail-title").textContent = `${summary.game} · player ${summary.player}`;
    const metadata = element("detail-metadata");
    metadata.replaceChildren();
    addDetail(metadata, "Feedback", summary.feedback_mode);
    const playerZeroAlgorithm = dashboardData.algorithmLabels[summary.algorithm_player_0] || summary.algorithm_player_0;
    const playerOneAlgorithm = dashboardData.algorithmLabels[summary.algorithm_player_1] || summary.algorithm_player_1;
    addDetail(metadata, "Profile", `${playerZeroAlgorithm} vs ${playerOneAlgorithm}`);
    addDetail(metadata, "Horizon", summary.horizon);
    addDetail(metadata, "Seed", summary.seed);
    addDetail(metadata, "Replicate", summary.replicate);
    addDetail(metadata, "Replicates in group", summary.replicate_count);
    addDetail(metadata, "Stationary solver", summary.stationary_method);

    const regrets = element("detail-regrets");
    regrets.replaceChildren();
    Object.entries(summary).filter(([name]) => name.startsWith("average_") && name.endsWith("_regret")).forEach(([name, value]) => {
        const metric = document.createElement("div");
        const label = document.createElement("span");
        const number = document.createElement("strong");
        label.textContent = name.replaceAll("_", " ");
        number.textContent = Number(value).toFixed(6);
        metric.append(label, number);
        regrets.append(metric);
    });

    const download = element("detail-download");
    download.href = summary.download_url;
    download.download = summary.experiment;
    const heatmap = element("detail-heatmap");
    heatmap.src = summary.joint_actions_url;
    const heatmapDownload = element("detail-heatmap-download");
    heatmapDownload.href = summary.joint_actions_url;
    heatmapDownload.download = `${summary.run_id}_joint_actions.png`;
    panel.scrollIntoView({behavior: "smooth", block: "nearest"});
}

function reuseSelectedExperiment() {
    if (!selectedSummary) {
        return;
    }
    element("game").value = selectedSummary.game;
    element("feedback-mode").value = selectedSummary.feedback_mode;
    updateAlgorithmsForFeedbackMode();
    updateReplicateVisibility();
    element("algorithm_player_0").value = selectedSummary.algorithm_player_0;
    element("algorithm_player_1").value = selectedSummary.algorithm_player_1;
    element("horizon").value = selectedSummary.horizon;
    element("seed").value = selectedSummary.seed;
    element("replicate").value = selectedSummary.replicate;
    element("replicates").value = 1;
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
    label.textContent = `${figure.source} · Player ${figure.player} · ${figure.regret}`;
    const image = document.createElement("img");
    image.src = figure.url;
    image.alt = `${figure.source} ${figure.regret} regret for player ${figure.player}`;
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

function installConfirmations() {
    document.querySelectorAll("form[data-confirm]").forEach((form) => {
        form.addEventListener("submit", (event) => {
            if (!window.confirm(form.dataset.confirm)) {
                event.preventDefault();
            }
        });
    });
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

element("feedback-mode")?.addEventListener("change", () => {
    updateAlgorithmsForFeedbackMode();
    updateReplicateVisibility();
});
element("swap-players")?.addEventListener("click", () => {
    swapPlayerValues();
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

[
    "filter-game",
    "filter-source",
    "filter-regret",
    "filter-player",
    "filter-view",
    "filter-feedback",
    "filter-player-algorithm",
    "filter-opponent-algorithm",
].forEach((id) => element(id)?.addEventListener("change", applyFilters));
["filter-horizon", "filter-seed"].forEach((id) => element(id)?.addEventListener("input", applyFilters));

restoreFormState();
installFormPersistence();
installConfirmations();
installTableSorting();
applyFilters();
pollActiveJobs();
