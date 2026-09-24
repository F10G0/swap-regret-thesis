"use strict";

const dashboardDataElement = document.getElementById("dashboard-data");
const dashboardData = dashboardDataElement
    ? JSON.parse(dashboardDataElement.textContent)
    : {mode: "fixed", gameDefinitions: {}, gamePresentations: {}, summaries: [], algorithms: {}, algorithmLabels: {}};
const onePlayerMode = dashboardData.mode === "adversarial";
let browsingNavigationPending = false;
let lastAnnouncedBrowsingSelection = null;
const formStorageKey = onePlayerMode ? "swap-regret-adversarial-form" : "swap-regret-experiment-form";
const generalSeedStorageKey = "swap-regret-experiment-seed";
let filteredDeletePending = false;
let filterSelectionRevision = 0;
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
    return form ? [...form.elements].filter((field) => field.name && field.type !== "hidden" && !["algorithm_names", "seed"].includes(field.name)) : [];
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

function loadGeneralSeed() {
    const seed = element("experiment-seed");
    const stored = restoreLocalValue(generalSeedStorageKey, "experiment seed");
    if (!seed || stored === null) return;
    const fallback = seed.value;
    seed.value = stored;
    if (!seed.checkValidity()) seed.value = fallback;
}

function saveGeneralSeed() {
    const seed = element("experiment-seed");
    if (seed) saveLocalValue(generalSeedStorageKey, seed.value, "experiment seed");
}

function restoreFormState() {
    const state = restoreLocalJson(formStorageKey, "experiment parameters");
    if (state) {
        if ("seed" in state) {
            delete state.seed;
            saveLocalJson(formStorageKey, state, "experiment parameters");
        }
        for (const control of formFields()) {
            const key = control.name in state ? control.name : control.id;
            let value = state[key];
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
    loadGeneralSeed();
}

function installFormPersistence() {
    const form = element("experiment-form");
    if (form) {
        form.addEventListener("input", saveFormState);
        form.addEventListener("change", saveFormState);
        form.addEventListener("submit", saveFormState);
    }
    const seed = element("experiment-seed");
    if (seed) {
        seed.addEventListener("input", saveGeneralSeed);
        seed.addEventListener("change", saveGeneralSeed);
    }
}

function updateAlgorithmSelect(select, algorithms, feedbackMode) {
    replaceSelectOptions(select, algorithms, dashboardData.algorithmLabels[feedbackMode] || {});
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
        updateAlgorithmSelect(select, algorithms, feedback ? feedback.value : "");
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
    playerAlgorithmSelects().forEach((select) => updateAlgorithmSelect(select, algorithms, feedbackSelect.value));
}

function updateDashboardForGame(preferredAlgorithms = null) {
    updatePlayerControls(preferredAlgorithms);
    const game = element("game");
    const description = element("game-description");
    if (game && description) {
        description.textContent = gamePresentation(game.value).description;
    }
}

function updateEnvironmentDescription() {
    const environment = element("adversarial-environment");
    const description = element("environment-description");
    if (environment && description) {
        description.textContent = (dashboardData.environmentDescriptions || {})[environment.value] || "";
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

function navigateBrowsing(url) {
    if (browsingNavigationPending) return;
    browsingNavigationPending = true;
    window.location.assign(url);
}

function browsingSelectionMatches(state) {
    const selected = dashboardData.browsingState;
    const context = element("filter-context");
    if (!selected || !context || !state) return false;
    const profiles = values => [...(values || [])].sort().join("\u0000");
    return state.scope === selected.scope && context.value === selected.context
        && state.comparisonMode === selected.comparisonMode
        && state.feedback === selected.feedback && state.horizon === selected.horizon
        && profiles(state.profiles) === profiles(selected.profiles)
        && state.metric === selected.metric && state.view === selected.view
        && (onePlayerMode ? state.action === selected.action : state.player === selected.player);
}

function browsingUrlForSelection(state) {
    const current = new URLSearchParams(window.location.search);
    const context = element("filter-context").value;
    const params = new URLSearchParams();
    for (const [key, value] of [
        ["mode", dashboardData.mode], ["scope", state.scope], ["context", context],
        ["compare", state.comparisonMode], ["feedback", state.feedback],
        ["horizon", state.horizon],
    ]) params.append(key, value);
    [...state.profiles].sort().forEach(profile => params.append("profile", profile));
    params.append("metric", state.metric);
    params.append("view", state.view);
    params.append(onePlayerMode ? "action" : "player", onePlayerMode ? state.action : state.player);
    const sort = current.get("sort");
    const regretSort = /^(average|sqrt_scaling)_(.+)$/.exec(sort || "");
    const sortVisible = !regretSort
        || ((state.metric === "all" || state.metric === regretSort[2])
            && (["all", "horizon_scaling", regretSort[1]].includes(state.view)));
    if (sort && sortVisible && (onePlayerMode || state.player !== "all")) {
        params.append("sort", sort);
        params.append("dir", current.get("dir") || "asc");
    }
    params.append("page", "1");
    params.append("page_size", current.get("page_size") || "25");
    return `/?${params.toString()}`;
}

function handleServerBrowsingSelection(state) {
    updateFilteredDeletion();
    const context = element("filter-context");
    if (!context || !context.value) {
        if (dashboardData.browsingBootstrap && !element("filter-scope").options.length) {
            navigateBrowsing(dashboardData.browsingDefaultUrl);
            return true;
        }
        return false;
    }
    const signature = JSON.stringify([
        context.value, state.scope, state.comparisonMode, state.feedback, state.horizon,
        [...state.profiles].sort(), state.metric, state.view,
        onePlayerMode ? state.action : state.player,
    ]);
    if (!dashboardData.browsingBootstrap && lastAnnouncedBrowsingSelection === null) {
        lastAnnouncedBrowsingSelection = signature;
        return false;
    }
    if (signature === lastAnnouncedBrowsingSelection) return false;
    lastAnnouncedBrowsingSelection = signature;
    if (dashboardData.browsingBootstrap || !browsingSelectionMatches(state)) {
        navigateBrowsing(browsingUrlForSelection(state));
    }
    return true;
}

function filteredDeletionState() {
    if (dashboardData.browsingBootstrap) return null;
    const state = dashboardData.browsingState;
    if (!state || !state.context) return null;
    return {
        mode: dashboardData.mode, scope: state.scope, context: state.context,
        comparisonMode: state.comparisonMode, feedback: state.feedback,
        horizon: state.horizon, profiles: [...state.profiles],
        player: onePlayerMode ? null : state.player,
        action: onePlayerMode ? state.action : null,
    };
}

function filteredDeletionData(form, state) {
    const data = new FormData(form);
    data.delete("group_id");
    Object.entries(state).forEach(([key, value]) => {
        if (Array.isArray(value)) value.forEach(item => data.append(key, item));
        else if (value != null) data.append(key, String(value));
    });
    return data;
}

function filteredDeletionNotice(message, error = false) {
    const form = element("delete-filtered-experiments");
    if (!form) return;
    let notice = element("filtered-deletion-status");
    if (!notice) {
        notice = document.createElement("p");
        notice.id = "filtered-deletion-status";
        notice.setAttribute("role", "status");
        form.after(notice);
    }
    notice.hidden = !message;
    notice.className = error ? "notice notice-error" : "hint";
    notice.textContent = message;
}

function updateFilteredDeletion() {
    const form = element("delete-filtered-experiments");
    if (!form) return;
    const button = form.querySelector("button");
    const busy = document.querySelector('[data-job-id][data-status="queued"], [data-job-id][data-status="running"]');
    button.disabled = Boolean(busy) || filteredDeletePending || !filteredDeletionState();
}

async function submitFilteredDeletion(event) {
    event.preventDefault();
    event.stopImmediatePropagation(); // Bypass the generic pre-preview form confirmation.
    if (filteredDeletePending) return;
    const form = event.currentTarget;
    const state = filteredDeletionState();
    if (!state) {
        filteredDeletionNotice("Select an available result set before deleting.", true);
        return;
    }
    const revision = filterSelectionRevision;
    const action = form.getAttribute("action");
    let deleted = false;
    filteredDeletePending = true;
    updateFilteredDeletion();
    filteredDeletionNotice("");
    try {
        const previewResponse = await fetch(`${action}/preview`, {
            method: "POST", headers: {Accept: "application/json"},
            body: filteredDeletionData(form, state),
        });
        const preview = await previewResponse.json().catch(() => ({}));
        if (!previewResponse.ok) throw new Error(preview.error || "Could not preview filtered deletion.");
        if (revision !== filterSelectionRevision) {
            filteredDeletionNotice("Filters changed; review the selection and try again.", true);
            return;
        }
        if (!Number.isSafeInteger(preview.count) || preview.count < 0
            || !/^[0-9a-f]{64}$/.test(preview.digest || "")) {
            throw new Error("The deletion preview was invalid. Please try again.");
        }
        if (preview.count === 0) {
            filteredDeletionNotice("No filtered experiments match the current selection.");
            return;
        }
        const noun = preview.count === 1 ? "experiment" : "experiments";
        if (!window.confirm(`Delete ${preview.count} filtered ${noun} and all of their replicates? Generated figures and caches will also be cleared. This cannot be undone.`)) return;
        const data = filteredDeletionData(form, state);
        data.set("digest", preview.digest);
        const response = await fetch(action, {
            method: "POST", headers: {Accept: "application/json"}, body: data,
        });
        const result = await response.json().catch(() => ({}));
        if (response.status === 409 && result.reconfirmation_required) {
            filteredDeletionNotice(result.error || "Results changed; please review and confirm again.", true);
            return;
        }
        if (!response.ok) throw new Error(result.error || "Could not delete filtered experiments.");
        deleted = true;
    } catch (error) {
        filteredDeletionNotice(error.message || "Could not delete filtered experiments.", true);
    } finally {
        filteredDeletePending = false;
        updateFilteredDeletion();
    }
    if (deleted) window.location.reload();
}

function installServerSorting() {
    const table = element("summary-table");
    const state = dashboardData.browsingState;
    if (!table || !state || (!onePlayerMode && state.player === "all")) return;
    table.querySelectorAll("th[data-sort]:not([hidden])").forEach((header) => {
        header.tabIndex = 0;
        header.title = "Sort all filtered groups by this column";
        if (new URLSearchParams(window.location.search).get("sort") === header.dataset.sort) {
            header.dataset.direction = new URLSearchParams(window.location.search).get("dir") === "desc"
                ? "descending" : "ascending";
        }
        const sort = () => {
            const params = new URLSearchParams(window.location.search);
            const ascending = header.dataset.direction !== "ascending";
            params.set("sort", header.dataset.sort);
            params.set("dir", ascending ? "asc" : "desc");
            params.set("page", "1");
            navigateBrowsing(`/?${params.toString()}`);
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
let selectedDetailRow = null;
let detailRequestRevision = 0;

function renderExperimentDetail(summary) {
    const gameLabel = gamePresentation(summary.game).label;
    element("detail-title").textContent = `${gameLabel} · player ${summary.player}`;
    const metadata = element("detail-metadata");
    metadata.replaceChildren();
    addDetail(metadata, "Feedback", summary.feedback_label);
    addDetail(metadata, "Profile", summary.profile_label);
    addDetail(metadata, "Horizon", summary.horizon);
    addDetail(metadata, "Seed", summary.seed);
    addDetail(metadata, "Replicates", `${summary.replicate_label} (n=${summary.replicate_count})`);
    addDetail(metadata, "Stationary solver", summary.stationary_method);

    const regrets = element("detail-regrets");
    regrets.replaceChildren();
    Object.entries(summary.display_regrets).forEach(([name, value]) => {
        const kind = name.split("_").pop();
        const view = name.startsWith("average_") ? "average" : "sqrt_scaling";
        const metric = document.createElement("div");
        const label = document.createElement("span");
        const number = document.createElement("strong");
        label.textContent = kind + (view === "average" ? " R/T" : " R/√T");
        number.textContent = Number(value).toFixed(6);
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
    const distanceUnavailable = summary.equilibrium_distance_unavailable;
    convergence.hidden = !distanceAvailable && !distanceUnavailable;
    element("detail-equilibrium-distance-card").hidden = !distanceAvailable;
    const unavailableNotice = element("detail-equilibrium-distance-unavailable");
    unavailableNotice.textContent = distanceUnavailable || "";
    unavailableNotice.hidden = !distanceUnavailable;
    const distanceDownload = element("detail-equilibrium-distance-download");
    if (distanceAvailable) {
        setHeatmapSource(distanceImage, summary.equilibrium_distance_url, "Computing equilibrium distances…");
        distanceImage.alt = `Mean CE and CCE L1 distance by horizon for ${gameLabel}`;
        distanceDownload.href = summary.equilibrium_distance_pdf_url;
        distanceDownload.download = `${summary.group_id}_mean_equilibrium_distance.pdf`;
    }
}

async function showExperimentDetail(row) {
    const panel = element("experiment-detail");
    if (!row || !panel || row.hidden) {
        return;
    }

    const revision = ++detailRequestRevision;
    selectedDetailRow = row;
    selectedSummary = null;
    panel.hidden = false;
    panel.setAttribute("aria-busy", "true");
    element("detail-title").textContent = `${gamePresentation(row.dataset.scope).label} · player ${row.dataset.player}`;
    element("detail-content").hidden = true;
    element("reuse-experiment").disabled = true;
    const status = element("detail-status");
    status.className = "notice";
    status.textContent = "Loading result details…";
    status.hidden = false;
    panel.scrollIntoView({behavior: "smooth", block: "nearest"});

    let failureMessage = "Could not load result details. Select the row again or refresh results.";
    try {
        const response = await fetch(row.dataset.detailUrl, {headers: {Accept: "application/json"}});
        const detail = await response.json().catch(() => ({}));
        if (revision !== detailRequestRevision || selectedDetailRow !== row || panel.hidden || row.hidden) return;
        if (!response.ok) {
            if (response.status === 404) failureMessage = "This result is no longer available. Refresh results.";
            throw new Error(failureMessage);
        }
        if (detail.group_id !== row.dataset.resultKey || String(detail.player) !== row.dataset.player) {
            failureMessage = "Result details changed. Refresh results.";
            throw new Error(failureMessage);
        }
        renderExperimentDetail(detail);
        selectedSummary = detail;
        element("detail-content").hidden = false;
        status.hidden = true;
        status.textContent = "";
        element("reuse-experiment").disabled = false;
        panel.removeAttribute("aria-busy");
    } catch (_) {
        if (revision !== detailRequestRevision || selectedDetailRow !== row || panel.hidden || row.hidden) return;
        status.className = "notice notice-error";
        status.textContent = failureMessage;
        panel.removeAttribute("aria-busy");
    }
}

function reuseSelectedExperiment() {
    if (!selectedSummary) {
        return;
    }
    element("game").value = selectedSummary.game;
    element("feedback-mode").value = selectedSummary.feedback_mode;
    updateDashboardForGame(selectedSummary.algorithm_profile);
    element("horizon").value = selectedSummary.horizon;
    element("experiment-seed").value = selectedSummary.seed;
    element("replicates").value = selectedSummary.replicate_count;
    saveFormState();
    saveGeneralSeed();
    element("experiment-form").scrollIntoView({behavior: "smooth"});
}

function setBusy(busy) {
    if (element("busy-indicator")) {
        element("busy-indicator").hidden = !busy;
    }
    document.querySelectorAll("[data-busy-control]").forEach((control) => {
        control.disabled = busy;
    });
    updateFilteredDeletion();
}

async function pollActiveJobs() {
    if (jobPollInFlight) return;
    window.clearTimeout(jobPollTimer);
    jobPollTimer = null;
    const activeJobs = [...document.querySelectorAll('[data-job-id][data-status="queued"], [data-job-id][data-status="running"]')];
    const pollStatus = element("job-poll-status");
    if (activeJobs.length === 0) {
        pollStatus.hidden = true;
        pollStatus.textContent = "";
        setBusy(false);
        return;
    }

    jobPollInFlight = true;
    try {
        const responses = await Promise.all(activeJobs.map((job) => fetch(job.dataset.statusUrl)));
        if (responses.some((response) => !response.ok)) {
            throw new Error("job status request failed");
        }
        const jobs = await Promise.all(responses.map((response) => response.json()));
        jobs.forEach((job) => updateJobElement(document.querySelector(`[data-job-id="${job.id}"]`), job));
        pollStatus.hidden = true;
        pollStatus.textContent = "";

        const terminalJobs = jobs.filter((job) => ["succeeded", "failed", "cancelled"].includes(job.status));
        if (terminalJobs.length > 0) {
            element("refresh-results-notice").hidden = false;
        }
    } catch (error) {
        pollStatus.textContent = "Job status is temporarily unavailable. Retrying…";
        pollStatus.hidden = false;
        console.warn("Could not refresh job status", error);
    } finally {
        jobPollInFlight = false;
        const busy = Boolean(document.querySelector('[data-job-id][data-status="queued"], [data-job-id][data-status="running"]'));
        setBusy(busy);
        if (busy) jobPollTimer = window.setTimeout(pollActiveJobs, 1200);
    }
}

listen("feedback-mode", "change", updateAlgorithmsForFeedbackMode);
listen("game", "change", () => {
    updateDashboardForGame();
});
listen("adversarial-environment", "change", updateEnvironmentDescription);
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
    if (row.dataset.detailUrl === undefined) {
        return;
    }
    const interactive = event => event.target.closest("a, button, form, input, select, textarea, label");
    const showDetail = () => showExperimentDetail(row);
    row.addEventListener("click", event => {
        if (!interactive(event)) showDetail();
    });
    row.addEventListener("keydown", (event) => {
        if (!interactive(event) && (event.key === "Enter" || event.key === " ")) {
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
installServerSorting();
const filteredDeleteForm = element("delete-filtered-experiments");
if (filteredDeleteForm) filteredDeleteForm.addEventListener("submit", submitFilteredDeletion, true);
document.addEventListener("results-filter-change", (event) => {
    if (handleServerBrowsingSelection(event.detail)) {
        filterSelectionRevision += 1;
        filteredDeletionNotice("");
    }
});
updateFilteredDeletion();
updateEnvironmentDescription();
pollActiveJobs();
