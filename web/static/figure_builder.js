/* One shared result selection; cache probes never render figures. */
(() => {
    const form = document.getElementById("figure-builder");
    const filters = document.getElementById("result-filters");
    if (!form || !filters) return;
    const filter = name => document.getElementById(`filter-${name}`);
    const builder = name => document.getElementById(`builder-${name}`);
    const onePlayer = form.elements.mode.value === "adversarial";
    const storageKey = `swap-regret-profile-batch-filters-${form.elements.mode.value}`;
    let localSaved = {};
    try { localSaved = JSON.parse(localStorage.getItem(storageKey)) || {}; } catch (_) {}
    const browsingData = JSON.parse(document.getElementById("dashboard-data").textContent);
    const urlAuthoritative = browsingData.serverBrowsing && !browsingData.browsingBootstrap;
    const urlState = browsingData.browsingState || {};
    const saved = urlAuthoritative
        ? {...localSaved, ...urlState,
           selections: {...(localSaved.selections || {}), ...(urlState.selections || {})}}
        : localSaved;
    const emptyUrlSelection = urlAuthoritative && Array.isArray(saved.profiles) && saved.profiles.length === 0;
    const selections = new Map(Object.entries(saved.selections || {}));
    const feedbackLabels = {full_information: "Full information", bandit: "Bandit feedback", both: "Both"};
    let catalog = {contexts: [], metrics: [], views: []};
    let currentContext = null, selectionKey = "";
    let compatibleProfileIds = [];
    let selectionRevision = 0, displayRevision = 0;
    let probingRevision = null, generatingRevision = null;
    let exporting = false, figures = [];
    let profileMetric = saved.profileMetric || "external";
    let selectedView = saved.view || "all";
    let standardView = selectedView === "log_log_fit" ? "all" : selectedView;
    let selectedAction = saved.selectedAction || "";
    let selectedHorizon = saved.selectedHorizon || "";
    const comparisonModes = onePlayer ? ["regrets", "profiles", "actions", "horizons"] : ["regrets", "profiles", "horizons"];

    if (comparisonModes.includes(saved.comparisonMode)) {
        filter(`compare-${saved.comparisonMode}`).checked = true;
    }

    const comparisonMode = () => filters.querySelector('input[name="comparison_mode"]:checked').value;
    const profiles = () => [...filter("profiles").selectedOptions].map(option => option.value);
    const selectionValid = () => currentContext && profiles().length > 0
        && (onePlayer || filter("player").value !== "all")
        && (comparisonMode() === "profiles" || profiles().length === 1)
        && (comparisonMode() === "horizons" || filter("horizon").value !== "all")
        && (!onePlayer || comparisonMode() === "actions" || filter("action").value !== "all");
    const selected = () => {
        const selectedProfiles = profiles();
        return {
            scope: filter("scope").value, feedback: filter("feedback").value,
            player: onePlayer ? "0" : filter("player").value,
            action: onePlayer ? filter("action").value : null,
            horizon: filter("horizon").value,
            metric: filter("metric").value, view: filter("view").value,
            profiles: selectedProfiles.includes("all") ? compatibleProfileIds : selectedProfiles,
            comparisonMode: comparisonMode(), resultKeys: currentContext ? currentContext.result_keys : [],
        };
    };

    function setOptions(select, entries, preferred = "", enabled = true) {
        select.disabled = true;
        select.replaceChildren(...entries.map(([value, label]) => new Option(label, String(value))));
        if (entries.length) {
            select.value = entries.some(([value]) => String(value) === preferred) ? preferred : String(entries[0][0]);
        }
        select.disabled = !enabled || !entries.length;
    }

    function clearOptions(...selects) {
        selects.forEach(select => {
            select.disabled = true;
            select.replaceChildren();
        });
    }

    function remember() {
        if (selectionKey) selections.set(selectionKey, profiles());
    }

    function announceSelection() {
        const state = selected();
        document.dispatchEvent(new CustomEvent("results-filter-change", {detail: state}));
        try {
            localStorage.setItem(storageKey, JSON.stringify({...state, context: filter("context").value,
                profileMetric, selectedAction, selectedHorizon, selections: Object.fromEntries(selections)}));
        } catch (_) {}
    }

    function updateButtons() {
        const pending = probingRevision === selectionRevision || generatingRevision !== null;
        builder("generate").disabled = pending || !selectionValid() || figures.length > 0;
        builder("download").disabled = exporting || visibleFigures().length === 0;
        const count = selected().profiles.length;
        filter("selection-status").textContent = `${count} profile${count === 1 ? "" : "s"} selected.`;
        announceSelection();
    }

    function clearFigures() {
        figures = [];
        builder("figure").replaceChildren();
        builder("export-status").textContent = "";
        displayRevision += 1;
    }

    function showFigures(items) {
        figures = items;
        builder("figure").replaceChildren(...figures.map(figureCard));
        filterFigureCards();
        displayRevision += 1;
    }

    const figureVisible = figure => (filter("metric").value === "all" || figure.metric === filter("metric").value)
        && (comparisonMode() === "horizons"
            || (filter("view").value === "all" ? figure.view !== "log_log_fit" : figure.view === filter("view").value));
    const visibleFigures = () => figures.filter(figureVisible);

    function filterFigureCards() {
        for (const card of builder("figure").children) {
            card.hidden = !figureVisible(card.dataset);
        }
    }

    function updateMetricControl() {
        const logLogFit = comparisonMode() === "regrets" && filter("view").value === "log_log_fit";
        const allRegrets = comparisonMode() === "horizons" || (comparisonMode() === "regrets" && !logLogFit);
        const entries = allRegrets ? [["all", "All regrets"]]
            : [["all", "All regrets"], ...catalog.metrics.map(metric => [metric.id, metric.label])];
        setOptions(filter("metric"), entries, allRegrets ? "all" : profileMetric, Boolean(currentContext));
        if (allRegrets) filter("metric").disabled = true;
    }

    function updateComparisonControls() {
        const mode = comparisonMode();
        const single = mode !== "profiles";
        const profileSelect = filter("profiles");
        const chosen = profiles()[0] || (profileSelect.options.length ? profileSelect.options[0].value : "");
        profileSelect.multiple = !single;
        profileSelect.size = single ? 1 : Math.min(Math.max(profileSelect.options.length, 2), 8);
        filter("profiles-label").textContent = single ? "Algorithm profile" : "Algorithm profiles";
        if (single) [...profileSelect.options].forEach(option => option.selected = !emptyUrlSelection && option.value === chosen);
        comparisonModes.forEach(value => {
            filter(`compare-${value}`).disabled = !catalog.contexts.some(context => context.comparison_modes.includes(value));
        });
        if (!["regrets", "horizons"].includes(mode)) {
            const metric = [...filter("metric").options].find(option => option.value === profileMetric)
                || [...filter("metric").options].find(option => option.value !== "all");
            if (metric) filter("metric").value = profileMetric = metric.value;
        }
    }

    async function selectionChanged() {
        const revision = ++selectionRevision;
        clearFigures();
        if (!selectionValid()) {
            builder("status").textContent = !onePlayer && filter("player").value === "all"
                ? "Select a numbered player to generate figures." : "No figures to display.";
            updateButtons();
            return;
        }
        probingRevision = revision;
        builder("status").textContent = "Checking generated figures…";
        updateButtons();
        try {
            const response = await fetch(form.dataset.cacheUrl, {
                method: "POST", body: new URLSearchParams(new FormData(form)),
            });
            const result = await response.json().catch(() => ({}));
            if (revision !== selectionRevision) return;
            if (!response.ok) throw new Error(result.error || "Could not check generated figures.");
            probingRevision = null;
            if (result.cached) {
                showFigures(result.figures);
                builder("status").textContent = "Cached figures loaded.";
            } else {
                builder("status").textContent = "No generated figures for this selection.";
            }
            updateButtons();
        } catch (error) {
            if (revision !== selectionRevision) return;
            probingRevision = null;
            builder("status").textContent = error.message;
            updateButtons();
        }
    }

    function updateProfiles() {
        currentContext = catalog.contexts.find(context => context.id === filter("context").value) || null;
        const feedback = filter("feedback").value;
        const feedbackProfiles = currentContext ? currentContext.profiles.filter(profile =>
            feedback === "both" || profile.feedback_mode === feedback) : [];
        if (onePlayer) {
            if (comparisonMode() === "actions") {
                if (filter("action").value && filter("action").value !== "all") selectedAction = filter("action").value;
                setOptions(filter("action"), [["all", "All actions"]], "all", false);
            } else {
                const actions = [...new Set(feedbackProfiles.flatMap(profile => profile.actions))]
                    .sort((left, right) => left - right).map(action => [String(action), `K=${action}`]);
                setOptions(filter("action"), actions, selectedAction, Boolean(currentContext));
                selectedAction = filter("action").value;
            }
        }
        const action = onePlayer ? filter("action").value : "";
        if (comparisonMode() === "horizons") {
            if (filter("horizon").value && filter("horizon").value !== "all") selectedHorizon = filter("horizon").value;
            setOptions(filter("horizon"), [["all", "All horizons"]], "all", false);
        } else {
            const horizons = [...new Set(feedbackProfiles.flatMap(profile => {
                if (!onePlayer) return profile.horizons;
                if (action === "all") return Object.values(profile.availability).flat();
                return profile.availability[action] || [];
            }))].sort((left, right) => left - right).map(horizon => [String(horizon), `T=${horizon}`]);
            setOptions(filter("horizon"), horizons, selectedHorizon, Boolean(currentContext));
            selectedHorizon = filter("horizon").value;
        }
        const horizon = filter("horizon").value;
        const entries = feedbackProfiles.filter(profile => {
            if (["regrets", "horizons"].includes(comparisonMode())
                    && !catalog.metrics.every(metric => profile.metrics.includes(metric.id))) return false;
            if (!onePlayer) return horizon === "all" ? profile.horizons.length >= 2 : profile.horizons.map(String).includes(horizon);
            if (comparisonMode() === "actions") {
                return Object.values(profile.availability).some(values => values.map(String).includes(horizon));
            }
            const values = profile.availability[action] || [];
            return horizon === "all" ? values.length >= 2 : values.map(String).includes(horizon);
        });
        const unique = new Map(entries.map(profile => [profile.id, profile.label]));
        compatibleProfileIds = [...unique.keys()];
        selectionKey = currentContext ? `${currentContext.id}:${feedback}:${comparisonMode()}:${action}:${horizon}` : "";
        const remembered = Array.isArray(selections.get(selectionKey)) ? selections.get(selectionKey) : [];
        const profileEntries = [...unique].sort(([left], [right]) => left.localeCompare(right));
        const completeProfileSet = ["regrets", "horizons"].includes(comparisonMode())
            && remembered.length > 1 && remembered.length === compatibleProfileIds.length
            && compatibleProfileIds.every(value => remembered.includes(value));
        if (["regrets", "horizons"].includes(comparisonMode()) && profileEntries.length) {
            profileEntries.unshift(["all", "All algorithm profiles"]);
        }
        filter("profiles").disabled = true;
        filter("profiles").multiple = comparisonMode() === "profiles";
        filter("profiles").replaceChildren(...profileEntries.map(([value, label]) => new Option(label, value)));
        const profileOptions = [...filter("profiles").options];
        if (comparisonMode() === "profiles") {
            profileOptions.forEach(option => option.selected = remembered.includes(option.value));
            if (!emptyUrlSelection && !filter("profiles").selectedOptions.length && compatibleProfileIds.length) {
                filter("profiles").value = compatibleProfileIds[0];
            }
        } else {
            filter("profiles").value = emptyUrlSelection ? ""
                : completeProfileSet ? "all"
                    : remembered.find(value => profileOptions.some(option => option.value === value))
                        || compatibleProfileIds[0] || "";
        }
        filter("profiles").disabled = filter("profiles").options.length === 0;

        const hasFilterContext = Boolean(currentContext);
        const horizonComparison = comparisonMode() === "horizons";
        const views = catalog.views.filter(view => view.id !== "horizon_scaling"
            && (comparisonMode() === "regrets" || view.id !== "log_log_fit"));
        const viewEntries = horizonComparison ? [["horizon_scaling", "Horizon scaling"]]
            : [["all", "Both views"], ...views.map(view => [view.id, view.label])];
        setOptions(filter("view"), viewEntries,
            horizonComparison ? "horizon_scaling" : comparisonMode() === "regrets" ? selectedView : standardView,
            hasFilterContext && !horizonComparison);
        selectedView = filter("view").value;
        updateMetricControl();
        updateComparisonControls();
        if (emptyUrlSelection) filter("profiles").selectedIndex = -1;
        remember();
        selectionChanged();
    }

    function updateContexts(preferred = {}) {
        const wantedFeedback = preferred.feedback || filter("feedback").value || "full_information";
        const wantedPlayer = onePlayer ? "0" : preferred.player || filter("player").value;
        const wantedContext = preferred.context || filter("context").value;
        currentContext = null;
        clearOptions(filter("feedback"), onePlayer ? filter("action") : filter("player"), filter("horizon"), filter("context"),
            filter("profiles"), filter("metric"), filter("view"));
        filter("profiles").size = 1;
        comparisonModes.forEach(value => filter(`compare-${value}`).disabled = true);

        let available = catalog.contexts.filter(context => context.scope === filter("scope").value
            && context.comparison_modes.includes(comparisonMode()));
        const feedbacks = ["full_information", "bandit"].filter(value =>
            available.some(context => context.feedback_modes.includes(value)));
        setOptions(filter("feedback"), [...feedbacks.map(value => [value, feedbackLabels[value]]), ["both", "Both"]],
            wantedFeedback);
        available = available.filter(context => filter("feedback").value === "both"
            || context.feedback_modes.includes(filter("feedback").value));
        if (!onePlayer) {
            const players = [...new Set(available.map(context => context.player))].sort((a, b) => a - b);
            const playerEntries = players.map(value => [String(value), `Player ${value}`]);
            if (players.length) playerEntries.push(["all", "All players"]);
            setOptions(filter("player"), playerEntries, wantedPlayer);
        }
        const contexts = catalog.contexts.filter(context => context.scope === filter("scope").value
            && context.comparison_modes.includes(comparisonMode())
            && (filter("feedback").value === "both" || context.feedback_modes.includes(filter("feedback").value))
            && (onePlayer || filter("player").value === "all"
                || String(context.player) === filter("player").value));
        const entries = contexts.map(context => [context.id, context.batch_label]);
        setOptions(filter("context"), entries, wantedContext);
        filter("batches").hidden = entries.length <= 1;
        updateProfiles();
    }

    function figureCard(result) {
        const card = document.createElement("article");
        card.className = "figure-card";
        card.dataset.filename = result.filename;
        card.dataset.metric = result.metric;
        card.dataset.view = result.view;
        const open = document.createElement("button");
        open.type = "button";
        open.className = "figure-open";
        const title = document.createElement("span");
        title.textContent = result.title;
        const image = document.createElement("img");
        image.src = result.url;
        image.alt = result.title;
        open.append(title, image);
        const actions = document.createElement("div");
        actions.className = "figure-actions";
        for (const [label, url, filename] of [["Download PDF", result.pdf_url, result.pdf_filename],
                ["Download PNG preview", result.url, result.filename]]) {
            const link = document.createElement("a");
            link.textContent = label;
            link.href = url;
            link.download = filename;
            actions.append(link);
        }
        card.append(open, actions);
        return card;
    }

    form.addEventListener("submit", async event => {
        event.preventDefault();
        if (!selectionValid() || probingRevision === selectionRevision || generatingRevision !== null || figures.length) return;
        const revision = selectionRevision;
        generatingRevision = revision;
        builder("status").textContent = "Generating figures…";
        updateButtons();
        try {
            const response = await fetch(form.getAttribute("action"), {method: "POST", body: new URLSearchParams(new FormData(form))});
            const result = await response.json().catch(() => ({}));
            if (revision !== selectionRevision) return;
            if (!response.ok) throw new Error(result.error || "Figure generation failed.");
            showFigures(result.figures);
            builder("status").textContent = `${figures.length} figure${figures.length === 1 ? "" : "s"} generated.`;
        } catch (error) {
            if (revision === selectionRevision) builder("status").textContent = error.message;
        } finally {
            if (generatingRevision === revision) {
                generatingRevision = null;
                updateButtons();
            }
        }
    });

    builder("download").addEventListener("click", async () => {
        const displayedFigures = visibleFigures();
        if (exporting || !displayedFigures.length) return;
        exporting = true;
        const revision = displayRevision;
        builder("download").disabled = true;
        builder("export-status").textContent = `Combining ${displayedFigures.length} PDFs…`;
        const body = new URLSearchParams({_csrf_token: form.elements._csrf_token.value, mode: "figure_builder"});
        displayedFigures.forEach(figure => body.append("filenames", figure.pdf_filename));
        try {
            const response = await fetch(form.dataset.exportUrl, {method: "POST", body});
            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.error || "PDF download failed.");
            }
            const blob = await response.blob();
            if (revision !== displayRevision) return;
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = "filtered-regret-figures.pdf";
            document.body.append(link);
            link.click();
            link.remove();
            window.setTimeout(() => URL.revokeObjectURL(url), 60000);
            builder("export-status").textContent = "Merged PDF downloaded.";
        } catch (error) {
            if (revision === displayRevision) builder("export-status").textContent = error.message;
        } finally {
            exporting = false;
            builder("download").disabled = !visibleFigures().length;
        }
    });

    filter("scope").addEventListener("change", () => { remember(); updateContexts(); });
    filter("feedback").addEventListener("change", () => { remember(); updateContexts(); });
    if (onePlayer) {
        filter("action").addEventListener("change", () => { remember(); selectedAction = filter("action").value; updateProfiles(); });
    } else {
        filter("player").addEventListener("change", () => { remember(); updateContexts(); });
    }
    filter("horizon").addEventListener("change", () => {
        remember();
        selectedHorizon = filter("horizon").value;
        updateProfiles();
    });
    filter("context").addEventListener("change", () => { remember(); updateProfiles(); });
    filter("metric").addEventListener("change", () => {
        profileMetric = filter("metric").value;
        filterFigureCards();
        displayRevision += 1;
        updateButtons();
    });
    filter("view").addEventListener("change", () => {
        selectedView = filter("view").value;
        if (selectedView !== "log_log_fit") standardView = selectedView;
        updateMetricControl();
        filterFigureCards();
        displayRevision += 1;
        updateButtons();
    });
    for (const mode of comparisonModes) {
        filter(`compare-${mode}`).addEventListener("change", () => {
            if (!filter(`compare-${mode}`).checked) return;
            remember();
            if (mode === "regrets" && !filter("metric").disabled) profileMetric = filter("metric").value;
            updateContexts();
        });
    }
    filter("profiles").addEventListener("change", () => { remember(); selectionChanged(); });

    fetch(filters.dataset.optionsUrl).then(async response => {
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Could not load available profiles.");
        catalog = result;
        const scopes = new Map(catalog.contexts.map(context => [context.scope, context.scope_label]));
        setOptions(filter("scope"), [...scopes], saved.scope || "");
        if (filter("scope").disabled) {
            builder("status").textContent = "No figures to display.";
            updateButtons();
        } else {
            updateContexts(saved);
        }
    }).catch(error => {
        builder("status").textContent = error.message;
        updateButtons();
    });
})();
