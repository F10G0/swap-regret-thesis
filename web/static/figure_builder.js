/* One shared result selection; cache probes never render figures. */
(() => {
    const form = document.getElementById("figure-builder");
    const filters = document.getElementById("result-filters");
    if (!form || !filters) return;
    const filter = name => document.getElementById(`filter-${name}`);
    const builder = name => document.getElementById(`builder-${name}`);
    const onePlayer = form.elements.mode.value === "adversarial";
    const storageKey = `swap-regret-shared-filters-${form.elements.mode.value}`;
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(storageKey)) || {}; } catch (_) {}
    const selections = new Map(Object.entries(saved.selections || {}));
    const feedbackLabels = {full_information: "Full information", bandit: "Bandit"};
    let catalog = {contexts: [], metrics: [], views: []};
    let currentContext = null, selectionKey = "";
    let selectionRevision = 0, displayRevision = 0;
    let probingRevision = null, generatingRevision = null;
    let exporting = false, figures = [];
    let profileMetric = saved.profileMetric || "external";
    let selectedView = saved.view || "all";
    let selectedAction = saved.selectedAction || "";
    const comparisonModes = onePlayer ? ["regrets", "profiles", "actions"] : ["regrets", "profiles"];

    if (comparisonModes.includes(saved.comparisonMode)) {
        filter(`compare-${saved.comparisonMode}`).checked = true;
    }

    const comparisonMode = () => filters.querySelector('input[name="comparison_mode"]:checked').value;
    const profiles = () => [...filter("profiles").selectedOptions].map(option => option.value);
    const selectionValid = () => currentContext && profiles().length > 0
        && (comparisonMode() === "profiles" || profiles().length === 1)
        && (comparisonMode() === "regrets" || filter("metric").value !== "all")
        && (!onePlayer || comparisonMode() === "actions" || filter("action").value !== "all");
    const selected = () => ({
        scope: filter("scope").value, feedback: filter("feedback").value,
        player: onePlayer ? "0" : filter("player").value,
        action: onePlayer ? filter("action").value : null,
        metric: filter("metric").value, view: filter("view").value, profiles: profiles(),
        comparisonMode: comparisonMode(), resultKeys: currentContext ? currentContext.result_keys : [],
    });

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
                profileMetric, selectedAction, selections: Object.fromEntries(selections)}));
        } catch (_) {}
    }

    function updateButtons() {
        const pending = probingRevision === selectionRevision || generatingRevision !== null;
        builder("generate").disabled = pending || !selectionValid() || figures.length > 0;
        builder("download").disabled = exporting || figures.length === 0;
        const count = profiles().length;
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
        displayRevision += 1;
    }

    function updateComparisonControls() {
        const mode = comparisonMode();
        const single = mode !== "profiles";
        const profileSelect = filter("profiles");
        const chosen = profiles()[0] || (profileSelect.options.length ? profileSelect.options[0].value : "");
        profileSelect.multiple = !single;
        profileSelect.size = single ? 1 : Math.min(Math.max(profileSelect.options.length, 2), 8);
        filter("profiles-label").textContent = single ? "Algorithm profile" : "Algorithm profiles";
        if (single) [...profileSelect.options].forEach(option => option.selected = option.value === chosen);
        const comparisonEnabled = profileSelect.options.length > 0;
        comparisonModes.forEach(value => filter(`compare-${value}`).disabled = !comparisonEnabled);
        filter("metric").disabled = !filter("metric").options.length || mode === "regrets";
        if (mode === "regrets" && filter("metric").options.length) {
            filter("metric").value = "all";
        } else {
            const metric = [...filter("metric").options].find(option => option.value === profileMetric)
                || [...filter("metric").options].find(option => option.value !== "all");
            if (metric) filter("metric").value = profileMetric = metric.value;
        }
    }

    async function selectionChanged() {
        const revision = ++selectionRevision;
        clearFigures();
        if (!selectionValid()) {
            builder("status").textContent = "No figures to display.";
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
        if (onePlayer) {
            if (comparisonMode() === "actions") {
                if (filter("action").value && filter("action").value !== "all") selectedAction = filter("action").value;
                setOptions(filter("action"), [["all", "All actions"]], "all", false);
            } else {
                const actions = currentContext ? currentContext.actions.map(action => [String(action), `K=${action}`]) : [];
                setOptions(filter("action"), actions, selectedAction, Boolean(currentContext));
                selectedAction = filter("action").value;
            }
        }
        const action = onePlayer ? filter("action").value : "";
        const entries = currentContext ? currentContext.profiles.filter(profile => !onePlayer
            || action === "all" || profile.actions.map(String).includes(action)) : [];
        const unique = new Map(entries.map(profile => [profile.id, profile.label]));
        selectionKey = currentContext ? (onePlayer ? `${currentContext.id}:${action}` : currentContext.id) : "";
        const remembered = Array.isArray(selections.get(selectionKey)) ? selections.get(selectionKey) : [];
        filter("profiles").disabled = true;
        filter("profiles").replaceChildren(...[...unique].sort(([left], [right]) => left.localeCompare(right))
            .map(([value, label]) => new Option(label, value)));
        [...filter("profiles").options].forEach(option => option.selected = remembered.includes(option.value));
        filter("profiles").disabled = filter("profiles").options.length === 0;

        const hasFilterContext = Boolean(currentContext);
        const metrics = comparisonMode() === "regrets" ? [["all", "All regrets"]]
            : catalog.metrics.map(metric => [metric.id, metric.label]);
        setOptions(filter("metric"), metrics, comparisonMode() === "regrets" ? "all" : profileMetric, hasFilterContext);
        setOptions(filter("view"), [["all", "Both views"], ...catalog.views.map(view => [view.id, view.label])],
            selectedView, hasFilterContext);
        updateComparisonControls();
        remember();
        selectionChanged();
    }

    function updateContexts(preferred = {}) {
        const wantedFeedback = preferred.feedback || filter("feedback").value || "full_information";
        const wantedPlayer = onePlayer ? "0" : preferred.player || filter("player").value;
        const wantedContext = preferred.context || filter("context").value;
        currentContext = null;
        clearOptions(filter("feedback"), onePlayer ? filter("action") : filter("player"), filter("context"),
            filter("profiles"), filter("metric"), filter("view"));
        filter("profiles").size = 1;
        comparisonModes.forEach(value => filter(`compare-${value}`).disabled = true);

        let available = catalog.contexts.filter(context => context.scope === filter("scope").value);
        const feedbacks = [...new Set(available.map(context => context.feedback_mode))];
        setOptions(filter("feedback"), feedbacks.map(value => [value, feedbackLabels[value] || value]), wantedFeedback);
        available = available.filter(context => context.feedback_mode === filter("feedback").value);
        if (!onePlayer) {
            const players = [...new Set(available.map(context => context.player))].sort((a, b) => a - b);
            setOptions(filter("player"), players.map(value => [String(value), `Player ${value}`]), wantedPlayer);
        }
        const contexts = catalog.contexts.filter(context => context.scope === filter("scope").value
            && context.feedback_mode === filter("feedback").value
            && (onePlayer || String(context.player) === filter("player").value));
        const entries = contexts.map(context => [context.id, context.batch_label]);
        setOptions(filter("context"), entries, wantedContext);
        filter("batches").hidden = entries.length <= 1;
        updateProfiles();
    }

    function figureCard(result) {
        const card = document.createElement("article");
        card.className = "figure-card";
        card.dataset.filename = result.filename;
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
            const response = await fetch(form.action, {method: "POST", body: new URLSearchParams(new FormData(form))});
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
        if (exporting || !figures.length) return;
        exporting = true;
        const revision = displayRevision;
        builder("download").disabled = true;
        builder("export-status").textContent = `Combining ${figures.length} PDFs…`;
        const body = new URLSearchParams({_csrf_token: form.elements._csrf_token.value, mode: "figure_builder"});
        figures.forEach(figure => body.append("filenames", figure.pdf_filename));
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
            builder("download").disabled = !figures.length;
        }
    });

    filter("scope").addEventListener("change", () => { remember(); updateContexts(); });
    filter("feedback").addEventListener("change", () => { remember(); updateContexts(); });
    if (onePlayer) {
        filter("action").addEventListener("change", () => { remember(); selectedAction = filter("action").value; updateProfiles(); });
    } else {
        filter("player").addEventListener("change", () => { remember(); updateContexts(); });
    }
    filter("context").addEventListener("change", () => { remember(); updateProfiles(); });
    filter("metric").addEventListener("change", () => {
        profileMetric = filter("metric").value;
        selectionChanged();
    });
    filter("view").addEventListener("change", () => {
        selectedView = filter("view").value;
        selectionChanged();
    });
    for (const mode of comparisonModes) {
        filter(`compare-${mode}`).addEventListener("change", () => {
            if (!filter(`compare-${mode}`).checked) return;
            remember();
            if (mode === "regrets" && !filter("metric").disabled) profileMetric = filter("metric").value;
            updateProfiles();
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
