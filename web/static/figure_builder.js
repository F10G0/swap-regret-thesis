/* One shared result selection; the backend owns compatible replicate groups. */
(() => {
    const form = document.getElementById("figure-builder");
    const filters = document.getElementById("result-filters");
    if (!form || !filters) return;
    const filter = (name) => document.getElementById(`filter-${name}`);
    const builder = (name) => document.getElementById(`builder-${name}`);
    const storageKey = `swap-regret-shared-filters-${form.elements.mode.value}`;
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(storageKey)) || {}; } catch (_) {}
    const selections = new Map(Object.entries(saved.selections || {}));
    const feedbackLabels = {full_information: "Full information", bandit: "Bandit"};
    let catalog = {contexts: [], scaling: []};
    let currentContext = null, selectionKey = "";
    let generationRevision = 0, displayRevision = 0;
    let busy = false, exporting = false;
    let figures = [];
    const profiles = () => [...filter("profiles").selectedOptions].map(option => option.value);
    const selected = () => ({
        scope: filter("scope").value, feedback: filter("feedback").value, player: filter("player").value,
        metric: filter("metric").value, view: filter("view").value, profiles: profiles(),
        resultKeys: currentContext ? currentContext.result_keys : [],
    });
    const visibleFigures = () => figures.filter(figure =>
        (filter("metric").value === "all" || figure.metric === filter("metric").value)
        && (filter("view").value === "all" || figure.view === filter("view").value));
    const canGenerate = () => currentContext && profiles().length > 0;

    function options(select, entries, preferred = select.value) {
        select.replaceChildren(...entries.map(([value, label]) => new Option(label, String(value))));
        if (entries.some(([value]) => String(value) === preferred)) select.value = preferred;
    }

    function remember() {
        if (selectionKey) selections.set(selectionKey, profiles());
    }

    function announceSelection() {
        const state = selected();
        document.dispatchEvent(new CustomEvent("results-filter-change", {detail: state}));
        try {
            localStorage.setItem(storageKey, JSON.stringify({...state, context: filter("context").value,
                selections: Object.fromEntries(selections)}));
        } catch (_) {}
    }

    function updateDisplay() {
        displayRevision += 1;
        builder("export-status").textContent = "";
        const visible = visibleFigures();
        const visibleNames = new Set(visible.map(figure => figure.filename));
        [...builder("figure").children].forEach(card => card.hidden = !visibleNames.has(card.dataset.filename));
        builder("download").disabled = exporting || !visible.length;
        builder("generate").disabled = busy || !canGenerate();
        const count = profiles().length;
        filter("selection-status").textContent = `${count} profile${count === 1 ? "" : "s"} selected.`;
        builder("status").textContent = figures.length
            ? `${figures.length} figures generated · ${visible.length} visible. The merged PDF follows the displayed order.`
            : !count ? "Select at least one algorithm profile. Empty selections show no results."
            : !currentContext ? "These are action-space scaling results; no round-by-round figure collection is available."
            : busy ? "Rendering all three regrets and both views…"
            : "Generate all six figures for the shared selection.";
        announceSelection();
    }

    function invalidate() {
        generationRevision += 1;
        figures = [];
        builder("figure").replaceChildren();
        updateDisplay();
    }

    function updateProfiles() {
        currentContext = catalog.contexts.find(context => context.id === filter("context").value) || null;
        const scaling = (catalog.scaling || []).filter(context => context.scope === filter("scope").value
            && context.feedback_mode === filter("feedback").value);
        const entries = currentContext ? currentContext.profiles : scaling.flatMap(context => context.profiles);
        const unique = new Map(entries.map(profile => [profile.id, profile.label]));
        selectionKey = currentContext ? currentContext.id : `${filter("scope").value}/${filter("feedback").value}/scaling`;
        const remembered = Array.isArray(selections.get(selectionKey)) ? selections.get(selectionKey) : [];
        options(filter("profiles"), [...unique].sort(([left], [right]) => left.localeCompare(right)));
        [...filter("profiles").options].forEach(option => option.selected = remembered.includes(option.value));
        invalidate();
        if (!entries.length) builder("status").textContent = "No compatible saved results are available.";
    }

    function updateContexts(preferred = {}) {
        let available = [...catalog.contexts, ...(catalog.scaling || [])].filter(context => context.scope === filter("scope").value);
        const feedbacks = [...new Set(available.map(context => context.feedback_mode))];
        options(filter("feedback"), feedbacks.map(value => [value, feedbackLabels[value] || value]),
            preferred.feedback || filter("feedback").value || "full_information");
        available = available.filter(context => context.feedback_mode === filter("feedback").value);
        const players = [...new Set(available.map(context => context.player))].sort((a, b) => a - b);
        options(filter("player"), players.map(value => [String(value), `Player ${value}`]), preferred.player || filter("player").value);
        const contexts = catalog.contexts.filter(context => context.scope === filter("scope").value
            && context.feedback_mode === filter("feedback").value && String(context.player) === filter("player").value);
        const entries = contexts.map(context => [context.id, context.batch_label]);
        if (available.some(context => !context.id)) entries.push(["scaling", "Action-space scaling results"]);
        options(filter("context"), entries, preferred.context || filter("context").value);
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
        if (busy || !canGenerate()) return;
        busy = true;
        const revision = generationRevision;
        builder("generate").disabled = true;
        builder("status").textContent = "Rendering all three regrets and both views…";
        try {
            const response = await fetch(form.action, {method: "POST", body: new URLSearchParams(new FormData(form))});
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(result.error || "Figure generation failed.");
            if (revision !== generationRevision) return;
            figures = result.figures;
            builder("figure").replaceChildren(...figures.map(figureCard));
            updateDisplay();
        } catch (error) {
            if (revision === generationRevision) builder("status").textContent = error.message;
        } finally {
            busy = false;
            builder("generate").disabled = !canGenerate();
        }
    });

    builder("download").addEventListener("click", async () => {
        const visible = visibleFigures();
        if (exporting || !visible.length) return;
        exporting = true;
        const revision = displayRevision;
        builder("download").disabled = true;
        builder("export-status").textContent = `Combining ${visible.length} PDFs…`;
        const body = new URLSearchParams({_csrf_token: form.elements._csrf_token.value, mode: "figure_builder"});
        visible.forEach(figure => body.append("filenames", figure.pdf_filename));
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

    for (const name of ["scope", "feedback", "player"]) {
        filter(name).addEventListener("change", () => { remember(); updateContexts(); });
    }
    filter("context").addEventListener("change", () => { remember(); updateProfiles(); });
    for (const name of ["metric", "view"]) filter(name).addEventListener("change", updateDisplay);
    filter("profiles").addEventListener("change", () => { remember(); invalidate(); });
    for (const [name, selected] of [["select-all", true], ["clear-all", false]]) {
        filter(name).addEventListener("click", () => {
            [...filter("profiles").options].forEach(option => option.selected = selected);
            remember();
            invalidate();
        });
    }

    fetch(filters.dataset.optionsUrl).then(async response => {
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Could not load available profiles.");
        catalog = result;
        options(filter("metric"), [["all", "All regrets"], ...catalog.metrics.map(metric => [metric.id, metric.label])], saved.metric || "all");
        options(filter("view"), [["all", "Both views"], ...catalog.views.map(view => [view.id, view.label])], saved.view || "all");
        const scopes = new Map([...catalog.contexts, ...(catalog.scaling || [])].map(context => [context.scope, context.scope_label]));
        options(filter("scope"), [...scopes], saved.scope || "");
        updateContexts(saved);
    }).catch(error => { builder("status").textContent = error.message; });
})();
