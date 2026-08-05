"use strict";

const playerCountInput = document.getElementById("custom-player-count");
const actionCountContainer = document.getElementById("custom-action-counts");
const payoffStructureSelect = document.getElementById("custom-payoff-structure");
const payoffStructureHint = document.getElementById("custom-payoff-structure-hint");

function renderActionCounts() {
    if (!playerCountInput || !actionCountContainer) {
        return;
    }
    const previous = [...actionCountContainer.querySelectorAll("input")].map((input) => input.value);
    let initial = [];
    try {
        initial = JSON.parse(actionCountContainer.dataset.initialCounts || "[]");
    } catch (error) {
        console.warn("Could not restore action counts", error);
    }
    const playerCount = Math.max(2, Number(playerCountInput.value) || 2);
    const symmetricZeroSum = payoffStructureSelect?.value === "zero_sum";
    const fields = [];
    const fieldCount = symmetricZeroSum ? 1 : playerCount;
    for (let player = 0; player < fieldCount; player += 1) {
        const field = document.createElement("div");
        field.className = "field";
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.id = `custom-player-${player}-actions`;
        input.name = "action_counts";
        input.type = "number";
        input.min = "1";
        input.max = "100";
        input.required = true;
        input.value = previous[player] || initial[player] || "2";
        label.htmlFor = input.id;
        label.textContent = symmetricZeroSum
            ? "Actions per player"
            : `Player ${player} actions`;
        field.append(label, input);
        fields.push(field);
    }
    actionCountContainer.replaceChildren(...fields);
    delete actionCountContainer.dataset.initialCounts;
}

function updatePayoffStructure() {
    const zeroSum = payoffStructureSelect?.value === "zero_sum";
    if (playerCountInput) {
        if (zeroSum) {
            playerCountInput.value = "2";
        }
        playerCountInput.readOnly = zeroSum;
    }
    if (payoffStructureHint) {
        payoffStructureHint.textContent = zeroSum
            ? "Both players share one action set. Centered payoffs satisfy A = −Aᵀ; displayed payoffs use u and 1 − u."
            : "General-sum payoffs are sampled independently for every player.";
    }
    renderActionCounts();
}

playerCountInput?.addEventListener("input", renderActionCounts);
payoffStructureSelect?.addEventListener("change", updatePayoffStructure);
updatePayoffStructure();

const payoffInspector = document.getElementById("payoff-inspector");
const payoffPlayerSelect = document.getElementById("payoff-player");
const payoffRowPlayerSelect = document.getElementById("payoff-row-player");
const payoffColumnPlayerSelect = document.getElementById("payoff-column-player");
const payoffFixedActionContainer = document.getElementById("payoff-fixed-actions");
const payoffSliceStatus = document.getElementById("payoff-slice-status");
const payoffTableScroll = document.getElementById("payoff-table-scroll");
let payoffSliceRequest = 0;

function payoffActionCounts() {
    if (!payoffInspector) {
        return [];
    }
    try {
        return JSON.parse(payoffInspector.dataset.actionCounts);
    } catch (error) {
        console.warn("Could not read payoff tensor action counts", error);
        return [];
    }
}

function actionSelect(player, count, value) {
    const select = document.createElement("select");
    select.dataset.player = String(player);
    for (let action = 0; action < count; action += 1) {
        const option = document.createElement("option");
        option.value = String(action);
        option.textContent = `Action ${action}`;
        select.append(option);
    }
    select.value = value;
    return select;
}

function renderFixedActionControls() {
    if (!payoffFixedActionContainer) {
        return;
    }
    const previousValues = new Map(
        [...payoffFixedActionContainer.querySelectorAll("select")].map((select) => [Number(select.dataset.player), select.value])
    );
    const rowPlayer = Number(payoffRowPlayerSelect.value);
    const columnPlayer = Number(payoffColumnPlayerSelect.value);
    const fields = [];
    payoffActionCounts().forEach((count, player) => {
        if (player === rowPlayer || player === columnPlayer) {
            return;
        }
        const field = document.createElement("div");
        const label = document.createElement("label");
        const select = actionSelect(player, count, previousValues.get(player) || "0");
        field.className = "field";
        select.id = `payoff-fixed-player-${player}`;
        label.htmlFor = select.id;
        label.textContent = `Player ${player} fixed action`;
        select.addEventListener("change", loadPayoffSlice);
        field.append(label, select);
        fields.push(field);
    });
    payoffFixedActionContainer.replaceChildren(...fields);
}

function keepPayoffAxesDistinct(changedSelect) {
    if (payoffRowPlayerSelect.value !== payoffColumnPlayerSelect.value) {
        return;
    }
    const otherSelect = changedSelect === payoffRowPlayerSelect ? payoffColumnPlayerSelect : payoffRowPlayerSelect;
    const replacement = [...otherSelect.options].find((option) => option.value !== changedSelect.value);
    otherSelect.value = replacement.value;
}

function payoffSliceParameters() {
    const parameters = new URLSearchParams({
        payoff_player: payoffPlayerSelect.value,
        row_player: payoffRowPlayerSelect.value,
        column_player: payoffColumnPlayerSelect.value,
    });
    const fixedActions = Array(payoffActionCounts().length).fill("0");
    payoffFixedActionContainer.querySelectorAll("select").forEach((select) => {
        fixedActions[Number(select.dataset.player)] = select.value;
    });
    fixedActions.forEach((action) => parameters.append("fixed_action", action));
    return parameters;
}

function renderPayoffTable(payload) {
    const table = document.createElement("table");
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    const corner = document.createElement("th");
    corner.textContent = `P${payload.row_player} ↓ / P${payload.column_player} →`;
    headRow.append(corner);
    payload.values[0].forEach((_, columnAction) => {
        const heading = document.createElement("th");
        heading.textContent = `Action ${columnAction}`;
        headRow.append(heading);
    });
    head.append(headRow);

    const body = document.createElement("tbody");
    for (let rowAction = 0; rowAction < payload.values.length; rowAction += 1) {
        const row = document.createElement("tr");
        const heading = document.createElement("th");
        heading.textContent = `Action ${rowAction}`;
        row.append(heading);
        payload.values[rowAction].forEach((rawValue) => {
            const value = Number(rawValue);
            const cell = document.createElement("td");
            const intensity = Math.max(0, Math.min(1, value));
            cell.className = "payoff-cell";
            cell.textContent = value.toFixed(4);
            cell.title = String(value);
            cell.style.backgroundColor = `rgba(37, 99, 235, ${0.08 + 0.82 * intensity})`;
            cell.style.color = intensity > 0.68 ? "#ffffff" : "#17201c";
            row.append(cell);
        });
        body.append(row);
    }
    table.className = "payoff-table";
    table.append(head, body);
    payoffTableScroll.replaceChildren(table);
    payoffTableScroll.setAttribute("aria-busy", "false");

    const fixed = payload.fixed_actions
        .map((action, player) => player === payload.row_player || player === payload.column_player ? null : `P${player}=A${action}`)
        .filter(Boolean)
        .join(", ");
    payoffSliceStatus.textContent = `Player ${payload.payoff_player} payoff${fixed ? ` · fixed ${fixed}` : ""}`;
}

async function loadPayoffSlice() {
    if (!payoffInspector || !payoffTableScroll) {
        return;
    }
    const request = ++payoffSliceRequest;
    payoffSliceStatus.textContent = "Loading payoff slice…";
    payoffTableScroll.replaceChildren();
    payoffTableScroll.setAttribute("aria-busy", "true");
    try {
        const response = await fetch(`${payoffInspector.dataset.sliceUrl}?${payoffSliceParameters()}`);
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || "Could not load payoff slice");
        }
        if (request === payoffSliceRequest) {
            renderPayoffTable(payload);
        }
    } catch (error) {
        if (request === payoffSliceRequest) {
            payoffSliceStatus.textContent = error.message;
            payoffTableScroll.setAttribute("aria-busy", "false");
        }
    }
}

function installPayoffInspector() {
    if (!payoffInspector) {
        return;
    }
    payoffPlayerSelect.addEventListener("change", loadPayoffSlice);
    [payoffRowPlayerSelect, payoffColumnPlayerSelect].forEach((select) => {
        select.addEventListener("change", () => {
            keepPayoffAxesDistinct(select);
            renderFixedActionControls();
            loadPayoffSlice();
        });
    });
    renderFixedActionControls();
    loadPayoffSlice();
}

installPayoffInspector();

document.querySelectorAll("img[data-heatmap-source]").forEach((image) => {
    setHeatmapSource(image, image.dataset.heatmapSource);
});
