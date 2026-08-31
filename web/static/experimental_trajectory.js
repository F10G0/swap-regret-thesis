"use strict";

const trajectoryDataElement = document.getElementById(
    "experimental-trajectory-data",
);
const trajectoryData = trajectoryDataElement
    ? JSON.parse(trajectoryDataElement.textContent)
    : {
        trajectoryComparisonCandidates: [],
        trajectoryComparisonUrl: "",
        gamePresentations: {},
    };
const finalIntervalSegmentsStorageKey =
    "swap-regret-final-interval-segments";
const focusFinalIntervalStorageKey =
    "swap-regret-focus-final-interval";
const trajectoryComparisonViewStorageKey =
    "swap-regret-trajectory-comparison-view";

function gamePresentation(game) {
    return trajectoryData.gamePresentations[game] || {
        label: game,
        description: "",
    };
}

function finalIntervalSegments() {
    const input = element("final-interval-segments");
    if (!input) {
        return 10;
    }
    const minimum = Number(input.min) || 1;
    const maximum = Number(input.max) || 50;
    const fallback = Number(input.defaultValue) || 10;
    const parsed = Number(input.value);
    const value = input.value !== "" && Number.isFinite(parsed)
        ? Math.trunc(parsed)
        : fallback;
    const normalized = Math.min(maximum, Math.max(minimum, value));
    input.value = normalized;
    return normalized;
}

function saveFinalIntervalSegments() {
    saveLocalValue(finalIntervalSegmentsStorageKey, String(finalIntervalSegments()), "final-interval segments");
}

function restoreFinalIntervalSegments() {
    const input = element("final-interval-segments");
    if (!input) {
        return;
    }
    const stored = restoreLocalValue(finalIntervalSegmentsStorageKey, "final-interval segments");
    if (stored !== null) {
        input.value = stored;
    }
    finalIntervalSegments();
}

function focusFinalInterval() {
    const input = element("focus-final-interval");
    return input ? input.checked : false;
}

function saveFocusFinalInterval() {
    saveLocalValue(focusFinalIntervalStorageKey, focusFinalInterval() ? "1" : "0", "final-interval focus preference");
}

function restoreFocusFinalInterval() {
    const input = element("focus-final-interval");
    if (!input) {
        return;
    }
    input.checked = restoreLocalValue(focusFinalIntervalStorageKey, "final-interval focus preference") === "1";
}

function trajectoryComparisonView() {
    const select = element("trajectory-comparison-view");
    return select ? select.value : "geometry";
}

function saveTrajectoryComparisonView() {
    saveLocalValue(trajectoryComparisonViewStorageKey, trajectoryComparisonView(), "trajectory-comparison view");
}

function restoreTrajectoryComparisonView() {
    const select = element("trajectory-comparison-view");
    if (!select) {
        return;
    }
    const stored = restoreLocalValue(trajectoryComparisonViewStorageKey, "trajectory-comparison view");
    if ([...select.options].some((option) => option.value === stored)) {
        select.value = stored;
    }
}

const trajectoryComparisonCandidates = new Map(
    (trajectoryData.trajectoryComparisonCandidates || []).map(
        (candidate) => [candidate.group_id, candidate],
    ),
);
const pendingTrajectoryComparisonMembers = new Set();
let renderedTrajectoryComparisonSignature = null;
let trajectoryComparisonRequestId = 0;

function trajectoryCompatibilitySignature(candidate) {
    return JSON.stringify(candidate.compatibility_key);
}

function currentTrajectoryComparisonSignature() {
    return JSON.stringify({
        members: [...pendingTrajectoryComparisonMembers].sort(),
        finalIntervalSegments: finalIntervalSegments(),
        focusFinalInterval: focusFinalInterval(),
        comparisonView: trajectoryComparisonView(),
    });
}

function updateTrajectoryComparisonDirtyState() {
    const dirty = element("trajectory-comparison-dirty");
    if (!dirty) {
        return;
    }
    if (pendingTrajectoryComparisonMembers.size === 0) {
        dirty.hidden = true;
        return;
    }
    const changed = currentTrajectoryComparisonSignature()
        !== renderedTrajectoryComparisonSignature;
    dirty.hidden = !changed;
    dirty.textContent = changed ? "Changes not rendered" : "Up to date";
}

function trajectoryCandidateLabel(candidate) {
    const presentation = gamePresentation(candidate.game).label;
    const replicates = candidate.replicate_indices.join(", ");
    return `${presentation} · ${candidate.label} · seed ${candidate.seed} · replicates ${replicates}`;
}

function renderTrajectoryComparisonCandidates() {
    const select = element("trajectory-comparison-candidate");
    if (!select) {
        return;
    }
    const selected = [...pendingTrajectoryComparisonMembers];
    const compatibility = selected.length
        ? trajectoryCompatibilitySignature(
            trajectoryComparisonCandidates.get(selected[0]),
        )
        : null;
    const options = [...trajectoryComparisonCandidates.values()].map(
        (candidate) => {
            const option = document.createElement("option");
            option.value = candidate.group_id;
            option.textContent = trajectoryCandidateLabel(candidate);
            option.disabled = pendingTrajectoryComparisonMembers.has(
                candidate.group_id,
            ) || (
                compatibility !== null
                && trajectoryCompatibilitySignature(candidate)
                    !== compatibility
            );
            return option;
        },
    );
    select.replaceChildren(...options);
    const available = options.find((option) => !option.disabled);
    if (available) {
        select.value = available.value;
    }
    element("trajectory-comparison-add").disabled = !available;
}

function renderPendingTrajectoryComparisonMembers() {
    const list = element("trajectory-comparison-selected");
    if (!list) {
        return;
    }
    const members = [...pendingTrajectoryComparisonMembers].map(
        (groupId) => trajectoryComparisonCandidates.get(groupId),
    );
    list.replaceChildren(...members.map((member) => {
        const item = document.createElement("li");
        const label = document.createElement("span");
        const remove = document.createElement("button");
        label.textContent = member.label;
        remove.type = "button";
        remove.dataset.groupId = member.group_id;
        remove.textContent = "Remove";
        item.append(label, remove);
        return item;
    }));
    const context = element("trajectory-comparison-compatibility");
    if (context) {
        if (members.length) {
            const first = members[0];
            context.textContent = `${gamePresentation(first.game).label} · ${first.feedback_mode} · ${first.regret_evaluation} · horizon ${first.horizon} · base seed ${first.seed} · replicate indices ${first.replicate_indices.join(", ")}`;
        } else {
            context.textContent =
                "Add an experiment to establish the compatibility context.";
        }
    }
    element("trajectory-comparison-generate").disabled =
        members.length === 0;
    renderTrajectoryComparisonCandidates();
    updateTrajectoryComparisonDirtyState();
}

function addTrajectoryComparisonMember() {
    const select = element("trajectory-comparison-candidate");
    const groupId = select ? select.value : "";
    if (!groupId || !trajectoryComparisonCandidates.has(groupId)) {
        return;
    }
    pendingTrajectoryComparisonMembers.add(groupId);
    renderPendingTrajectoryComparisonMembers();
}

function renderAuthoritativeComparisonMembers(members) {
    const list = element("trajectory-comparison-rendered-members");
    if (!list) {
        return;
    }
    list.replaceChildren(...members.map((member) => {
        const item = document.createElement("li");
        const swatch = document.createElement("span");
        const description = document.createElement("span");
        swatch.className = "trajectory-member-color";
        swatch.style.backgroundColor = member.color;
        description.textContent = `${member.label} · replicates ${member.replicate_indices.join(", ")}`;
        item.append(swatch, description);
        return item;
    }));
}

async function generateTrajectoryComparison() {
    if (
        !trajectoryData.trajectoryComparisonUrl
        || pendingTrajectoryComparisonMembers.size === 0
    ) {
        return;
    }
    const requestId = ++trajectoryComparisonRequestId;
    const requestSignature = currentTrajectoryComparisonSignature();
    const source = new URL(
        trajectoryData.trajectoryComparisonUrl,
        window.location.href,
    );
    [...pendingTrajectoryComparisonMembers].sort().forEach((groupId) => {
        source.searchParams.append("member", groupId);
    });
    source.searchParams.set(
        "final_interval_segments",
        finalIntervalSegments(),
    );
    source.searchParams.set(
        "focus_final_interval",
        focusFinalInterval() ? "1" : "0",
    );
    source.searchParams.set(
        "comparison_view",
        trajectoryComparisonView(),
    );
    const result = element("trajectory-comparison-result");
    const image = element("trajectory-comparison-image");
    const frame = image.closest(".heatmap-frame");
    const status = frame.querySelector(".heatmap-loading");
    result.hidden = false;
    frame.classList.add("is-loading");
    frame.classList.remove("has-error");
    element("trajectory-comparison-generate").disabled = true;

    const poll = async () => {
        try {
            const response = await fetch(source, {
                cache: "no-store",
                headers: {Accept: "application/json"},
            });
            if (requestId !== trajectoryComparisonRequestId) {
                return;
            }
            const payload = await response.json();
            if (response.status === 202) {
                status.textContent = payload.message
                    || "Computing shared trajectory comparison…";
                const retryAfter = Number(
                    response.headers.get("Retry-After"),
                ) || 2;
                window.setTimeout(
                    poll,
                    Math.max(1, retryAfter) * 1000,
                );
                return;
            }
            if (!response.ok) {
                throw new Error(
                    payload.error
                    || `Comparison request failed (${response.status})`,
                );
            }
            image.src = `${payload.image_url}?v=${Date.now()}`;
            const download = element("trajectory-comparison-download");
            download.href = payload.pdf_url;
            download.download =
                `trajectory_comparison_${payload.artifact_id}.pdf`;
            renderAuthoritativeComparisonMembers(payload.members);
            renderedTrajectoryComparisonSignature = requestSignature;
            frame.classList.remove("is-loading", "has-error");
            updateTrajectoryComparisonDirtyState();
        } catch (error) {
            frame.classList.remove("is-loading");
            frame.classList.add("has-error");
            status.textContent = error.message;
        } finally {
            element("trajectory-comparison-generate").disabled =
                pendingTrajectoryComparisonMembers.size === 0;
        }
    };
    await poll();
}

listen("final-interval-segments", "change", () => {
    saveFinalIntervalSegments();
    updateTrajectoryComparisonDirtyState();
});
listen("focus-final-interval", "change", () => {
    saveFocusFinalInterval();
    updateTrajectoryComparisonDirtyState();
});
listen("trajectory-comparison-view", "change", () => {
    saveTrajectoryComparisonView();
    updateTrajectoryComparisonDirtyState();
});
listen("trajectory-comparison-add", "click", addTrajectoryComparisonMember);
listen(
    "trajectory-comparison-selected",
    "click",
    (event) => {
        const button = event.target.closest("button[data-group-id]");
        if (!button) {
            return;
        }
        pendingTrajectoryComparisonMembers.delete(button.dataset.groupId);
        renderPendingTrajectoryComparisonMembers();
    },
);
listen("trajectory-comparison-generate", "click", generateTrajectoryComparison);

restoreFinalIntervalSegments();
restoreFocusFinalInterval();
restoreTrajectoryComparisonView();
renderPendingTrajectoryComparisonMembers();
