"use strict";

const adversarialDataElement = document.getElementById("adversarial-data");
const adversarialData = JSON.parse(
    adversarialDataElement ? adversarialDataElement.textContent : "{}"
);
const adversarialForm = document.getElementById("adversarial-form");
const adversarialFormStorageKey = "swap-regret-adversarial-form";
const adversarialFields = [...adversarialForm.elements].filter(
    (field) => field.name && field.type !== "hidden"
);

function saveAdversarialForm() {
    const state = Object.fromEntries(
        adversarialFields.map((field) => [field.name, field.value])
    );
    saveLocalJson(adversarialFormStorageKey, state, "adversarial parameters");
}

function restoreAdversarialForm() {
    const state = restoreLocalJson(adversarialFormStorageKey, "adversarial parameters");
    if (!state) {
        return;
    }
    const feedbackSelect = document.getElementById("adversarial-feedback-mode");
    if (
        feedbackSelect
        && [...feedbackSelect.options].some(
            (option) => option.value === state.feedback_mode
        )
    ) {
        feedbackSelect.value = state.feedback_mode;
    }
    updateAdversarialAlgorithms(state.algorithm_name);
    for (const field of adversarialFields) {
        if (
            ["feedback_mode", "algorithm_name"].includes(field.name)
            || state[field.name] === undefined
        ) {
            continue;
        }
        const fallback = field.value;
        field.value = state[field.name];
        if (!field.checkValidity()) {
            field.value = fallback;
        }
    }
}

function updateAdversarialAlgorithms(preferredAlgorithm = null) {
    const feedbackSelect = document.getElementById("adversarial-feedback-mode");
    const feedbackMode = feedbackSelect ? feedbackSelect.value : null;
    const select = document.getElementById("adversarial-algorithm");
    const algorithms = adversarialData.algorithms[feedbackMode] || [];
    if (!select) {
        return;
    }
    const selected = preferredAlgorithm || select.value;
    select.replaceChildren(...algorithms.map((algorithm) => {
        const option = document.createElement("option");
        option.value = algorithm;
        option.textContent = adversarialData.algorithmLabels[algorithm] || algorithm;
        return option;
    }));
    select.value = algorithms.includes(selected) ? selected : algorithms[0] || "";
}

function updateAdversarialEnvironment() {
    const environment = document.getElementById("adversarial-environment").value;
    const randomWalk = environment === adversarialData.randomWalkEnvironment;
    for (const [fieldId, enabled] of [
        ["adversarial-initialization-field", randomWalk],
        ["adversarial-environment-seed-field", randomWalk],
        ["adversarial-memory-field", !randomWalk],
    ]) {
        const field = document.getElementById(fieldId);
        field.hidden = !enabled;
        field.querySelectorAll("input, select").forEach((input) => {
            input.disabled = !enabled;
        });
    }
    document.getElementById("historical-frequency-rule").hidden = randomWalk;
    document.getElementById("random-walk-rule").hidden = !randomWalk;
}

restoreAdversarialForm();
updateAdversarialEnvironment();
const adversarialFeedbackSelect = document.getElementById(
    "adversarial-feedback-mode"
);
if (adversarialFeedbackSelect) {
    adversarialFeedbackSelect.addEventListener(
        "change",
        () => updateAdversarialAlgorithms()
    );
}
document.getElementById("adversarial-environment").addEventListener(
    "change",
    updateAdversarialEnvironment
);
for (const event of ["input", "change", "submit"]) {
    adversarialForm.addEventListener(event, saveAdversarialForm);
}

const adversarialJobs = [...document.querySelectorAll("[data-adversarial-job]")];
const activeAdversarialJobs = adversarialJobs.filter((element) =>
    ["queued", "running"].includes(element.dataset.status)
);

async function refreshAdversarialJob(element) {
    const response = await fetch(element.dataset.statusUrl, {
        headers: {Accept: "application/json"},
    });
    if (!response.ok) {
        return false;
    }
    const job = await response.json();
    element.dataset.status = job.status;
    const status = element.querySelector("[data-job-status]");
    const message = element.querySelector("[data-job-message]");
    if (status) {
        status.textContent = job.status;
    }
    if (message) {
        message.textContent = job.message;
    }
    return !["queued", "running"].includes(job.status);
}

if (activeAdversarialJobs.length > 0) {
    const poll = async () => {
        try {
            const completed = await Promise.all(
                activeAdversarialJobs.map(refreshAdversarialJob)
            );
            if (completed.some(Boolean)) {
                window.location.reload();
                return;
            }
        } catch (error) {
            console.warn("Could not refresh adversarial job status", error);
        }
        window.setTimeout(poll, 1000);
    };
    window.setTimeout(poll, 1000);
}
