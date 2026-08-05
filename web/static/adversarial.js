"use strict";

const adversarialForm = document.getElementById("adversarial-form");
const adversarialFormStorageKey = "swap-regret-adversarial-form";
const adversarialFields = [...adversarialForm.elements].filter(
    (field) => field.name && field.type !== "hidden"
);

function saveAdversarialForm() {
    const state = Object.fromEntries(
        adversarialFields.map((field) => [field.name, field.value])
    );
    try {
        window.localStorage.setItem(adversarialFormStorageKey, JSON.stringify(state));
    } catch (error) {
        console.warn("Could not save adversarial parameters", error);
    }
}

function restoreAdversarialForm() {
    let state;
    try {
        state = JSON.parse(window.localStorage.getItem(adversarialFormStorageKey));
    } catch (error) {
        console.warn("Could not restore adversarial parameters", error);
        return;
    }
    if (!state) {
        return;
    }
    for (const field of adversarialFields) {
        if (state[field.name] === undefined) {
            continue;
        }
        const fallback = field.value;
        field.value = state[field.name];
        if (!field.checkValidity()) {
            field.value = fallback;
        }
    }
}

restoreAdversarialForm();
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
