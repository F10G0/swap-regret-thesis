"use strict";

const themeStorageKey = "swap-regret-primary-theme";
const primaryThemes = new Set(["green", "blue", "purple", "orange", "red"]);

function element(id) {
    return document.getElementById(id);
}

function applyPrimaryTheme(theme, persist = false) {
    const selectedTheme = primaryThemes.has(theme) ? theme : "green";
    document.documentElement.dataset.theme = selectedTheme;
    const themeSelect = element("primary-theme");
    if (themeSelect) {
        themeSelect.value = selectedTheme;
    }
    if (!persist) {
        return;
    }
    try {
        window.localStorage.setItem(themeStorageKey, selectedTheme);
    } catch (error) {
        console.warn("Could not save the primary color", error);
    }
}

function installThemeSelector() {
    let storedTheme = "";
    try {
        storedTheme = window.localStorage.getItem(themeStorageKey) || "";
    } catch (error) {
        console.warn("Could not restore the primary color", error);
    }
    applyPrimaryTheme(storedTheme || document.documentElement.dataset.theme);
    element("primary-theme")?.addEventListener("change", (event) => {
        applyPrimaryTheme(event.target.value, true);
    });
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

installThemeSelector();
installConfirmations();
