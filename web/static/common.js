"use strict";

const themeStorageKey = "swap-regret-primary-theme";
const primaryThemes = new Set(["green", "blue", "purple", "orange", "red"]);

function applyPrimaryTheme(theme, persist = false) {
    const selectedTheme = primaryThemes.has(theme) ? theme : "green";
    document.documentElement.dataset.theme = selectedTheme;
    const themeSelect = document.getElementById("primary-theme");
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
    document.getElementById("primary-theme")?.addEventListener("change", (event) => {
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

installThemeSelector();
installConfirmations();
