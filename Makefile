.DEFAULT_GOAL := help

PYTHON ?= python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
GAMES_LEARNING_COMMIT := 6ca238a9c8716cd34fc3fcbc949bba7a7ea68dc5
GAMES_LEARNING_EDITABLE := git+https://github.com/TUM-DSS/games_learning.git@$(GAMES_LEARNING_COMMIT)\#egg=games_learning

RESULTS_DIR ?= results
RAW_DIR ?= $(RESULTS_DIR)/raw
FIGURE_DIR ?= $(RESULTS_DIR)/figures
REPORT_PATH ?= $(RESULTS_DIR)/index.html

.PHONY: help install-games-learning install install-dev
.PHONY: run full bandit web plot report precompute-equilibria
.PHONY: test smoke clean clean-results reset

##@ General

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n"} /^##@ / {printf "\n%s:\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Setup

install-games-learning: ## Install the pinned authoritative game/equilibrium backend
	$(PIP) install --no-deps -e "$(GAMES_LEARNING_EDITABLE)"

install: install-games-learning ## Install the project in editable mode
	$(PIP) install -e .

install-dev: install-games-learning ## Install the project with test dependencies
	$(PIP) install -e ".[test]"

##@ Run

run: ## Run all experiments, generate plots, and build the report
	$(PYTHON) -m main

full: ## Run all full-information cross-play experiments
	$(PYTHON) -m experiments.scenarios.full_information_cross_play

bandit: ## Run all bandit-feedback cross-play experiments
	$(PYTHON) -m experiments.scenarios.bandit_cross_play

web: ## Start the local experiment dashboard
	$(PYTHON) -m web.app

##@ Generated outputs

plot: ## Regenerate plots from existing raw results
	$(PYTHON) -m experiments.plots.plot_regret

report: ## Build the HTML report from existing figures
	$(PYTHON) -m experiments.build_report

precompute-equilibria: ## Regenerate static CE/CCE profile-weight heatmaps
	$(PYTHON) -m web.precompute_equilibrium_figures --force --workers 4

##@ Validation

test: ## Run the complete test suite without creating caches
	PYTHONDONTWRITEBYTECODE=1 $(PYTEST) -q -p no:cacheprovider

smoke: ## Run only the end-to-end smoke tests
	PYTHONDONTWRITEBYTECODE=1 $(PYTEST) -q -p no:cacheprovider tests/test_smoke.py

##@ Cleanup

clean: ## Remove Python and pytest caches
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -maxdepth 1 -type f -name "*-pulp.*" -delete
	find results web/static/equilibria data/custom_games -type d \( -name ".figures-*" -o -name ".equilibrium-*" -o -name ".equilibrium-convergence-*" -o -name ".precompute-equilibrium-*" -o -name ".custom-game-*" \) -prune -exec rm -rf {} + 2>/dev/null || true

clean-results: ## Remove generated results while preserving .gitkeep files
	@if [ -d "$(RAW_DIR)" ]; then \
		find "$(RAW_DIR)" -mindepth 1 -maxdepth 1 -type f ! -name ".gitkeep" -delete; \
	fi
	@if [ -d "$(FIGURE_DIR)" ]; then \
		find "$(FIGURE_DIR)" -type f -name "*.png" -delete; \
	fi
	@if [ -f "$(REPORT_PATH)" ]; then \
		rm -f -- "$(REPORT_PATH)"; \
	fi

reset: clean clean-results ## Remove all generated caches and results
