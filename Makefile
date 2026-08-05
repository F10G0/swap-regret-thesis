.DEFAULT_GOAL := help

PYTHON ?= python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
GAMES_LEARNING_COMMIT := 6ca238a9c8716cd34fc3fcbc949bba7a7ea68dc5
GAMES_LEARNING_EDITABLE := git+https://github.com/TUM-DSS/games_learning.git@$(GAMES_LEARNING_COMMIT)\#egg=games_learning

RESULTS_DIR ?= results
RAW_DIR ?= $(RESULTS_DIR)/raw
FIGURE_DIR ?= $(RESULTS_DIR)/figures
ADVERSARIAL_DIR ?= $(RESULTS_DIR)/adversarial
EXPERIMENTAL_TRAJECTORIES ?= 0

.PHONY: help all install install-experimental
.PHONY: web plot precompute-equilibria
.PHONY: test clean reset

##@ General

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n"} /^##@ / {printf "\n%s:\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

all: install plot ## Install the default build and refresh existing plots

##@ Setup

install: ## Install project and tests without experimental trajectories
	$(PIP) install --no-deps -e "$(GAMES_LEARNING_EDITABLE)"
	EXPERIMENTAL_TRAJECTORIES="$(EXPERIMENTAL_TRAJECTORIES)" $(PIP) install -e ".[test]"

install-experimental: EXPERIMENTAL_TRAJECTORIES := 1
install-experimental: install ## Install project and tests with experimental trajectories

##@ Run

web: ## Start the local experiment dashboard
	$(PYTHON) -m web.app

##@ Generated outputs

plot: ## Regenerate plots from existing raw results
	$(PYTHON) -m experiments.plots.plot_regret
	$(PYTHON) -m experiments.plots.plot_adversarial

precompute-equilibria: ## Regenerate static CE/CCE profile-weight heatmaps
	$(PYTHON) -m web.precompute_equilibrium_figures --force --workers 4

##@ Validation

test: ## Run the complete test suite without creating caches
	PYTHONDONTWRITEBYTECODE=1 $(PYTEST) -q -p no:cacheprovider

##@ Cleanup

clean: ## Remove Python and pytest caches
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" \) -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*-pulp.*" \) -delete
	find "$(RESULTS_DIR)" web/static/equilibria data/custom_games -type d \( -name ".figures-*" -o -name ".equilibrium-*" -o -name ".equilibrium-convergence-*" -o -name ".precompute-equilibrium-*" -o -name ".custom-game-*" \) -prune -exec rm -rf {} + 2>/dev/null || true

reset: clean ## Remove caches and generated results
	@if [ -d "$(RAW_DIR)" ]; then \
		find "$(RAW_DIR)" -mindepth 1 -maxdepth 1 -type f ! -name ".gitkeep" -delete; \
	fi
	@if [ -d "$(FIGURE_DIR)" ]; then \
		find "$(FIGURE_DIR)" -type f \( -name "*.png" -o -name "*.pdf" \) -delete; \
	fi
	@if [ -d "$(ADVERSARIAL_DIR)" ]; then \
		find "$(ADVERSARIAL_DIR)" -type f \( -name "*.csv" -o -name "*.png" -o -name "*.pdf" \) -delete; \
	fi
