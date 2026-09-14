.DEFAULT_GOAL := help

PYTHON ?= python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
LOCKFILE := requirements.lock

# Set numerical-library limits before Python imports NumPy/BLAS. Spawned
# replicate workers inherit these; explicit make/environment overrides work.
export OMP_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export NUMEXPR_NUM_THREADS ?= 1

RESULTS_DIR ?= results
RAW_DIR ?= $(RESULTS_DIR)/raw
FIGURE_DIR ?= $(RESULTS_DIR)/figures
ADVERSARIAL_DIR ?= $(RESULTS_DIR)/adversarial
CUSTOM_GAME_DIR ?= data/custom_games

.PHONY: help all install
.PHONY: web plot
.PHONY: test clean reset

##@ General

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n"} /^##@ / {printf "\n%s:\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

all: install plot ## Install and refresh existing plots

##@ Setup

install: ## Install project and tests
	$(PIP) install --requirement "$(LOCKFILE)"

##@ Run

web: ## Start the local experiment dashboard
	$(PYTHON) -m web.app

##@ Generated outputs

plot: ## Regenerate plots from existing raw results
	$(PYTHON) -m experiments.plots.plot_regret
	$(PYTHON) -m experiments.plots.plot_adversarial
	$(PYTHON) -m experiments.plots.plot_adversarial_scaling

##@ Validation

test: ## Run the complete test suite without creating caches
	PYTHONDONTWRITEBYTECODE=1 $(PYTEST) -q -p no:cacheprovider

##@ Cleanup

clean: ## Remove Python caches and temporary staging files
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" \) -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find "$(RESULTS_DIR)" "$(CUSTOM_GAME_DIR)" -type d \( -name ".figures-*" -o -name ".equilibrium-convergence-*" -o -name ".custom-game-*" \) -prune -exec rm -rf {} + 2>/dev/null || true

reset: clean ## Remove all experiment-derived results, figures, and caches
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m experiments.cleanup \
		--preserve "$(CUSTOM_GAME_DIR)" \
		"$(RAW_DIR)" "$(FIGURE_DIR)" "$(ADVERSARIAL_DIR)" "$(RESULTS_DIR)/cache"
