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
.PHONY: web
.PHONY: test clean reset

##@ General

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n"} /^##@ / {printf "\n%s:\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

all: install ## Install project dependencies

##@ Setup

install: ## Install project and tests
	$(PIP) install --requirement "$(LOCKFILE)"

##@ Run

web: ## Start the local experiment dashboard
	$(PYTHON) -m web.app

##@ Validation

test: ## Run the complete Python and Node/jsdom test suite without creating caches
	@command -v node >/dev/null 2>&1 || { echo "Node.js >=12.22.0 is required for make test. Install Node.js, then run npm install." >&2; exit 1; }
	@node -e "require('jsdom')" >/dev/null 2>&1 || { echo "jsdom is required for make test. Run npm install in the project root." >&2; exit 1; }
	PYTHONDONTWRITEBYTECODE=1 $(PYTEST) -q -p no:cacheprovider

##@ Cleanup

clean: ## Remove Python caches and selected figure/game staging directories
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" \) -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find "$(RESULTS_DIR)" "$(CUSTOM_GAME_DIR)" -type d \( -name ".figures-*" -o -name ".equilibrium-convergence-*" -o -name ".custom-game-*" \) -prune -exec rm -rf {} + 2>/dev/null || true

reset: clean ## Remove all experiment-derived results, figures, and caches
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m experiments.cleanup \
		--preserve "$(CUSTOM_GAME_DIR)" \
		"$(RAW_DIR)" "$(FIGURE_DIR)" "$(ADVERSARIAL_DIR)" "$(RESULTS_DIR)/cache"
