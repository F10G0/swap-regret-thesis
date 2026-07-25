.DEFAULT_GOAL := help

PYTHON ?= python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest

RESULTS_DIR ?= results
RAW_DIR ?= $(RESULTS_DIR)/raw
FIGURE_DIR ?= $(RESULTS_DIR)/figures
REPORT_PATH ?= $(RESULTS_DIR)/index.html

.PHONY: help all install install-dev run full bandit plot report web test smoke
.PHONY: clean clean-results reset rerun reweb

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

all: test ## Run the default validation target

install: ## Install the project in editable mode
	$(PIP) install -e .

install-dev: ## Install the project with test dependencies
	$(PIP) install -e ".[test]"

run: ## Run all experiments, generate plots, and build the report
	$(PYTHON) -m main

full: ## Run all full-information cross-play experiments
	$(PYTHON) -m experiments.scenarios.full_information_cross_play

bandit: ## Run all bandit-feedback cross-play experiments
	$(PYTHON) -m experiments.scenarios.bandit_cross_play

plot: ## Regenerate plots from existing raw results
	$(PYTHON) -m experiments.plots.plot_regret

report: ## Build the HTML report from existing figures
	$(PYTHON) -m experiments.build_report

web: ## Start the local experiment dashboard
	$(PYTHON) -m web.app

test: ## Run the complete test suite without creating caches
	PYTHONDONTWRITEBYTECODE=1 $(PYTEST) -q -p no:cacheprovider

smoke: ## Run only the end-to-end smoke tests
	PYTHONDONTWRITEBYTECODE=1 $(PYTEST) -q -p no:cacheprovider tests/test_smoke.py

clean: ## Remove Python and pytest caches
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +

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

rerun: reset run ## Reset the workspace and run the complete pipeline

reweb: reset web ## Reset the workspace and start the dashboard
