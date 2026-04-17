# Bandit Algorithms Thesis Project

## Overview

This repository contains the implementation and experimental framework for a Bachelor's thesis on **no-regret and no-swap-regret algorithms** in bandit and online learning settings.

The project is designed as a **modular and extensible experimental framework**, supporting:

- Implementation of bandit algorithms (ETC, UCB, Exp3, swap-regret variants)
- Simulation of stochastic and adversarial environments
- Reproducible experimental evaluation (multi-run averaging)
- Regret analysis and visualization

The current version includes a working baseline using the **Explore-Then-Commit (ETC)** algorithm on a Bernoulli bandit.

---

## Repository Structure

```
thesis_bandits/
├── README.md
├── config.py
├── main.py
├── requirements.txt
├── pytest.ini
│
├── algorithms/
├── environments/
├── experiments/
├── plots/
│
├── tests/
└── results/
```

---

## Module Description

### Root Files

- **main.py**  
  Entry point for experiments.  
  Runs:
  - single experiment (debug)
  - multiple experiments (evaluation)  
  and triggers plotting.

- **config.py**  
  Centralized configuration:
  - environment parameters (e.g., probabilities)
  - horizon
  - algorithm parameters (e.g., `m`)
  - number of runs
  - random seeds

- **requirements.txt**  
  Python dependencies:
  - numpy
  - matplotlib
  - pytest

- **pytest.ini**  
  Configures pytest to include project root in Python path.

---

### algorithms/

Implements bandit learning algorithms.

- **base.py**  
  Abstract interface for all algorithms:
  - `select_action()`
  - `update(action, reward)`
  - `reset()`

- **etc.py**  
  Explore-Then-Commit (ETC) algorithm:
  - fixed exploration phase
  - commit to empirically best arm

Future extensions:
- `ucb.py`
- `exp3.py`
- swap-regret algorithms

---

### environments/

Defines reward-generating processes.

- **base.py**  
  Abstract environment interface:
  - `pull(action)`
  - `arm_mean(action)`
  - `optimal_mean()`
  - `optimal_arm()`

- **bernoulli_bandit.py**  
  Stochastic bandit with Bernoulli rewards:
  - each arm has fixed success probability
  - rewards sampled independently

Future extensions:
- Gaussian bandits
- adversarial bandits
- game environments (multi-agent)

---

### experiments/

Controls execution and evaluation.

- **runner.py**  
  Simulation logic:
  - runs interaction loop
  - records actions, rewards, regret
  - supports single-run and multi-run experiments

- **metrics.py**  
  Evaluation utilities:
  - instantaneous regret
  - cumulative regret
  - averaging across runs

---

### plots/

Visualization of results.

- **plot_regret.py**  
  Plotting functions:
  - single-run cumulative regret
  - average cumulative regret
  - multi-algorithm comparison (future)

All figures are saved to `results/figures/`.

---

### tests/

Unit tests (using pytest).

- **test_etc.py**  
  Verifies:
  - correct exploration (round-robin)
  - correct commit behavior
  - correct reset behavior

---

### results/

Stores experiment outputs.

- **figures/**  
  Generated plots (e.g., regret curves)

- **raw/**  
  Raw data (reserved for future use)

---

## Experiment Workflow

The typical experiment pipeline is:

```
Algorithm → Environment → Runner → Metrics → Plots
```

1. Algorithm selects an action
2. Environment returns a stochastic reward
3. Runner records interaction
4. Metrics compute regret
5. Plots visualize results

---

## Usage

### Install dependencies

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run experiments

```
python main.py
```

### Run tests

```
pytest -v
```

---

## Current Status

- [x] Project structure initialized
- [x] Bernoulli bandit environment implemented
- [x] Explore-Then-Commit (ETC) implemented
- [x] Single and multiple experiments supported
- [x] Regret computation and plotting
- [x] Unit tests for ETC

---

## Next Steps

- [ ] Implement UCB algorithm
- [ ] Implement Exp3 algorithm
- [ ] Add multi-algorithm comparison plots
- [ ] Extend to adversarial and game settings
- [ ] Implement no-swap-regret algorithms

---

## Notes

- Algorithms are implemented as **independent modules**, enabling easy integration into external frameworks.
- The project is structured for **scalability and reproducibility**, which is essential for experimental research.
