# Bandit Algorithms Thesis Project

## Overview

This repository contains the implementation and experimental framework for a Bachelor's thesis on **no-regret and no-swap-regret algorithms** in bandit and online learning settings.

The project is designed as a **modular and extensible experimental framework**, supporting:

- Implementation of bandit algorithms and no-regret learning baselines
- Simulation of stochastic and adversarial environments
- Reproducible experimental evaluation
- Regret analysis and visualization
- Unit-tested algorithm components

The current version includes working baselines using **Explore-Then-Commit (ETC)**, **Upper Confidence Bound (UCB)**, **Phased UCB**, and **phased elimination** algorithms on a Bernoulli bandit.

---

## Repository Structure

```text
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
  Runs experiments and triggers plotting.

- **config.py**  
  Centralized configuration:
  - environment parameters
  - horizon
  - algorithm parameters
  - number of runs
  - random seeds

- **requirements.txt**  
  Python dependencies:
  - numpy
  - matplotlib
  - pytest

- **pytest.ini**  
  Configures pytest to include the project root in the Python path.

---

### algorithms/

Implements bandit learning algorithms and wrapper functions for experiment setup.

The package exposes algorithm constructors through `make_*` wrapper functions, while internal algorithm classes remain modular.

Current structure:

- **base.py**  
  Abstract interface for bandit algorithms.

- **empirical_mean_base.py**  
  Shared base class for algorithms that maintain:
  - pull counts
  - reward sums
  - empirical means

- **etc.py / etc_wrappers.py**  
  Explore-Then-Commit (ETC):
  - fixed round-robin exploration
  - commit to empirically best arm
  - doubling-trick wrapper for horizon-dependent ETC

- **ucb.py / ucb_wrappers.py**  
  Upper Confidence Bound (UCB):
  - initialization by playing each arm once
  - UCB index-based action selection
  - standard, delta-based, and asymptotically optimal variants

- **phased_ucb.py / phased_ucb_wrappers.py**  
  Phased UCB:
  - UCB-based arm selection at phase boundaries
  - exponential and count-doubling phase schedules

- **elimination.py / elimination_wrappers.py**  
  Phased elimination:
  - round-robin exploration over active arms
  - elimination using phase-wise empirical means
  - commitment when only one arm remains

- **doubling_trick.py**  
  Generic doubling-trick wrapper for restarting horizon-dependent algorithms in exponentially growing epochs.

Future extensions:
- `exp3.py`
- internal-regret algorithms
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
- game environments

---

### experiments/

Controls execution and evaluation.

- **runner.py**  
  Simulation logic:
  - runs interaction loop
  - records actions, rewards, and regret
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
  - multi-algorithm comparison

All figures are saved to `results/figures/`.

---

### tests/

Unit tests using pytest.

- **test_etc.py**  
  Verifies:
  - round-robin exploration
  - commit behavior
  - empirical statistics
  - reset behavior
  - doubling-trick ETC wrapper

- **test_ucb.py**  
  Verifies:
  - initialization phase
  - UCB action selection
  - beta schedules
  - reset behavior
  - invalid parameter handling

- **test_phased_ucb.py**  
  Verifies:
  - initialization phase
  - phase scheduling
  - exponential phase lengths
  - count-doubling phase lengths
  - UCB selection at phase boundaries

- **test_elimination.py**  
  Verifies:
  - phase-based round-robin behavior
  - phase-local statistics
  - elimination rule
  - phase transitions
  - commitment when one arm remains

---

### results/

Stores experiment outputs.

- **figures/**  
  Generated plots, such as regret curves.

- **raw/**  
  Raw data, reserved for future use.

---

## Experiment Workflow

The typical experiment pipeline is:

```text
Algorithm → Environment → Runner → Metrics → Plots
```

1. Algorithm selects an action
2. Environment returns a reward
3. Runner records the interaction
4. Metrics compute regret
5. Plots visualize results

---

## Usage

### Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run experiments

```bash
python main.py
```

### Run tests

```bash
pytest -v
```

---

## Current Status

- [x] Project structure initialized
- [x] Bernoulli bandit environment implemented
- [x] Explore-Then-Commit (ETC) implemented
- [x] ETC doubling-trick wrapper implemented
- [x] Upper Confidence Bound (UCB) implemented
- [x] Phased UCB implemented
- [x] Phased elimination algorithm implemented
- [x] Single-run and multi-run experiments supported
- [x] Regret computation and plotting
- [x] Unit tests for ETC, UCB, Phased UCB, and elimination

---

## Next Steps

- [ ] Implement Exp3 algorithm
- [ ] Add richer multi-algorithm comparison plots
- [ ] Extend to adversarial and game settings
- [ ] Implement internal-regret and no-swap-regret algorithms

---

## Notes

- Algorithms are implemented as independent modules with wrapper-based public constructors.
- The project is structured for modularity, reproducibility, and future extension to stronger regret notions.
