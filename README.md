# Swap Regret Thesis Project

Implementation and empirical evaluation of no-regret and no-swap-regret learning algorithms in repeated games.

This repository contains a modular experimental framework developed for my bachelor's thesis. The project focuses on regret minimization, swap regret reductions, correlated equilibrium, and repeated game dynamics.

---

## Implemented Algorithms

### External Regret

Full-information:
- Hedge

Partial-information:
- Exp3

### Swap Regret

Full-information:
- Blum-Mansour reduction
- Ito reduction

Partial-information:
- Blum-Mansour reduction
- Ito reduction

---

## Implemented Environments

### Full-information

- Repeated Game

### Partial-information

- Bandit Repeated Game

---

## Regret Metrics

The framework currently supports:

- External Regret
- Internal Regret
- Swap Regret

All regret notions are computed from a common cumulative replacement-gain matrix.

---

## Experimental Games

Current benchmark games include:

- Rock-Paper-Scissors
- Dominant Coordination
- Cyclic Dominance

---

## Repository Structure

```text
.
├── algorithms
├── environments
├── experiments
├── metrics
├── results
│   ├── raw
│   └── figures
│
├── config.py
├── main.py
├── requirements.txt
└── Makefile
```

---

## Installation

```bash
make install
```

or

```bash
pip install -r requirements.txt
```

---

## Running Experiments

Run the full experimental pipeline:

```bash
make run
```

This will:

1. Execute all configured experiments.
2. Save raw CSV results.
3. Generate regret plots.

Generated files are stored in:

```text
results/
├── raw/
└── figures/
```

---

## Research Topics

The project investigates:

- External regret minimization
- Internal regret minimization
- Swap regret minimization
- Reduction-based learning algorithms
- Correlated equilibrium
- Repeated games
- Online learning

The codebase is designed to support both theoretical investigation and empirical comparison of regret-minimizing algorithms.
