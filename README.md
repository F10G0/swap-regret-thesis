# Swap Regret Thesis Project

Implementation and empirical evaluation of no-regret and no-swap-regret learning algorithms.

This repository contains:
- literature notes and references,
- thesis proposal and related documents,
- implementations of online learning and bandit algorithms,
- ongoing work on reduction-based no-swap-regret algorithms.

The project focuses on:
- swap regret,
- correlated equilibrium,
- reduction-based online learning algorithms,
- computational efficiency and practical behavior of learning algorithms.

---

## Repository Structure

```text
swap-regret-thesis/
├── algorithms/
├── environments/
├── experiments/
├── results/
├── tests/
│
├── docs/
│   ├── proposal/
│   ├── references/
│   ├── thesis/
│   └── progress.md
│
├── config.py
├── main.py
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Implemented Algorithms

Current implementations mainly include no-external-regret and bandit algorithms:
- Explore-Then-Commit (ETC)
- UCB and phased UCB
- elimination-based methods
- Exp3 and Exp3-IX
- doubling-trick wrappers

Future work includes:
- Hedge,
- full-information learners,
- reduction-based no-swap-regret algorithms.

---

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run experiments:

```bash
python main.py
```

Run tests:

```bash
pytest -v
```

---

## Notes

This repository is under active development as part of a Bachelor's thesis project on swap regret and online learning.
