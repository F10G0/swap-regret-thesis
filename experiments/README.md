# Experiments

This directory contains the experimental framework used to evaluate learning algorithms in repeated games.

## Structure

```text
experiments/
├── README.md
├── __init__.py
├── games.py
├── recorder.py
├── runner.py
├── plots
│   └── plot_regret.py
└── scenarios
    └── self_play.py
```

## Files

### games.py

Provides payoff generators for benchmark games used in experiments.

Current games include:
- Rock-Paper-Scissors
- Dominant Coordination
- Cyclic Dominance

All payoff generators return normalized payoff tensors that can be used to construct game environments.

### recorder.py

Utilities for recording experimental results and exporting them to CSV files.

### runner.py

Core experiment runner.

Responsible for:
- executing repeated interactions,
- collecting feedback from environments,
- updating algorithms,
- updating regret trackers,
- recording experimental data.

### scenarios/

Contains experiment configurations.

Current scenarios:
- `self_play.py` — all players use the same learning algorithm and learn simultaneously.

### plots/

Contains visualization utilities.

Current plots:
- `plot_regret.py` — generates regret curves from recorded CSV results.

## Output

Experimental outputs are written to:

```text
results/
├── raw/
└── figures/
```

where

- `raw/` contains CSV files,
- `figures/` contains generated plots.
