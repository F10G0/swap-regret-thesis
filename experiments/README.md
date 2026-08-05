# Experiments

This package builds games, runs learners, records CSV results, and creates the standard figures.

Experiments are configured and queued from `make web`. Use `make plot` only to rebuild figures from existing CSVs. Existing run IDs are never overwritten.

Each plot is saved as a PNG dashboard preview and a same-stem PDF for publication.

Adversarial runs are isolated under `results/adversarial/`. They use full-information learners, run once per submission, and accept any memory-window length, with 0 meaning full history. Their figures show both `R/t` and `R/sqrt(t)` for expected and realized regret, plus learner and punished-action frequencies.

Dashboard full-information uses replicate index 0. Adversarial submissions have no replicate setting. Bandit batches use indices 0 through n−1.

## Games

Built-in games are RPS, RPSLS, Matching Pennies, and five 21 × 21 Bertrand variants. All payoffs are normalized per player to `[0, 1]`.

`GameCatalog` also loads compressed custom games from `data/custom_games/`. Custom games may have 2–8 players, heterogeneous action counts, and at most 1,000,000 payoff values. The dashboard creates reproducible general-sum games or two-player zero-sum games. Zero-sum payoffs are stored in the equivalent `[0,1]` constant-sum form `u` and `1-u`.

## Runs and Seeds

A run is identified by its game and payoff digest, feedback mode, regret evaluation, ordered algorithm profile, horizon, base seed, replicate, and stationary solver.

Player `i` in replicate `r` of a `p`-player game receives:

```text
base_seed + r * p + i
```

This makes every learner stream distinct and reproducible.

Each round samples a joint action, collects the permitted feedback, updates the requested regret trackers, and then updates the learners. Feedback and regret evaluation are independent: a bandit learner still receives only its realized reward even when expected regret is evaluated offline.

## Results and Figures

Raw runs are written atomically to `results/raw/`. Each row contains the run identity, round data, action and payoff, and cumulative and average regret.

The plotting layer provides:

- regret curves with replicate means and 95% confidence intervals;
- separate expected and realized figures when both were recorded;
- joint-action heatmaps for built-in games and custom two-player zero-sum games;
- precomputed maximum CE/CCE profile-weight heatmaps;
- full-space CE/CCE L1-distance plots for equilibrium convergence.

Equilibrium distance is computed for each replicate before aggregation. Projected trajectories are optional and generated only from the experimental dashboard.

```text
results/
├── raw/             experiment CSVs
├── figures/         generated plots and lazy detail figures
└── adversarial/     stress-test CSVs and figures
```

Game definitions live in `games.py` and `game_catalog.py`; execution and recording in `runner.py` and `recorder.py`; result validation in `results.py`. See [../metrics/README.md](../metrics/README.md) for the analysis functions.
