# Experiments

Experiment construction, repeated-game execution, recording, result loading, and visualization.

## Structure

```text
experiments/
├── README.md
├── __init__.py
├── build_report.py
├── games.py
├── recorder.py
├── results.py
├── runner.py
├── spec.py
├── plots/
│   ├── __init__.py
│   ├── plot_joint_actions.py
│   └── plot_regret.py
└── scenarios/
    ├── __init__.py
    ├── bandit_cross_play.py
    ├── cross_play.py
    ├── fieldnames.py
    └── full_information_cross_play.py
```

## Benchmark Games

`games.py` provides normalized payoff tensors for:

- Rock–Paper–Scissors;
- Dominant Coordination with 9 actions;
- Cyclic Dominance with 9 actions.

Every factory returns a tensor with shape `(n_players, action_1, ..., action_n)` and payoffs in `[0, 1]`. The current scenario registry and CSV schema are deliberately two-player, although the environment tensor representation supports more players.

## Experiment Identity

`ExperimentSpec` defines a two-player run by:

- game;
- feedback mode;
- ordered player algorithm profile;
- horizon;
- base seed;
- replicate index.
- configured stationary-distribution solver.

The complete configuration is hashed into a deterministic run ID. Feedback modes, horizons, seeds, profiles, and replicates therefore cannot overwrite or accidentally merge with one another.

Player seeds are deterministic:

```text
player_seed = base_seed + replicate * number_of_players + player
```

This gives every player and bandit replicate a distinct reproducible random stream.

## Cross-Play Scenarios

`scenarios/cross_play.py` contains common construction and execution. An `AlgorithmFactory` records whether a learner constructor receives the experiment horizon.

Full-information profiles can use:

- Hedge;
- full-information Blum–Mansour;
- full-information Ito;
- Regret Matching;
- Stationary Regret Matching.

Bandit profiles can use:

- Exp3;
- Exp3-IX;
- bandit Blum–Mansour;
- bandit Ito;
- LCE-IX.

The two players may use different algorithms. Learning rates are never experiment parameters; each exponential-weights learner derives its schedule from its configured horizon and local update count.

Useful entry points:

```bash
make full
make bandit
```

The full-information batch runs every ordered algorithm pair once. The bandit batch runs every ordered pair for `BANDIT_REPLICATES`, configured in `config.py`.

## Runner

For every round, `runner.py`:

1. samples one action from each learner;
2. calls `game.step(actions)` once with the joint action;
3. queries each player's environment feedback according to the experiment's explicit feedback mode;
4. obtains evaluation-only deviation payoffs when bandit feedback does not contain them;
5. updates both expected and realized regret trackers;
6. updates each learner using only its permitted feedback;
7. records only expected regret for full-information runs or realized regret for bandit runs.

The runner accepts an optional cancellation callback. Cancellation is checked before each round and raises `ExperimentCancelled`.

## Recording

`CsvRecorder` writes to a temporary file and atomically publishes the final CSV only after the experiment exits successfully. Failed or cancelled experiments therefore do not leave partial result files.

Every row contains:

- run identity and feedback mode;
- base seed, replicate, and stationary solver;
- game and ordered algorithm profile;
- horizon, round, and player;
- sampled action and realized payoff;
- cumulative and time-average external, internal, and swap regret for the recorded regret source.

## Regret Quantities

Full-information experiments record expected regret:

```text
G_expected[i, j] = sum_t p_t[i] * (r_t[j] - r_t[i]).
```

Bandit experiments record realized regret:

```text
G_realized[i, j] = sum_{t: a_t = i} (r_t[j] - r_t[i]).
```

Both trackers use the complete unilateral-deviation payoff vector for evaluation. In bandit runs, that vector is obtained separately from the environment and is never exposed to the learner.

One full-information run computes the expected-regret increment directly from the mixed strategy and payoff vector. Repeated independently seeded bandit runs are averaged to estimate expected realized-regret performance.

For the applicable source, each CSV records:

- cumulative external, internal, and swap regret;
- time-average external, internal, and swap regret.

## Result Loading

`results.py` centralizes:

- expected/realized regret column names;
- CSV identity and schema validation;
- streaming row iteration;
- efficient final-round loading for the dashboard.

CSV rows from one file must share identical run metadata. The loader also checks horizon, round, player, stationary solver, and feedback-specific regret columns.

## Regret Plots

`plots/plot_regret.py`:

- streams and downsamples long runs to at most 2,000 points per player;
- keeps expected and realized sources separate;
- groups runs that differ only by replicate;
- computes pointwise means and 95% confidence intervals;
- creates time-average regret and `regret / sqrt(t)` scaling views;
- uses stable profile colors and a zero-reference line;
- places the legend below the data axes;
- expands the PNG height for large legends so entries never cover or clip the curves.

Command-line plotting fails on malformed result files so data problems are visible. Dashboard plotting logs and skips malformed files, reports them in the result warnings, and continues rendering valid experiments. Dashboard publication uses a temporary directory so a rendering failure does not destroy the last valid figures.

Generate or rebuild figures with:

```bash
make plot
```

## Joint-Action Heatmaps

`plots/plot_joint_actions.py` streams one result file, counts the realized two-player joint actions, and plots their empirical frequency. The dashboard generates these heatmaps lazily, caches them under `results/figures/details/`, and regenerates them when the source CSV changes.

## Static Report

`build_report.py` creates `results/index.html` from the top-level regret figures:

```bash
make report
```

The interactive dashboard provides richer filtering, result details, downloads, and joint-action heatmaps.

## Output Layout

```text
results/
├── raw/
│   └── <deterministic-run-id>.csv
├── figures/
│   ├── <regret-figure>.png
│   └── details/
│       └── <run-id>_joint_actions.png
└── index.html
```

Example run IDs:

```text
rps_full_information_hedge_vs_bm_<hash>
rps_bandit_exp3_vs_lce_ix_<hash>
```
