# Experiments

Game construction, execution, atomic recording, result loading, and plotting. Pure game analysis belongs in `metrics/`.

## Built-in Games

Every factory returns `(2, actions_player_0, actions_player_1)` payoffs normalized per player to `[0, 1]`.

| ID | Actions | Definition |
|---|---:|---|
| `rps` | 3 × 3 | Upstream Rock–Paper–Scissors |
| `rpsls` | 5 × 5 | Local standard RPSLS relation |
| `matching_pennies` | 2 × 2 | Upstream Matching Pennies |
| `bertrand_standard_o1` | 21 × 21 | Costs `(0, 0)`, prices `[0.05, 1]`, demand `1` |
| `bertrand_linear_o2` | 21 × 21 | Costs `(0, 0)`, prices `[0, 1]`, `alpha=.48`, `beta=.9`, `gamma=.6` |
| `bertrand_logit_o3` | 21 × 21 | Costs `(1, 1)`, prices `[1, 2]`, `alpha=(2, 2)`, `mu=.25` |
| `bertrand_linear_o2_prime` | 21 × 21 | O2 with costs `(0, .2)` |
| `bertrand_logit_o3_prime` | 21 × 21 | Costs `(.5, 1)`, prices `[.5, 2]`, `alpha=(1.5, 2)`, `mu=.25` |

RPS, Matching Pennies, and Bertrand use the pinned `games_learning` definitions. RPSLS is local because upstream has no implementation; its order is Rock, Paper, Scissors, Lizard, Spock.

RPS provides strict CE/CCE separation: uniform probability over `(Rock, Rock)`, `(Paper, Paper)`, and `(Scissors, Scissors)` is a CCE but not a CE.

## Custom Games

`GameCatalog` combines built-in factories with compressed `.npz` files in `data/custom_games/`. Custom games use independent uniform `[0, 1)` payoffs from `numpy.random.default_rng(seed)`.

Validation enforces:

- 2–8 players and one action count per player;
- 1–100 actions per player;
- at most 1,000,000 payoff values;
- finite `[0, 1]` payoffs;
- safe 1–64 character names and non-negative seeds;
- consistent file metadata and tensor shape.

Creation is atomic and never overwrites an existing name. Invalid catalog files are skipped with warnings.

## Run Identity and Seeding

`ExperimentSpec` includes game, feedback, regret evaluation, ordered n-player algorithm profile, horizon, base seed, replicate, and stationary solver. The complete configuration is deterministically hashed; long profiles use RM/SRM or compact filename forms.

```text
player_seed = base_seed + replicate * number_of_players + player_id
```

Players therefore accept one base seed input but receive distinct reproducible streams.

## Algorithm Registries

| Full information | Bandit |
|---|---|
| `hedge` | `exp3` |
| `bm` | `exp3_ix` |
| `ito` | `bm` |
| `regret_matching` | `ito` |
| `stationary_regret_matching` | `lce_ix` |

`make full` and `make bandit` run every built-in ordered two-player pair. Batch commands fail on an existing run. The dashboard queues only missing runs and supports explicit n-player profiles.

## Runner

Each round:

1. samples every player's action;
2. stores the joint action in the environment;
3. obtains permitted learner feedback;
4. updates offline expected and realized regret trackers;
5. updates each learner;
6. records the selected regret source or both.

Feedback and evaluation are independent. A bandit learner always receives one scalar payoff; exact deviation payoffs are evaluator-only. Defaults remain expected for full information and realized for bandit feedback.

The runner validates player/action counts and checks optional cancellation before every round.

## CSV Results

`CsvRecorder` writes `<run-id>.csv.tmp` and publishes the final CSV only after success.

Each row records:

- run identity, feedback, regret evaluation, and reproducibility fields;
- JSON n-player algorithm profile plus player-specific algorithm;
- horizon, round, player, action, and payoff;
- cumulative and average external, internal, and swap regret for the selected source(s).

`results.py` validates complete files. `load_final_result_rows()` reads final player rows from the end, while `iter_result_rows()` streams full trajectories. Legacy two-player profiles and pre-evaluation CSVs remain readable.

## Figures

### Regret

`plot_regret.py` groups matching replicates, separates expected/realized sources, and plots pointwise means with 95% confidence intervals. Long files are reduced to at most 2,000 points per player while retaining first and final rounds. Legends use abbreviated algorithm labels.

### Joint actions

Built-in two-player replicate distributions are averaged and plotted with action 0 at the lower-left. The dashboard caches the PNG lazily.

### Maximum CE/CCE profile weight

Each heatmap cell is optimized independently, so the matrix is not an equilibrium distribution. Static fixed-blue, lower-origin assets live in `web/static/equilibria/`; `make precompute-equilibria` writes only that directory.

### Equilibrium convergence

- Distance uses iteration 1, powers of ten from 100, and the final round.
- CE/CCE L1 distance is computed per replicate, then averaged with a 95% confidence interval.
- The projected mean trajectory uses 2–50 uniform checkpoints, default 10.
- Hide first removes round 1 before PCA fitting and drawing, without changing distance.
- Built-in and custom n-player games are supported; matrix heatmaps remain built-in two-player only.

Distance and trajectory are separate lazy caches. Trajectory filenames include point count and `from_round_1` or `hide_round_1`.

## Outputs and Structure

```text
results/
├── raw/<run-id>.csv
├── figures/<regret-figure>.png
├── figures/details/<lazy-detail-cache>.png
└── index.html

experiments/
├── games.py, game_catalog.py
├── spec.py, runner.py, recorder.py
├── result_schema.py, results.py, result_trajectories.py
├── scenarios/
└── plots/
```

`results/figures/details/` is safe to delete because the dashboard regenerates it from source CSVs.
