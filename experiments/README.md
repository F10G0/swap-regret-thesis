# Experiments

Experiment construction, execution, CSV recording, and plotting. Configure runs with `make web`; existing run IDs are never overwritten.

## Games and Runs

Built-ins are Matching Pennies, RPS, and RPSLS, defined locally with per-player normalization to [0, 1]. `GameCatalog` also reads compressed custom games from `data/custom_games/`. The dashboard creates seeded random general-sum games or symmetric two-player zero-sum games whose centered matrix satisfies `A = -A.T`.

A fixed-game run ID encodes the game and payoff digest, feedback, ordered learner profile, horizon, base seed, replicate, stationary solver, and recorded runtime-environment fingerprint. It does not encode the source revision; keep outputs from incompatible scientific versions separate. Player `i` in replicate `r` of a `p`-player game receives seed:

```text
base_seed + r * p + i
```

Multiple stationary-distribution computation methods are implemented, but an experiment collection is expected to use one method consistently. In one-player experiments, the stationary method is not intended as an arbitrary per-run comparison dimension.

Full-information, bandit, and one-player batches use the configured replicate indices `0..n-1`. The dashboard exposes one Seed for both experiment modes, defaulting to 42. One-player learner seeds and, for lazy-random-walk runs, environment seeds are derived from separate domains of that base seed and replicate index. Historical-frequency runs have no environment RNG, so their CSV environment-seed fields are blank. Multi-action batches reuse each applicable derived seed at every K; differing action counts can still change random-number consumption and trajectories. CSVs retain the base and effective learner seeds and, for random-walk runs, the base and effective environment seeds.

The Horizon field accepts one value or a comma-separated list. A horizon sweep creates ordinary, separately initialized runs for every horizon/replicate pair and preserves the same base seed and replicate indices across horizons. Each learner receives its run's exact configured horizon; no endpoint is borrowed from a longer run.

Feedback and evaluation remain separate: fixed-game and one-player runs compute action regret from the sampled actions, while bandit learners receive only their sampled reward. The evaluator uses the full payoff vector offline to update only the sampled source-action row of the cumulative replacement-gain matrix.

## Long horizons and replicate execution

Learners and cumulative action-regret state update on every round. Regret summaries and CSV rows are computed only at deterministic recording checkpoints. With the default 500-point budget, horizons up to 500 record every round; longer runs use at most 500 deduplicated geometric timestamps, always including `t=1` and `t=horizon`. Fixed games write one row per player at each checkpoint. The Python runners accept `max_recorded_points` for controlled dense/sparse comparisons. This storage choice does not change seeds or run identities, and existing results are never overwritten.

Fixed-game execution maintains cumulative sampled joint-action counts. Histograms use the same checkpoint sampling rule as regret trajectories but with a 20-point budget: every round through horizon 20, otherwise at most 20 deduplicated geometric timestamps including `t=1` and `t=horizon`. They are stored once in `joint_action_histograms` on the final player-0 row. Under the default policy, CSV action fields capture every sampled joint action through horizon 500; longer runs retain only checkpoint actions and cumulative histograms. Regret loaders accept increasing checkpoint times while checking endpoint coverage, consistent metadata, and one row per player per checkpoint. Plot loaders align dense runs with the default sparse schedule; if custom recording budgets differ, replicate curves use only shared observed timestamps, without interpolation or changing the set of replicates being averaged.

`experiments.parallel.run_replicates` runs independent keyword-argument tasks through a bounded spawn-based process pool. Results are returned in input order, seed assignment is unchanged, and each worker publishes its own CSV atomically. Progress and cancellation callbacks stay in the parent; cancellation or worker failure signals active peers to clean up incomplete outputs. Finished replicate CSVs are retained. Only one task per worker is submitted at a time and nested pools are disabled. Worker count is the minimum of the requested count (or available CPUs), available CPUs, task count, and 12. Each worker has its own learner/environment state: reduce `REPLICATE_WORKERS` for memory-heavy workloads. Numerical-library thread settings are inherited unchanged; for comparable serial and parallel runs, configure the same BLAS/OpenMP thread limits for both launches.

Web replicate batches use this executor automatically when their combined horizon is at least 20,000 rounds; smaller batches avoid process startup overhead. Set `DashboardService(replicate_workers=1)` or Flask `REPLICATE_WORKERS=1` for serial execution; `None` is automatic and positive integers request a bounded worker count. Multi-action one-player batches queue one ordinary trajectory per action-count/replicate pair. The web horizon limit defaults to 1,000,000.

`make web`, `make test`, and the other Make targets default `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to `1` before Python starts. Existing environment values or explicit Make overrides take precedence. This avoids numerical-library oversubscription across replicate processes. For a direct launch, set the same limits explicitly:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python3 -m web.app
```

Use identical limits when comparing serial and parallel results. Workers never change these variables after NumPy initializes; restart a running dashboard to adopt launch-time changes.

Both fixed-game and adversarial CSV loaders validate constant metadata once per file and compare raw constant fields on every subsequent row. Runtime fingerprints hash the JSON already canonicalized at the input boundary. The fixed-game histogram loader validates action-shape compatibility, checkpoint actions, canonical payload placement, decimal horizons, count dimensions, exact totals, and cumulative monotonicity.

## Outputs

```text
results/
├── raw/              fixed-game CSVs
├── figures/          generated detail figures
├── adversarial/raw/  one-player CSVs
└── cache/            Figure Builder and equilibrium-distance artifacts
```

Plots are saved as PNG previews and same-stem vector PDFs. Raw CSVs remain authoritative. Regret curves and full-space CE/CCE L1 distances are computed per replicate, then shown as means without confidence bands. Joint-action plots use each replicate’s final cumulative histogram, without solving an LP. Figure Builder compares selected algorithm profiles for one regret notion, all three regret notions for one profile, compatible one-player action spaces, or compatible configured horizons. Exact cached selections may be restored; cache misses require an explicit generation request.

Its exploratory Log-log fit view for regret-notion comparison applies to external, internal, or swap regret. This within-run empirical log-log fit uses the existing replicate-mean cumulative action-regret trajectory and requires every checkpoint with `t >= T/10` to have strictly positive replicate-mean regret before plotting `log(mean regret)` against `log(t)` and overlaying an OLS fit. It remains distinct from true horizon scaling.

Horizon comparison combines external, internal, and swap regret in one log-log figure using final endpoints from independently configured runs. For each notion independently, it plots the arithmetic replicate mean of final cumulative action regret at every configured horizon and fits an OLS power law only when there are at least three distinct horizons and all their means are strictly positive. A valid fit reports the empirical exponent `alpha` and fitted coefficient `c = exp(intercept)` on the plot, corresponding to `R_T ≈ c T^alpha`. The PDF information page additionally records log-log OLS `R²`, the fit point count and range, and final `R_T/T` replicate mean/sample-SD/SE at the largest horizon. Invalid notions are omitted without suppressing valid ones.

Ordinary Figure Builder PDF information pages report final replicate endpoint summaries in the plotted `R_T/T` or `R_T/sqrt(T)` normalization. Equilibrium-convergence information pages similarly report final CE and CCE full-space L1-distance summaries. In all endpoint summaries, SD is the sample standard deviation across replicates (`ddof=1`) and `SE = SD/sqrt(n)`; single-replicate SD and SE are reported as unavailable. These are descriptive Monte Carlo statistics, with no confidence interval or significance test implied. The `alpha`, `c`, and `R²` values are empirical finite-horizon fit diagnostics, not theoretical or asymptotic regret guarantees.

Fixed-game and one-player jobs publish ordinary CSVs without automatically rendering regret figures. The CSV loader checks constant seed/runtime metadata once per file, still checks metadata equality and observations on every row, and validates trajectory endpoints.

Duplicate physical CSV files for one replicate are an abnormal storage state. Statistical and dashboard summaries select one deterministic canonical file per replicate; maintenance and deletion membership includes every physical file in the scientific group. Consumers that require unambiguous replicate membership may reject a group with duplicate-replicate files.

Figure Builder’s regret-notion comparison uses solid External, dashed Swap, and dash-dot Internal curves, with Swap above External when they overlap. Profile and action-space comparisons use solid lines with deterministic color and marker identities; their markers are phase-staggered at 0.24 spacing. Action-space comparisons hold environment, feedback, horizon, base seed, replicate set, runtime environment, profile, regret notion, and view fixed while plotting one ordinary time trajectory per compatible K. CSVs retain learner actions and punishment or current-best-action data. Random-walk rewards are precomputed from the effective environment seed and can therefore be shared exactly across learners with the same K and replicate. CSVs also record the canonical runtime environment and its fingerprint; the fingerprint participates in run identity, so changes to recorded Python/package/lock fields change the run ID. It does not capture every numerical build, platform detail, thread setting, or source revision.

Core files are `games.py`, `game_catalog.py`, `runner.py`, `recorder.py`, `result_schema.py`, and `results.py`. Analysis is documented in [metrics](../metrics/README.md).
