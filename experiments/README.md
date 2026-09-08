# Experiments

Experiment construction, execution, CSV recording, and standard plots. Configure runs with `make web`; use `make plot` only to rebuild figures from saved CSVs. Existing run IDs are never overwritten.

## Games and Runs

Built-ins are RPS, RPSLS, Matching Pennies, and five normalized 21 × 21 Bertrand games. `GameCatalog` also reads compressed custom games from `data/custom_games/`. The dashboard creates reproducible general-sum games or symmetric two-player zero-sum games whose centered matrix satisfies `A = -A.T`.

A fixed-game run is identified by the game and payoff digest, feedback, regret evaluation, ordered learner profile, horizon, base seed, replicate, stationary solver, implementation version, and runtime-environment fingerprint. Older CSVs remain readable as legacy version 0. Player `i` in replicate `r` of a `p`-player game receives seed:

```text
base_seed + r * p + i
```

Full-information, bandit, and one-player batches use the configured replicate indices `0..n-1`. One-player learner and environment seeds are derived from separate domains plus the replicate index, so they remain independent even when the configured base seeds are equal. Action-space scaling batches apply the same derivation at every K, which preserves paired environment randomness across learners. CSVs retain both configured base seeds and effective derived seeds.

Feedback and evaluation remain separate: one-player and fixed-game runs can record expected regret, realized regret, or both, and a bandit learner still receives only its sampled reward when expected regret is evaluated offline.

## Long horizons and replicate execution

Learners and cumulative regret state update on every round. Only the selected regret trackers are allocated and updated: `expected`, `realized`, or both. Regret summaries and CSV rows are computed only at deterministic recording checkpoints. The default uses at most 2,000 timestamps (a mixture of logarithmic and linear spacing), always including `t=1` and `t=horizon`; short runs with at most 2,000 rounds remain dense. Fixed games write one row per player at each checkpoint. The Python runners accept `max_recorded_points` for controlled dense/sparse comparisons. This storage choice does not change seeds or run identities, and existing results are never overwritten.

Sparse fixed-game rows also contain `action_history`: a versioned, base64-encoded zlib block of little-endian uint32 actions for that player since the preceding checkpoint. This lossless, compact history keeps joint-action heatmaps and arbitrary CE/CCE trajectory checkpoints exact without per-round CSV rows or companion files. `load_result_action_profiles` reconstructs the complete action sequence; it never treats sparse sampled actions as the full history. Dense legacy CSVs remain supported. Regret loaders accept increasing checkpoint times while checking endpoint coverage, consistent metadata, and one row per player per checkpoint. Plot loaders align legacy dense runs with the default sparse schedule; if custom recording budgets differ, replicate curves use only shared observed timestamps, without interpolation or changing the set of replicates being averaged.

`experiments.parallel.run_replicates` runs independent keyword-argument tasks through a bounded spawn-based process pool. Results are returned in input order, seed assignment is unchanged, and each worker publishes its own CSV atomically. Progress and cancellation callbacks stay in the parent; cancellation or worker failure signals active peers to clean up incomplete outputs. Finished replicate CSVs are retained. Only one task per worker is submitted at a time and nested pools are disabled. Automatic mode uses all available logical CPUs (respecting CPU affinity where supported), bounded by the number of tasks; there is no fixed four-worker cap. Explicit worker requests are also bounded by CPUs and task count. Each worker has its own learner/environment state: reduce `REPLICATE_WORKERS` for memory-heavy workloads. Numerical-library thread settings are inherited unchanged to preserve reproducibility; for CPU-bound batches, configure the same BLAS/OpenMP thread limits for both serial and parallel launches.

Web replicate batches and action-space scaling use this executor automatically when their combined horizon is at least 20,000 rounds; smaller batches avoid process startup overhead. Set `DashboardService(replicate_workers=1)` or Flask `REPLICATE_WORKERS=1` for serial execution; `None` is automatic and positive integers request a bounded worker count. Scaling writes final rows in action-count/replicate order and records only endpoint summaries in its temporary trajectories. The web horizon limit defaults to 1,000,000.

`make web`, `make test`, and the other Make targets default `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to `1` before Python starts. Existing environment values or explicit Make overrides take precedence. This avoids numerical-library oversubscription across replicate processes. For a direct launch, set the same limits explicitly:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python3 -m web.app
```

Use identical limits for serial/parallel reproducibility checks. Workers never change these variables after NumPy initializes; restart a running dashboard to adopt launch-time changes.

Both fixed-game and adversarial CSV loaders validate constant metadata once per file and compare raw constant fields on every subsequent row. Runtime fingerprints hash the JSON already canonicalized at the input boundary. Dynamic checkpoint/player/action-block checks remain in the loaders. Action-history reconstruction trusts their grouping guarantees while checking action-shape compatibility, action bounds, compressed payloads, and complete dense histories. Empirical counting then consumes those validated profiles without per-round conversion or validation.

## Outputs

```text
results/
├── raw/          fixed-game CSVs
├── figures/      regret and detail plots
├── adversarial/  stress-test and action-scaling results
└── cache/        regenerable plotting and equilibrium data
```

Plots are saved as PNG previews and same-stem vector PDFs. Validated, downsampled fixed-game rows are cached under `results/cache/plot_rows/`, while raw CSVs remain authoritative. Regret curves and full-space CE/CCE L1 distances are computed per replicate, then shown as means with pointwise Student-t 95% confidence intervals. Regret figures with intervals also cache a mean-only pair for the per-figure web toggle. Expected and realized regret remain separate. Joint-action plots show the replicate mean; theoretical CE/CCE profile-weight heatmaps depend only on the game.

After a one-player batch, only its environment/feedback/action-count figure group is rebuilt, preserving unrelated figures. The collector reads just the first row of unrelated CSVs. Validated adversarial plotting trajectories are cached under `results/adversarial/cache/plot_rows/`; source path, size, modification/change timestamps, sampling budget, and cache version determine reuse. Missing or damaged caches are regenerated. The CSV loader checks constant seed/runtime metadata once per file, still checks metadata equality and observations on every row, and validates trajectory endpoints. Manual full rebuilds and deletion workflows continue to cover all groups, using the same cache.

Adversarial plots show replicate-mean environment-specific `R/t` and `R/sqrt(t)` diagnostics with Student-t 95% confidence intervals. The action-scaling view plots final target regret against K with the same interval convention. CSVs retain learner actions and punishment or current-best-action data. Random-walk rewards are precomputed from the effective environment seed and can therefore be shared exactly across learners with the same K and replicate. New CSVs also record the canonical runtime environment and its fingerprint; the fingerprint participates in run identity so incompatible dependency environments cannot silently share a run ID.

Core files are `games.py`, `game_catalog.py`, `runner.py`, `recorder.py`, `result_schema.py`, and `results.py`. Analysis is documented in [metrics](../metrics/README.md).
