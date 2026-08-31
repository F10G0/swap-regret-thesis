# Experiments

Experiment construction, execution, CSV recording, and standard plots. Configure runs with `make web`; use `make plot` only to rebuild figures from saved CSVs. Existing run IDs are never overwritten.

## Games and Runs

Built-ins are RPS, RPSLS, Matching Pennies, and five normalized 21 × 21 Bertrand games. `GameCatalog` also reads compressed custom games from `data/custom_games/`. The dashboard creates reproducible general-sum games or symmetric two-player zero-sum games whose centered matrix satisfies `A = -A.T`.

A fixed-game run is identified by the game and payoff digest, feedback, regret evaluation, ordered learner profile, horizon, base seed, replicate, stationary solver, and implementation version. Older CSVs remain readable as legacy version 0. Player `i` in replicate `r` of a `p`-player game receives seed:

```text
base_seed + r * p + i
```

Full-information, bandit, and one-player batches use the configured replicate indices `0..n-1`. In one-player batches, replicate `r` uses learner seed `base + r` and, for random walks, environment seed `base + r`. Action-space scaling batches apply the same schedule at every K.

Feedback and evaluation remain separate: one-player and fixed-game runs can record expected regret, realized regret, or both, and a bandit learner still receives only its sampled reward when expected regret is evaluated offline.

## Outputs

```text
results/
├── raw/          fixed-game CSVs
├── figures/      regret and detail plots
├── adversarial/  stress-test and action-scaling results
└── cache/        regenerable plotting and equilibrium data
```

Plots are saved as PNG previews and same-stem vector PDFs. Validated, downsampled fixed-game rows are cached under `results/cache/plot_rows/`, while raw CSVs remain authoritative. Regret curves and full-space CE/CCE L1 distances are computed per replicate, then shown as means with pointwise Student-t 95% confidence intervals. Regret figures with intervals also cache a mean-only pair for the per-figure web toggle. Expected and realized regret remain separate. Joint-action plots show the replicate mean; theoretical CE/CCE profile-weight heatmaps depend only on the game.

Adversarial plots show replicate-mean environment-specific `R/t` and `R/sqrt(t)` diagnostics with Student-t 95% confidence intervals. The action-scaling view plots final target regret against K with the same interval convention. CSVs retain learner actions and punishment or current-best-action data. Random-walk rewards are precomputed from the environment seed and can therefore be shared exactly across learners with the same K.

Core files are `games.py`, `game_catalog.py`, `runner.py`, `recorder.py`, `result_schema.py`, and `results.py`. Analysis is documented in [metrics](../metrics/README.md).
