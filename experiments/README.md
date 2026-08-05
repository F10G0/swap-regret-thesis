# Experiments

Experiment construction, execution, CSV recording, and standard plots. Configure runs with `make web`; use `make plot` only to rebuild figures from saved CSVs. Existing run IDs are never overwritten.

## Games and Runs

Built-ins are RPS, RPSLS, Matching Pennies, and five normalized 21 × 21 Bertrand games. `GameCatalog` also reads compressed custom games from `data/custom_games/`. The dashboard creates reproducible general-sum games or symmetric two-player zero-sum games whose centered matrix satisfies `A = -A.T`.

A fixed-game run is identified by the game and payoff digest, feedback, regret evaluation, ordered learner profile, horizon, base seed, replicate, and stationary solver. Player `i` in replicate `r` of a `p`-player game receives seed:

```text
base_seed + r * p + i
```

Full-information runs use replicate 0. Bandit batches use indices `0..n-1`. Adversarial submissions are single runs with separate applicable environment and learner seeds.

Feedback and evaluation remain separate: a bandit learner receives only its realized reward even when expected regret is evaluated offline.

## Outputs

```text
results/
├── raw/          fixed-game CSVs
├── figures/      regret and detail plots
└── adversarial/  stress-test CSVs and plots
```

Plots are saved as PNG previews and same-stem vector PDFs. They include replicate summaries, separate expected/realized regret, joint-action and CE/CCE profile-weight heatmaps, and full-space CE/CCE L1 distance. Equilibrium distances are computed per replicate before aggregation.

Adversarial plots show `R/t`, `R/sqrt(t)`, learner actions, and the punished or current best action. The random-walk reward trajectory is precomputed from its environment seed and can therefore be shared exactly across learners.

Core files are `games.py`, `game_catalog.py`, `runner.py`, `recorder.py`, and `results.py`. Analysis is documented in [metrics](../metrics/README.md).
