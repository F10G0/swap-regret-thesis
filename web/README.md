# Web Dashboard

Local Flask interface for experiments, custom games, and saved results.

```bash
make install
make web
```

The optional trajectory comparison is selected at installation with `make install-experimental`; it is not a runtime switch.

## Features

- Configure built-in or custom games and one learner per player.
- Queue, monitor, cancel, download, and delete fixed-game runs.
- Filter regret summaries and inspect joint-action and CE/CCE figures.
- Create reproducible general-sum or symmetric two-player zero-sum games.
- Run historical-frequency and lazy-random-walk stress tests from **Adversarial**.

The queue has one worker and reserves run IDs on submission. Fixed-game full-information uses replicate 0; bandit runs accept a replicate count. Compatible replicate groups share the game/payoff digest, feedback, evaluation, learner profile, horizon, base seed, and stationary solver. Adversarial runs are single experiments.

## Figures and Data

PNG previews have matching vector PDFs. Regret plots update after a run. Detail figures are generated lazily and cached. Custom-game equilibrium heatmaps stay with the game; built-in heatmaps can be rebuilt with `make precompute-equilibria`.

| Location | Contents |
|---|---|
| `results/raw/` | Fixed-game CSVs |
| `results/figures/` | Regret and detail figures |
| `results/adversarial/` | Stress-test CSVs and figures |
| `results/cache/` | Regenerable geometry caches |
| `data/custom_games/` | Custom games and cached heatmaps |
| `web/static/equilibria/` | Built-in equilibrium assets |

`make reset` removes experiment CSVs and figures but keeps custom games, their heatmaps, static assets, and geometry caches. The adversarial page's **Clear results** action removes both its CSVs and figures.

An experimental build adds `/experimental/trajectory-comparisons`; comparisons render only when **Generate** is pressed. See the [experimental guide](../experimental/equilibrium_trajectory/README.md).

POST parameters, CSRF tokens, and filenames are validated. For a stable session secret across restarts:

```bash
export SWAP_REGRET_WEB_SECRET="<stable-random-secret>"
make web
```
