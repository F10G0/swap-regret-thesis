# Web Dashboard

Local Flask interface for experiments, custom games, and saved results.

```bash
make install
make web
```

The optional trajectory comparison is selected at installation with `make install-experimental`; it is not a runtime switch.

The **Experiments** page uses one template, controller, result filters, figure grid and dialog, and sortable summary framework for fixed games and one-player environments. Both modes also share feedback, regret evaluation, horizon, seed, algorithm, and replicate controls.

## Features

- Configure built-in or custom games and one learner per player.
- Queue, monitor, cancel, download, and delete fixed-game runs.
- Filter figures and summaries by game or environment and feedback mode.
- Create reproducible general-sum or symmetric two-player zero-sum games.
- Switch the **Experiments** page between fixed games and one-player historical-frequency or lazy-random-walk environments.
- Sweep a configurable list of action counts and plot final regret against K.

The queue has one worker and reserves run IDs on submission. Fixed-game and one-player experiments accept a replicate count. Compatible fixed-game groups share the game/payoff digest, feedback, evaluation, learner profile, horizon, base seed, and stationary solver. Their regret and CE/CCE-distance views report replicate means with pointwise Student-t 95% confidence intervals.

One-player regret diagnostics are grouped by environment and feedback mode, with algorithm-only plot legends. Ordinary and action-scaling batches use matched base-plus-replicate seed schedules. Every experiment figure opens in the same viewer. Eligible figures have independent controls for switching between Student-t 95% confidence bands and cached mean-only figures; previews and PDF downloads always use the same selection.

The sidebar configures the next run. The global result filters control saved figures, summaries, and analysis: choosing one game shows its CE/CCE analysis, choosing one environment shows its rule, and choosing **All** hides the analysis panel. Completed jobs refresh the page automatically.

## Figures and Data

PNG previews have matching vector PDFs. Regret plots update after a run; a manual rebuild uses the same background queue and refreshes the page when finished. Detail figures are generated lazily and cached. Custom-game equilibrium heatmaps stay with the game; built-in heatmaps can be rebuilt with `make precompute-equilibria`.

| Location | Contents |
|---|---|
| `results/raw/` | Fixed-game CSVs |
| `results/figures/` | Regret and detail figures |
| `results/adversarial/` | Stress-test CSVs and figures |
| `results/cache/` | Regenerable plot and geometry caches |
| `data/custom_games/` | Custom games and cached heatmaps |
| `web/static/equilibria/` | Built-in equilibrium assets |

`make reset` removes experiment CSVs, figures, and their row cache, but keeps custom games, heatmaps, static assets, and geometry caches. **Clear results** in the one-player mode removes both its CSVs and figures.

An experimental build adds `/experimental/trajectory-comparisons`; comparisons render only when **Generate** is pressed. See the [experimental guide](../experimental/equilibrium_trajectory/README.md).

POST parameters, CSRF tokens, and filenames are validated. For a stable session secret across restarts:

```bash
export SWAP_REGRET_WEB_SECRET="<stable-random-secret>"
make web
```
