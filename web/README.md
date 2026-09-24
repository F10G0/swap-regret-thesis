# Web Dashboard

Local Flask interface for experiments, custom games, and saved results.

```bash
make install
make web
```

The **Experiments** page uses one template, controller, result filters, figure grid and dialog, and sortable summary framework for fixed games and one-player environments. Both modes share one general Seed, defaulting to 42, while their other form choices remain mode-specific.

## Features

- Configure built-in or custom games and one learner per player.
- Queue, monitor, and cancel jobs, and download recorded results.
- Filter figures and summaries by game or environment and by full-information, bandit, or both feedback modes.
- Create seeded random general-sum or symmetric two-player zero-sum games.
- Switch the **Experiments** page between fixed games and one-player historical-frequency or lazy-random-walk environments.
- Enter one or more horizons in either experiment mode, and one or more action counts in one-player mode, to queue ordinary experiments.

The queue has one worker and reserves run IDs on submission. Fixed-game and one-player experiments accept a replicate count. Compatible fixed-game groups share the game/payoff digest, feedback, learner profile, horizon, base seed, stationary solver, and runtime-environment fingerprint. Their action-regret views report replicate means, which estimate expected action regret, while full-space CE/CCE-distance views report mean distances.

One-player regret diagnostics are grouped by environment and feedback mode, with algorithm-only plot legends. Every action count uses the ordinary result schema and catalog. Multi-action batches derive learner seeds from the shared base seed and replicate; lazy-random-walk batches also derive environment seeds from a separate domain. CSVs record the applicable base and derived seeds. Generated figure cards open in the same viewer. Figures show means without confidence bands; previews and PDF downloads use the same selection.

The sidebar configures the next run. The page flows through **Execution / Job status**, **Analysis / Result filters**, **Visualization / Generated figures**, and **Data / Recorded output**. Result filters coordinate Figure Builder and ordinary summaries. Queue buttons submit in the background and update job status without navigating away. When a job finishes, **Refresh results** loads the latest results on demand without interrupting your current analysis.

Result filtering and sorting are server-authoritative, and dashboard browsing paginates complete scientific groups on the server. Fixed-game group details are fetched on demand rather than embedded in the initial page. **Delete filtered experiments** applies to all matching scientific groups across pages, not just the visible page: membership is recomputed at preview and again before deletion, so changes require a fresh review. Figure Builder continues to use the full compatible result catalog, independent of the current page.

## Figures and Data

PNG previews have matching vector PDFs. Experiment jobs save CSVs; regret figures are generated explicitly through Figure Builder. Detail figures are generated lazily and cached. Result details retain mean empirical joint-action heatmaps for supported two-player games and asynchronously computed full-space CE/CCE convergence figures.

The paired CE/CCE detail figure is unavailable when either analysis exceeds the fixed 128 MiB estimate of the current LP's dense allocations. The game, regret results, and other feasible views remain usable; analyses within the budget still use the unchanged full-space L1 LP solved numerically.

**Figure Builder** compares one or more algorithm profiles for one regret notion, external/internal/swap regret together for one profile, compatible action spaces, or compatible configured horizons. Its **Both** feedback filter can combine otherwise-compatible full-information and bandit profiles because reduction variants have unique result IDs. Horizon comparison plots replicate-mean final cumulative action regret from separately initialized ordinary runs; the other modes plot trajectories over time. Exact cached selections may restore automatically; a cache miss requires **Generate figures**. The collected-PDF download combines generated PDFs in their displayed order. Exporting does not rerun experiments or rebuild plots.

| Location | Contents |
|---|---|
| `results/raw/` | Fixed-game CSVs |
| `results/figures/` | Generated detail figures |
| `results/adversarial/raw/` | One-player CSVs |
| `results/cache/` | Figure Builder and equilibrium-distance artifacts |
| `data/custom_games/` | Saved custom games |

`make reset` and **Reset all experiment results** remove all experiment-derived data, generated figures, and experiment-dependent caches. Source and configuration inputs such as custom games, plus `.gitkeep` placeholders, are preserved.

POST parameters, CSRF tokens, and filenames are validated. For a stable session secret across restarts:

```bash
export SWAP_REGRET_WEB_SECRET="<stable-random-secret>"
make web
```
