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
- Filter figures and summaries by game or environment and feedback mode.
- Create reproducible general-sum or symmetric two-player zero-sum games.
- Switch the **Experiments** page between fixed games and one-player historical-frequency or lazy-random-walk environments.
- Enter one action count for an ordinary one-player experiment or multiple counts for action-space scaling.

The queue has one worker and reserves run IDs on submission. Fixed-game and one-player experiments accept a replicate count. Compatible fixed-game groups share the game/payoff digest, feedback, learner profile, horizon, base seed, stationary solver, and runtime-environment fingerprint. Their strategy-weighted regret and full-space CE/CCE-distance views report replicate means.

One-player regret diagnostics are grouped by environment and feedback mode, with algorithm-only plot legends. Ordinary and action-space scaling batches derive learner and environment seeds from distinct domains of the shared base seed plus the replicate index, while retaining the base and derived seeds in their CSVs. The same **Actions** field selects the ordinary or scaling workflow; there is no separate scaling form. Generated figure cards open in the same viewer. Figures show means without confidence bands; previews and PDF downloads use the same selection.

The sidebar configures the next run. The page flows through **Execution / Job status**, **Analysis / Result filters**, **Visualization / Generated figures**, and **Data / Recorded output**; relevant one-player scaling results appear separately under **Scaling / Action-space scaling**. Result filters coordinate Figure Builder and ordinary summaries, while scaling remains separate. Queue buttons submit in the background and update job status without navigating away. When a job finishes, **Refresh results** loads the latest results on demand without interrupting your current analysis.

## Figures and Data

PNG previews have matching vector PDFs. Ordinary experiment jobs save CSVs; regret figures are generated explicitly through Figure Builder. Detail figures are generated lazily and cached. Result details retain mean empirical joint-action heatmaps for supported two-player games and asynchronously computed full-space CE/CCE convergence figures. Action-space scaling keeps its visible generated figures and downloads.

**Figure Builder** compares either one or more algorithm profiles for the selected regret filters, or external, internal, and swap regret together for one profile. Exact cached selections may restore automatically; a cache miss requires **Generate figures**. The collected-PDF download combines generated PDFs in their displayed order. Exporting does not rerun experiments or rebuild plots.

| Location | Contents |
|---|---|
| `results/raw/` | Fixed-game CSVs |
| `results/figures/` | Generated detail figures |
| `results/adversarial/` | One-player CSVs and action-space scaling figures |
| `results/cache/` | Figure Builder and equilibrium-distance artifacts |
| `data/custom_games/` | Saved custom games |

`make reset` and **Reset all experiment results** remove all experiment-derived data, generated figures, and experiment-dependent caches. Source and configuration inputs such as custom games, plus `.gitkeep` placeholders, are preserved.

POST parameters, CSRF tokens, and filenames are validated. For a stable session secret across restarts:

```bash
export SWAP_REGRET_WEB_SECRET="<stable-random-secret>"
make web
```
