# Web Dashboard

Local Flask interface for experiments, custom games, queued execution, and recorded results.

```bash
make web
```

The installed entry point is `swap-regret-web`. Submission limits are 100,000 rounds and 100 bandit replicates.

## Pages

### Dashboard

- Built-in and stored custom-game selection.
- Compact Game, Feedback, and Regret evaluation cards.
- One algorithm selector per player and Player 0 → all synchronization.
- Paired Horizon/Seed and Starting replicate/Bandit replicates controls.
- Single-profile or built-in two-player all-pairs submission.
- Queue progress, cancellation, raw downloads, and deletion.
- Regret-figure and final-summary filters, full-size figures, sorting, and parameter reuse.
- Theoretical and empirical detail figures.

Form values, both filter sets, trajectory controls, and the interface theme persist in `localStorage`. The theme selector at the top of the sidebar offers green, blue, purple, orange, and red accents; plot palettes are independent.

### Custom Games

`/custom-games` creates reproducible random `.npz` games under `data/custom_games/`.

- 2–8 players, 1–100 actions each, and at most 1,000,000 payoff values.
- Stored-game listing, download, and deletion.
- Tensor statistics and selectable two-dimensional payoff slices.
- Payoff owner, row player, column player, and fixed actions for remaining players.
- Row action 0 displayed at the top.

A game cannot be deleted while results reference it. Result reset preserves custom games.

Custom games receive regret figures, full-dimensional CE/CCE distance, and projected trajectories. They do not receive matrix heatmaps.

## Queue

`JobManager` has one FIFO worker; submissions can continue while another job runs. It tracks progress, timestamps, cancellation, reservations, and the latest 20 jobs.

Run IDs are reserved at queue time to prevent duplicates. Queued cancellation releases reservations immediately; running jobs stop at round boundaries. Result reset and custom-game deletion require an idle queue.

## Results and Replicates

`ResultIndex` caches final CSV rows by path, modification time, and size. Malformed files become warnings instead of blocking valid results.

Runs group only when these match:

```text
game, feedback, regret evaluation, ordered algorithm profile,
horizon, base seed, stationary solver
```

Replicate is the varying dimension. Final regret is averaged with a 95% confidence interval; constituent CSVs remain separate.

## Figure Lifecycle

### Regret figures

Experiment completion rebuilds the affected game's top-level figures atomically. Rebuild figures is only needed after importing/changing CSVs or deliberately removing top-level plots.

### Static equilibrium heatmaps

Built-in maximum CE/CCE profile-weight PNGs live under `web/static/equilibria/`. They are fixed-blue, lower-origin assets independent of results:

```bash
make precompute-equilibria
```

Result reset does not remove them.

### Lazy detail figures

- Built-in two-player joint-action heatmaps average replicate distributions.
- CE/CCE distance supports built-in and custom n-player games.
- Distance starts at iteration 1 and is computed per replicate before averaging.
- Projected mean trajectories use 2–50 uniform nodes.
- Hide first excludes round 1 before PCA fitting and drawing.

Distance and trajectory use a dedicated two-worker executor and independent caches. A trajectory cache name includes its point count and `from_round_1` or `hide_round_1`; distance is shared across trajectory settings. Requests return HTTP 202 while work continues, and the previous image is hidden during replacement.

Deleting results invalidates their detail caches. Reset cancels tracked convergence work and prevents stale publication.

## Module Ownership

| Module | Responsibility |
|---|---|
| `routes.py` | Thin HTTP handlers |
| `view_models.py` | Template and JSON presentation data |
| `services.py` | Submission, cleanup, custom games, and figure publication |
| `jobs.py` | FIFO jobs, progress, cancellation, and reservations |
| `result_index.py` | Cached final-row loading |
| `result_groups.py` | Replicate aggregation and confidence intervals |
| `experiment_modes.py` | Feedback-mode runners and algorithm registries |
| `presentations.py` | Built-in game labels and descriptions |
| `validation.py` | Form parsing and leaf-filename validation |
| `equilibrium_figures.py` | Static asset naming |

Browser code:

- `common.js`: theme persistence and confirmations;
- `dashboard.js`: form/filter persistence, jobs, figures, and result details;
- `custom_games.js`: action fields and payoff slices;
- `dashboard.css`: shared responsive layout and theme variables.

## Security and Data

POST forms use session-backed CSRF tokens. Serve/download routes accept validated leaf filenames. All game, algorithm, tensor, and experiment inputs are validated before mutation.

Set a stable session secret when needed:

```bash
export SWAP_REGRET_WEB_SECRET="<stable-random-secret>"
make web
```

| Location | Purpose | Removed by result reset |
|---|---|---|
| `results/raw/` | Experiment CSVs | Yes |
| `results/figures/` | Regret and detail figures | Yes |
| `results/index.html` | Static report | Yes |
| `data/custom_games/` | Custom games | No |
| `web/static/equilibria/` | Precomputed equilibrium assets | No |
