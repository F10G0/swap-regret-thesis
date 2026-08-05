# Web Dashboard

Local Flask interface for running experiments, managing custom games, and exploring results.

```bash
make install
make web
```

Trajectory comparison is excluded by default. Include it at installation time with `make install-experimental`; it is not a runtime switch.

## What the Dashboard Does

- configures built-in or custom games and one algorithm per player;
- queues experiments, reports progress, and supports cancellation;
- filters regret figures and final summaries;
- downloads or deletes recorded runs;
- displays joint-action, equilibrium-weight, and CE/CCE-distance figures;
- creates and inspects reproducible general-sum or two-player zero-sum payoff tensors;
- runs an isolated one-player adaptive stress test from the **Adversarial** page.

The queue has one experiment worker. Run IDs are reserved when submitted, so duplicate jobs cannot overwrite results. Replicates are grouped only when the game/payoff digest, feedback, evaluation, algorithm profile, horizon, base seed, and stationary solver match.

Bandit experiments select a replicate count and use indices 0 through n−1. Fixed-game full-information uses index 0. Adversarial submissions are single runs and have no replicate setting.

## Figures

Regret figures are updated after experiments finish. The dashboard previews PNG files and downloads matching vector PDFs. Joint-action and full-space equilibrium-distance figures are generated lazily and cached under `results/figures/details/`. CE/CCE profile-weight heatmaps for custom zero-sum games are cached under `data/custom_games/.equilibria/`. Built-in static heatmaps can be rebuilt with:

```bash
make precompute-equilibria
```

With the optional build enabled, `/experimental/trajectory-comparisons` adds explicit comparison selection and a manual **Generate** action. Changed settings are rendered only when **Generate** is pressed again. See [the experimental guide](../experimental/equilibrium_trajectory/README.md).

## Data

| Location | Contents |
|---|---|
| `results/raw/` | Experiment CSVs |
| `results/figures/` | Regret and detail figures |
| `results/adversarial/` | Adversarial stress-test CSVs and figures |
| `results/cache/experimental/` | Regenerable trajectory geometry cache |
| `data/custom_games/` | Custom payoff tensors |
| `data/custom_games/.equilibria/` | Cached custom-game CE/CCE heatmaps |
| `web/static/equilibria/` | Precomputed equilibrium PNG/PDF pairs |

Reset removes experiment CSVs and figures, but keeps custom games and their heatmaps, static equilibrium assets, and the regenerable geometry cache.

The adversarial page selects a learner and memory-window length, then shows expected/realized `R/t`, `R/sqrt(t)`, and action-frequency diagnostics. Clearing adversarial results removes both its CSVs and figures. Game, CE/CCE, and trajectory controls are omitted because the adaptive scenario has no fixed payoff tensor.

## Structure and Security

`routes.py` handles HTTP requests, `services.py` coordinates operations, `jobs.py` owns the queue, and `result_index.py` / `result_groups.py` load and aggregate results. Optional trajectory logic remains under `experimental/equilibrium_trajectory/`.

POST requests use CSRF tokens, filenames are validated, and all submitted parameters are checked before use. To keep sessions stable across restarts, set:

```bash
export SWAP_REGRET_WEB_SECRET="<stable-random-secret>"
make web
```
