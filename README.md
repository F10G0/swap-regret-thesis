# Swap Regret Thesis Project

Framework for external, internal, and swap-regret experiments in finite repeated games and one-player stress tests. It produces seeded CSV results, CE/CCE analyses, PNG/PDF figures, custom games, and a local Flask dashboard.

## Quick Start

Python 3.10 or newer is required. Use an isolated Python environment. Node.js
12.22.0 or newer is needed only for frontend tests, not for the Flask dashboard.

```bash
make install
npm install
make test
make web
```

`make install` uses the pinned Python `requirements.lock`. `npm install` installs
the frontend-test-only `jsdom` dependency declared in `package.json`. The direct
`jsdom` version is pinned, but there is no npm lockfile yet, so transitive npm
dependencies are not frozen. `make test` checks Node.js and `jsdom` before running
the complete pytest suite; a successful run should have no unintended frontend
skips. Direct `python3 -m pytest` can still skip frontend tests when those
prerequisites are absent. To run the dashboard without testing, `make install`
followed by `make web` is sufficient; Flask normally serves it at
`http://127.0.0.1:5000/`.

## Scope

| Feedback | Display name (`experiment ID`) |
|---|---|
| Full information | Hedge (`hedge`), OptHedge (`optimistic_hedge`), BM-Hedge (`bm_hedge`), BM-OptHedge (`bm_optimistic_hedge`), Ito-Hedge (`ito_hedge`), RM (`regret_matching`), SRM (`stationary_regret_matching`) |
| Bandit | EXP3 (`auer_exp3`), EXP3-IX (`exp3_ix`), BM-EXP3 (`bm_exp3`), Ito-Tsallis (`ito_tsallis`), LCE-IX (`lce_ix`) |

Display labels differ from class names: OptHedge is `OptimisticHedge`, EXP3 is the Auer et al. `AuerExp3`, and Ito-Tsallis uses `TsallisINF` inner learners. There is no generic `exp3` experiment ID.

Built-in games are Matching Pennies, RPS, and RPSLS, defined locally as fixed payoff matrices. The dashboard also creates random general-sum games and symmetric two-player zero-sum games. One-player experiments cover a historical-frequency adversary and an action-independent lazy reward walk. Multiple action counts or horizons can be queued as batches of ordinary runs.

Fixed-game and one-player experiments compute external, internal, and swap action regret from the sampled action trajectory. Bandit learners receive only their sampled reward; the evaluator uses the full payoff vector offline. With the default recording policy, regret trajectories record every round through horizon 500 and at most 500 geometric checkpoints for longer runs. Empirical joint-action and equilibrium-distance trajectories use the same sampling rule with a 20-point budget. Regret figures report replicate-mean action regret, a Monte Carlo estimate of expected action regret, without confidence bands or interval toggles. Empirical joint-action heatmaps show the same recorded play, and equilibrium convergence uses full-space L1 distance to CE and CCE.

Figure Builder can compare replicate-mean final cumulative action regret across configured horizons. Each point averages separately initialized experiment runs for exactly that horizon, so horizon-dependent learner tuning is respected. Its fitted slope is an empirical finite-horizon scaling exponent, not a proof of asymptotic regret order.

Payoffs are finite values in `[0, 1]`. Fixed-game tensors have shape:

```text
(n_players, actions_player_0, ..., actions_player_(n-1))
```

## Commands

| Command | Purpose |
|---|---|
| `make all` | Install project dependencies |
| `make install` | Install Python project and test dependencies |
| `make web` | Start the local dashboard (no Node.js required) |
| `make test` | Require Node.js/jsdom, then run the complete test suite |
| `make clean` | Remove Python caches and selected figure/game staging directories |
| `make reset` | Remove all experiment-derived results, figures, and caches |

`make all` intentionally does not run the test suite. Run `make help` for the live command list.

## Data

```text
results/
├── raw/              fixed-game CSVs
├── figures/          generated detail figures
├── adversarial/raw/  one-player CSVs
└── cache/            Figure Builder and equilibrium-distance artifacts
```

Generated files under `results/` are local artifacts and are not guaranteed to be version-controlled.

Run IDs encode the configured experiment fields and a recorded runtime fingerprint, but not the source-code revision. Result loaders assume stored files are compatible with the current implementation; keep results from incompatible scientific versions separate. Defaults such as the horizon, seeds, replicate count, tolerances, and stationary solver live in `config.py`.

The install commands use `requirements.lock`. Each CSV records the Python version and implementation, NumPy/SciPy/Matplotlib versions (and available VCS commits), the lock-file hash when available, and their runtime fingerprint. This is not a complete machine fingerprint. The dashboard uses one shared experiment Seed, defaulting to 42. One-player learner seeds and, for lazy-random-walk runs, environment seeds use separate domains of that base seed and replicate; CSVs record the applicable base and effective seeds.

CE/CCE incentive constraints and L1 projection are implemented locally using NumPy and SciPy’s `linprog(method="highs")`. Empirical heatmaps count recorded joint actions directly and require no equilibrium LP.

`make reset` preserves source and configuration inputs, including saved custom games, along with `.gitkeep` placeholders. Everything derived from experiment results is removed so that no generated figure or cache outlives its source data.

## Research Materials

[Documentation and research materials](docs/README.md) describes the submitted thesis PDF and source archive in `docs/thesis/`, the earlier proposal in `docs/proposal/`, and the selected papers and project description in `docs/references/`. These historical materials are separate from generated experiment outputs in `results/`; later code or documentation changes do not alter the submitted thesis.

## Guides

- [Algorithms](algorithms/README.md)
- [Environments](environments/README.md)
- [Experiments](experiments/README.md)
- [Metrics](metrics/README.md)
- [Web dashboard](web/README.md)
