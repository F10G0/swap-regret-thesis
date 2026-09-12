# Swap Regret Thesis Project

Framework for external, internal, and swap-regret experiments in finite repeated games and one-player stress tests. It provides reproducible CSV results, CE/CCE analysis, publication-ready plots, custom games, and a local Flask dashboard.

## Quick Start

Python 3.10 or newer is required.

```bash
make install
make test
make web
```

## Scope

| Feedback | Learners |
|---|---|
| Full information | Hedge, Regret Matching, SRM, Blum–Mansour, Ito |
| Bandit | AuerExp3, Exp3-IX, Blum–Mansour, Ito, LCE-IX |

Built-in games are RPS and RPSLS, defined locally as fixed payoff matrices. Matching Pennies is a local test-only fixture. The dashboard also creates random general-sum games and symmetric two-player zero-sum games. One-player experiments cover a historical-frequency adversary, an action-independent lazy reward walk, and replicated action-space sweeps over configurable K values.

Fixed-game and one-player experiments use strategy-weighted external, internal, and swap regret. Bandit learners receive only their sampled reward; the evaluator uses the full payoff vector offline. Figures report replicate means, without confidence bands or interval toggles. Empirical joint-action heatmaps show recorded play, and equilibrium convergence uses full-space L1 distance to CE and CCE.

Payoffs are finite values in `[0, 1]`. Fixed-game tensors have shape:

```text
(n_players, actions_player_0, ..., actions_player_(n-1))
```

## Commands

| Command | Purpose |
|---|---|
| `make all` | Install and refresh existing plots |
| `make install` | Install project and tests |
| `make web` | Start the dashboard |
| `make plot` | Rebuild plots from saved CSVs |
| `make test` | Run the complete test suite |
| `make clean` | Remove caches and temporary files |
| `make reset` | Also remove generated experiment results |

`make all` intentionally does not run the test suite. Run `make help` for the live command list.

## Data

```text
results/
├── raw/          fixed-game CSVs
├── figures/      PNG previews and vector PDFs
├── adversarial/  stress-test CSVs and figures
└── cache/        regenerable plot and equilibrium-distance caches
```

Run IDs include the complete experiment identity and an implementation version, so results from changed code do not collide. Incompatible historical CSVs are left untouched and excluded from current analysis; no migration is performed. Defaults such as the horizon, seeds, replicate count, tolerances, and stationary solver live in `config.py`.

The install commands use `requirements.lock`. Each CSV records the canonical Python and numerical-package environment plus its fingerprint, and that fingerprint participates in the run ID. One-player learner and environment randomness use domain-separated, replicate-specific seeds; CSVs retain both the user-supplied base seeds and the effective derived seeds.

CE/CCE incentive constraints and L1 projection are implemented locally using NumPy and SciPy’s `linprog(method="highs")`. Empirical heatmaps count recorded joint actions directly and require no equilibrium LP.

## Guides

- [Algorithms](algorithms/README.md)
- [Environments](environments/README.md)
- [Experiments](experiments/README.md)
- [Metrics](metrics/README.md)
- [Web dashboard](web/README.md)
