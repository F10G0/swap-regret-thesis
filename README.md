# Swap Regret Thesis Project

Framework for external, internal, and swap-regret experiments in finite repeated games and one-player stress tests. It provides reproducible CSV results, CE/CCE analysis, publication-ready plots, custom games, and a local Flask dashboard.

## Quick Start

Python 3.10 or newer is required.

```bash
make install
make test
make web
```

The experimental trajectory comparison is excluded by default. Use `make install-experimental` to enable it and `make install` to disable it again.

## Scope

| Feedback | Learners |
|---|---|
| Full information | Hedge, Regret Matching, SRM, Blum–Mansour, Ito |
| Bandit | Exp3, Exp3-IX, Blum–Mansour, Ito, LCE-IX |

Built-in games include RPS, RPSLS, Matching Pennies, and five Bertrand variants. The dashboard also creates random general-sum games and symmetric two-player zero-sum games. One-player experiments cover a historical-frequency adversary, an action-independent lazy reward walk, and replicated action-space sweeps over configurable K values.

Feedback and regret evaluation are independent. Fixed-game and one-player experiments can record expected regret, realized regret, or both; the two sources are plotted separately. Both support configurable replicates with pointwise Student-t 95% confidence intervals and per-figure interval toggles. Equilibrium convergence uses full-space L1 distance to CE and CCE; optional 2-D trajectories are interpretive views only.

Payoffs are finite values in `[0, 1]`. Fixed-game tensors have shape:

```text
(n_players, actions_player_0, ..., actions_player_(n-1))
```

## Commands

| Command | Purpose |
|---|---|
| `make all` | Install the default build and refresh existing plots |
| `make install` | Install without experimental trajectories |
| `make install-experimental` | Install with experimental trajectories |
| `make web` | Start the dashboard |
| `make plot` | Rebuild plots from saved CSVs |
| `make precompute-equilibria` | Rebuild static CE/CCE heatmaps |
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
└── cache/        regenerable plot and geometry caches
```

Run IDs include the complete experiment identity and an implementation version, so results from changed code do not collide. Older CSVs load as legacy version 0. Defaults such as the horizon, seeds, replicate count, tolerances, and stationary solver live in `config.py`.

CE/CCE constraints use the pinned `TUM-DSS/games_learning` commit `6ca238a9c8716cd34fc3fcbc949bba7a7ea68dc5`.

## Guides

- [Algorithms](algorithms/README.md)
- [Environments](environments/README.md)
- [Experiments](experiments/README.md)
- [Metrics](metrics/README.md)
- [Web dashboard](web/README.md)
- [Experimental trajectories](experimental/equilibrium_trajectory/README.md)
