# Swap Regret Thesis Project

Framework for studying external, internal, and swap regret in finite repeated games and a one-player adaptive stress test. It includes reproducible experiments, regret plots, CE/CCE distance analysis, custom games, and a local Flask dashboard.

## Quick Start

Python 3.10 or newer is required.

```bash
make install
make test
make web
```

The experimental trajectory comparison is excluded by default. To include it, install with:

```bash
make install-experimental
```

Reinstall with `make install` to disable it again.

## Included Algorithms

| Feedback | Algorithms |
|---|---|
| Full information | Hedge, Regret Matching, SRM, Blum–Mansour, Ito |
| Bandit | Exp3, Exp3-IX, Blum–Mansour, Ito, LCE-IX |

Experiments require a positive horizon. See [algorithms/README.md](algorithms/README.md) for the learner interface and update rules.

## Games

A payoff tensor has shape:

```text
(n_players, actions_player_0, ..., actions_player_(n-1))
```

Payoffs must be finite values in `[0, 1]`. Built-in games include RPS, RPSLS, Matching Pennies, and five Bertrand variants. The dashboard can also create random general-sum games with 2–8 players or two-player zero-sum games. CE/CCE profile-weight heatmaps for custom zero-sum games are generated once and cached with the saved game.

See [environments/README.md](environments/README.md) and [experiments/README.md](experiments/README.md).

## Regret and Equilibria

For cumulative replacement gains `G[i, j]`:

```text
external regret = max_j sum_i G[i, j]
internal regret = max_{i,j} G[i, j]
swap regret     = sum_i max_j G[i, j]
```

Feedback and evaluation are separate. Either feedback mode can record expected regret, realized regret, or both. When both are recorded, they are plotted in separate figures.

Figures are stored as PNG previews for the dashboard and matching vector PDFs for publication-quality downloads.

The core equilibrium output is full-space L1 distance to CE and CCE. Low-dimensional trajectory projections are optional visualizations and should not replace the distance plots when making convergence claims.

The CE/CCE constraints come from the pinned `TUM-DSS/games_learning` checkout at commit `6ca238a9c8716cd34fc3fcbc949bba7a7ea68dc5`.

## Commands

| Command | Purpose |
|---|---|
| `make install` | Install project and tests without trajectory projection |
| `make install-experimental` | Install project and tests with trajectory projection |
| `make web` | Start the dashboard |
| `make plot` | Rebuild standard and adversarial plots from CSVs |
| `make precompute-equilibria` | Rebuild static CE/CCE heatmaps |
| `make test` | Run the complete test suite |
| `make clean` | Remove temporary caches |
| `make reset` | Remove caches and generated results |

Run `make help` for the live command list.

## Results

```text
results/
├── raw/                 experiment CSVs
├── figures/             regret and lazy detail figures
├── adversarial/         isolated stress-test CSVs and figures
└── cache/experimental/  optional trajectory geometry cache
```

Fixed-game run IDs include the game, payoff fingerprint, feedback, evaluation, algorithm profile, horizon, seed, replicate, and stationary solver. Adversarial run IDs use the learner, action count, memory window, horizon, and seed. Existing runs are never overwritten.

## Configuration

Defaults live in `config.py`:

| Setting | Default |
|---|---:|
| `SEED` | `42` |
| `HORIZON` | `1_000` |
| `BANDIT_REPLICATES` | `20` |
| `ADVERSARIAL_ACTIONS` | `3` |
| `ADVERSARIAL_MEMORY_WINDOW` | `0` (full history) |
| `NUMERICAL_TOLERANCE` | `1e-10` |
| `EQUILIBRIUM_LP_TOLERANCE` | `1e-8` |
| `STATIONARY_METHOD` | `"solve"` |

## More Documentation

- [Algorithms](algorithms/README.md)
- [Environments](environments/README.md)
- [Experiments and results](experiments/README.md)
- [Metrics](metrics/README.md)
- [Web dashboard](web/README.md)
- [Experimental trajectories](experimental/equilibrium_trajectory/README.md)
