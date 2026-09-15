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
| Full information | Hedge, Optimistic Hedge, Regret Matching, SRM, Blum–Mansour, BM-Optimistic-Hedge, Ito |
| Bandit | AuerExp3, Exp3-IX, Blum–Mansour, Ito, LCE-IX |

Built-in games are RPS and RPSLS, defined locally as fixed payoff matrices. Matching Pennies is a local test-only fixture. The dashboard also creates random general-sum games and symmetric two-player zero-sum games. One-player experiments cover a historical-frequency adversary and an action-independent lazy reward walk; multiple K values can be queued as one ordinary-result batch.

Fixed-game and one-player experiments use realized external, internal, and swap regret from the sampled action trajectory. Bandit learners receive only their sampled reward; the evaluator uses the full payoff vector offline. Results record every round through horizon 500 and at most 500 geometric checkpoints for longer runs. Figures report replicate means, without confidence bands or interval toggles. Empirical joint-action heatmaps show the same recorded play, and equilibrium convergence uses full-space L1 distance to CE and CCE.

Payoffs are finite values in `[0, 1]`. Fixed-game tensors have shape:

```text
(n_players, actions_player_0, ..., actions_player_(n-1))
```

## Commands

| Command | Purpose |
|---|---|
| `make all` | Install project dependencies |
| `make install` | Install project and tests |
| `make web` | Start the dashboard |
| `make test` | Run the complete test suite |
| `make clean` | Remove Python caches and temporary staging files |
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

Run IDs include the complete experiment identity and numerical runtime fingerprint. Results on disk are assumed to come from the current implementation; clear them before making an incompatible scientific change. Defaults such as the horizon, seeds, replicate count, tolerances, and stationary solver live in `config.py`.

The install commands use `requirements.lock`. Each CSV records the canonical Python and numerical-package environment plus its fingerprint, and that fingerprint participates in the run ID. The dashboard uses one shared experiment Seed, defaulting to 42. One-player learner and environment randomness use domain-separated, replicate-specific seeds derived from that base seed; CSVs retain the base and effective derived seeds.

CE/CCE incentive constraints and L1 projection are implemented locally using NumPy and SciPy’s `linprog(method="highs")`. Empirical heatmaps count recorded joint actions directly and require no equilibrium LP.

`make reset` preserves source and configuration inputs, including saved custom games, along with `.gitkeep` placeholders. Everything derived from experiment results is removed so that no generated figure or cache outlives its source data.

## Guides

- [Algorithms](algorithms/README.md)
- [Environments](environments/README.md)
- [Experiments](experiments/README.md)
- [Metrics](metrics/README.md)
- [Web dashboard](web/README.md)
