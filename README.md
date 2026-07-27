# Swap Regret Thesis Project

Framework for external-, internal-, and swap-regret learning in finite repeated games. It includes reproducible experiments, CE/CCE analysis through `TUM-DSS/games_learning`, plots, custom n-player games, and a local Flask dashboard.

## Quick Start

Python 3.10 or newer is required.

```bash
make install-dev
make test
make web
```

Run the complete configured batch with `make run`. Existing run IDs are never overwritten; use `make reset` before a fresh batch.

## Algorithms

| Feedback | Objective | Implementations |
|---|---|---|
| Full information | External | Hedge |
| Bandit | External | Exp3, Exp3-IX |
| Full information | Internal | Regret Matching (RM), Stationary Regret Matching (SRM) |
| Full information | Swap | Blum–Mansour (BM), Ito |
| Bandit | Swap | BM, Ito, LCE-IX |

Experiments provide a horizon rather than a learning rate. `horizon=0` gives exponential-weights learners an anytime schedule. See [algorithms/README.md](algorithms/README.md).

## Games

A game is one finite payoff tensor:

```text
payoff_tensor[player, action_player_0, ..., action_player_(n-1)]
shape = (n_players, actions_player_0, ..., actions_player_(n-1))
```

Payoffs must be finite and in `[0, 1]`. The environment, runner, metrics, and convergence analysis support heterogeneous action counts and more than two players. See [environments/README.md](environments/README.md).

### Built-in benchmarks

| ID | Role | Source |
|---|---|---|
| `rps` | Symmetric zero-sum CE/CCE benchmark | `games_learning` |
| `rpsls` | Five-action learning-dynamics extension | Local standard relation |
| `matching_pennies` | Two-action asymmetric zero-sum control | `games_learning` |
| `bertrand_standard_o1` | Homogeneous-good Bertrand baseline | `games_learning` |
| `bertrand_linear_o2` / `_prime` | Symmetric/asymmetric linear-demand pricing | `games_learning` |
| `bertrand_logit_o3` / `_prime` | Symmetric/asymmetric logit-demand pricing | `games_learning` |

RPSLS follows the five-action relation used by [Leme, Piliouras, and Schneider (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/8b9062635eafbc3677429496a23e424b-Paper-Conference.pdf). Built-in payoffs are normalized per player.

### Custom games

The dashboard stores reproducible random games as compressed `.npz` files under `data/custom_games/`:

- 2–8 players;
- 1–100 actions per player;
- at most 1,000,000 payoff values;
- independent uniform `[0, 1)` payoffs from a non-negative seed.

Large action products can make CE/CCE LP figures expensive even when the game itself runs normally.

## Regret

For cumulative replacement gains `G[i, j]`:

```text
external regret = max_j sum_i G[i, j]
internal regret = max_{i,j} G[i, j]
swap regret     = sum_i max_j G[i, j]
```

Feedback and regret evaluation are independent. Either feedback mode can record expected regret, realized regret, or both. Bandit learners still receive only their realized payoff; exact counterfactuals are used only by the offline evaluator.

Matching replicates are aggregated pointwise in regret plots and by final value in the dashboard, with 95% confidence intervals. See [metrics/README.md](metrics/README.md).

## CE/CCE Analysis

The pinned authoritative backend is `TUM-DSS/games_learning` at commit `6ca238a9c8716cd34fc3fcbc949bba7a7ea68dc5`.

```python
from metrics.equilibrium import equilibrium_profile_weights, optimize_equilibrium
from metrics.equilibrium_distance import equilibrium_l1_distance
```

- `optimize_equilibrium(...)` returns the upstream CE or CCE distribution directly.
- `equilibrium_profile_weights(...)` maximizes every profile independently; its result is not one equilibrium distribution.
- `equilibrium_l1_distance(...)` measures full-dimensional L1 distance to the CE or CCE polytope.
- Projected trajectories use deterministic two-component PCA only for visualization.

Pip creates the ignored `src/games-learning/` editable checkout because the published wheel omits required packages. It is an unmodified dependency, not project-owned code. If imports fail in an editor, select the interpreter used by `make install-dev` and restart the language server.

## Experiments and Results

Run identity includes game, feedback, regret evaluation, ordered algorithm profile, horizon, seed, replicate, and stationary solver. Player seeds are deterministic and distinct.

```text
results/
├── raw/<run-id>.csv
├── figures/<regret-figure>.png
├── figures/details/<lazy-detail-cache>.png
└── index.html
```

CSV and figure publication is atomic. `results/figures/details/` is a regenerable cache. See [experiments/README.md](experiments/README.md).

## Dashboard

`make web` starts the dashboard. It provides:

- built-in and custom-game selection;
- feedback, regret-evaluation, and per-player algorithm controls;
- Player 0 → all synchronization;
- a FIFO experiment queue with progress and cancellation;
- persistent form, filter, trajectory, and five-color interface-theme settings;
- replicate-mean regret summaries and 95% confidence intervals;
- custom-game creation, tensor inspection, download, and deletion;
- lazy equilibrium-distance and projected-trajectory figures for built-in and custom n-player games;
- joint-action and maximum-profile-weight heatmaps for built-in two-player games.

Heatmaps use a fixed blue palette and lower origin; the interface theme changes dashboard accents only. Set `SWAP_REGRET_WEB_SECRET` if Flask sessions must survive server restarts. See [web/README.md](web/README.md).

## Commands

| Command | Purpose |
|---|---|
| `make install` / `install-dev` | Install runtime or development dependencies |
| `make run` | Run batch experiments, plots, and static report |
| `make full` / `bandit` | Run one configured feedback-mode batch |
| `make web` | Start the dashboard |
| `make plot` / `report` | Rebuild plots or the static report |
| `make precompute-equilibria` | Regenerate `web/static/equilibria/` only |
| `make test` / `smoke` | Run all tests or smoke tests |
| `make clean` | Remove caches and interrupted staging artifacts |
| `make clean-results` | Remove generated results, preserving custom games and static equilibrium assets |
| `make reset` | Run both cleanup targets |

Run `make help` for the live command list.

## Configuration

| Name | Default |
|---|---:|
| `SEED` | `42` |
| `HORIZON` | `1_000` |
| `BANDIT_REPLICATES` | `20` |
| `NUMERICAL_TOLERANCE` | `1e-10` |
| `EQUILIBRIUM_LP_TOLERANCE` | `1e-8` |
| `STATIONARY_METHOD` | `"solve"` |

Output paths are also defined in `config.py`.

## Structure

```text
algorithms/    learners and stationary solvers
environments/  payoff tensors and feedback
experiments/   games, execution, recording, and plots
metrics/       regret and equilibrium analysis
web/           Flask dashboard
tests/         unit, integration, web, and smoke tests
```

Run `make test` for the complete suite.
