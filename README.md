# Swap Regret Thesis Project

Implementation and empirical evaluation of external-, internal-, and swap-regret minimization in repeated games.

The project compares full-information and bandit-feedback learning procedures, stationary-distribution reductions, regret matching, and their empirical behavior in fixed two-player benchmark games.

## Quick Start

The project requires Python 3.10 or newer.

```bash
make install-dev
make test
make web
```

The local dashboard then provides experiment configuration, execution, result inspection, and plotting. To run the complete configured batch pipeline instead:

```bash
make run
```

`make run` expects the corresponding run IDs not to exist already. Use `make rerun` when a completely fresh configured run is intended.

## Implemented Algorithms

### External regret

Full information:

- Hedge

Bandit feedback:

- Exp3
- Exp3-IX

### Internal regret

Full information:

- Hart–Mas-Colell inertia-based regret matching from equation (2.2)
- Hart–Mas-Colell stationary regret matching from equation (3.1)

### Swap regret

Full information:

- Blum–Mansour reduction
- Ito reduction

Bandit feedback:

- Blum–Mansour reduction
- Ito reduction
- LCE-IX

Algorithms receive a horizon rather than a learning rate. Exponential-weights learners derive their rates internally; `horizon=0` selects an anytime schedule based on the learner's own update count. See [algorithms/README.md](algorithms/README.md) for the formulas and reduction details.

## Environments

Games are represented by one NumPy payoff tensor with shape `(n_players, action_1, ..., action_n)`. The leading dimension identifies the player and the remaining dimensions identify the joint action.

The stateful environment interface separates the round transition from player feedback:

- `step(actions)` validates and stores one joint action.
- `RepeatedGame.feedback(player)` returns that player's unilateral-deviation payoff vector.
- `BanditRepeatedGame.feedback(player)` returns only that player's realized payoff.
- `deviation_payoffs(player)` exposes the same counterfactual payoff vector for regret evaluation. In bandit runs, this evaluation-only information is never passed to the learner.

Payoff vectors are computed on demand and are not cached. See [environments/README.md](environments/README.md) for the exact interface and invariants.

## Regret Metrics

The metrics layer tracks cumulative replacement gains

```text
G[i, j] = cumulative gain from replacing action i with action j.
```

It derives:

- external regret as `max_j sum_i G[i, j]`,
- internal regret as `max_{i,j} G[i, j]`,
- swap regret as `sum_i max_j G[i, j]`.

Full-information experiments record expected regret from the learner's mixed strategy and payoff vector. Bandit experiments record realized regret from the sampled action and evaluation-only deviation payoffs. Repeated bandit runs are averaged to estimate expected performance.

## Benchmark Games

- Rock–Paper–Scissors
- Dominant Coordination with 9 actions
- Cyclic Dominance with 9 actions

All generated payoff tensors are normalized to `[0, 1]`.

## CE and CCE Analysis

Finite-game correlated equilibria (CE) and coarse correlated equilibria
(CCE) are represented as linear feasibility polytopes. The reusable
metrics utilities build their incentive constraints and use
SciPy/HiGHS to maximize each joint-action probability independently:

```python
from metrics import equilibrium_profile_weights

ce_weights = equilibrium_profile_weights(payoff_tensor, equilibrium="ce")
cce_weights = equilibrium_profile_weights(payoff_tensor, equilibrium="cce")
```

CE constraints condition deviations on a recommended action; CCE
constraints describe unconditional fixed deviations. Consequently,
`CE` is contained in `CCE`.

An entry `weights[a]` is the maximum probability that any equilibrium of
the selected type can assign to profile `a`. Different entries can be
attained by different equilibria, so the returned array is not itself
an equilibrium distribution and generally does not sum to one. The
solver supports finite n-player games with heterogeneous action counts.

## Experiments and Outputs

The experiment layer supports heterogeneous two-player cross-play, deterministic run identities, independent player and replicate seeds, stationary-solver metadata, atomic CSV recording, result validation, and regret plotting.

`make run` performs four stages:

1. Run all configured full-information cross-play profiles.
2. Run all configured bandit profiles across the configured replicates.
3. Generate regret figures.
4. Build the static HTML report.

Generated artifacts are stored in:

```text
results/
├── raw/                 # Per-round CSV results
├── figures/             # Regret plots
│   └── details/         # Lazy empirical and theoretical heatmaps
└── index.html           # Static report
```

Matching replicate curves are averaged pointwise. A 95% confidence interval is shown when a group contains multiple replicates. Figure legends are placed outside the data axes, and the PNG height expands with the number of profiles so the legend cannot obscure the curves.

See [experiments/README.md](experiments/README.md) for the result schema, regret definitions, scenario entry points, and plotting behavior.

## Web Dashboard

Start the local Flask dashboard with:

```bash
make web
```

It supports:

- game, feedback mode, player algorithm, horizon, seed, and replicate selection;
- consecutive bandit replicate batches;
- one profile or all missing algorithm pairs;
- first-in, first-out experiment queueing while another experiment is running;
- job progress and safe cancellation;
- persistent form parameters across reloads;
- expected- and realized-regret figure filtering;
- responsive plot comparison and full-size PNG inspection;
- result filtering, sorting, comparable-group minimum highlighting, and exact parameter reuse;
- raw CSV downloads and lazily cached empirical joint-action heatmaps;
- lazily cached maximum CE and CCE profile-weight heatmaps for each game;
- asynchronous plot rebuilding and figure-inventory updates without page navigation;
- individual experiment deletion and generated-result reset.

The dashboard executes one background job at a time. Its default limits are 100,000 rounds and 100 bandit replicates per submitted batch. Set `SWAP_REGRET_WEB_SECRET` to a stable random value when browser sessions should survive server restarts.

## Repository Structure

```text
.
├── algorithms/          # Learners, reductions, and stationary solvers
├── environments/        # Full-information and bandit repeated games
├── experiments/         # Games, scenarios, recording, results, and plots
├── metrics/             # Regret and equilibrium analysis
├── results/             # Generated CSV, PNG, and HTML artifacts
├── tests/               # Algorithm, environment, metric, experiment, and web tests
├── web/                 # Flask dashboard
├── config.py            # Experiment, numerical, and output configuration
├── main.py              # Complete batch pipeline
├── Makefile
└── pyproject.toml
```

## Commands

| Command | Purpose |
|---|---|
| `make help` | List the available Make targets |
| `make all` | Run the default validation target (`make test`) |
| `make install` | Install the project in editable mode |
| `make install-dev` | Install the project and test dependency |
| `make run` | Run all configured experiments, plots, and the report |
| `make full` | Run all full-information cross-play profiles |
| `make bandit` | Run all bandit profiles and replicates |
| `make plot` | Rebuild regret plots from existing CSV files |
| `make report` | Rebuild the static HTML report |
| `make web` | Start the local dashboard |
| `make test` | Run the complete test suite without cache artifacts |
| `make smoke` | Run only the end-to-end smoke tests |
| `make clean` | Remove Python and pytest caches |
| `make clean-results` | Remove generated CSV, PNG, and report files |
| `make reset` | Run both cleanup targets |
| `make rerun` | Reset and run the complete pipeline |
| `make reweb` | Reset and start the dashboard |

## Configuration

The main defaults are defined in `config.py`:

- `SEED`: reproducible base seed;
- `HORIZON`: experiment horizon;
- `BANDIT_REPLICATES`: number of bandit repetitions in the batch scenario;
- `STATIONARY_METHOD`: `solve`, `pinv`, or `iteration`;
- numerical and equilibrium-LP validation tolerances;
- output directories.

The selected stationary method is included in each run ID and CSV so results produced with different solvers cannot overwrite or merge with one another.

## Testing

Run the complete suite with:

```bash
make test
```

The tests cover initialization, numerical safety, stationary distributions,
CE/CCE constraints and optimization, regret matching, reduction behavior,
environment feedback, regret metrics, experiment identity and atomic
recording, plotting layout, cancellation, dashboard validation, and
end-to-end execution.
