# Metrics

Game-analysis utilities independent of experiment execution, CSV formats, and web presentation.

```python
from metrics.regret import RegretBundles
from metrics.empirical_distribution import empirical_distribution_trajectory
from metrics.equilibrium import optimize_equilibrium
from metrics.equilibrium_distance import equilibrium_l1_distance
```

`metrics.__init__` has no eager re-exports, so regret-only code does not import PuLP, SciPy, or `games_learning`.

## Replacement Regret

For cumulative replacement gains `G[i, j]`:

```text
external regret = max_j sum_i G[i, j]
internal regret = max_{i,j} G[i, j]
swap regret     = sum_i max_j G[i, j]
```

- Expected update: `G[i, j] += strategy[i] * (payoff[j] - payoff[i])`.
- Realized update: only row `action` receives `payoff[j] - payoff[action]`.
- `RegretBundles` updates both trackers; experiments decide which summaries to record.

Trackers validate finite shapes and normalized strategies. `summary(time)` returns cumulative and average values.

## Empirical Distributions

`empirical_distribution_trajectory(action_profiles, action_shape, checkpoints=None)` supports n-player heterogeneous action spaces. It returns:

- `action_shape`;
- selected `horizons`;
- flattened C-order probability `vectors`;
- `distributions` reshaped to `(n_checkpoints, *action_shape)`.

Default checkpoints are iteration 1, powers of ten from 100, and the final iteration. `uniform_checkpoints(horizon, count)` instead spreads points from iteration 1 to final. Matching replicate trajectories can be averaged with `mean_empirical_distribution_trajectory(...)`.

## CE/CCE Adapter

`equilibrium.py` is a thin adapter over `games_learning.utils.equilibrium`:

```text
ce  -> coarse=False
cce -> coarse=True
```

`optimize_equilibrium(...)` returns one upstream feasible or objective-maximizing distribution directly.

- `max_equilibrium_profile_weight(...)` maximizes one profile.
- `equilibrium_profile_weights(...)` maximizes every profile independently.
- `create_equilibrium_lp(...)` exposes the authoritative upstream LP for extensions.

Profile weights have the full joint-action shape but do not form one equilibrium distribution.

## Equilibrium Distance

`equilibrium_l1_distance(payoff_tensor, empirical_distribution, equilibrium)` solves:

```text
min_q ||q - empirical_distribution||_1
subject to q being a CE or CCE
```

It validates the empirical distribution with `EQUILIBRIUM_LP_TOLERANCE`, adds absolute-deviation variables to the upstream LP, and returns `EquilibriumDistanceResult(distance, nearest_distribution)`.

## Projection and Convergence

`LinearProjection2D.fit()` performs deterministic two-component PCA/SVD with stable component signs. `project_equilibrium_region(...)` optimizes upstream equilibria in sampled projection directions and forms the displayed boundary.

Projection is visualization only. Hide first is a presentation-layer operation that removes round 1 before fitting; full-dimensional distances are unchanged.

`equilibrium_convergence.py` provides:

- CE and CCE distance trajectories;
- per-replicate distance means and 95% confidence intervals;
- projected mean empirical trajectories;
- projected CE and CCE regions.

Distances are computed separately for every replicate, never only from the mean distribution.

## Numerical Ownership

- `NUMERICAL_TOLERANCE = 1e-10`: learner and stationary-distribution checks.
- `EQUILIBRIUM_LP_TOLERANCE = 1e-8`: empirical-distribution checks for LP distance.
- Upstream `games_learning`: equilibrium constraints and returned CE/CCE distributions.

## Structure

```text
metrics/
├── regret.py
├── empirical_distribution.py
├── equilibrium.py
├── equilibrium_distance.py
├── equilibrium_projection.py
└── equilibrium_convergence.py
```
