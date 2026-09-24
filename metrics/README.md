# Metrics

Utilities for regret, empirical play, and equilibrium convergence.

`RegretBundle(n_actions)` maintains one cumulative action-regret replacement-gain matrix. Runners update it every round; under the default recording policy, they extract summaries at every round through horizon 500, or at most 500 geometric checkpoints for longer runs.

`RegretBundle.update` checks the sampled action's range but assumes the payoff vector has the expected shape and finite values. Fixed-game result loaders validate stored cumulative histogram trajectories before equilibrium analysis.

## Regret

For sampled action `I_t` and payoff vector `u_t`:

```text
G[i,j] += 1{I_t=i} * (u_t(j) - u_t(i))
external regret = max_j sum_i G[i,j]
internal regret = max_{i,j} G[i,j]
swap regret     = sum_i max_j G[i,j]
```

Each round therefore updates only `G[I_t, :]`. The evaluator uses the full payoff vector offline, while bandit learners still observe only their sampled reward. Regret and empirical-play analyses refer to the same sampled-action trajectory. Replicate curves and final summaries report replicate-mean action regret, a Monte Carlo estimate of expected action regret, without confidence bands. Theoretical pseudo-regret instead maximizes over deviations after taking expectation; these plots average each run's realized maximum.

## Empirical Play and Equilibria

`joint_action_distribution(...)` normalizes the final stored cumulative joint-action histogram. `mean_joint_action_distribution(...)` averages these distributions across replicates for empirical heatmaps. Cumulative histograms and equilibrium distances use the regret checkpoint sampling rule with a separate 20-point budget. Neither operation solves an equilibrium LP.

`equilibrium_l1_distance(...)` solves `min_{q in E} ||q - empirical||_1`, where `E` is CE or CCE. The local implementation in `equilibrium_distance.py` constructs incentive constraints and uses `scipy.optimize.linprog(method="highs")`. Prepared LPs reuse fixed coefficient matrices across checkpoints.

Before building dense incentive rows, a deterministic 128 MiB budget estimates the current LP's dense row arrays, stacked copy, and other fixed-size dense intermediates separately for CE and CCE. This is not a peak-RSS estimate. Over-budget equilibrium-distance analysis is unavailable, but game validity and regret learning are unaffected; within-budget analyses retain the same exact full-space formulation solved numerically.

Distances are measured in the full joint-distribution space. `equilibrium_distance_trajectory(...)` computes them per replicate, and `aggregate_equilibrium_distance_trajectories(...)` then averages them; this is not the distance of the replicate-mean distribution. Figures show mean CE/CCE distances without confidence bands.
