# Metrics

Utilities for regret, empirical play, and equilibrium convergence.

`RegretBundle(n_actions)` maintains one cumulative strategy-weighted replacement-gain matrix. Runners update it every round and extract summaries only at recording checkpoints.

Regret updates trust learner strategies and environment payoff vectors; they perform only the replacement-gain arithmetic. Fixed-game result loaders validate stored cumulative histogram trajectories before equilibrium analysis.

## Regret

For the played strategy `p_t` and the payoff vector `u_t`:

```text
G[i,j] += p_t[i] * (u_t(j) - u_t(i))
external regret = max_j sum_i G[i,j]
internal regret = max_{i,j} G[i,j]
swap regret     = sum_i max_j G[i,j]
```

Evaluation is independent of feedback: bandit learners still observe only their sampled reward. Replicate regret curves and final summaries report sample means, without confidence bands.

## Empirical Play and Equilibria

`joint_action_distribution(...)` normalizes the final stored cumulative joint-action histogram. `mean_joint_action_distribution(...)` averages these distributions across replicates for empirical heatmaps. Neither operation solves an equilibrium LP.

`equilibrium_l1_distance(...)` solves `min_{q in E} ||q - empirical||_1`, where `E` is CE or CCE. The local implementation in `equilibrium_distance.py` constructs incentive constraints and uses `scipy.optimize.linprog(method="highs")`. Prepared LPs reuse fixed coefficient matrices across checkpoints.

Distances are measured in the full joint-distribution space. `equilibrium_convergence.py` computes distances per replicate and then averages them; it does not measure the distance of the replicate-mean distribution. Figures show mean CE/CCE distances without confidence bands.
