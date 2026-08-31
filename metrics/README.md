# Metrics

Utilities for regret, empirical play, and equilibrium convergence.

## Regret

For cumulative replacement gains `G[i, j]`:

```text
external regret = max_j sum_i G[i, j]
internal regret = max_{i,j} G[i, j]
swap regret     = sum_i max_j G[i, j]
```

Expected regret weights deviations by the played strategy. Realized regret updates only the selected action's row. `RegretBundles` can track both together.

Replicate regret curves and final summaries report the sample mean with pointwise two-sided Student-t 95% confidence intervals.

## Empirical Play and Equilibria

`empirical_distribution_trajectory(...)` converts joint-action histories into full joint-distribution vectors at deterministic checkpoints. The optional comparison view adds checkpoints within the final logarithmic interval; see the [experimental guide](../experimental/equilibrium_trajectory/README.md).

The pinned `games_learning` LP adapter uses `coarse=False` for CE and `coarse=True` for CCE. Its main operations are:

- `optimize_equilibrium(...)`: return a feasible or objective-maximizing distribution.
- `equilibrium_profile_weights(...)`: maximize each profile independently for a heatmap.
- `equilibrium_l1_distance(...)`: solve `min_q ||q - empirical||_1`.

Distances are measured in the full joint-distribution space. Replicate summaries average per-replicate distances and use Student-t 95% confidence intervals; they do not measure the distance of a replicate mean.

`regret.py` contains the trackers, `empirical_distribution.py` the trajectory construction, and the `equilibrium*.py` files the LP and aggregation logic.
