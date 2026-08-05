# Metrics

Analysis utilities for regret, empirical play, and equilibrium convergence.

## Regret

For cumulative replacement gains `G[i, j]`:

```text
external regret = max_j sum_i G[i, j]
internal regret = max_{i,j} G[i, j]
swap regret     = sum_i max_j G[i, j]
```

Expected regret weights deviations by the played strategy. Realized regret updates only the chosen action's row. `RegretBundles` can maintain both at once.

## Empirical Play

`empirical_distribution_trajectory(...)` converts joint-action histories into full joint-distribution vectors. By default it records round 1, powers of ten from 100, and the final round.

The optional comparison view uses denser checkpoints only in the last logarithmic interval; its exact focus rule is documented in [the experimental guide](../experimental/equilibrium_trajectory/README.md).

## CE and CCE

`equilibrium.py` adapts the pinned `games_learning` equilibrium LPs:

```text
CE  -> coarse=False
CCE -> coarse=True
```

`optimize_equilibrium(...)` returns one feasible or objective-maximizing distribution. `equilibrium_profile_weights(...)` maximizes each profile independently, so its output is a heatmap, not one equilibrium distribution.

`equilibrium_l1_distance(...)` solves for the nearest CE or CCE distribution:

```text
min_q ||q - empirical_distribution||_1
```

Distances are computed in the full joint-distribution space. Replicate summaries average per-replicate distances and report Student-t 95% confidence intervals; they do not replace those distances with the distance of a replicate mean.

## Files

| File | Purpose |
|---|---|
| `regret.py` | Regret trackers |
| `empirical_distribution.py` | Checkpoints and empirical trajectories |
| `equilibrium.py` | CE/CCE LP adapter |
| `equilibrium_distance.py` | Full-space L1 distance |
| `equilibrium_convergence.py` | Replicate aggregation |
| `confidence.py` | Confidence intervals |

Experiment CSV handling belongs to [experiments](../experiments/README.md); projected trajectories are isolated in the optional experimental package.
