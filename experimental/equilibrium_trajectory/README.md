# Experimental Equilibrium Trajectories

Optional 2-D comparisons of empirical joint-distribution trajectories. They aid interpretation; full-space CE/CCE L1 distance remains the convergence measure.

```bash
make install-experimental  # enable
make install               # disable
```

The enabled build adds `/experimental/trajectory-comparisons`. Rendering occurs only when **Generate** is pressed.

## Comparisons and Checkpoints

Each member is a result group and may contain multiple replicates. Members must share the game/payoff digest, feedback, evaluation, horizon, base seed, replicate count and indices, and derived player-seed schedule. For player `i`, replicate `r`, and `p` players, that seed is `base_seed + r * p + i`.

Let `L` be the largest power of ten strictly below horizon `T`. Checkpoints contain the powers of ten through `L`, then `final_interval_segments` integer subdivisions of `[L, T]`. For `T = 10000` and four segments:

```text
1, 10, 100, 1000, 3250, 5500, 7750, 10000
```

The accepted range is 1–50; `T = 1` has only checkpoint 1. **Focus final log interval** fits axes and limits from `[L, T]`. The preceding logarithmic point is excluded from fitting but its incoming segment remains visible and may begin off-screen.

## Projection

The geometry-aware view finds the affine dimensions of CE and CCE and renders each as a point, line, or region. Comparison members share equilibrium-centered axes.

When CE is a point inside line-like CCE, the specialized coordinates are:

```text
x[e,t] = v_x^T (mu[e,t] - mu_CE)
y[e,t] = d_CCE(mu[e,t])
```

CE is therefore `(0, 0)`, CCE is `y = 0`, and height is the full-space CCE distance. The unified view uses one shared horizontal direction, CE/CCE support intervals, and the same distance vertically.

## Cost and Interpretation

Cached projected equilibrium shapes require 0 LP solves for a point, 2 endpoint solves for a line, and adaptive support reconstruction for a region. Region metadata separates support queries from LP solves; 512 queries is a safety cap. The unified view uses four support LPs plus one CCE-distance LP per represented checkpoint and member.

Geometry is cached under `results/cache/experimental/equilibrium_trajectory/`; rendered PNG/PDF pairs are under `results/figures/details/experimental/equilibrium_trajectory/`.

Finite-round membership can occur briefly because empirical distributions lie on a rational lattice. Overlap in 2-D does not prove full-space membership, so use the distance figures for convergence claims.
