# Experimental Equilibrium Trajectories

Optional two-dimensional comparisons of empirical joint-distribution trajectories. These plots aid interpretation; full-space CE/CCE L1 distance remains the convergence measure.

## Enable the Feature

```bash
make install-experimental  # on
make install               # off again
```

The standard build is off. An enabled build adds `/experimental/trajectory-comparisons`, where rendering occurs only when **Generate** is pressed.

## Valid Comparisons

Each member is a result group and may contain several replicates. Members must have the same:

- game and payoff digest;
- feedback and regret evaluation;
- horizon and base seed;
- replicate count and exact replicate indices;
- derived player-seed schedule.

For player `i`, replicate `r`, and `p` players, the seed is `base_seed + r * p + i`. Matching the complete schedule prevents seed-population differences from being mistaken for algorithm effects.

## Checkpoints and Focus

Let `L` be the largest power of ten strictly below horizon `T`. Keep the logarithmic checkpoints through `L`, then divide `[L, T]` into `final_interval_segments` integer segments. For example, `T = 10000` and four segments gives:

```text
1, 10, 100, 1000, 3250, 5500, 7750, 10000
```

The UI accepts 1–50 segments. `T = 1` produces only checkpoint 1.

**Focus final log interval** fits axes and limits from `[L, T]`. The preceding logarithmic point `P`, when present, is excluded from fitting but its incoming segment `P -> L` is still drawn and may start off-screen.

## Projection Modes

The geometry-aware view finds the affine dimensions of CE and CCE, then renders each projected set as a point, line, or region. Trajectories in a comparison share the same equilibrium-centered axes.

When CE is a point inside a line-like CCE, the specialized coordinates are:

```text
x[e,t] = v_x^T (mu[e,t] - mu_CE)
y[e,t] = d_CCE(mu[e,t])
```

Thus CE is `(0, 0)`, CCE is the line `y = 0`, and height is the true full-space CCE distance.

The unified view instead fits one shared horizontal direction, shows the CE and CCE support intervals on that axis, and again uses full-space CCE distance vertically.

## Cost

Once equilibrium geometry is cached, rendering a projected equilibrium shape uses:

| Shape | LP work |
|---|---|
| Point | 0 solves |
| Line | 2 endpoint solves |
| Region | Adaptive support reconstruction |

Adaptive regions stop when their hull edges are certified. Metadata separates support queries from LP solves; the 512-query cap is only a numerical safety fallback. The unified view uses four support LPs plus one CCE-distance LP per represented checkpoint and member.

Geometry is cached by payoff digest in `results/cache/experimental/equilibrium_trajectory/`. Rendered comparisons live as PNG/PDF pairs under `results/figures/details/experimental/equilibrium_trajectory/`.

Exact finite-round CE/CCE membership can appear briefly because empirical distributions lie on a rational lattice. Also, overlap in a two-dimensional projection does not prove full-space membership. Use the distance figures for convergence claims.
