# EXP10/EXP11 Rationale: Scaling SGD Noise

The goal is to test whether the amount of stochasticity itself controls
optimization behavior once the deterministic full-gradient component is fixed.

For each step we compute:

```text
g_batch = g_full + noise
```

EXP10 uses raw noise:

```text
g_alpha = g_full + alpha * noise
```

EXP11 normalizes the noise to the full-gradient norm:

```text
g_alpha = g_full + alpha * ||g_full|| / ||noise|| * noise
```

The alpha grid is:

```text
[0.01, 0.1, 0.3, 0.5, 0.8, 1, 2, 3, 5, 10, 100]
```

Interpretation:

- if alpha below 1 wins, SGD noise is harmful in this local regime;
- if alpha around 1 wins, ordinary SGD noise scale is near-optimal;
- if alpha above 1 wins, extra stochasticity improves convergence/generalization;
- if alpha 100 diverges, the experiment identifies an instability boundary.

These experiments do not use multiple seeds by design; each alpha is a fixed
trajectory to show the deterministic slice through noise scale.
