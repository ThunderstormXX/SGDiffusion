# EXP9: Theory-Denoised SGD

Goal: turn the descriptive SGD-diffusion theory into a prototype optimizer.

EXP7 removed the whole update component in theory-selected directions and hurt
loss. EXP9 is more precise: it keeps deterministic full-gradient drift and
removes only the stochastic minibatch-noise component in selected directions.

For a minibatch gradient:

```text
g_batch(theta) = g_full(theta) + noise(theta)
```

the theory-denoised update is

```text
g_modified = g_batch - P noise
```

where `P` projects onto theory-predicted high-variance Hessian directions.

Controls:

- `baseline_sgd`: ordinary SGD;
- `denoise_theory_high_variance`: remove noise in theory-selected directions;
- `denoise_random`: remove noise in random Hessian-basis directions;
- `denoise_sharp`: remove noise in sharp Hessian directions.

This is computationally expensive because it uses the full-batch gradient at
each step. It is a proof-of-concept optimizer, not a production optimizer.
