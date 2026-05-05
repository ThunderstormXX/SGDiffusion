# EXP8: Theory-Guided Diffusion Amplification

Goal: test whether the theory can be used constructively, not only
diagnostically. EXP7 showed that suppressing directions with high predicted SGD
variance can reduce diffusion but slightly hurts loss. EXP8 tests the opposite
intervention: amplify minibatch noise specifically in those theory-selected
directions.

For each minibatch gradient we decompose

```text
g_batch(theta) = g_full(theta) + noise(theta)
```

and update with

```text
theta <- theta - eta * (g_full + noise + (alpha - 1) P noise)
```

where `P` projects onto one of:

- theory-predicted high-variance Hessian directions;
- random Hessian-basis directions;
- sharp Hessian directions.

Compared methods:

- `baseline_sgd`: ordinary minibatch SGD;
- `amplify_theory_high_variance`: amplify noise in theory-selected directions;
- `amplify_random`: random-direction control;
- `amplify_sharp`: sharp-direction control.

Primary metrics:

- train/test loss over time;
- final train/test loss and generalization gap;
- variance in theory-high-variance and sharp subspaces;
- whether theory-guided amplification improves or hurts loss relative to
  controls.

This is a small MLP-386 full-lite experiment intended to test the intervention
mechanism before scaling it up.
