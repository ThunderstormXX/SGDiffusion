# EXP8 Rationale: Theory-Guided Diffusion Amplification

## Motivation

The earlier experiments validate descriptive claims:

- EXP2 checks whether Hessian/gradient-noise statistics predict covariance
  structure in selected eigendirections.
- EXP6 checks whether flat-direction displacement directly correlates with
  generalization; in the current MLP setting it mostly does not.
- EXP7 performs a causal suppression test: removing theory-predicted
  high-variance directions nearly eliminates the corresponding diffusion and
  slightly worsens optimization/test loss.

EXP8 turns this into a constructive intervention. If the high-variance
directions identified by the theory are functionally relevant, then amplifying
the stochastic component specifically in those directions may change convergence
more effectively than isotropic/random controls.

## Hypothesis

Theory-selected high-variance directions are not arbitrary noise directions.
They are directions where SGD naturally injects stochasticity due to the
interaction of Hessian geometry and minibatch gradient noise. Amplifying noise
in this subspace should affect optimization differently from amplifying random
or sharp directions.

## Intervention

At each step we compute both a minibatch gradient and the full-batch gradient:

```text
noise = g_batch(theta) - g_full(theta)
```

Then only the noise component is modified:

```text
g_modified = g_full + noise + (alpha - 1) P noise
```

where `P` is the projector onto the selected eigenspace. This avoids confusing
"amplify diffusion" with simply changing the deterministic gradient.

## What Would Support The Practical Claim?

Evidence in favor:

- theory-guided amplification improves train loss or time-to-threshold relative
  to baseline;
- it outperforms random-direction and sharp-direction amplification controls;
- the effect appears in loss curves, not only in parameter-space variance.

Evidence against:

- theory-guided amplification only increases variance without improving loss;
- random or sharp controls perform equally well;
- all amplification variants destabilize or degrade optimization.

## Relationship to the Paper

EXP8 is not needed to prove that SGD is not Brownian motion. Its purpose is to
show how the theory could guide optimizer interventions: identify where SGD
injects stochasticity and selectively control that stochasticity.
