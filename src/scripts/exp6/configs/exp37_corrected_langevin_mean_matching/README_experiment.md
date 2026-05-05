# EXP37 corrected Langevin one-step mean matching

Theory equation tested:

`M1 = L1`, `M2 = L2 - 1/2 L1 L1`, which gives the corrected generator drift
`b_corr(w) = -g(w) - eta/2 H(w)g(w)`.

Expected result:

The flow of the corrected continuous generator over time `eta` should match the
discrete SGD Euler mean map better than the standard generator at finite `eta`.

Interpretation:

The correction is a generator/backward-error correction. It should not be read
as changing the conditional covariance.

Limitation:

This is a synthetic nonlinear-loss test; it isolates the `Hg` mechanism.
