# EXP38 corrected generator distribution matching

Theory equation tested:

`M2 = L2 - 1/2 L1 L1`, implemented as the corrected drift
`-g - eta/2 Hg` with centered diffusion covariance `D`.

Expected result:

The corrected generator should reduce moment evolution error, especially the
ensemble mean error, compared with the standard generator.

Interpretation:

This tests generator-level distribution evolution, not one-step conditional
likelihood.

Limitation:

The surrogate remains Gaussian, so higher cumulants are not expected to match.
