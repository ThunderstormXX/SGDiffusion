# EXP39 drift correction ablation

Theory equation tested:

The corrected generator changes drift by `eta/2 Hg`; it does not add `gg^T` to
conditional covariance.

Expected result:

The corrected drift should improve mean evolution. The raw `D + gg^T`
covariance model should not improve conditional likelihood.

Interpretation:

This separates the new PDF correction from the older naive modified Langevin
idea.

Limitation:

The experiment is local and synthetic, designed as a diagnostic ablation.
