# EXP43 MLP mean trajectory and covariance

Theory equation tested:

The drift-corrected Langevin approximation uses

`f_eta(w) = grad Lbar(w) + eta/2 H(w) grad Lbar(w)`

so one discrete surrogate step is

`w_{t+1} = w_t - eta g(w_t) - eta^2/2 H(w_t)g(w_t) + eta D(w_t)^{1/2} xi_t`.

Expected result:

The drift-corrected Langevin ensemble mean should track the exact SGD ensemble
mean better than standard Langevin at finite `eta`. Covariance may or may not
improve; it is reported separately.

Interpretation:

This experiment tests whether the `eta^2/2 H g` drift correction improves the
projected ensemble mean trajectory of Langevin dynamics relative to exact SGD
on the same MLP setup. This is not a new optimizer.

Limitation:

For tractability, `g`, `H`, and centered gradient covariance `D` are estimated
at the reference point `w0` and frozen for the Langevin surrogates.
