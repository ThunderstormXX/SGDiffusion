# EXP40 semigroup log matching

Theory equation tested:

`sum_k eta^k M_k = log(1 + sum_k eta^k L_k)`.

Expected result:

The corrected generator should be closer to the matrix logarithm of the exact
one-step SGD transition matrix than the standard generator.

Interpretation:

This is the most direct finite-state check of the logarithmic matching idea.

Limitation:

The grid discretization and matrix logarithm require regularization; results
should be interpreted as a controlled diagnostic, not a large-model experiment.
