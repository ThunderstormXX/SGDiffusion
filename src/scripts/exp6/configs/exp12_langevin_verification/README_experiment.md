# EXP12 Langevin Verification

This is a diagnostic experiment for the EXP1B Langevin baselines. It does not
try to tune the figure. It audits the implemented noise scaling, compares
one-step update moments against true minibatch SGD, compares short/medium/long
horizon errors, and sweeps a small multiplier on the modified Langevin noise.

The output is intended to determine whether modified Langevin is locally
accurate but long-horizon unstable, miscalibrated in noise amplitude, affected
by projection metrics, or simply not better than the standard baseline in this
setup.
