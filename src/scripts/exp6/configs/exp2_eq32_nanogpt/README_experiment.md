# EXP2 NanoGPT Eq. 32 Validation

Second EXP2 variant using the Shakespeare/NanoGPT setup from
`src/scripts/exp2/bash/setup2/config.sh`.

The full config reuses the existing setup2 checkpoint
`src/scripts/exp2/exp_results/setup2/initial_after_sgd_and_gd.pt` as the
reference point, then recomputes Hessian/gradient-noise statistics and SGD
trajectory covariance inside the exp6 reproducible pipeline.

