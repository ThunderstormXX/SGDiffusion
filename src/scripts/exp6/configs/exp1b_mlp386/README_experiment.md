# EXP1B MLP-386 Falsification Test

Starts MLP-386 from a fixed early reference point and compares replacement SGD,
without-replacement SGD, standard Langevin, and modified Langevin-like dynamics.
Langevin trajectories are integrated with Euler-Maruyama substeps `dt = eta / M`
inside each outer SGD step, then compared to SGD at times `eta * n`. The default
configs use a fast full-run mode: Langevin drift and centered gradient-noise
coefficients are frozen at the reference point while the SDE is still integrated
with `M` Euler-Maruyama substeps. Set `langevin_drift_mode: current`,
`langevin_coefficient_update_every: 1`, and `langevin_noise_mode: current` for the
exact but much slower per-`dt` coefficient update.
