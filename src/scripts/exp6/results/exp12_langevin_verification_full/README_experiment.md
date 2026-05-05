# EXP12: Langevin implementation verification

Diagnostic audit for standard and modified Langevin baselines at finite learning rate.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp12_langevin_verification_full/config.yaml
```

## Artifacts

- `config.yaml`: exact configuration used for this run.
- `environment.json`: Python, package, hardware, git metadata.
- `runtime.json`: runtime and RSS memory snapshot.
- `metrics.json`: machine-readable primary metrics.
- `raw_outputs.npz`: raw trajectories/statistics.
- `figure_data.csv`: plotted data with mean/std/95% CI when applicable.
- `make_figure.py`: figure generation from saved artifacts only.

## Primary Metrics

```json
{
  "implementation_verdict": "scaling_mismatch_with_langevin_sde",
  "modified_best_calibration": 0.75,
  "modified_best_calibration_cov_error": 0.7977017642747799,
  "modified_best_horizon_calibration": "modified_c_0.5",
  "modified_calibration_improves_local_match": true,
  "modified_cov_error": 1.0010095738267235,
  "modified_long_horizon_error": 0.050199154764413834,
  "modified_mean_update_error": 0.5322616796437822,
  "modified_noise_scaling_mode": "lr_times_reestimated_empirical_minibatch_noise_std",
  "modified_projection_error": 0.04204655811190605,
  "modified_short_horizon_error": 0.5529146194458008,
  "pass": true,
  "standard_cov_error": 1.1799686342956466,
  "standard_long_horizon_error": 0.03486575186252594,
  "standard_mean_update_error": 0.5579449409679734,
  "standard_noise_scaling_mode": "lr_times_empirical_minibatch_noise_std",
  "standard_projection_error": 0.04431967809796333,
  "standard_short_horizon_error": 0.6760289669036865
}
```