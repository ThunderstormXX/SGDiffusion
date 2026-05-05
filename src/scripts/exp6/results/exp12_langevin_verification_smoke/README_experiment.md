# EXP12: Langevin implementation verification smoke

Diagnostic audit for standard and modified Langevin baselines: noise scaling, one-step matching, horizon errors, and modified-noise calibration.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp12_langevin_verification_smoke/config.yaml
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
  "implementation_verdict": "projection_metric_misleading",
  "modified_best_calibration": 0.5,
  "modified_best_calibration_cov_error": 0.8519664773815401,
  "modified_best_horizon_calibration": "modified_c_0.5",
  "modified_calibration_improves_local_match": false,
  "modified_cov_error": 0.6778457567050307,
  "modified_long_horizon_error": 0.23899970948696136,
  "modified_mean_update_error": 0.7768677723624848,
  "modified_noise_scaling_mode": "Euler-Maruyama with dt=lr: lr * sqrt(D + grad_bar^2)",
  "modified_projection_error": 0.22684143483638763,
  "modified_short_horizon_error": 0.7160745859146118,
  "pass": true,
  "standard_cov_error": 0.7897527223379495,
  "standard_long_horizon_error": 0.28986507654190063,
  "standard_mean_update_error": 0.6021553051006283,
  "standard_noise_scaling_mode": "Euler-Maruyama with dt=lr: lr * sqrt(D)",
  "standard_projection_error": 0.13406796753406525,
  "standard_short_horizon_error": 0.7461097836494446
}
```