# EXP36: EXP36 real learning-rate scaling full

Real MLP386 eta-scaling of raw and centered minibatch-gradient update moments.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp36_real_learning_rate_scaling_full/config.yaml
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
  "centered_loglog_slope": 2.0000000000000018,
  "centered_prediction_relative_error": 0.015873015873015917,
  "dim": 386,
  "mean_grad_norm": 0.05486754246595449,
  "mode": "mlp386",
  "n_runs": 64,
  "pass": true,
  "raw_loglog_slope": 2.000000000000002,
  "raw_over_centered_ratio_at_max_eta": 1.2031710649168759,
  "raw_prediction_relative_error": 0.016722762670235244
}
```