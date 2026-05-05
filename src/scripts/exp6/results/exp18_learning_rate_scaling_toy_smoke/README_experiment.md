# EXP18: EXP18 learning-rate scaling toy smoke

Smoke test for eta scaling of centered covariance and raw second moment.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp18_learning_rate_scaling_toy_smoke/config.yaml
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
  "centered_loglog_slope": 2.0000000000000013,
  "centered_prediction_relative_error": 0.007941683687769601,
  "dim": 6,
  "mean_grad_norm": 7.2313128242911375,
  "mode": "toy",
  "n_runs": 5000,
  "pass": true,
  "raw_loglog_slope": 2.0000000000000013,
  "raw_over_centered_ratio_at_max_eta": 35.572561097639195,
  "raw_prediction_relative_error": 0.00010954846819879027
}
```