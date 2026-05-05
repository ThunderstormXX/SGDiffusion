# EXP18: EXP18 learning-rate scaling toy full

Full toy test for eta scaling of centered covariance and raw second moment.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp18_learning_rate_scaling_toy_full/config.yaml
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
  "centered_loglog_slope": 2.0000000000000027,
  "centered_prediction_relative_error": 0.0015828212007895541,
  "dim": 10,
  "mean_grad_norm": 3.997408931250685,
  "mode": "toy",
  "n_runs": 50000,
  "pass": true,
  "raw_loglog_slope": 2.0000000000000013,
  "raw_over_centered_ratio_at_max_eta": 7.384708994459518,
  "raw_prediction_relative_error": 0.0006361098733915923
}
```