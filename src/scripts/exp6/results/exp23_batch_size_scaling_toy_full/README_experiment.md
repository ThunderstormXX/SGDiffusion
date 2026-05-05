# EXP23: EXP23 batch-size scaling toy full

Full test for batch-size scaling of centered covariance and raw moment.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp23_batch_size_scaling_toy_full/config.yaml
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
  "centered_large_batch_ratio_to_b1": 0.0009748562768758786,
  "large_batch_raw_over_deterministic_floor": 1.0000374949684594,
  "mean_grad_norm": 7.99481786250137,
  "mode": "toy",
  "pass": true,
  "raw_large_batch_ratio_to_b1": 0.9621122666749982
}
```