# EXP23: EXP23 batch-size scaling toy smoke

Smoke test for batch-size scaling of centered covariance and raw moment.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp23_batch_size_scaling_toy_smoke/config.yaml
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
  "centered_large_batch_ratio_to_b1": 0.0038578852586874236,
  "large_batch_raw_over_deterministic_floor": 1.0000307750309478,
  "mean_grad_norm": 14.462625648582275,
  "mode": "toy",
  "pass": true,
  "raw_large_batch_ratio_to_b1": 0.9930244352274746
}
```