# EXP20: EXP20 anisotropy toy full

Full test for raw update anisotropy parallel to the mean gradient.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp20_noise_anisotropy_toy_full/config.yaml
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
  "dim": 10,
  "mean_grad_norm": 7.99481786250137,
  "mode": "toy",
  "orthogonal_centered_covariance_mean": 0.00022536883879140522,
  "orthogonal_raw_second_moment_mean": 0.0002253718399858702,
  "parallel_centered_covariance": 0.00022524744449224543,
  "parallel_over_orthogonal_raw_ratio": 256.30811808166504,
  "parallel_raw_second_moment": 0.05776463217538053,
  "pass": true
}
```