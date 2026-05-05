# EXP20: EXP20 anisotropy toy smoke

Smoke test for raw update anisotropy parallel to the mean gradient.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp20_noise_anisotropy_toy_smoke/config.yaml
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
  "dim": 6,
  "mean_grad_norm": 14.462625648582275,
  "mode": "toy",
  "orthogonal_centered_covariance_mean": 0.00022753629790823012,
  "orthogonal_raw_second_moment_mean": 0.00022761060261417335,
  "parallel_centered_covariance": 0.00022312561000350378,
  "parallel_over_orthogonal_raw_ratio": 827.9093567285179,
  "parallel_raw_second_moment": 0.18844094759489058,
  "pass": true
}
```