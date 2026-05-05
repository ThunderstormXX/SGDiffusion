# EXP1A: Toy falsification test

Fast toy test where multiplicative discrete SGD noise is non-stationary while standard additive Langevin is stationary.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp1a_toy_smoke/config.yaml
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
  "modified_langevin_final_mean_path_error_to_sgd": 0.08180334315451229,
  "modified_langevin_late_variance_slope": -0.13512485501995386,
  "pass": true,
  "sgd_late_variance_slope": -0.018294615928616576,
  "standard_langevin_final_mean_path_error_to_sgd": 0.03130168732749242,
  "standard_langevin_late_variance_slope": 0.000713476521930245,
  "wasserstein_modified_to_sgd": 0.08180827931054228,
  "wasserstein_standard_to_sgd": 1.2015244869869757
}
```