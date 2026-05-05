# EXP3: Sampling robustness

Replacement vs no-replacement SGD covariance and late-slope comparison.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp3_sampling_full/config.yaml
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
  "no_replacement_late_slope": -1.9758040390584936e-07,
  "pass": true,
  "reference_full_gradient_norm": 0.221458300948143,
  "replacement_late_slope": 9.383114626293431e-08,
  "replacement_vs_no_replacement_wasserstein": 0.002768145074696804,
  "sgd_no_replacement_final_loss_ci95_high": 0.8065591116505204,
  "sgd_no_replacement_final_loss_ci95_low": 0.8053473628056707,
  "sgd_no_replacement_final_loss_mean": 0.8059532372280955,
  "sgd_no_replacement_final_loss_std": 0.0034972890828088702,
  "sgd_no_replacement_final_mean_path_error_to_sgd": 0.004866390954703093,
  "sgd_no_replacement_final_variance": 6.698790457448922e-06,
  "sgd_replacement_final_loss_ci95_high": 0.8075335159019726,
  "sgd_replacement_final_loss_ci95_low": 0.8061342264472348,
  "sgd_replacement_final_loss_mean": 0.8068338711746037,
  "sgd_replacement_final_loss_std": 0.0040385594379098435,
  "sgd_replacement_final_mean_path_error_to_sgd": 0.0,
  "sgd_replacement_final_variance": 6.46879561827518e-05
}
```