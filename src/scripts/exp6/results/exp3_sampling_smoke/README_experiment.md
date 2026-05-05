# EXP3: Sampling robustness smoke

Compare replacement and no-replacement SGD ensembles from the same MLP-386 reference point.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp3_sampling_smoke/config.yaml
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
  "no_replacement_late_slope": 3.647580186547549e-09,
  "pass": true,
  "reference_full_gradient_norm": 0.10600215941667557,
  "replacement_late_slope": 8.81074246342797e-08,
  "replacement_vs_no_replacement_wasserstein": 0.0012131405140583713,
  "sgd_no_replacement_final_loss_ci95_high": 2.3282148328991776,
  "sgd_no_replacement_final_loss_ci95_low": 2.3281810792712325,
  "sgd_no_replacement_final_loss_mean": 2.328197956085205,
  "sgd_no_replacement_final_loss_std": 2.1091623835178826e-05,
  "sgd_no_replacement_final_mean_path_error_to_sgd": 0.0017074075294658542,
  "sgd_no_replacement_final_variance": 5.70441138769695e-10,
  "sgd_replacement_final_loss_ci95_high": 2.3288237812417076,
  "sgd_replacement_final_loss_ci95_low": 2.326803628375399,
  "sgd_replacement_final_loss_mean": 2.327813704808553,
  "sgd_replacement_final_loss_std": 0.0012623325828768474,
  "sgd_replacement_final_mean_path_error_to_sgd": 0.0,
  "sgd_replacement_final_variance": 3.6982066831114935e-06
}
```