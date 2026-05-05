# EXP5: Non-stationary regime validation

Early/mid-training point test for non-zero-gradient theory improvement over stationary approximation.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp5_nonstationary_full/config.yaml
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
  "fit_improvement": -5.3536606780350436e-05,
  "modified_langevin_final_loss_ci95_high": 2.278110782038228,
  "modified_langevin_final_loss_ci95_low": 2.2774972322274687,
  "modified_langevin_final_loss_mean": 2.2778040071328483,
  "modified_langevin_final_loss_std": 0.0007667775347656021,
  "modified_langevin_final_mean_path_error_to_sgd": 0.0058895801194012165,
  "modified_langevin_final_variance": 1.4229708540369757e-05,
  "pass": false,
  "reference_full_gradient_norm": 0.06345392763614655,
  "sgd_replacement_final_loss_ci95_high": 2.2782589141197906,
  "sgd_replacement_final_loss_ci95_low": 2.27723511619688,
  "sgd_replacement_final_loss_mean": 2.277747015158335,
  "sgd_replacement_final_loss_std": 0.0012794808728839025,
  "sgd_replacement_final_mean_path_error_to_sgd": 0.0,
  "sgd_replacement_final_variance": 1.1672078471747227e-05,
  "standard_langevin_final_loss_ci95_high": 2.2779297386070843,
  "standard_langevin_final_loss_ci95_low": 2.277420425965314,
  "standard_langevin_final_loss_mean": 2.277675082286199,
  "standard_langevin_final_loss_std": 0.0006365082101463125,
  "standard_langevin_final_mean_path_error_to_sgd": 0.005637603811919689,
  "standard_langevin_final_variance": 1.1480385182949249e-05,
  "wasserstein_modified_to_sgd": 0.0011733802345285464,
  "wasserstein_standard_to_sgd": 0.001119843627748196
}
```