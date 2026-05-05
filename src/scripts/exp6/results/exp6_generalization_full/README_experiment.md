# EXP6: Diffusion vs generalization

Correlate Hessian flat/sharp eigenspace displacement with final test performance and generalization gap.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp6_generalization_full/config.yaml
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
  "final_test_loss_ci95_high": 1.1441610621543261,
  "final_test_loss_ci95_low": 1.1431816417469052,
  "final_test_loss_mean": 1.1436713519506156,
  "final_test_loss_std": 0.003997634316003631,
  "final_train_loss_ci95_high": 1.0022218001189471,
  "final_train_loss_ci95_low": 1.0017628560659646,
  "final_train_loss_mean": 1.0019923280924559,
  "final_train_loss_std": 0.0018732410325813706,
  "flat_displacement_mean": 0.0012421391534189752,
  "flat_eigenvalue_abs_max": 1.1366931573775219e-07,
  "generalization_gap_ci95_high": 0.14210548485928015,
  "generalization_gap_ci95_low": 0.1412525628570394,
  "generalization_gap_mean": 0.14167902385815978,
  "generalization_gap_std": 0.0034813142948601247,
  "pass": true,
  "pearson_flat_displacement_generalization_gap": -0.012117979697099525,
  "pearson_flat_displacement_generalization_gap_pvalue": 0.8470013303997846,
  "pearson_flat_displacement_test_loss": -0.05815575969167727,
  "pearson_sharp_displacement_generalization_gap": 0.23793355078334547,
  "pearson_total_displacement_generalization_gap": 0.04867937094113394,
  "regression_gap_flat_coef": -0.04911005438962225,
  "regression_gap_intercept": 0.11195426257059851,
  "regression_gap_r2": 0.05667751179293479,
  "regression_gap_sharp_coef": 0.057118988889187906,
  "regression_gap_train_loss_coef": 0.01435772929394408,
  "sharp_displacement_mean": 0.2696026051416993,
  "sharp_eigenvalue_max": 1.9151133298873901,
  "sharp_eigenvalue_min": 0.07089203596115112,
  "spearman_flat_displacement_generalization_gap": -0.003703650720988784,
  "spearman_flat_displacement_generalization_gap_pvalue": 0.9529771073322686,
  "spearman_flat_displacement_test_loss": -0.0362003604943923,
  "spearman_sharp_displacement_generalization_gap": 0.24126270336152522,
  "spearman_total_displacement_generalization_gap": 0.010317008024756977,
  "total_displacement_mean": 3.739494327455759
}
```