# EXP8: Theory-guided diffusion amplification smoke

Smoke test amplifying minibatch noise in theory-predicted high-variance Hessian directions.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp8_diffusion_amplification_smoke/config.yaml
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
  "amplification_factor": 2.0,
  "amplify_random_generalization_gap_ci95_high": 0.015009436654493674,
  "amplify_random_generalization_gap_ci95_low": 0.013264622641160623,
  "amplify_random_generalization_gap_mean": 0.014137029647827148,
  "amplify_random_generalization_gap_std": 0.0010902816399805252,
  "amplify_random_sharp_final_variance": 0.00022768194321542978,
  "amplify_random_test_loss_ci95_high": 2.3167528617947757,
  "amplify_random_test_loss_ci95_low": 2.314010986827864,
  "amplify_random_test_loss_delta": 2.1457672119140625e-06,
  "amplify_random_test_loss_mean": 2.31538192431132,
  "amplify_random_test_loss_std": 0.001713314950878429,
  "amplify_random_theory_high_variance_final_variance": 6.798630547422135e-09,
  "amplify_random_total_displacement_mean": 0.12637217218677202,
  "amplify_random_total_displacement_std": 0.015446847359961954,
  "amplify_random_train_loss_ci95_high": 2.302424526013955,
  "amplify_random_train_loss_ci95_low": 2.3000652633130305,
  "amplify_random_train_loss_mean": 2.3012448946634927,
  "amplify_random_train_loss_std": 0.0014742320883786213,
  "amplify_sharp_generalization_gap_ci95_high": 0.015801407266074177,
  "amplify_sharp_generalization_gap_ci95_low": 0.01262325309871261,
  "amplify_sharp_generalization_gap_mean": 0.014212330182393393,
  "amplify_sharp_generalization_gap_std": 0.001985932661718308,
  "amplify_sharp_sharp_final_variance": 0.0009114564745686948,
  "amplify_sharp_test_loss_ci95_high": 2.3174598630369676,
  "amplify_sharp_test_loss_ci95_low": 2.3134317779440705,
  "amplify_sharp_test_loss_delta": 6.604194641113281e-05,
  "amplify_sharp_test_loss_mean": 2.315445820490519,
  "amplify_sharp_test_loss_std": 0.002517028856660675,
  "amplify_sharp_theory_high_variance_final_variance": 6.843272615242313e-09,
  "amplify_sharp_total_displacement_mean": 0.13517118245363235,
  "amplify_sharp_total_displacement_std": 0.023005128173932788,
  "amplify_sharp_train_loss_ci95_high": 2.303069678740306,
  "amplify_sharp_train_loss_ci95_low": 2.299397301875946,
  "amplify_sharp_train_loss_mean": 2.301233490308126,
  "amplify_sharp_train_loss_std": 0.002294757515531751,
  "amplify_theory_high_variance_generalization_gap_ci95_high": 0.015009627394199266,
  "amplify_theory_high_variance_generalization_gap_ci95_low": 0.013264749792893831,
  "amplify_theory_high_variance_generalization_gap_mean": 0.014137188593546549,
  "amplify_theory_high_variance_generalization_gap_std": 0.0010903213741861675,
  "amplify_theory_high_variance_sharp_final_variance": 0.0002277592575410381,
  "amplify_theory_high_variance_test_loss_ci95_high": 2.3167441794256223,
  "amplify_theory_high_variance_test_loss_ci95_low": 2.3140153776625936,
  "amplify_theory_high_variance_test_loss_mean": 2.315379778544108,
  "amplify_theory_high_variance_test_loss_std": 0.0017051459001601495,
  "amplify_theory_high_variance_theory_high_variance_final_variance": 2.718821612290867e-08,
  "amplify_theory_high_variance_total_displacement_mean": 0.12634951993823051,
  "amplify_theory_high_variance_total_displacement_std": 0.015366695383281595,
  "amplify_theory_high_variance_train_loss_ci95_high": 2.302410377415171,
  "amplify_theory_high_variance_train_loss_ci95_low": 2.300074802485952,
  "amplify_theory_high_variance_train_loss_mean": 2.3012425899505615,
  "amplify_theory_high_variance_train_loss_std": 0.0014594303144449142,
  "amplify_theory_test_loss_delta": 0.0,
  "amplify_theory_train_loss_delta": 0.0,
  "amplify_theory_variance_ratio": 3.998638870628376,
  "baseline_sgd_generalization_gap_ci95_high": 0.015009631165347493,
  "baseline_sgd_generalization_gap_ci95_low": 0.013264746021745604,
  "baseline_sgd_generalization_gap_mean": 0.014137188593546549,
  "baseline_sgd_generalization_gap_std": 0.001090326087139688,
  "baseline_sgd_sharp_final_variance": 0.00022775899560656399,
  "baseline_sgd_test_loss_ci95_high": 2.316744152729349,
  "baseline_sgd_test_loss_ci95_low": 2.3140154043588668,
  "baseline_sgd_test_loss_mean": 2.315379778544108,
  "baseline_sgd_test_loss_std": 0.0017051125367686952,
  "baseline_sgd_theory_high_variance_final_variance": 6.7993677355104865e-09,
  "baseline_sgd_total_displacement_mean": 0.1263493224978447,
  "baseline_sgd_total_displacement_std": 0.015366724252687407,
  "baseline_sgd_train_loss_ci95_high": 2.302410374398325,
  "baseline_sgd_train_loss_ci95_low": 2.300074805502798,
  "baseline_sgd_train_loss_mean": 2.3012425899505615,
  "baseline_sgd_train_loss_std": 0.001459426544172364,
  "pass": true,
  "sharp_eigenvalue_max": 0.23689129948616028,
  "sharp_eigenvalue_min": 0.15011925995349884,
  "theory_high_variance_eigenvalue_abs_max": 1.4826238839305006e-05,
  "theory_high_variance_eigenvalue_abs_min": 3.266654147182635e-08,
  "theory_high_variance_predicted_max": 0.0014353571459650993,
  "theory_high_variance_predicted_min": 0.0013418883318081498
}
```