# EXP7: Hessian-guided diffusion control smoke

Smoke test for suppressing SGD updates along Hessian-flat directions in MLP-386.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp7_diffusion_control_smoke/config.yaml
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
  "baseline_sgd_flat_final_variance": 1.6320429233854412e-14,
  "baseline_sgd_generalization_gap_ci95_high": -0.0006158513603214759,
  "baseline_sgd_generalization_gap_ci95_low": -0.0027385788383479577,
  "baseline_sgd_generalization_gap_mean": -0.0016772150993347168,
  "baseline_sgd_generalization_gap_std": 0.0015316275452281055,
  "baseline_sgd_sharp_final_variance": 0.00021203856158535928,
  "baseline_sgd_test_loss_ci95_high": 2.302161581094491,
  "baseline_sgd_test_loss_ci95_low": 2.3011696867392186,
  "baseline_sgd_test_loss_mean": 2.301665633916855,
  "baseline_sgd_test_loss_std": 0.0007156890049324759,
  "baseline_sgd_theory_high_variance_final_variance": 7.680333702353437e-09,
  "baseline_sgd_total_displacement_mean": 0.26273864321410656,
  "baseline_sgd_total_displacement_std": 0.010853477922176766,
  "baseline_sgd_train_loss_ci95_high": 2.3040539043493964,
  "baseline_sgd_train_loss_ci95_low": 2.3026317936829828,
  "baseline_sgd_train_loss_mean": 2.3033428490161896,
  "baseline_sgd_train_loss_std": 0.0010261062202233172,
  "flat_eigenvalue_abs_max": 7.033096327546673e-12,
  "flat_variance_reduction_fraction": 0.9998760638464586,
  "pass": true,
  "sharp_eigenvalue_max": 0.20786572992801666,
  "sharp_eigenvalue_min": 0.17053331434726715,
  "suppress_flat_flat_final_variance": 1.2393615354146492e-16,
  "suppress_flat_generalization_gap_ci95_high": -0.0006158513603214759,
  "suppress_flat_generalization_gap_ci95_low": -0.0027385788383479577,
  "suppress_flat_generalization_gap_mean": -0.0016772150993347168,
  "suppress_flat_generalization_gap_std": 0.0015316275452281055,
  "suppress_flat_sharp_final_variance": 0.00021203856158535928,
  "suppress_flat_test_loss_ci95_high": 2.302161581094491,
  "suppress_flat_test_loss_ci95_low": 2.3011696867392186,
  "suppress_flat_test_loss_delta": 0.0,
  "suppress_flat_test_loss_mean": 2.301665633916855,
  "suppress_flat_test_loss_std": 0.0007156890049324759,
  "suppress_flat_test_std_reduction_fraction": 0.0,
  "suppress_flat_theory_high_variance_final_variance": 7.680344360494473e-09,
  "suppress_flat_total_displacement_mean": 0.26273863948881626,
  "suppress_flat_total_displacement_std": 0.010853479177706177,
  "suppress_flat_train_loss_ci95_high": 2.3040539043493964,
  "suppress_flat_train_loss_ci95_low": 2.3026317936829828,
  "suppress_flat_train_loss_delta": 0.0,
  "suppress_flat_train_loss_mean": 2.3033428490161896,
  "suppress_flat_train_loss_std": 0.0010261062202233172,
  "suppress_sharp_flat_final_variance": 1.5949225515049688e-14,
  "suppress_sharp_generalization_gap_ci95_high": -0.007499521729001519,
  "suppress_sharp_generalization_gap_ci95_low": -0.00906752968167353,
  "suppress_sharp_generalization_gap_mean": -0.008283525705337524,
  "suppress_sharp_generalization_gap_std": 0.0011313765880498101,
  "suppress_sharp_sharp_final_variance": 1.5540785592608954e-15,
  "suppress_sharp_test_loss_ci95_high": 2.3092538424452713,
  "suppress_sharp_test_loss_ci95_low": 2.308114855674846,
  "suppress_sharp_test_loss_mean": 2.3086843490600586,
  "suppress_sharp_test_loss_std": 0.0008218217031116712,
  "suppress_sharp_theory_high_variance_final_variance": 7.707963156633468e-09,
  "suppress_sharp_total_displacement_mean": 0.19328922219574451,
  "suppress_sharp_total_displacement_std": 0.009436571560620693,
  "suppress_sharp_train_loss_ci95_high": 2.317430125611343,
  "suppress_sharp_train_loss_ci95_low": 2.3165056239194493,
  "suppress_sharp_train_loss_mean": 2.316967874765396,
  "suppress_sharp_train_loss_std": 0.0006670626689352575,
  "suppress_theory_high_variance_flat_final_variance": 1.6373072331526267e-14,
  "suppress_theory_high_variance_generalization_gap_ci95_high": -0.000615763002109601,
  "suppress_theory_high_variance_generalization_gap_ci95_low": -0.002738369173335956,
  "suppress_theory_high_variance_generalization_gap_mean": -0.0016770660877227783,
  "suppress_theory_high_variance_generalization_gap_std": 0.0015315400178189486,
  "suppress_theory_high_variance_sharp_final_variance": 0.00021203838696237653,
  "suppress_theory_high_variance_test_loss_ci95_high": 2.3021617423154366,
  "suppress_theory_high_variance_test_loss_ci95_low": 2.301169883146142,
  "suppress_theory_high_variance_test_loss_delta": 1.7881393432617188e-07,
  "suppress_theory_high_variance_test_loss_mean": 2.301665812730789,
  "suppress_theory_high_variance_test_loss_std": 0.0007156636169287975,
  "suppress_theory_high_variance_theory_high_variance_final_variance": 5.640979495477952e-16,
  "suppress_theory_high_variance_total_displacement_mean": 0.2627384830266237,
  "suppress_theory_high_variance_total_displacement_std": 0.01085342879851643,
  "suppress_theory_high_variance_train_loss_ci95_high": 2.3040538949469234,
  "suppress_theory_high_variance_train_loss_ci95_low": 2.3026318626901006,
  "suppress_theory_high_variance_train_loss_delta": 2.9802322387695312e-08,
  "suppress_theory_high_variance_train_loss_mean": 2.303342878818512,
  "suppress_theory_high_variance_train_loss_std": 0.0010260496447604378,
  "theory_high_variance_eigenvalue_abs_max": 2.195132537963218e-06,
  "theory_high_variance_eigenvalue_abs_min": 1.5143589848776173e-07,
  "theory_high_variance_predicted_max": 0.0017813908634707332,
  "theory_high_variance_predicted_min": 0.0011422814568504691,
  "theory_high_variance_reduction_fraction": 0.9999999265529375
}
```