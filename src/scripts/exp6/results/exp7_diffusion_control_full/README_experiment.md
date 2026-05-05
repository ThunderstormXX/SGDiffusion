# EXP7: Hessian-guided diffusion control

Interventional MLP-386 test suppressing SGD update components along Hessian-flat directions.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp7_diffusion_control_full/config.yaml
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
  "baseline_sgd_flat_final_variance": 8.830055575614892e-10,
  "baseline_sgd_generalization_gap_ci95_high": 0.1531999331281839,
  "baseline_sgd_generalization_gap_ci95_low": 0.1512846925292156,
  "baseline_sgd_generalization_gap_mean": 0.15224231282869974,
  "baseline_sgd_generalization_gap_std": 0.003384993904216082,
  "baseline_sgd_sharp_final_variance": 0.0002909599570557475,
  "baseline_sgd_test_loss_ci95_high": 1.235586425262881,
  "baseline_sgd_test_loss_ci95_low": 1.2334548725702437,
  "baseline_sgd_test_loss_mean": 1.2345206489165623,
  "baseline_sgd_test_loss_std": 0.0037673036353652466,
  "baseline_sgd_theory_high_variance_final_variance": 1.6155738194356672e-05,
  "baseline_sgd_total_displacement_mean": 3.874776596824328,
  "baseline_sgd_total_displacement_std": 0.011549485468521378,
  "baseline_sgd_train_loss_ci95_high": 1.0828047850180185,
  "baseline_sgd_train_loss_ci95_low": 1.081751887157707,
  "baseline_sgd_train_loss_mean": 1.0822783360878627,
  "baseline_sgd_train_loss_std": 0.0018608903971840712,
  "flat_eigenvalue_abs_max": 4.889341376212997e-09,
  "flat_variance_reduction_fraction": 0.9999810234196745,
  "pass": true,
  "sharp_eigenvalue_max": 1.58993399143219,
  "sharp_eigenvalue_min": 0.4514719843864441,
  "suppress_flat_flat_final_variance": 1.6756425890926407e-14,
  "suppress_flat_generalization_gap_ci95_high": 0.1531999491163531,
  "suppress_flat_generalization_gap_ci95_low": 0.15128471876100313,
  "suppress_flat_generalization_gap_mean": 0.15224233393867811,
  "suppress_flat_generalization_gap_std": 0.0033849757996574754,
  "suppress_flat_sharp_final_variance": 0.00029095998615957797,
  "suppress_flat_test_loss_ci95_high": 1.2355864395786365,
  "suppress_flat_test_loss_ci95_low": 1.23345488060623,
  "suppress_flat_test_loss_delta": 1.1175870895385742e-08,
  "suppress_flat_test_loss_mean": 1.2345206600924332,
  "suppress_flat_test_loss_std": 0.0037673147342220225,
  "suppress_flat_test_std_reduction_fraction": -2.946100938538976e-06,
  "suppress_flat_theory_high_variance_final_variance": 1.615592191228643e-05,
  "suppress_flat_total_displacement_mean": 3.8747765719890594,
  "suppress_flat_total_displacement_std": 0.011549462627928095,
  "suppress_flat_train_loss_ci95_high": 1.0828047821867945,
  "suppress_flat_train_loss_ci95_low": 1.0817518701207158,
  "suppress_flat_train_loss_delta": -9.934107536579972e-09,
  "suppress_flat_train_loss_mean": 1.0822783261537552,
  "suppress_flat_train_loss_std": 0.0018609155044396165,
  "suppress_sharp_flat_final_variance": 8.959535890973314e-10,
  "suppress_sharp_generalization_gap_ci95_high": 0.15611521955359428,
  "suppress_sharp_generalization_gap_ci95_low": 0.15523207877051384,
  "suppress_sharp_generalization_gap_mean": 0.15567364916205406,
  "suppress_sharp_generalization_gap_std": 0.0015608619454402478,
  "suppress_sharp_sharp_final_variance": 1.1165416871237663e-13,
  "suppress_sharp_test_loss_ci95_high": 1.2540526167859896,
  "suppress_sharp_test_loss_ci95_low": 1.2529364872663953,
  "suppress_sharp_test_loss_mean": 1.2534945520261924,
  "suppress_sharp_test_loss_std": 0.001972645954861859,
  "suppress_sharp_theory_high_variance_final_variance": 1.7943464627023786e-05,
  "suppress_sharp_total_displacement_mean": 3.806496108571688,
  "suppress_sharp_total_displacement_std": 0.01171549462539344,
  "suppress_sharp_train_loss_ci95_high": 1.0982355699657835,
  "suppress_sharp_train_loss_ci95_low": 1.0974062357624932,
  "suppress_sharp_train_loss_mean": 1.0978209028641384,
  "suppress_sharp_train_loss_std": 0.001465764261789226,
  "suppress_theory_high_variance_flat_final_variance": 8.828044961717296e-10,
  "suppress_theory_high_variance_generalization_gap_ci95_high": 0.15426533643094334,
  "suppress_theory_high_variance_generalization_gap_ci95_low": 0.15234310702793166,
  "suppress_theory_high_variance_generalization_gap_mean": 0.1533042217294375,
  "suppress_theory_high_variance_generalization_gap_std": 0.003397345907978559,
  "suppress_theory_high_variance_sharp_final_variance": 0.0002934534568339586,
  "suppress_theory_high_variance_test_loss_ci95_high": 1.2453081680828342,
  "suppress_theory_high_variance_test_loss_ci95_low": 1.2431773420677572,
  "suppress_theory_high_variance_test_loss_delta": 0.009722106158733368,
  "suppress_theory_high_variance_test_loss_mean": 1.2442427550752957,
  "suppress_theory_high_variance_test_loss_std": 0.003766019306329106,
  "suppress_theory_high_variance_theory_high_variance_final_variance": 2.764984049282316e-14,
  "suppress_theory_high_variance_total_displacement_mean": 3.8492533614238105,
  "suppress_theory_high_variance_total_displacement_std": 0.011460403905733624,
  "suppress_theory_high_variance_train_loss_ci95_high": 1.0914562107002739,
  "suppress_theory_high_variance_train_loss_ci95_low": 1.0904208559914428,
  "suppress_theory_high_variance_train_loss_delta": 0.008660197257995605,
  "suppress_theory_high_variance_train_loss_mean": 1.0909385333458583,
  "suppress_theory_high_variance_train_loss_std": 0.0018298846526032806,
  "theory_high_variance_eigenvalue_abs_max": 0.0011349732521921396,
  "theory_high_variance_eigenvalue_abs_min": 6.560068044336731e-08,
  "theory_high_variance_predicted_max": 0.005059540271759033,
  "theory_high_variance_predicted_min": 0.0014868632424622774,
  "theory_high_variance_reduction_fraction": 0.9999999982885437
}
```