# EXP9: Theory-denoised SGD

Prototype optimizer using full-gradient control variates to remove minibatch noise in theory-predicted high-variance directions.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp9_theory_denoised_sgd_full/config.yaml
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
  "baseline_sgd_generalization_gap_ci95_high": 0.07832929978756763,
  "baseline_sgd_generalization_gap_ci95_low": 0.07473926256111606,
  "baseline_sgd_generalization_gap_mean": 0.07653428117434184,
  "baseline_sgd_generalization_gap_std": 0.0031725137128968306,
  "baseline_sgd_sharp_final_variance": 0.0004632033815141767,
  "baseline_sgd_test_loss_ci95_high": 1.890681464632573,
  "baseline_sgd_test_loss_ci95_low": 1.884026071270086,
  "baseline_sgd_test_loss_mean": 1.8873537679513295,
  "baseline_sgd_test_loss_std": 0.005881367065399877,
  "baseline_sgd_theory_high_variance_final_variance": 1.056117184816685e-06,
  "baseline_sgd_total_displacement_mean": 3.8075301249821982,
  "baseline_sgd_total_displacement_std": 0.03260813413248772,
  "baseline_sgd_train_loss_ci95_high": 1.8140285850683966,
  "baseline_sgd_train_loss_ci95_low": 1.807610388485579,
  "baseline_sgd_train_loss_mean": 1.8108194867769878,
  "baseline_sgd_train_loss_std": 0.005671756415512834,
  "denoise_random_generalization_gap_ci95_high": 0.07836322796871276,
  "denoise_random_generalization_gap_ci95_low": 0.07480696029931297,
  "denoise_random_generalization_gap_mean": 0.07658509413401286,
  "denoise_random_generalization_gap_std": 0.0031426715758749893,
  "denoise_random_sharp_final_variance": 0.00047296579577960074,
  "denoise_random_test_loss_ci95_high": 1.890662743944027,
  "denoise_random_test_loss_ci95_low": 1.8840773559628945,
  "denoise_random_test_loss_delta": 1.628200213121822e-05,
  "denoise_random_test_loss_mean": 1.8873700499534607,
  "denoise_random_test_loss_std": 0.005819503352487126,
  "denoise_random_theory_high_variance_final_variance": 8.60648697198485e-07,
  "denoise_random_total_displacement_mean": 3.8074745535850525,
  "denoise_random_total_displacement_std": 0.03233693341026633,
  "denoise_random_train_loss_ci95_high": 1.8139773994139978,
  "denoise_random_train_loss_ci95_low": 1.8075925122248977,
  "denoise_random_train_loss_mean": 1.8107849558194478,
  "denoise_random_train_loss_std": 0.005642320924549569,
  "denoise_sharp_generalization_gap_ci95_high": 0.07838036046703449,
  "denoise_sharp_generalization_gap_ci95_low": 0.07598124517719158,
  "denoise_sharp_generalization_gap_mean": 0.07718080282211304,
  "denoise_sharp_generalization_gap_std": 0.002120096722052684,
  "denoise_sharp_sharp_final_variance": 0.00131161545868963,
  "denoise_sharp_test_loss_ci95_high": 1.8881355537562456,
  "denoise_sharp_test_loss_ci95_low": 1.8819538540056462,
  "denoise_sharp_test_loss_delta": -0.0023090640703835597,
  "denoise_sharp_test_loss_mean": 1.885044703880946,
  "denoise_sharp_test_loss_std": 0.005462764308762194,
  "denoise_sharp_theory_high_variance_final_variance": 1.0720976888478617e-06,
  "denoise_sharp_total_displacement_mean": 3.8192588686943054,
  "denoise_sharp_total_displacement_std": 0.036504590023591965,
  "denoise_sharp_train_loss_ci95_high": 1.8112350514070714,
  "denoise_sharp_train_loss_ci95_low": 1.8044927507105943,
  "denoise_sharp_train_loss_mean": 1.8078639010588329,
  "denoise_sharp_train_loss_std": 0.005958167023574221,
  "denoise_theory_high_variance_generalization_gap_ci95_high": 0.07832934679182248,
  "denoise_theory_high_variance_generalization_gap_ci95_low": 0.0747386592468433,
  "denoise_theory_high_variance_generalization_gap_mean": 0.07653400301933289,
  "denoise_theory_high_variance_generalization_gap_std": 0.0031730883989840204,
  "denoise_theory_high_variance_sharp_final_variance": 0.0004632033815141767,
  "denoise_theory_high_variance_test_loss_ci95_high": 1.8906813938984015,
  "denoise_theory_high_variance_test_loss_ci95_low": 1.8840257049035294,
  "denoise_theory_high_variance_test_loss_mean": 1.8873535494009654,
  "denoise_theory_high_variance_test_loss_std": 0.005881628315558843,
  "denoise_theory_high_variance_theory_high_variance_final_variance": 5.662215585289232e-08,
  "denoise_theory_high_variance_total_displacement_mean": 3.8075297276178994,
  "denoise_theory_high_variance_total_displacement_std": 0.03260848646727154,
  "denoise_theory_high_variance_train_loss_ci95_high": 1.814028665055453,
  "denoise_theory_high_variance_train_loss_ci95_low": 1.8076104277078122,
  "denoise_theory_high_variance_train_loss_mean": 1.8108195463816326,
  "denoise_theory_high_variance_train_loss_std": 0.0056717924393621735,
  "denoise_theory_test_loss_delta": -2.1855036402840256e-07,
  "denoise_theory_train_loss_delta": 5.960464477539063e-08,
  "denoise_theory_variance_reduction_fraction": 0.946386483747331,
  "pass": false,
  "sharp_eigenvalue_max": 0.17162853479385376,
  "sharp_eigenvalue_min": 0.11290927231311798,
  "theory_high_variance_eigenvalue_abs_max": 7.057883067318471e-06,
  "theory_high_variance_eigenvalue_abs_min": 1.029333862589965e-08,
  "theory_high_variance_predicted_max": 0.004156754817813635,
  "theory_high_variance_predicted_min": 0.0019306050380691886
}
```