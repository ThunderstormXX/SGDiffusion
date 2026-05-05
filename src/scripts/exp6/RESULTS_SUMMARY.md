# EXP6 Results Summary

This file is generated from saved `metrics.json` artifacts only.

| Experiment | Result dir | Primary metrics | Pass |
|---|---|---|---|
| `exp10_noise_alpha_sweep_full` | `src/scripts/exp6/results/exp10_noise_alpha_sweep_full` | alpha_0.01_diverged=0, alpha_0.01_final_test_accuracy=0.816, alpha_0.01_final_test_loss=0.723, alpha_0.01_final_train_accuracy=0.9922, alpha_0.01_final_train_loss=0.05815 | `True` |
<!-- config: src/scripts/exp6/results/exp10_noise_alpha_sweep_full/config.yaml -->
| `exp11_normalized_noise_alpha_sweep_full` | `src/scripts/exp6/results/exp11_normalized_noise_alpha_sweep_full` | alpha_0.01_diverged=0, alpha_0.01_final_test_accuracy=0.822, alpha_0.01_final_test_loss=0.7183, alpha_0.01_final_train_accuracy=0.9922, alpha_0.01_final_train_loss=0.0482 | `False` |
<!-- config: src/scripts/exp6/results/exp11_normalized_noise_alpha_sweep_full/config.yaml -->
| `exp1a_toy_full` | `src/scripts/exp6/results/exp1a_toy_full` | modified_langevin_late_variance_slope=-3.435e-28, sgd_late_variance_slope=-1.437e-31, standard_langevin_late_variance_slope=-1.014e-05, wasserstein_modified_to_sgd=1.308e-21, wasserstein_standard_to_sgd=1.971 | `False` |
<!-- config: src/scripts/exp6/results/exp1a_toy_full/config.yaml -->
| `exp1a_toy_smoke` | `src/scripts/exp6/results/exp1a_toy_smoke` | modified_langevin_final_mean_path_error_to_sgd=0.0818, modified_langevin_late_variance_slope=-0.1351, sgd_late_variance_slope=-0.01829, standard_langevin_final_mean_path_error_to_sgd=0.0313, standard_langevin_late_variance_slope=0.0007135 | `True` |
<!-- config: src/scripts/exp6/results/exp1a_toy_smoke/config.yaml -->
| `exp1b_mlp386_full` | `src/scripts/exp6/results/exp1b_mlp386_full` | modified_langevin_final_loss_ci95_high=2.26, modified_langevin_final_loss_ci95_low=2.255, modified_langevin_final_loss_mean=2.258, modified_langevin_final_loss_std=0.01513, modified_langevin_final_variance=0.0006055 | `True` |
<!-- config: src/scripts/exp6/results/exp1b_mlp386_full/config.yaml -->
| `exp1b_mlp386_smoke` | `src/scripts/exp6/results/exp1b_mlp386_smoke` | modified_langevin_final_loss_ci95_high=2.284, modified_langevin_final_loss_ci95_low=2.282, modified_langevin_final_loss_mean=2.283, modified_langevin_final_loss_std=0.001091, modified_langevin_final_mean_path_error_to_sgd=0.002235 | `False` |
<!-- config: src/scripts/exp6/results/exp1b_mlp386_smoke/config.yaml -->
| `exp2_eq32_full` | `src/scripts/exp6/results/exp2_eq32_full` | all_directions_log_correlation=0.02289, all_directions_relative_error=4.912, gamma_estimate=0.007826, heldout_log_correlation=0.9128, heldout_relative_error_after_train_scalar=0.2171 | `True` |
<!-- config: src/scripts/exp6/results/exp2_eq32_full/config.yaml -->
| `exp2_eq32_smoke` | `src/scripts/exp6/results/exp2_eq32_smoke` | all_directions_log_correlation=0.1614, all_directions_relative_error=16.62, gamma_estimate=0.01313, heldout_log_correlation=-0.03482, heldout_relative_error_after_train_scalar=2.633 | `False` |
<!-- config: src/scripts/exp6/results/exp2_eq32_smoke/config.yaml -->
| `exp3_sampling_full` | `src/scripts/exp6/results/exp3_sampling_full` | no_replacement_late_slope=-1.976e-07, reference_full_gradient_norm=0.2215, replacement_late_slope=9.383e-08, replacement_vs_no_replacement_wasserstein=0.002768, sgd_no_replacement_final_loss_ci95_high=0.8066 | `True` |
<!-- config: src/scripts/exp6/results/exp3_sampling_full/config.yaml -->
| `exp3_sampling_smoke` | `src/scripts/exp6/results/exp3_sampling_smoke` | no_replacement_late_slope=3.648e-09, reference_full_gradient_norm=0.106, replacement_late_slope=8.811e-08, replacement_vs_no_replacement_wasserstein=0.001213, sgd_no_replacement_final_loss_ci95_high=2.328 | `True` |
<!-- config: src/scripts/exp6/results/exp3_sampling_smoke/config.yaml -->
| `exp4_scaling_full` | `src/scripts/exp6/results/exp4_scaling_full` | final_loss_ci95_high=2.101, final_loss_ci95_low=2.099, final_loss_mean=2.1, final_loss_std=0.002817, n_parameters=8970 | `True` |
<!-- config: src/scripts/exp6/results/exp4_scaling_full/config.yaml -->
| `exp5_nonstationary_full` | `src/scripts/exp6/results/exp5_nonstationary_full` | fit_improvement=-5.354e-05, modified_langevin_final_loss_ci95_high=2.278, modified_langevin_final_loss_ci95_low=2.277, modified_langevin_final_loss_mean=2.278, modified_langevin_final_loss_std=0.0007668 | `False` |
<!-- config: src/scripts/exp6/results/exp5_nonstationary_full/config.yaml -->
| `exp5_nonstationary_smoke` | `src/scripts/exp6/results/exp5_nonstationary_smoke` | fit_improvement=-0.0005049, modified_langevin_final_loss_ci95_high=2.313, modified_langevin_final_loss_ci95_low=2.311, modified_langevin_final_loss_mean=2.312, modified_langevin_final_loss_std=0.001072 | `False` |
<!-- config: src/scripts/exp6/results/exp5_nonstationary_smoke/config.yaml -->
| `exp6_generalization_full` | `src/scripts/exp6/results/exp6_generalization_full` | final_test_loss_ci95_high=1.144, final_test_loss_ci95_low=1.143, final_test_loss_mean=1.144, final_test_loss_std=0.003998, final_train_loss_ci95_high=1.002 | `True` |
<!-- config: src/scripts/exp6/results/exp6_generalization_full/config.yaml -->
| `exp6_generalization_smoke` | `src/scripts/exp6/results/exp6_generalization_smoke` | final_test_loss_ci95_high=2.314, final_test_loss_ci95_low=2.311, final_test_loss_mean=2.313, final_test_loss_std=0.001586, final_train_loss_ci95_high=2.278 | `True` |
<!-- config: src/scripts/exp6/results/exp6_generalization_smoke/config.yaml -->
| `exp7_diffusion_control_full` | `src/scripts/exp6/results/exp7_diffusion_control_full` | baseline_sgd_flat_final_variance=8.83e-10, baseline_sgd_generalization_gap_ci95_high=0.1532, baseline_sgd_generalization_gap_ci95_low=0.1513, baseline_sgd_generalization_gap_mean=0.1522, baseline_sgd_generalization_gap_std=0.003385 | `True` |
<!-- config: src/scripts/exp6/results/exp7_diffusion_control_full/config.yaml -->
| `exp7_diffusion_control_smoke` | `src/scripts/exp6/results/exp7_diffusion_control_smoke` | baseline_sgd_flat_final_variance=1.632e-14, baseline_sgd_generalization_gap_ci95_high=-0.0006159, baseline_sgd_generalization_gap_ci95_low=-0.002739, baseline_sgd_generalization_gap_mean=-0.001677, baseline_sgd_generalization_gap_std=0.001532 | `True` |
<!-- config: src/scripts/exp6/results/exp7_diffusion_control_smoke/config.yaml -->
| `exp8_diffusion_amplification_full` | `src/scripts/exp6/results/exp8_diffusion_amplification_full` | amplification_factor=3, amplify_random_generalization_gap_ci95_high=0.06911, amplify_random_generalization_gap_ci95_low=0.06624, amplify_random_generalization_gap_mean=0.06768, amplify_random_generalization_gap_std=0.002531 | `False` |
<!-- config: src/scripts/exp6/results/exp8_diffusion_amplification_full/config.yaml -->
| `exp8_diffusion_amplification_smoke` | `src/scripts/exp6/results/exp8_diffusion_amplification_smoke` | amplification_factor=2, amplify_random_generalization_gap_ci95_high=0.01501, amplify_random_generalization_gap_ci95_low=0.01326, amplify_random_generalization_gap_mean=0.01414, amplify_random_generalization_gap_std=0.00109 | `True` |
<!-- config: src/scripts/exp6/results/exp8_diffusion_amplification_smoke/config.yaml -->
| `exp9_theory_denoised_sgd_full` | `src/scripts/exp6/results/exp9_theory_denoised_sgd_full` | baseline_sgd_generalization_gap_ci95_high=0.07833, baseline_sgd_generalization_gap_ci95_low=0.07474, baseline_sgd_generalization_gap_mean=0.07653, baseline_sgd_generalization_gap_std=0.003173, baseline_sgd_sharp_final_variance=0.0004632 | `False` |
<!-- config: src/scripts/exp6/results/exp9_theory_denoised_sgd_full/config.yaml -->
| `exp9_theory_denoised_sgd_smoke` | `src/scripts/exp6/results/exp9_theory_denoised_sgd_smoke` | baseline_sgd_generalization_gap_ci95_high=0.0246, baseline_sgd_generalization_gap_ci95_low=0.02351, baseline_sgd_generalization_gap_mean=0.02405, baseline_sgd_generalization_gap_std=0.000679, baseline_sgd_sharp_final_variance=0.0001841 | `False` |
<!-- config: src/scripts/exp6/results/exp9_theory_denoised_sgd_smoke/config.yaml -->

## Caveats

- Smoke runs are deliberately small and validate the pipeline, not the final paper-scale claims.
- EXP4 is implemented as a lightweight HVP diagnostic, not the requested 1M+ large-model experiment.
- Metrics and figures are regenerated from raw saved artifacts; no manual figure editing is used.