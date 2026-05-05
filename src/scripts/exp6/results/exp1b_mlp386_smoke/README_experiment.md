# EXP1B: MLP-386 real-model falsification smoke

Small MNIST/MLP-386 ensemble from an early reference point, comparing SGD and Langevin-like baselines.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp1b_mlp386_smoke/config.yaml
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
  "high_lr": 0.09,
  "high_lr_modified_langevin_closer": false,
  "high_lr_modified_langevin_error_ratio": 1.0302922114636257,
  "lr_0.01_langevin_coefficient_update_every": 8,
  "lr_0.01_langevin_discretization": "Euler-Maruyama with M*dt=eta and noise scale sqrt(eta*dt)",
  "lr_0.01_langevin_drift_mode": "reference",
  "lr_0.01_langevin_dt": 0.00125,
  "lr_0.01_langevin_noise_mode": "reference",
  "lr_0.01_langevin_noise_scale": 0.0035355339059327377,
  "lr_0.01_langevin_substeps": 8,
  "lr_0.01_langevin_time_per_sgd_step": 0.01,
  "lr_0.01_modified_closer_ratio": 1.0982119235300782,
  "lr_0.01_modified_langevin_final_loss_ci95_high": 2.290415123172661,
  "lr_0.01_modified_langevin_final_loss_ci95_low": 2.2900606840172806,
  "lr_0.01_modified_langevin_final_loss_mean": 2.2902379035949707,
  "lr_0.01_modified_langevin_final_loss_std": 0.0002021811340097657,
  "lr_0.01_modified_langevin_final_mean_path_error_to_sgd": 0.001106758019886911,
  "lr_0.01_modified_langevin_final_variance": 7.596679552079877e-07,
  "lr_0.01_pass": false,
  "lr_0.01_reference_full_gradient_norm": 0.11714503914117813,
  "lr_0.01_relative_error_modified": 0.0004478524128595988,
  "lr_0.01_relative_error_standard": 0.0004078014482123157,
  "lr_0.01_relative_improvement": -0.09821192353007831,
  "lr_0.01_sgd_replacement_final_loss_ci95_high": 2.290714494445199,
  "lr_0.01_sgd_replacement_final_loss_ci95_low": 2.2899999220587075,
  "lr_0.01_sgd_replacement_final_loss_mean": 2.290357208251953,
  "lr_0.01_sgd_replacement_final_loss_std": 0.0004076103140407404,
  "lr_0.01_sgd_replacement_final_mean_path_error_to_sgd": 0.0,
  "lr_0.01_sgd_replacement_final_variance": 1.0062025239676586e-06,
  "lr_0.01_standard_langevin_final_loss_ci95_high": 2.290377216365238,
  "lr_0.01_standard_langevin_final_loss_ci95_low": 2.2900898170209922,
  "lr_0.01_standard_langevin_final_loss_mean": 2.290233516693115,
  "lr_0.01_standard_langevin_final_loss_std": 0.00016393991592394893,
  "lr_0.01_standard_langevin_final_mean_path_error_to_sgd": 0.0009584423387423158,
  "lr_0.01_standard_langevin_final_variance": 5.691751994163496e-07,
  "lr_0.01_wasserstein_modified_to_sgd": 0.0004478524128595988,
  "lr_0.01_wasserstein_standard_to_sgd": 0.0004078014482123157,
  "lr_0.03_langevin_coefficient_update_every": 8,
  "lr_0.03_langevin_discretization": "Euler-Maruyama with M*dt=eta and noise scale sqrt(eta*dt)",
  "lr_0.03_langevin_drift_mode": "reference",
  "lr_0.03_langevin_dt": 0.00375,
  "lr_0.03_langevin_noise_mode": "reference",
  "lr_0.03_langevin_noise_scale": 0.010606601717798213,
  "lr_0.03_langevin_substeps": 8,
  "lr_0.03_langevin_time_per_sgd_step": 0.03,
  "lr_0.03_modified_closer_ratio": 1.030978737818809,
  "lr_0.03_modified_langevin_final_loss_ci95_high": 2.2926437762045886,
  "lr_0.03_modified_langevin_final_loss_ci95_low": 2.292014274332521,
  "lr_0.03_modified_langevin_final_loss_mean": 2.292329025268555,
  "lr_0.03_modified_langevin_final_loss_std": 0.0003590839229353745,
  "lr_0.03_modified_langevin_final_mean_path_error_to_sgd": 0.006188482046127319,
  "lr_0.03_modified_langevin_final_variance": 5.233825959294336e-06,
  "lr_0.03_pass": false,
  "lr_0.03_reference_full_gradient_norm": 0.1024748682975769,
  "lr_0.03_relative_error_modified": 0.002545595254438619,
  "lr_0.03_relative_error_standard": 0.002469105483032763,
  "lr_0.03_relative_improvement": -0.030978737818809013,
  "lr_0.03_sgd_replacement_final_loss_ci95_high": 2.29423326321927,
  "lr_0.03_sgd_replacement_final_loss_ci95_low": 2.29212919405612,
  "lr_0.03_sgd_replacement_final_loss_mean": 2.293181228637695,
  "lr_0.03_sgd_replacement_final_loss_std": 0.0012002147138172286,
  "lr_0.03_sgd_replacement_final_mean_path_error_to_sgd": 0.0,
  "lr_0.03_sgd_replacement_final_variance": 8.269208592537325e-06,
  "lr_0.03_standard_langevin_final_loss_ci95_high": 2.2926590503310247,
  "lr_0.03_standard_langevin_final_loss_ci95_low": 2.292095702781768,
  "lr_0.03_standard_langevin_final_loss_mean": 2.2923773765563964,
  "lr_0.03_standard_langevin_final_loss_std": 0.0003213478099731606,
  "lr_0.03_standard_langevin_final_mean_path_error_to_sgd": 0.0056882197968661785,
  "lr_0.03_standard_langevin_final_variance": 3.888923401973443e-06,
  "lr_0.03_wasserstein_modified_to_sgd": 0.002545595254438619,
  "lr_0.03_wasserstein_standard_to_sgd": 0.002469105483032763,
  "lr_0.09_langevin_coefficient_update_every": 8,
  "lr_0.09_langevin_discretization": "Euler-Maruyama with M*dt=eta and noise scale sqrt(eta*dt)",
  "lr_0.09_langevin_drift_mode": "reference",
  "lr_0.09_langevin_dt": 0.01125,
  "lr_0.09_langevin_noise_mode": "reference",
  "lr_0.09_langevin_noise_scale": 0.03181980515339464,
  "lr_0.09_langevin_substeps": 8,
  "lr_0.09_langevin_time_per_sgd_step": 0.09,
  "lr_0.09_modified_closer_ratio": 1.0302922114636257,
  "lr_0.09_modified_langevin_final_loss_ci95_high": 2.281196550696116,
  "lr_0.09_modified_langevin_final_loss_ci95_low": 2.2783291298458765,
  "lr_0.09_modified_langevin_final_loss_mean": 2.279762840270996,
  "lr_0.09_modified_langevin_final_loss_std": 0.0016356499850090321,
  "lr_0.09_modified_langevin_final_mean_path_error_to_sgd": 0.028460372239351273,
  "lr_0.09_modified_langevin_final_variance": 4.7102803364396095e-05,
  "lr_0.09_pass": false,
  "lr_0.09_reference_full_gradient_norm": 0.1024748682975769,
  "lr_0.09_relative_error_modified": 0.010827745714535316,
  "lr_0.09_relative_error_standard": 0.010509392960617939,
  "lr_0.09_relative_improvement": -0.030292211463625644,
  "lr_0.09_sgd_replacement_final_loss_ci95_high": 2.276712043475425,
  "lr_0.09_sgd_replacement_final_loss_ci95_low": 2.2724375651366353,
  "lr_0.09_sgd_replacement_final_loss_mean": 2.27457480430603,
  "lr_0.09_sgd_replacement_final_loss_std": 0.0024382714627252514,
  "lr_0.09_sgd_replacement_final_mean_path_error_to_sgd": 0.0,
  "lr_0.09_sgd_replacement_final_variance": 5.0378574087517336e-05,
  "lr_0.09_standard_langevin_final_loss_ci95_high": 2.280629380706799,
  "lr_0.09_standard_langevin_final_loss_ci95_low": 2.278613153930652,
  "lr_0.09_standard_langevin_final_loss_mean": 2.2796212673187255,
  "lr_0.09_standard_langevin_final_loss_std": 0.0011501071758982183,
  "lr_0.09_standard_langevin_final_mean_path_error_to_sgd": 0.02717990055680275,
  "lr_0.09_standard_langevin_final_variance": 3.499818194541149e-05,
  "lr_0.09_wasserstein_modified_to_sgd": 0.010827745714535316,
  "lr_0.09_wasserstein_standard_to_sgd": 0.010509392960617939,
  "lr_grid": [
    0.01,
    0.03,
    0.09
  ],
  "modified_better_at_high_lr": false,
  "modified_langevin_closer_fraction": 0.0,
  "n_learning_rates": 3,
  "pass": false,
  "relative_error_modified": 0.010827745714535316,
  "relative_error_standard": 0.010509392960617939,
  "relative_improvement": -0.030292211463625644
}
```