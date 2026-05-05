# EXP43: EXP43 MLP mean trajectory and covariance full

Compares projected ensemble mean trajectories and second moments for exact SGD, standard Langevin, and drift-corrected Langevin on MLP/MNIST.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp43_mlp_langevin_mean_covariance_full/config.yaml
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
  "batch_size": 64,
  "cov_error_corrected": 8.058247970000319e-05,
  "cov_error_standard": 5.5165854500748674e-05,
  "cov_improvement": 0.6845886935488099,
  "dim": 386,
  "eta": 0.05,
  "final_mean_error_corrected": 0.009663975355892003,
  "final_mean_error_standard": 0.014990830092483677,
  "full_gradient_norm": 0.08353198992150364,
  "hessian_gradient_norm": 0.009293009013768101,
  "local_coefficients": "frozen_at_w0",
  "mean_error_corrected": 0.0036318546386334936,
  "mean_error_standard": 0.006781716519385785,
  "mean_improvement": 1.8672874313982581,
  "model": "MLP-386",
  "n_noise_batches": 48,
  "n_runs": 32,
  "num_steps": 40,
  "pass": true,
  "selected_eigenvalue_0": 0.16065747562932742,
  "selected_eigenvalue_1": 0.19652660239778239
}
```