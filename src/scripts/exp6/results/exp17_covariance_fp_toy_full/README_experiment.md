# EXP17: Covariance FP toy

Ensemble covariance evolution test showing that the discrete/raw-moment FP recursion matches exact SGD better than the standard FP truncation.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp17_covariance_fp_toy_full/config.yaml
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
  "dim": 10,
  "discrete_better_fraction": 1.0,
  "eig_max": 3.0,
  "eig_min": 0.3,
  "eta": 0.15,
  "final_discrete_covariance_trace": 0.19919716754254102,
  "final_discrete_relative_frobenius_error": 0.009775679496240778,
  "final_empirical_covariance_trace": 0.199067376096016,
  "final_error_improvement_standard_over_discrete": 6.455754713161803,
  "final_standard_covariance_trace": 0.18306051587301592,
  "final_standard_relative_frobenius_error": 0.0631093889822156,
  "initial_scale": 1.0,
  "mean_error_improvement_standard_over_discrete": 5.78790472781566,
  "n_runs": 50000,
  "num_steps": 1000,
  "pass": true,
  "sigma_g": 0.5
}
```