# EXP17: Covariance FP toy smoke

Smoke test comparing empirical SGD covariance evolution with standard FP and discrete/raw-moment FP recursions.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp17_covariance_fp_toy_smoke/config.yaml
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
  "dim": 6,
  "discrete_better_fraction": 1.0,
  "eig_max": 3.0,
  "eig_min": 0.3,
  "eta": 0.15,
  "final_discrete_covariance_trace": 0.13182926823208474,
  "final_discrete_relative_frobenius_error": 0.013556739310341836,
  "final_empirical_covariance_trace": 0.13147509275581165,
  "final_error_improvement_standard_over_discrete": 4.0719881609412605,
  "final_standard_covariance_trace": 0.12212098427805262,
  "final_standard_relative_frobenius_error": 0.05520288197267894,
  "initial_scale": 1.0,
  "mean_error_improvement_standard_over_discrete": 4.39119447898976,
  "n_runs": 10000,
  "num_steps": 100,
  "pass": true,
  "sigma_g": 0.5
}
```