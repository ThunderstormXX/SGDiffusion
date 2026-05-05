# EXP16: Langevin trajectory evidence toy

Trajectory log-evidence comparison of standard vs modified Gaussian Langevin surrogates on stochastic quadratic SGD.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp16_langevin_evidence_toy_full/config.yaml
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
  "eig_max": 3.0,
  "eig_min": 0.3,
  "eta": 0.03,
  "final_log_evidence_modified_langevin_mean": 27734.39367800492,
  "final_log_evidence_standard_langevin_mean": 27783.119742089715,
  "final_log_ratio_modified_minus_standard_ci95_high": -42.98288045212696,
  "final_log_ratio_modified_minus_standard_ci95_low": -54.46924771746302,
  "final_log_ratio_modified_minus_standard_mean": -48.72606408479499,
  "final_log_ratio_modified_minus_standard_std": 8.287845086243175,
  "final_log_ratio_positive_fraction": 0.0,
  "max_formula_error": 2.325917236589703e-14,
  "mean_formula_error": 5.2023542808712904e-15,
  "n_runs": 8,
  "num_steps": 1000,
  "pass": true,
  "ridge": 1e-08,
  "sigma_g": 0.5,
  "sigma_h": 0.2
}
```