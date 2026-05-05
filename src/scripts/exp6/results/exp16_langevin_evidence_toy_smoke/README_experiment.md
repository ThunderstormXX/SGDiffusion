# EXP16: Langevin trajectory evidence toy smoke

Smoke test for trajectory log-evidence of standard vs modified Gaussian Langevin surrogates on stochastic quadratic SGD.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp16_langevin_evidence_toy_smoke/config.yaml
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
  "eig_max": 3.0,
  "eig_min": 0.3,
  "eta": 0.03,
  "final_log_evidence_modified_langevin_mean": 1632.7693683652478,
  "final_log_evidence_standard_langevin_mean": 1658.0440991584405,
  "final_log_ratio_modified_minus_standard_ci95_high": 1.7391904008993322,
  "final_log_ratio_modified_minus_standard_ci95_low": -52.28865198728338,
  "final_log_ratio_modified_minus_standard_mean": -25.274730793192024,
  "final_log_ratio_modified_minus_standard_std": 19.491558023245915,
  "final_log_ratio_positive_fraction": 0.0,
  "max_formula_error": 1.226796442210798e-14,
  "mean_formula_error": 2.799288578714254e-15,
  "n_runs": 2,
  "num_steps": 100,
  "pass": true,
  "ridge": 1e-08,
  "sigma_g": 0.5,
  "sigma_h": 0.2
}
```