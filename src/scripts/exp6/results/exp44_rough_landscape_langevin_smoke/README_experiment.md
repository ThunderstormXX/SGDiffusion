# EXP44: EXP44 rough fluctuating landscape smoke

Smoke reproduction of the rough fluctuating 1D landscape comparison between exact SGD and standard Langevin.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp44_rough_landscape_langevin_smoke/config.yaml
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
  "corrected_histogram_l1_distance": 0.10820512820512818,
  "corrected_mean_error_l1": 0.00176471883784818,
  "corrected_variance_error_l1": 0.00046531730246941223,
  "eta": 0.1,
  "final_corrected_mean_error": 0.0049156281657471435,
  "final_corrected_variance_error": 0.0010765039934949733,
  "final_mean_drift_corrected_langevin": -0.36111849369006904,
  "final_mean_error": 0.004234035634521849,
  "final_mean_exact_sgd": -0.3660341218558162,
  "final_mean_standard_langevin": -0.36180008622129434,
  "final_variance_drift_corrected_langevin": 0.039888012824273594,
  "final_variance_error": 0.001922920377030568,
  "final_variance_exact_sgd": 0.03881150883077862,
  "final_variance_standard_langevin": 0.04073442920780919,
  "histogram_l1_distance": 0.11435897435897432,
  "langevin_substeps": 50,
  "mean_error_l1": 0.0015048348692162977,
  "mean_improvement_standard_over_corrected": 0.8527334989245238,
  "n_modes": 20,
  "n_samples": 3000,
  "num_steps": 30,
  "pass": true,
  "variance_error_l1": 0.0006228809849009551,
  "variance_improvement_standard_over_corrected": 1.3386155674748423
}
```