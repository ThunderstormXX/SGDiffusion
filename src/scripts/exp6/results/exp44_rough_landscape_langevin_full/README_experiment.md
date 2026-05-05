# EXP44: EXP44 rough fluctuating landscape full

Rough fluctuating 1D landscape comparison between exact discrete SGD and standard Langevin.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp44_rough_landscape_langevin_full/config.yaml
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
  "corrected_histogram_l1_distance": 0.08538461538461536,
  "corrected_mean_error_l1": 0.0008922311374029143,
  "corrected_variance_error_l1": 0.0007293338233004204,
  "eta": 0.1,
  "final_corrected_mean_error": 0.002114339204981497,
  "final_corrected_variance_error": 0.0010589606126330166,
  "final_mean_drift_corrected_langevin": -0.4408750153548109,
  "final_mean_error": 0.0016455702734289868,
  "final_mean_exact_sgd": -0.4387606761498294,
  "final_mean_standard_langevin": -0.4404062464232584,
  "final_variance_drift_corrected_langevin": 0.014571932525599263,
  "final_variance_error": 0.0010364601294738165,
  "final_variance_exact_sgd": 0.01563089313823228,
  "final_variance_standard_langevin": 0.014594433008758463,
  "histogram_l1_distance": 0.060615384615384654,
  "langevin_substeps": 100,
  "mean_error_l1": 0.0017977572860221125,
  "mean_improvement_standard_over_corrected": 2.0149008599441873,
  "n_modes": 20,
  "n_samples": 10000,
  "num_steps": 50,
  "pass": true,
  "variance_error_l1": 0.0010684332647961496,
  "variance_improvement_standard_over_corrected": 1.4649440772693336
}
```