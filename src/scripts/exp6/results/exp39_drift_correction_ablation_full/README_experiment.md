# EXP39: EXP39 drift correction ablation full

Ablates standard drift, corrected Hg drift, wrong raw covariance, and full corrected generator.


## Theory equation tested

`g g^T` is not conditional covariance; finite-eta correction enters through `eta/2 H g` drift.

## Expected result

Corrected drift improves mean error; raw covariance worsens conditional likelihood.

## Interpretation

Separates the new log-generator correction from the earlier naive modified Langevin model.

## Limitation

Local synthetic diagnostic.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp39_drift_correction_ablation_full/config.yaml
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
  "corrected_drift_mean_error": 0.007721336948268362,
  "dim": 6,
  "eta": 0.15,
  "n_runs": 60000,
  "pass": true,
  "raw_covariance_nll_delta_vs_standard": 0.733879169008163,
  "standard_mean_error": 0.04091814381627217
}
```