# EXP37: EXP37 corrected Langevin one-step mean matching smoke

Smoke test for log-generator drift correction -eta/2 H g in finite-eta mean-map matching.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp37_corrected_langevin_mean_matching_smoke/config.yaml
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
  "corrected_error_slope": 3.111059301038597,
  "dim": 4,
  "final_improvement_standard_over_corrected": 4.2896359339572,
  "n_points": 256,
  "pass": true,
  "standard_error_slope": 1.947019697365687,
  "substeps": 96
}
```