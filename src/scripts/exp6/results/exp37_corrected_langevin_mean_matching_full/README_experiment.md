# EXP37: EXP37 corrected Langevin one-step mean matching full

Validates that the logarithmically matched corrected generator improves finite-eta mean-map matching against discrete SGD.


## Theory equation tested

`M1=L1`, `M2=L2-1/2 L1 L1`, giving corrected drift `b=-g-eta/2 H g`.

## Expected result

The corrected generator flow over one macro-step `eta` has smaller mean-map error than the standard generator.

## Interpretation

This is a generator/backward-error correction, not a conditional-covariance correction.

## Limitation

Synthetic nonlinear loss; isolates the `H g` term.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp37_corrected_langevin_mean_matching_full/config.yaml
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
  "corrected_error_slope": 3.0874722836227773,
  "dim": 6,
  "final_improvement_standard_over_corrected": 3.8090714250569473,
  "n_points": 2048,
  "pass": true,
  "standard_error_slope": 1.952949068002339,
  "substeps": 160
}
```