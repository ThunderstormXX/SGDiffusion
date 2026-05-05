# EXP38: EXP38 corrected generator distribution matching smoke

Smoke test for corrected generator moment matching over short nonlinear stochastic trajectories.


## Theory equation tested

Corrected generator with drift `-g-eta/2 H g` and centered diffusion covariance `D`.

## Expected result

Corrected generator reduces ensemble moment error, especially mean drift error.

## Interpretation

Tests distribution evolution rather than one-step likelihood.

## Limitation

Gaussian surrogate still does not model higher cumulants.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp38_corrected_generator_distribution_smoke/config.yaml
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
  "dim": 4,
  "eta": 0.1,
  "final_covariance_improvement_standard_over_corrected": 1.8672820577066842,
  "final_mean_improvement_standard_over_corrected": 1.0448716086621934,
  "n_runs": 4000,
  "num_steps": 12,
  "pass": true
}
```