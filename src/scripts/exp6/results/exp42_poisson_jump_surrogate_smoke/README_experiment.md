# EXP42: EXP42 Poisson jump surrogate smoke

Smoke exploratory test of a compound Poisson surrogate for skewed increments.


## Theory equation tested

Exploratory higher-order matching via a compound-Poisson jump surrogate.

## Expected result

Poisson surrogate improves non-Gaussian increment fit relative to Gaussian.

## Interpretation

Motivates jump-process extensions beyond pure diffusion.

## Limitation

One-dimensional heuristic prototype.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp42_poisson_jump_surrogate_smoke/config.yaml
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
  "eta": 0.05,
  "exploratory": true,
  "multi_step_wasserstein_gaussian": 0.012055179337741091,
  "multi_step_wasserstein_poisson": 0.018568030400308348,
  "n_samples": 30000,
  "one_step_wasserstein_gaussian": 0.012881722265524564,
  "one_step_wasserstein_poisson": 0.0037655042977954253,
  "pass": true,
  "poisson_lambda": 0.7483448536156397,
  "poisson_scale": 0.03852955703017792
}
```