# EXP42: EXP42 Poisson jump surrogate full

Exploratory compound-Poisson surrogate matched to skewed SGD increment moments.


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
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp42_poisson_jump_surrogate_full/config.yaml
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
  "multi_step_wasserstein_gaussian": 0.012413391837518608,
  "multi_step_wasserstein_poisson": 0.005939282323739401,
  "n_samples": 200000,
  "one_step_wasserstein_gaussian": 0.013016740901394399,
  "one_step_wasserstein_poisson": 0.0004889720370456009,
  "pass": true,
  "poisson_lambda": 0.6978344398493487,
  "poisson_scale": 0.04025548121258792
}
```