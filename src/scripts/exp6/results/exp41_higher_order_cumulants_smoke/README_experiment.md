# EXP41: EXP41 higher-order cumulant diagnostic smoke

Smoke diagnostic showing Gaussian corrected Langevin misses non-Gaussian increment cumulants.


## Theory equation tested

Second-order Gaussian corrected Langevin does not match higher cumulants.

## Expected result

Skewed SGD increments have nonzero skew/kurtosis missed by Gaussian surrogate.

## Interpretation

A limitation/Pawula diagnostic, not a failure of the second-order result.

## Limitation

Synthetic non-Gaussian minibatch noise.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp41_higher_order_cumulants_smoke/config.yaml
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
  "empirical_kurtosis": 9.29467908053652,
  "empirical_skewness": -2.0154932715217004,
  "eta": 0.05,
  "gaussian_kurtosis": 3.0597169835492073,
  "gaussian_skewness": 0.013173156685950854,
  "histogram_kl_empirical_to_gaussian": 0.5141971045945979,
  "n_samples": 30000,
  "pass": true
}
```