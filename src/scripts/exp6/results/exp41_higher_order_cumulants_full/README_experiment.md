# EXP41: EXP41 higher-order cumulant diagnostic full

Tests Pawula-style limitation: Gaussian corrected Langevin matches second-order structure but misses skew/kurtosis.


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
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp41_higher_order_cumulants_full/config.yaml
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
  "empirical_kurtosis": 9.219502937023723,
  "empirical_skewness": -2.027602887041882,
  "eta": 0.05,
  "gaussian_kurtosis": 3.0010796742390347,
  "gaussian_skewness": 0.0009684323159862717,
  "histogram_kl_empirical_to_gaussian": 0.48855976567268333,
  "n_samples": 200000,
  "pass": true
}
```