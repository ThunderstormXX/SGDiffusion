# EXP1A: Toy falsification test

Full toy ensemble comparing discrete SGD, standard Langevin, and modified Langevin.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp1a_toy_full/config.yaml
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
  "modified_langevin_late_variance_slope": -3.4348383772790827e-28,
  "pass": false,
  "sgd_late_variance_slope": -1.437092683266531e-31,
  "standard_langevin_late_variance_slope": -1.0138760887267962e-05,
  "wasserstein_modified_to_sgd": 1.3076627828388103e-21,
  "wasserstein_standard_to_sgd": 1.9706587424580482
}
```