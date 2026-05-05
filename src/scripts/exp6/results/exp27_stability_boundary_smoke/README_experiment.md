# EXP27: EXP27 stability boundary smoke

Smoke phase diagram for discrete SGD vs standard Langevin stability.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp27_stability_boundary_smoke/config.yaml
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
  "d": 0.2,
  "empirical_unstable_fraction_in_mismatch_region": 0.6666666666666666,
  "lambda": 1.0,
  "mismatch_grid_points": 3,
  "n_runs": 8000,
  "num_steps": 300,
  "pass": true
}
```