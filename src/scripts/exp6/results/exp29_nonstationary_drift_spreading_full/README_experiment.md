# EXP29: EXP29 non-stationary drift spreading full

Full test for eta^2 Gamma mu(t)^2 drift-induced variance source.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp29_nonstationary_drift_spreading_full/config.yaml
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
  "drift_over_diffusion_prediction_final": 1078.2006666666193,
  "eta": 0.02,
  "final_drift_induced_prediction_flat": 7.770244799999792,
  "final_flat_variance": 8.14442429694302,
  "final_pure_diffusion_prediction_flat": 0.007200000000000123,
  "flat_variance_power_law_tail": 3.0410521338091914,
  "n_runs": 50000,
  "num_steps": 900,
  "pass": true
}
```