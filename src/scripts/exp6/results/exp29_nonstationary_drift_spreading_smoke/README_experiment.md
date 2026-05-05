# EXP29: EXP29 non-stationary drift spreading smoke

Smoke test for eta^2 Gamma mu(t)^2 drift-induced variance source.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp29_nonstationary_drift_spreading_smoke/config.yaml
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
  "drift_over_diffusion_prediction_final": 82.8340000000001,
  "eta": 0.02,
  "final_drift_induced_prediction_flat": 0.16766799999999993,
  "final_flat_variance": 0.17330565244648688,
  "final_pure_diffusion_prediction_flat": 0.0019999999999999966,
  "flat_variance_power_law_tail": 3.0043141672668603,
  "n_runs": 10000,
  "num_steps": 250,
  "pass": true
}
```