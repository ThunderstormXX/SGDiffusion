# EXP33: EXP33 real Hessian eigenbasis full

MLP386 covariance decoupling in the Hessian eigenbasis.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp33_real_hessian_eigenbasis_full/config.yaml
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
  "diagonal_mass_fraction_final": 0.8538728169342625,
  "dim": 386,
  "final_offdiag_over_diag": 0.17113460010401918,
  "model": "MLP-386",
  "n_directions": 12,
  "n_runs": 32,
  "num_steps": 20,
  "pass": true
}
```