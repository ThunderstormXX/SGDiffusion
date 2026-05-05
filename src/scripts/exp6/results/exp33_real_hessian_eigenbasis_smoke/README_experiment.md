# EXP33: EXP33 real Hessian eigenbasis smoke

MLP386 covariance decoupling in the Hessian eigenbasis.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp33_real_hessian_eigenbasis_smoke/config.yaml
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
  "diagonal_mass_fraction_final": 0.7300609906427431,
  "dim": 386,
  "final_offdiag_over_diag": 0.369748572813901,
  "model": "MLP-386",
  "n_directions": 8,
  "n_runs": 16,
  "num_steps": 8,
  "pass": true
}
```