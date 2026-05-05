# EXP4: Scaling experiment lite

Lightweight HVP scaling diagnostic: wider MLP, top Hessian-vector direction, random flat proxy directions, and SGD trajectory variance.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp4_scaling_full/config.yaml
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
  "final_loss_ci95_high": 2.1006810446922914,
  "final_loss_ci95_low": 2.0993008831794127,
  "final_loss_mean": 2.099990963935852,
  "final_loss_std": 0.0028166561487317885,
  "n_parameters": 8970,
  "pass": true,
  "random_flat_proxy_final_variance": 4.712688337349391e-07,
  "random_flat_proxy_late_slope": 1.3688168950771854e-09,
  "sharp_final_variance": 0.00016550849250052124,
  "sharp_late_slope": -5.793765559003623e-07,
  "status": "exp4_lite_hvp_power_iteration",
  "top_hvp_eigenvalue_estimate": 0.1678711473941803
}
```