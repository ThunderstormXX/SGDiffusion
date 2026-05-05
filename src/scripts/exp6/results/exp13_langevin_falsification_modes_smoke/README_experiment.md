# EXP13: Langevin falsification in Hessian modes smoke

Smoke test for SGD vs standard/modified Langevin variance dynamics in flat and sharp Hessian eigenspaces.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp13_langevin_falsification_modes_smoke/config.yaml
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
  "diffusive_eigenvalue_abs_max": 4.137706127949059e-06,
  "diffusive_predicted_variance_max": 0.05376385897397995,
  "diffusive_predicted_variance_min": 0.042810920625925064,
  "flat_eigenvalue_abs_max": 7.03435660415197e-12,
  "gd_diffusive_final_variance": 0.0,
  "gd_diffusive_late_mean_variance": 0.0,
  "gd_diffusive_late_slope": -0.0,
  "gd_flat_final_variance": 0.0,
  "gd_flat_late_mean_variance": 0.0,
  "gd_flat_late_slope": -0.0,
  "gd_sharp_final_variance": 0.0,
  "gd_sharp_late_mean_variance": 0.0,
  "gd_sharp_late_slope": -0.0,
  "modified_langevin_diffusive_final_variance": 7.724234762918059e-08,
  "modified_langevin_diffusive_late_mean_variance": 5.8874874753200857e-08,
  "modified_langevin_diffusive_late_slope": 3.541097637338455e-09,
  "modified_langevin_flat_final_variance": 3.954360838775983e-09,
  "modified_langevin_flat_late_mean_variance": 5.488590471003363e-09,
  "modified_langevin_flat_late_slope": -3.9341319091335e-10,
  "modified_langevin_sharp_final_variance": 6.889646465424448e-05,
  "modified_langevin_sharp_late_mean_variance": 4.76472923764959e-05,
  "modified_langevin_sharp_late_slope": 4.746951162815098e-06,
  "pass": false,
  "reference_full_grad_norm": 0.11288345605134964,
  "sgd_diffusive_final_variance": 8.125720540874681e-09,
  "sgd_diffusive_late_mean_variance": 7.874337626390115e-09,
  "sgd_diffusive_late_slope": 3.056699338088718e-11,
  "sgd_flat_final_variance": 5.455045705039384e-16,
  "sgd_flat_late_mean_variance": 7.179706958998402e-16,
  "sgd_flat_late_slope": -2.0292004357901728e-17,
  "sgd_sharp_final_variance": 9.78919051703997e-05,
  "sgd_sharp_late_mean_variance": 0.00010455735173309222,
  "sgd_sharp_late_slope": -6.778918759664486e-08,
  "sgd_vs_standard_diffusive_slope_ratio": 0.003986096244730547,
  "sgd_vs_standard_flat_slope_ratio": -2.163634740601613e-06,
  "sgd_vs_standard_sharp_variance_ratio": 4.819009392729343,
  "sharp_eigenvalue_max": 0.19789715111255646,
  "sharp_eigenvalue_min": 0.15169718861579895,
  "standard_langevin_diffusive_final_variance": 1.0067153510817661e-07,
  "standard_langevin_diffusive_late_mean_variance": 7.608851859686183e-08,
  "standard_langevin_diffusive_late_slope": 7.668403245730825e-09,
  "standard_langevin_flat_final_variance": 2.1496127100562035e-09,
  "standard_langevin_flat_late_mean_variance": 2.204112226067423e-09,
  "standard_langevin_flat_late_slope": 9.378664511672336e-12,
  "standard_langevin_sharp_final_variance": 2.0313698769314215e-05,
  "standard_langevin_sharp_late_mean_variance": 1.6703736037015915e-05,
  "standard_langevin_sharp_late_slope": 1.1649407269942473e-06
}
```