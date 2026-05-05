# EXP5: Learning-rate scaling of flat/sharp variance

Real-model learning-rate scaling test for non-stationary diffusion in flat directions and bounded variance in sharp directions.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp5_lr_scaling_full/config.yaml
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
  "flat_slope": 6.26043869522697e-12,
  "flat_slope_lr_correlation": 0.936324441122511,
  "flat_slope_vs_lr_fit_slope": 7.398610066991139e-11,
  "lr_0.0125_flat_late_plateau_mean": 2.554869510396074e-12,
  "lr_0.0125_flat_late_slope": 1.0267307837263931e-14,
  "lr_0.0125_sharp_late_plateau_mean": 6.481334276031703e-05,
  "lr_0.0125_sharp_late_slope": 5.289148248266413e-08,
  "lr_0.025_flat_late_plateau_mean": 1.1583488508659645e-11,
  "lr_0.025_flat_late_slope": 5.188002805134319e-14,
  "lr_0.025_sharp_late_plateau_mean": 0.00015922037709970027,
  "lr_0.025_sharp_late_slope": 4.6053173718973286e-08,
  "lr_0.05_flat_late_plateau_mean": 6.505702226933252e-11,
  "lr_0.05_flat_late_slope": 4.0366247670842204e-13,
  "lr_0.05_sharp_late_plateau_mean": 0.0003125373332295567,
  "lr_0.05_sharp_late_slope": -6.536502041854088e-08,
  "lr_0.1_flat_late_plateau_mean": 6.59616972153998e-10,
  "lr_0.1_flat_late_slope": 6.26043869522697e-12,
  "lr_0.1_sharp_late_plateau_mean": 0.0004068756534252316,
  "lr_0.1_sharp_late_slope": -1.4257173461373921e-06,
  "lr_grid": [
    0.0125,
    0.025,
    0.05,
    0.1
  ],
  "n_runs": 24,
  "pass": true,
  "primary_lr": 0.1,
  "sharp_slope": -1.4257173461373921e-06,
  "slope_ratio": 4.391079839344025e-06,
  "steps": 400
}
```