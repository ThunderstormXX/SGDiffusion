# EXP15: Gradient-noise Hessian alignment

Diagnostic for Var(G_i) versus Hessian eigenvalue using full MLP EXP2 artifacts.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp15_gradient_noise_hessian_alignment_full/config.yaml
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
  "gamma_ci95_high": 0.013583145819569578,
  "gamma_ci95_low": 0.009575078577706992,
  "gamma_cv": 1.477585720559552,
  "gamma_mean": 0.011579112198638285,
  "gamma_std": 0.01710913084146485,
  "grad_noise_lambda_log_correlation": 0.9673088895287669,
  "linear_fit_intercept": -3.8540439621296676e-05,
  "linear_fit_r2": 0.862582167038862,
  "linear_fit_slope": 0.011517673440482296,
  "n_directions": 280,
  "pass": true,
  "positive_lambda_max": 1.8141762018203735,
  "positive_lambda_min": 1.2636708568436461e-08,
  "source_result_dir": "src/scripts/exp6/results/exp2_eq32_full"
}
```