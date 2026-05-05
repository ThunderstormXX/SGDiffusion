# EXP40: EXP40 semigroup log matching smoke

Smoke test for comparing exact log transition matrix against standard/corrected generators.


## Theory equation tested

`sum_k eta^k M_k = log(1 + sum_k eta^k L_k)` on a finite transition matrix.

## Expected result

Corrected generator is closer to `log(P_sgd)/eta` than standard generator.

## Interpretation

Direct finite-state semigroup-log check.

## Limitation

Sensitive to grid resolution and transition-matrix regularization.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp40_semigroup_log_matching_smoke/config.yaml
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
  "corrected_better_fraction": 1.0,
  "final_improvement_standard_over_corrected": 2.719933854607187,
  "n_grid": 61,
  "pass": true,
  "sigma": 0.35
}
```