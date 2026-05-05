# EXP35: EXP35 real sampling replacement full

MLP386 with-replacement vs without-replacement ensemble covariance.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp35_real_sampling_replacement_full/config.yaml
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
  "final_trace_with_replacement": 0.0006460768894018354,
  "final_trace_without_replacement": 2.9293820768496493e-07,
  "model": "MLP-386",
  "n_runs": 8,
  "num_steps": 48,
  "pass": true,
  "without_over_with_trace_final": 0.00045341075108905256
}
```