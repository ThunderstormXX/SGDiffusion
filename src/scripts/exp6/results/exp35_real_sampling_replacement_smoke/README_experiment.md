# EXP35: EXP35 real sampling replacement smoke

MLP386 with-replacement vs without-replacement ensemble covariance.

## Reproduce

```bash
bash /Users/igoreshka/Desktop/SGDiffusion/src/scripts/exp6/scripts/run_one.sh src/scripts/exp6/results/exp35_real_sampling_replacement_smoke/config.yaml
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
  "final_trace_with_replacement": 0.00017061380527766333,
  "final_trace_without_replacement": 2.4509021930715787e-08,
  "model": "MLP-386",
  "n_runs": 4,
  "num_steps": 16,
  "pass": true,
  "without_over_with_trace_final": 0.00014365204439833508
}
```