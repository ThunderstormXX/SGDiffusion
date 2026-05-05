# EXP18: Learning-Rate Scaling

Tests whether the update moment scales as eta^2 and whether the raw second
moment is described by `Sigma + g g^T`, while centered covariance is described
by `Sigma` only.

Run:

```bash
python -m src.scripts.exp6.src.run_experiment src/scripts/exp6/configs/exp18_learning_rate_scaling/smoke.yaml --make-figure
python -m src.scripts.exp6.src.run_experiment src/scripts/exp6/configs/exp18_learning_rate_scaling/full.yaml --make-figure
python -m src.scripts.exp6.src.run_experiment src/scripts/exp6/configs/exp18_learning_rate_scaling/mlp386_smoke.yaml --make-figure
python -m src.scripts.exp6.src.run_experiment src/scripts/exp6/configs/exp18_learning_rate_scaling/mlp386_full.yaml --make-figure
```

