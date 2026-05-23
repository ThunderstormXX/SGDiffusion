# exp8: GD/SGD MLP training trajectories

Default run:

```bash
src/scripts/exp8/sh_scripts/run_training_trajectories.sh
```

Pretrain a start point:

```bash
src/scripts/exp8/sh_scripts/pretrain_point.sh
```

Pretrain override example:

```bash
PRETRAIN_SGD_STEPS=2000 PRETRAIN_GD_EPOCHS=100 \
  src/scripts/exp8/sh_scripts/pretrain_point.sh
```

Run trajectories from the pretrained point:

```bash
src/scripts/exp8/sh_scripts/run_from_pretrained_point.sh
```

Main defaults:

- MNIST: 6400 train, 100 val, 100 test
- `FlexibleMLP`: `hidden_dim=48`, `num_hidden_layers=1`, `input_downsample=14` (9946 params)
- GD: full-batch, epochs equal to `SGD_ITERATIONS`
- SGD: `RUNS=10`, `SGD_ITERATIONS=1000`, replacement dataloader
- pretrain: `PRETRAIN_SGD_STEPS=10000`, then `PRETRAIN_GD_EPOCHS=500`
- learning rate: single `LR=0.1` for pretrain-SGD, pretrain-GD, GD, and SGD
- device: `mps`

Override example:

```bash
LR=0.1 RUNS=5 SGD_ITERATIONS=300 HIDDEN_DIM=64 DEVICE=mps \
  src/scripts/exp8/sh_scripts/run_training_trajectories.sh
```

Figures are written to `src/scripts/exp8/figures/training_trajectories/`.
The default pretrained checkpoint is written to
`src/scripts/exp8/results/pretrained_points/pretrained_point.pt`.
The pretrain loss figure is written to
`src/scripts/exp8/figures/pretrain/pretrain_losses.png`.
Pretrain metrics are written to `pretrain_metrics.jsonl` and
`pretrain_metrics.csv` in the selected logs directory.
The tracked oscillation plot is written to `oscillation_fraction.png`.

You can also pass a start point directly:

```bash
CHECKPOINT_IN=src/scripts/exp8/results/pretrained_points/pretrained_point.pt \
  src/scripts/exp8/sh_scripts/run_training_trajectories.sh
```
