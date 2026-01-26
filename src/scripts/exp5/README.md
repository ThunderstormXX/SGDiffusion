# Experiment 5: SGD -> GD -> SGD Tunnel Experiments

Multi-stage training pipelines with configurable tracking and aggregation.

## Available Experiments

| Preset | Dataset | N_RUNS | SGD1 | GD | SGD2 |
|--------|---------|--------|------|-----|------|
| `mnist_small` | MNIST | 5 | 100 steps | 10 epochs | 1000 steps |
| `mnist_medium` | MNIST | 20 | 1000 steps | 100 epochs | 1000 steps |
| `mnist_large` | MNIST | 1000 | 1000 steps | 100 epochs | 3000 steps |
| `shakespeare_small` | Shakespeare | 5 | 100 steps | 10 epochs | 1000 steps |
| `shakespeare_medium` | Shakespeare | 20 | 1000 steps | 100 epochs | 1000 steps |
| `shakespeare_large` | Shakespeare | 1000 | 1000 steps | 100 epochs | 3000 steps |

## Tracking Strategy

| Stage | What is tracked |
|-------|-----------------|
| Stage 1 (SGD) | **Weights only** |
| Stage 2 (GD) | **Metrics only** (+ weights for viz) |
| Stage 3 (SGD) | **Weights + Hessians** |

## Quick Start

```bash
# Run all 3 steps (experiment, visualize, aggregate)
./src/scripts/exp5/bash/run_shakespeare_small.sh

# Run on CUDA
DEVICE=cuda ./src/scripts/exp5/bash/run_shakespeare_small.sh
```

## Modular Step Execution

Each script supports selective step execution:

| Step | Name | Description |
|------|------|-------------|
| 0 | Experiment | Run all 3 stages for all seeds |
| 1 | Visualize | Generate weight trajectory plots |
| 2 | Aggregate | Create aggregated tensors |

### Examples

```bash
# Run ALL steps (default)
./run_shakespeare_small.sh

# Run ONLY step 0 (experiment)
./run_shakespeare_small.sh 0

# Run ONLY step 1 (visualize existing results)
./run_shakespeare_small.sh 1

# Run ONLY step 2 (aggregate tensors)
./run_shakespeare_small.sh 2

# Run steps 0 and 1
./run_shakespeare_small.sh 0 1

# Run steps 1 and 2 (reuse existing experiment)
./run_shakespeare_small.sh 1 2
```

## Execution Order

The experiment runs stages in the correct order for sampling:

1. **Stage 1 (SGD)** for ALL seeds → saves checkpoints
2. **Stage 2 (GD)** for ALL seeds → loads stage 1 checkpoints, saves new checkpoints
3. **Stage 3 (SGD)** for ALL seeds → loads stage 2 checkpoints

This ensures that trajectories are properly sampled from the same distribution at each stage transition.

## Output Structure

```
src/scripts/exp5/exp_results/shakespeare_small/
├── run_seed42/
│   ├── config.json
│   ├── metrics_stage1_sgd.json
│   ├── metrics_stage2_gd.json
│   ├── metrics_stage3_sgd.json
│   ├── weights_stage1_sgd.pt
│   ├── weights_stage2_gd.pt
│   ├── weights_stage3_sgd.pt
│   ├── hessian_stage3_sgd.pt
│   └── stage*.pt (checkpoints)
├── run_seed43/
│   └── ...
└── aggregated/
    ├── # Aggregated tensors
    ├── weights_stage1_sgd.pt          # (n_runs, n_steps, n_params)
    ├── weights_stage1_sgd_tensor_info.txt
    ├── weights_stage2_gd.pt
    ├── weights_stage2_gd_tensor_info.txt
    ├── weights_stage3_sgd.pt
    ├── weights_stage3_sgd_tensor_info.txt
    ├── hessians_stage3_sgd.pt         # (n_runs, n_steps, n_params, n_params)
    ├── hessians_stage3_sgd_tensor_info.txt
    │
    ├── # Visualization plots (10 total)
    ├── weight_percentile_0.png
    ├── weight_percentile_0_normalized.png
    ├── weight_percentile_20.png
    ├── weight_percentile_20_normalized.png
    ├── weight_percentile_40.png
    ├── weight_percentile_40_normalized.png
    ├── weight_percentile_60.png
    ├── weight_percentile_60_normalized.png
    ├── weight_percentile_80.png
    ├── weight_percentile_80_normalized.png
    ├── weight_trajectories_combined.png
    └── weight_trajectories_comparison.png
```

## Visualization

### 10 Plots Generated

1. **5 percentile plots** (raw X-axis: Step)
2. **5 percentile plots** (normalized X-axis: Step × Batch Size)

### X-axis Modes

| Mode | X-axis Label | Description |
|------|--------------|-------------|
| Raw | Step | Steps as-is |
| Normalized | Step × Batch Size | Steps scaled by batch size |

The normalized view makes computational cost comparable:
- SGD with batch 32: step×32
- GD with full batch (e.g., 2000): step×2000

### Plot Features

- **Mean ± std** across all runs
- **Red vertical lines** at stage boundaries
- **Stage labels** (SGD, GD, SGD)

## Aggregated Tensors

The aggregate step creates tensors with shape info:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `weights_{stage}.pt` | (n_runs, n_steps, n_params) | Weight trajectories |
| `hessians_{stage}.pt` | (n_runs, n_steps, n_params, n_params) | Hessian matrices |
| `grads_{stage}.pt` | (n_runs, n_steps, n_params) | Gradient vectors |

Each tensor has a `{name}_tensor_info.txt` file with metadata.

## Python Scripts

| Script | Description |
|--------|-------------|
| `run_stage.py` | Run single stage for all seeds |
| `aggregate_tensors.py` | Aggregate tensors from runs |
| `visualize_many_runs.py` | Generate visualization plots |
| `visualize.py` | Single-run visualization |

## Models

- **MNIST**: FlexibleMLP(hidden_dim=8, num_hidden_layers=1, input_downsample=6) → **386 parameters**
  - Training set: 6400 samples, Batch size: 64
  - Learning rates: SGD lr=0.1, GD lr=0.01
  
- **Shakespeare**: NanoGPT(n_embd=8, n_head=1, n_layer=1, mlp_ratio=1, block_size=16)
  - Batch size: 64
  - Learning rates: SGD lr=0.01, GD lr=0.001
