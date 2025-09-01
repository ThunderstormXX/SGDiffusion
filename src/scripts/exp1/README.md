# EXP1 - SGD/GD Optimization Analysis

Minimal implementation of SGD→GD training pipeline with Hessian analysis.

## Usage

```bash
# Run with MNIST dataset
bash src/scripts/exp1/bash/run_full_setup.sh mnist

# Run with MNIST_similar dataset  
bash src/scripts/exp1/bash/run_full_setup.sh mnist_similar
```

## Pipeline
1. `train_sgd.py` - SGD training → `initial_after_sgd.pt`
2. `train_gd.py` - GD from checkpoint → `initial_after_sgd_and_gd.pt`
3. `sgd_hessians.py` - Hessian trajectories → `weights_traj_lr*.pt`, `hessians_traj_lr*.pt`
4. `sgd_many_runs.py` - Multiple runs → `many_runs_weights_lr*.pt`

## Outputs
- Model checkpoints: `initial_after_sgd.pt`, `initial_after_sgd_and_gd.pt`
- Logs: `logs_sgd.json`, `logs_gd.json`
- Plots: `loss_acc_sgd.png`, `loss_acc_sgd_then_gd.png`
- Trajectories: `weights_traj_lr*.pt`, `hessians_traj_lr*.pt`, `many_runs_weights_lr*.pt`