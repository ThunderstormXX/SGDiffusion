import argparse
import copy
from pathlib import Path
import numpy as np
import torch

from dataloader import load_dataset
from model import build_model, count_parameters
from plotter import plot_param_trajectories, plot_pca_trajectories
from trainer import pick_device, set_seed, train_gd, train_sgd_runs


def parse_args():
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--architecture", default="flexible_mlp")
    parser.add_argument("--train_size", type=int, default=6400)
    parser.add_argument("--val_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--sgd_iterations", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--input_downsample", type=int, default=14)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--checkpoint_in", default="")
    parser.add_argument("--data_dir", default=str(root / "src/data/MNIST"))
    parser.add_argument("--figures_dir", default=str(root / "src/scripts/exp8/figures/training_trajectories"))
    parser.add_argument("--results_dir", default=str(root / "src/scripts/exp8/results/training_trajectories"))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    figures_dir = Path(args.figures_dir)
    results_dir = Path(args.results_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    train_ds, gd_loader, val_loader, test_loader = load_dataset(
        args.dataset, args.data_dir, args.train_size, args.val_size,
        args.test_size, args.batch_size, args.seed,
    )
    factory = lambda: build_model(
        args.architecture, args.hidden_dim, args.num_hidden_layers,
        args.input_downsample, args.dropout,
    )
    model = factory().to(device)
    if args.checkpoint_in:
        checkpoint = torch.load(args.checkpoint_in, map_location=device)
        state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state)
        print(f"loaded checkpoint: {args.checkpoint_in}")
    init_state = copy.deepcopy(model.state_dict())
    n_params = count_parameters(model)
    print(f"device={device} dataset={args.dataset} train={len(train_ds)} val={args.val_size} test={args.test_size}")
    print(f"architecture={args.architecture} params={n_params} lr={args.lr} gd_epochs={args.sgd_iterations}")
    gd_traj, gd_logs = train_gd(
        model, gd_loader, val_loader, test_loader, args.lr,
        args.sgd_iterations, device, args.eval_every,
    )
    sgd_traj, sgd_logs = train_sgd_runs(factory, init_state, train_ds, val_loader, test_loader, args, device)
    if gd_logs:
        epoch, loss, val_acc, test_acc = gd_logs[-1]
        print(f"GD final epoch={int(epoch)} loss={loss:.4f} val_acc={val_acc:.3f} test_acc={test_acc:.3f}")
    if sgd_logs:
        sgd_arr = np.array(sgd_logs, dtype=float)
        print(f"SGD final mean val_acc={sgd_arr[:, 2].mean():.3f} test_acc={sgd_arr[:, 4].mean():.3f}")
    np.savez_compressed(
        results_dir / "trajectories.npz",
        gd=gd_traj,
        sgd=sgd_traj,
        gd_logs=np.array(gd_logs, dtype=float),
        sgd_logs=np.array(sgd_logs, dtype=float),
    )
    plot_param_trajectories(gd_traj, sgd_traj, figures_dir / "param_percentile_trajectories.png")
    plot_pca_trajectories(gd_traj, sgd_traj, figures_dir / "pca_trajectory_cloud.png")
    print(f"saved figures to {figures_dir}")
    print(f"saved trajectories to {results_dir / 'trajectories.npz'}")


if __name__ == "__main__":
    main()
