#!/usr/bin/env python3
import os, sys, argparse, json, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import FlexibleMLP, FlexibleCNN
from src.utils import MNIST, load_similar_mnist_data
from src.utils import load_data_with_replacement, load_data

def get_device(force_auto: bool = True) -> torch.device:
    if force_auto:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_model(model_type: str):
    if model_type == "mlp":
        return FlexibleMLP(hidden_dim=8, num_hidden_layers=1, input_downsample=6)
    return FlexibleCNN(
        in_channels=1, conv_channels=[12], conv_kernels=[3], conv_strides=[1],
        conv_use_relu_list=[True], conv_dropouts=[0.0], conv_use_bn=True,
        pool_after=[False], gap_size=1, mlp_hidden_dims=[11],
        mlp_use_relu_list=[True], mlp_dropouts=[0.0], output_dim=10
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_train", choices=["mnist", "mnist_similar"], required=True)
    p.add_argument("--dataset_val", default="mnist")
    p.add_argument("--model", choices=["mlp", "cnn"], required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--optimizer", default="sgd")
    p.add_argument("--results_dir", default="src/scripts/exp_results")
    p.add_argument("--sample_size", type=int, default=6400)
    p.add_argument("--auto_device", action="store_true")
    p.add_argument("--data_loader", default='default')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.auto_device)

    load_data_fn = (
        load_data_with_replacement if args.data_loader == 'replacement'
        else load_data if args.data_loader == 'default'
        else None
    )
    train_ds, val_ds, train_loader, val_loader = load_data_fn(
        args.dataset_train, args.batch_size, args.sample_size, args.seed
    )

    model = create_model(args.model).to(device)
    optimizer = (
        optim.SGD(model.parameters(), lr=args.lr)
        if args.optimizer.lower() == "sgd"
        else optim.Adam(model.parameters(), lr=args.lr)
    )
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"SGD: {args.model}, {args.dataset_train}->{args.dataset_val}, "
        f"train_size={len(train_ds)}, val_size={len(val_ds)}, "
        f"lr={args.lr}, epochs={args.epochs}, batch_size={args.batch_size}, "
        f"seed={args.seed}, params={n_params}"
    )

    train_losses, val_losses, val_accs = [], [], []

    for epoch in tqdm(range(args.epochs), desc="Epochs (SGD)"):
        # --- Training ---
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_losses.append(running / max(1, len(train_loader)))

        # --- Validation ---
        model.eval()
        vloss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                vloss += criterion(out, y).item()
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
        val_losses.append(vloss / max(1, len(val_loader)))
        val_accs.append(correct / len(val_loader.dataset))

    # --- Save model + logs ---
    os.makedirs(args.results_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.results_dir, "initial_after_sgd.pt"))

    logs = {"train_losses": train_losses, "val_losses": val_losses, "val_accs": val_accs}
    with open(os.path.join(args.results_dir, "logs_sgd.json"), "w") as f:
        json.dump(logs, f)
    print(f"[saved] logs_sgd.json")

    # --- Plots ---
    epochs = np.arange(1, args.epochs + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(args.results_dir, "loss_acc_sgd.png")
    plt.savefig(fig_path)
    print(f"[saved] {fig_path}")


if __name__ == "__main__":
    main()
