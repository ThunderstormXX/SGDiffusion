#!/usr/bin/env python3
import os, sys, argparse, json, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import NanoGPT
from src.utils import load_shakespeare_data


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




def create_model(meta_path):
    """Create NanoGPT model based on metadata"""
    meta = torch.load(meta_path)
    model = NanoGPT(
        vocab_size=meta['vocab_size'],
        n_embd=8,
        n_head=1,
        n_layer=1,
        block_size=meta['block_size'],
        mlp_ratio=1
    )
    return model


def compute_accuracy(logits, targets):
    """Compute token-level accuracy"""
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets).float()
    return correct.mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_train", default="shakespeare")
    p.add_argument("--dataset_val", default="shakespeare")
    p.add_argument("--model", default="nanogpt")
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--optimizer", default="sgd")
    p.add_argument("--results_dir", default="src/scripts/exp2/exp_results")
    p.add_argument("--auto_device", action="store_true")
    p.add_argument("--data_loader", default='default')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # Load Shakespeare data
    train_path = 'src/data/shakespeare_train.pt'
    val_path = 'src/data/shakespeare_val.pt'
    meta_path = 'src/data/shakespeare_meta.pt'
    
    # Determine if replacement sampling should be used
    replacement = args.data_loader == 'replacement'
    train_ds, val_ds, train_loader, val_loader = load_shakespeare_data(
        train_path, val_path, args.batch_size, replacement=replacement, seed=args.seed
    )

    model = create_model(meta_path).to(device)
    optimizer = (
        optim.SGD(model.parameters(), lr=args.lr)
        if args.optimizer.lower() == "sgd"
        else optim.Adam(model.parameters(), lr=args.lr)
    )

    n_params = model.get_num_params()
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
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / max(1, len(train_loader)))

        # --- Validation ---
        model.eval()
        vloss, vacc = 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                vloss += loss.item()
                vacc += compute_accuracy(logits, y)
        
        val_losses.append(vloss / max(1, len(val_loader)))
        val_accs.append(vacc / max(1, len(val_loader)))

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
