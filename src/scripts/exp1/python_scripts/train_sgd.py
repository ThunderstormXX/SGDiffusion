#!/usr/bin/env python3
import os, sys, argparse, json
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # ← NEW

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import FlexibleMLP, FlexibleCNN
from src.utils import MNIST, load_similar_mnist_data

def get_device(force_auto: bool = False) -> torch.device:
    if force_auto:
        if torch.cuda.is_available(): return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

import random, numpy as np
from torch.utils.data import DataLoader, RandomSampler, BatchSampler

def seed_worker(worker_id: int):
    # уникальный сид для каждого воркера
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_data(dataset_train: str, dataset_val: str, batch_size: int, sample_size: int | None, seed: int):
    if dataset_train == "mnist":
        train_dataset, test_dataset, _, _ = MNIST(batch_size=batch_size, sample_size=sample_size)
    else:
        train_dataset, test_dataset, _, _ = load_similar_mnist_data(batch_size=batch_size, sample_size=sample_size or 1000)

    # генератор для контроля случайности
    g = torch.Generator().manual_seed(seed)

    # случайные батчи (с возвращением, так что каждый батч независим)
    sampler = RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset), generator=g)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    train_loader = DataLoader(train_dataset,
                              batch_sampler=batch_sampler,
                              num_workers=0,  # если хочешь ускорить, можно >0
                              worker_init_fn=seed_worker)

    val_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            worker_init_fn=seed_worker)

    # "снимок" всего трейна без shuffle — для full_imgs/full_lbls
    snapshot_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_imgs, all_lbls = [], []
    for imgs, lbls in snapshot_loader:
        all_imgs.append(imgs)
        all_lbls.append(lbls)
    full_batch_images = torch.cat(all_imgs, dim=0)
    full_batch_labels = torch.cat(all_lbls, dim=0)

    return train_dataset, test_dataset, train_loader, val_loader, full_batch_images, full_batch_labels


def create_model(model_type: str):
    if model_type == "mlp":
        return FlexibleMLP(hidden_dim=8, num_hidden_layers=1, input_downsample=6)
    return FlexibleCNN(in_channels=1, conv_channels=[12], conv_kernels=[3], conv_strides=[1],
                       conv_use_relu_list=[True], conv_dropouts=[0.0], conv_use_bn=True,
                       pool_after=[False], gap_size=1, mlp_hidden_dims=[11],
                       mlp_use_relu_list=[True], mlp_dropouts=[0.0], output_dim=10)

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
    p.add_argument("--sample_size", type=int, default=6000)
    p.add_argument("--auto_device", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.auto_device)

    train_ds, val_ds, train_loader, val_loader, full_imgs, full_lbls = load_data(
        args.dataset_train, args.dataset_val, args.batch_size, args.sample_size, args.seed
    )
    full_imgs, full_lbls = full_imgs.to(device), full_lbls.to(device)

    model = create_model(args.model).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr) if args.optimizer.lower() == "sgd" else optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"SGD: {args.model}, {args.dataset_train}->{args.dataset_val}, train_size={len(train_ds)}, val_size={len(val_ds)}, lr={args.lr}, epochs={args.epochs}, batch_size={args.batch_size}, seed={args.seed}, params={n_params}")

    train_losses, val_losses, val_accs = [], [], []

    for epoch in tqdm(range(args.epochs), desc="Epochs (SGD)"):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"Train epoch {epoch+1}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_losses.append(running / max(1, len(train_loader)))

        model.eval()
        vloss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validate", leave=False):
                x, y = x.to(device), y.to(device)
                out = model(x)
                vloss += criterion(out, y).item()
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
        val_losses.append(vloss / max(1, len(val_loader)))
        val_accs.append(correct / len(val_loader.dataset))

    os.makedirs(args.results_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.results_dir, "initial_after_sgd.pt"))

    logs = {"train_losses": train_losses, "val_losses": val_losses, "val_accs": val_accs}
    with open(os.path.join(args.results_dir, "logs_sgd.json"), "w") as f:
        json.dump(logs, f)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(train_losses, label="Train Loss"); plt.plot(val_losses, label="Val Loss"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(val_accs, label="Val Acc"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "loss_acc_sgd.png"))

if __name__ == "__main__":
    main()
