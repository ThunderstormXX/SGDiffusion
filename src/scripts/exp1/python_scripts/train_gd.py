#!/usr/bin/env python3
import os, sys, argparse, json, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import FlexibleMLP, FlexibleCNN
from src.utils import MNIST, load_similar_mnist_data


def get_device(force_auto: bool = False) -> torch.device:
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


def load_data(dataset_train: str, dataset_val: str, sample_size: int | None, seed: int):
    if dataset_train == "mnist":
        train_dataset, test_dataset, _, _ = MNIST(batch_size=1, sample_size=sample_size)
    else:
        train_dataset, test_dataset, _, _ = load_similar_mnist_data(
            batch_size=1, sample_size=sample_size or 1000
        )

    # --- для GD батч = весь датасет
    train_loader_full = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker
    )

    return train_dataset, test_dataset, train_loader_full, val_loader


def create_model(model_type):
    if model_type == 'mlp':
        return FlexibleMLP(hidden_dim=8, num_hidden_layers=1, input_downsample=6)
    else:
        return FlexibleCNN(
            in_channels=1, conv_channels=[12], conv_kernels=[3], conv_strides=[1],
            conv_use_relu_list=[True], conv_dropouts=[0.0], conv_use_bn=True,
            pool_after=[False], gap_size=1, mlp_hidden_dims=[11],
            mlp_use_relu_list=[True], mlp_dropouts=[0.0], output_dim=10
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train', choices=['mnist', 'mnist_similar'], required=True)
    parser.add_argument('--dataset_val', default='mnist')
    parser.add_argument('--model', choices=['mlp', 'cnn'], required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--checkpoint_in', required=True)
    parser.add_argument('--results_dir', default='src/scripts/exp_results')
    parser.add_argument('--sample_size', type=int, default=6400)
    args = parser.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # загружаем данные
    train_ds, val_ds, train_loader_full, val_loader = load_data(
        args.dataset_train, args.dataset_val, args.sample_size, args.seed
    )

    model = create_model(args.model).to(device)
    model.load_state_dict(torch.load(os.path.join(args.results_dir, args.checkpoint_in), map_location=device))

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"GD: {args.model}, {args.dataset_train}->{args.dataset_val}, "
          f"train_size={len(train_ds)}, val_size={len(val_ds)}, "
          f"lr={args.lr}, epochs={args.epochs}, batch_size=full, "
          f"seed={args.seed}, params={n_params}")

    train_losses, val_losses, val_accs = [], [], []

    for epoch in tqdm(range(args.epochs), desc="Epochs (GD)"):
        # --- Train (один батч = весь датасет)
        model.train()
        for data, target in train_loader_full:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validation
        model.eval()
        vloss, correct = 0.0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                vloss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=False)
                correct += (pred == target).sum().item()
        val_losses.append(vloss / max(1, len(val_loader)))
        val_accs.append(correct / len(val_loader.dataset))

    os.makedirs(args.results_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.results_dir, "initial_after_sgd_and_gd.pt"))

    logs = {'train_losses': train_losses, 'val_losses': val_losses, 'val_accs': val_accs}
    logs_path = os.path.join(args.results_dir, "logs_gd.json")
    with open(logs_path, 'w') as f:
        json.dump(logs, f)
    print(f"[saved] {logs_path}")

    # --- графики
    epochs = np.arange(1, args.epochs + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(args.results_dir, "loss_acc_gd.png")
    plt.savefig(fig_path)
    print(f"[saved] {fig_path}")


if __name__ == '__main__':
    main()
