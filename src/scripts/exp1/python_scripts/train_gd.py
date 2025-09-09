#!/usr/bin/env python3
import os, sys, argparse, json, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import FlexibleMLP, FlexibleCNN
from src.utils import MNIST, load_similar_mnist_data
from src.utils import load_data

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
    load_data_fn = load_data ## WITHOUT REPLACEMENT
    batch_size = args.sample_size
    train_ds, val_ds, train_loader_full, val_loader = load_data_fn(
        args.dataset_train, batch_size , args.sample_size, args.seed
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
    grad_norms = []  # --- NEW

    # --- NEW: tqdm по эпохам с динамическим постфиксом
    pbar = tqdm(range(args.epochs), desc="Epochs (GD)")
    for epoch in pbar:
        # --- Train (один батч = весь датасет)
        model.train()
        current_grad_norm = None   # --- NEW
        current_train_loss = None  # --- NEW
        for data, target in train_loader_full:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # --- NEW: считаем L2-норму полного градиента
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    pn = p.grad.detach().norm(2).item()
                    total_norm_sq += pn * pn
            grad_norm = float(total_norm_sq ** 0.5)
            grad_norms.append(grad_norm)
            current_grad_norm = grad_norm
            current_train_loss = float(loss.item())

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
        current_val_loss = vloss / max(1, len(val_loader))
        current_val_acc = correct / len(val_loader.dataset)
        val_losses.append(current_val_loss)
        val_accs.append(current_val_acc)

        # --- NEW: обновляем строку tqdm текущими метриками
        pbar.set_postfix({
            "grad": f"{current_grad_norm:.3e}" if current_grad_norm is not None else "n/a",
            "train": f"{current_train_loss:.4f}" if current_train_loss is not None else "n/a",
            "val": f"{current_val_loss:.4f}",
            "acc": f"{current_val_acc:.4f}",
        })

    os.makedirs(args.results_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.results_dir, "initial_after_sgd_and_gd.pt"))

    logs = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'grad_norms': grad_norms  # --- NEW
    }
    logs_path = os.path.join(args.results_dir, "logs_gd.json")
    with open(logs_path, 'w') as f:
        json.dump(logs, f)
    print(f"[saved] {logs_path}")

    # --- NEW: также сохраняем в .npy
    npy_path = os.path.join(args.results_dir, "grad_norms_gd.npy")
    np.save(npy_path, np.array(grad_norms, dtype=np.float64))
    print(f"[saved] {npy_path}")

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

    # --- NEW: лог-лог график нормы градиента (по шагам)
    plt.figure(figsize=(6, 4))
    steps = np.arange(1, len(grad_norms) + 1)
    plt.loglog(steps, grad_norms)
    plt.xlabel('Step')
    plt.ylabel('||grad||_2')
    plt.title('Gradient norm (log-log)')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    grad_fig_path = os.path.join(args.results_dir, "grad_norm_loglog_gd.png")
    plt.tight_layout()
    plt.savefig(grad_fig_path)
    print(f"[saved] {grad_fig_path}")

if __name__ == '__main__':
    main()
