#!/usr/bin/env python3
import os, sys, argparse, json, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import NanoGPT


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


def load_shakespeare_data(train_path, val_path, batch_size):
    """Load Shakespeare dataset for full batch GD"""
    X_train, Y_train = torch.load(train_path)
    X_val, Y_val = torch.load(val_path)
    
    # For GD, use full dataset as single batch
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False,
                             worker_init_fn=seed_worker, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           worker_init_fn=seed_worker, num_workers=0)
    
    return train_dataset, val_dataset, train_loader, val_loader


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train', default='shakespeare')
    parser.add_argument('--dataset_val', default='shakespeare')
    parser.add_argument('--model', default='nanogpt')
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--checkpoint_in', required=True)
    parser.add_argument('--results_dir', default='src/scripts/exp2/exp_results')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load Shakespeare data
    train_path = 'src/data/shakespeare_train.pt'
    val_path = 'src/data/shakespeare_val.pt'
    meta_path = 'src/data/shakespeare_meta.pt'
    
    train_ds, val_ds, train_loader_full, val_loader = load_shakespeare_data(
        train_path, val_path, batch_size=32  # val batch size
    )

    model = create_model(meta_path).to(device)
    model.load_state_dict(torch.load(os.path.join(args.results_dir, args.checkpoint_in), map_location=device))

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    n_params = model.get_num_params()
    print(f"GD: {args.model}, {args.dataset_train}->{args.dataset_val}, "
          f"train_size={len(train_ds)}, val_size={len(val_ds)}, "
          f"lr={args.lr}, epochs={args.epochs}, batch_size=full, "
          f"seed={args.seed}, params={n_params}")

    train_losses, val_losses, val_accs = [], [], []
    grad_norms = []

    # --- NEW: tqdm по эпохам с динамическим постфиксом
    pbar = tqdm(range(args.epochs), desc="Epochs (GD)")
    for epoch in pbar:
        # --- Train (один батч = весь датасет)
        model.train()
        current_grad_norm = None
        current_train_loss = None
        
        for data, target in train_loader_full:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logits, loss = model(data, target)
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
        vloss, vacc = 0.0, 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                logits, loss = model(data, target)
                vloss += loss.item()
                vacc += compute_accuracy(logits, target)
        
        current_val_loss = vloss / max(1, len(val_loader))
        current_val_acc = vacc / max(1, len(val_loader))
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
        'grad_norms': grad_norms
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
