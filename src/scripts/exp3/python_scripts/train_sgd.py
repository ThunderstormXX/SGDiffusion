#!/usr/bin/env python3
import os, sys, argparse, json, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# Import custom optimizers
from custom_optimizers import MUON, SignSGD

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import data utilities
from data_utils import load_dataset, get_model_for_dataset


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
    p.add_argument("--optimizer_params", type=str, default="{}", help="JSON string with optimizer parameters")
    p.add_argument("--results_dir", default="src/scripts/exp3/exp_results")
    p.add_argument("--auto_device", action="store_true")
    p.add_argument("--data_loader", default='default')
    p.add_argument('--device', default='cpu')
    p.add_argument('--save_trajectory', action='store_true', help="Save weight trajectory during training")
    p.add_argument('--sample_size', type=int, default=None, help="Limit dataset size for faster experiments")
    p.add_argument('--model_params', type=str, default=None, help="JSON string with model architecture parameters")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # Load dataset based on the specified name
    replacement = args.data_loader == 'replacement'
    train_ds, val_ds, train_loader, val_loader = load_dataset(
        args.dataset_train, args.batch_size, replacement=replacement, seed=args.seed, sample_size=args.sample_size
    )

    # Parse model parameters from JSON if provided
    model_params = None
    if args.model_params:
        try:
            model_params = json.loads(args.model_params)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse model_params: {e}")
    
    # Create appropriate model for the dataset
    model = get_model_for_dataset(args.dataset_train, device=device, model_params=model_params)
    
    # Parse optimizer parameters from JSON
    optimizer_params = json.loads(args.optimizer_params)
    
    # Get optimizer-specific parameters or use defaults
    opt_name = args.optimizer.lower()
    opt_config = optimizer_params.get(opt_name, {})
    
    # Set the learning rate from command line arguments if not in config
    if 'lr' not in opt_config:
        opt_config['lr'] = args.lr
    
    # Create the optimizer based on the name
    if opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), **opt_config)
    elif opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), **opt_config)
    elif opt_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), **opt_config)
    elif opt_name == "muon":
        optimizer = MUON(model.parameters(), **opt_config)
    elif opt_name == "signsgd":
        optimizer = SignSGD(model.parameters(), **opt_config)
    else:
        # Default to SGD if optimizer not recognized
        print(f"Warning: Optimizer {opt_name} not recognized, using SGD")
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Count model parameters in a universal way
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"SGD: {args.model}, {args.dataset_train}->{args.dataset_val}, "
        f"train_size={len(train_ds)}, val_size={len(val_ds)}, "
        f"lr={args.lr}, epochs={args.epochs}, batch_size={args.batch_size}, "
        f"seed={args.seed}, params={n_params}"
    )

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    # For weight trajectory tracking
    weight_trajectory = []
    if args.save_trajectory:
        # Save initial weights
        flat_weights = []
        for p in model.parameters():
            flat_weights.append(p.data.detach().cpu().flatten())
        weight_trajectory.append(torch.cat(flat_weights).numpy())

    for epoch in tqdm(range(args.epochs), desc=f"Epochs ({args.optimizer})"):
        # --- Training ---
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += compute_accuracy(logits, y)
        train_losses.append(running_loss / max(1, len(train_loader)))
        train_accs.append(running_acc / max(1, len(train_loader)))
        
        # Save weights for trajectory (every 10 epochs or last epoch)
        if args.save_trajectory and (epoch % 10 == 0 or epoch == args.epochs - 1):
            flat_weights = []
            for p in model.parameters():
                flat_weights.append(p.data.detach().cpu().flatten())
            weight_trajectory.append(torch.cat(flat_weights).numpy())

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

    logs = {
        "train_losses": train_losses, 
        "val_losses": val_losses, 
        "train_accs": train_accs,
        "val_accs": val_accs,
        "optimizer": args.optimizer,
        "optimizer_config": opt_config
    }
    
    log_filename = f"logs_{args.optimizer}.json"
    with open(os.path.join(args.results_dir, log_filename), "w") as f:
        json.dump(logs, f)
    print(f"[saved] {log_filename}")
    
    # Save weight trajectory if requested
    if args.save_trajectory:
        trajectory_filename = f"weight_trajectory_{args.optimizer}.npy"
        np.save(os.path.join(args.results_dir, trajectory_filename), np.array(weight_trajectory))
        print(f"[saved] {trajectory_filename}")
        
    # Save loss trajectory
    loss_trajectory_filename = f"loss_trajectory_{args.optimizer}.npz"
    np.savez(os.path.join(args.results_dir, loss_trajectory_filename), 
             train_losses=np.array(train_losses),
             val_losses=np.array(val_losses),
             train_accs=np.array(train_accs),
             val_accs=np.array(val_accs),
             train_steps=np.arange(1, args.epochs + 1))
    print(f"[saved] {loss_trajectory_filename}")

    # --- Plots ---
    epochs = np.arange(1, args.epochs + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.title(f"{args.optimizer.upper()} Training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.title(f"{args.optimizer.upper()} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    fig_path = os.path.join(args.results_dir, f"loss_acc_{args.optimizer}.png")
    plt.savefig(fig_path)
    print(f"[saved] {fig_path}")


if __name__ == "__main__":
    main()
