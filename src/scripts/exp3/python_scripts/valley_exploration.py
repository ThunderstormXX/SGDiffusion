#!/usr/bin/env python3
import os
import sys
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train', default='shakespeare')
    parser.add_argument('--dataset_val', default='shakespeare')
    parser.add_argument('--model', default='nanogpt')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--checkpoint_in', required=True)
    parser.add_argument('--results_dir', default='src/scripts/exp3/exp_results')
    parser.add_argument('--data_loader', default='default')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--sample_size', type=int, default=None, help="Limit dataset size for faster experiments")
    parser.add_argument('--model_params', type=str, default=None, help="JSON string with model architecture parameters")
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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
    
    # Create appropriate model for the dataset and load from checkpoint
    model = get_model_for_dataset(args.dataset_train, device=device, model_params=model_params)
    checkpoint_path = os.path.join(args.checkpoint_in)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Initialize SGD optimizer for valley exploration
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Count model parameters in a universal way
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Valley Exploration: {args.model}, {args.dataset_train}->{args.dataset_val}, "
        f"train_size={len(train_ds)}, val_size={len(val_ds)}, "
        f"lr={args.lr}, steps={args.steps}, batch_size={args.batch_size}, "
        f"seed={args.seed}, params={n_params}"
    )

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Track metrics
    train_losses = []
    val_losses = []
    val_accs = []
    weight_trajectory = []
    grad_norms = []
    
    # Save initial weights
    flat_weights = []
    for p in model.parameters():
        flat_weights.append(p.data.detach().cpu().flatten())
    weight_trajectory.append(torch.cat(flat_weights).numpy())

    # Initial validation
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

    # Training loop
    pbar = tqdm(range(args.steps), desc="Valley Exploration (SGD)")
    for step in pbar:
        # Training step
        model.train()
        
        # Get batch
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            
            # Calculate gradient norm
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    pn = p.grad.detach().norm(2).item()
                    total_norm_sq += pn * pn
            grad_norm = float(total_norm_sq ** 0.5)
            grad_norms.append(grad_norm)
            
            optimizer.step()
            train_losses.append(loss.item())
            break  # Only use one batch per step
        
        # Save weights for trajectory
        if step % 10 == 0 or step == args.steps - 1:
            flat_weights = []
            for p in model.parameters():
                flat_weights.append(p.data.detach().cpu().flatten())
            weight_trajectory.append(torch.cat(flat_weights).numpy())
        
        # Validation (every 10 steps)
        if step % 10 == 0 or step == args.steps - 1:
            model.eval()
            vloss, vacc = 0.0, 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits, loss = model(x, y)
                    vloss += loss.item()
                    vacc += compute_accuracy(logits, y)
            
            current_val_loss = vloss / max(1, len(val_loader))
            current_val_acc = vacc / max(1, len(val_loader))
            val_losses.append(current_val_loss)
            val_accs.append(current_val_acc)
            
            # Update progress bar
            pbar.set_postfix({
                "grad": f"{grad_norm:.3e}",
                "train": f"{train_losses[-1]:.4f}",
                "val": f"{current_val_loss:.4f}",
                "acc": f"{current_val_acc:.4f}",
            })

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.results_dir, "valley_exploration_final.pt"))
    
    # Save trajectory data
    trajectory_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'grad_norms': grad_norms
    }
    
    with open(os.path.join(args.results_dir, "valley_exploration_logs.json"), "w") as f:
        json.dump(trajectory_data, f)
    
    # Save weight trajectory as numpy array
    np.save(os.path.join(args.results_dir, "valley_weight_trajectory.npy"), np.array(weight_trajectory))
    
    # Plot training curves
    steps = np.arange(len(train_losses))
    
    # Ensure val_steps and val_losses have the same length
    if len(val_losses) > 0:
        # Create validation steps array with correct length
        val_steps = np.linspace(0, len(train_losses)-1, len(val_losses), dtype=int)
    
    plt.figure(figsize=(12, 8))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(steps, train_losses, label='Train Loss')
    plt.plot(val_steps, val_losses, label='Val Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss during Valley Exploration')
    
    # Accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(val_steps, val_accs, label='Val Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy during Valley Exploration')
    
    # Gradient norm plot
    plt.subplot(2, 2, 3)
    plt.plot(steps, grad_norms)
    plt.xlabel('Steps')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm during Valley Exploration')
    
    # Log-scale gradient norm
    plt.subplot(2, 2, 4)
    plt.semilogy(steps, grad_norms)
    plt.xlabel('Steps')
    plt.ylabel('Gradient Norm (log scale)')
    plt.title('Gradient Norm (Log Scale)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "valley_exploration_plots.png"))
    print(f"[saved] {os.path.join(args.results_dir, 'valley_exploration_plots.png')}")


if __name__ == "__main__":
    main()
