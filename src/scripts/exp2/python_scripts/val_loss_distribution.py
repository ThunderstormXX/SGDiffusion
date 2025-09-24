#!/usr/bin/env python3
import os, sys, argparse, json, random
import torch, torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import NanoGPT


def load_shakespeare_data(train_path, val_path, batch_size):
    """Load Shakespeare dataset"""
    X_train, Y_train = torch.load(train_path)
    X_val, Y_val = torch.load(val_path)
    
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
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


@torch.no_grad()
def evaluate_model(model, val_loader, device):
    """Evaluate model and return loss and accuracy"""
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        logits, loss = model(data, target)
        total_loss += loss.item()
        total_acc += compute_accuracy(logits, target)
        n_batches += 1
    
    return total_loss / n_batches, total_acc / n_batches


def load_weights_and_evaluate(weights_file, meta_path, val_loader, device, n_samples=100):
    """Load weights trajectory and evaluate validation metrics"""
    print(f"Loading weights from {weights_file}")
    weights_tensor = torch.load(weights_file, map_location='cpu')
    
    # weights_tensor shape: [n_runs, n_steps, n_params] or [n_steps, n_params]
    if weights_tensor.dim() == 3:
        n_runs, n_steps, n_params = weights_tensor.shape
        # Take first run for simplicity
        weights_tensor = weights_tensor[0]
    else:
        n_steps, n_params = weights_tensor.shape
    
    model_template = create_model(meta_path).to(device)
    
    # Sample steps to evaluate
    step_indices = np.linspace(0, n_steps-1, min(n_samples, n_steps), dtype=int)
    
    results = {
        'steps': [],
        'val_losses': [],
        'val_accs': []
    }
    
    for step_idx in tqdm(step_indices, desc="Evaluating weights"):
        # Load weights into model
        weights_vector = weights_tensor[step_idx]
        torch.nn.utils.vector_to_parameters(weights_vector, model_template.parameters())
        
        # Evaluate
        val_loss, val_acc = evaluate_model(model_template, val_loader, device)
        
        results['steps'].append(int(step_idx))
        results['val_losses'].append(val_loss)
        results['val_accs'].append(val_acc)
    
    return results


def plot_distributions(results, output_dir, prefix):
    """Plot loss and accuracy distributions"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Loss distribution at start
    start_losses = results['val_losses'][:10] if len(results['val_losses']) > 10 else results['val_losses'][:1]
    end_losses = results['val_losses'][-10:] if len(results['val_losses']) > 10 else results['val_losses'][-1:]
    
    start_accs = results['val_accs'][:10] if len(results['val_accs']) > 10 else results['val_accs'][:1]
    end_accs = results['val_accs'][-10:] if len(results['val_accs']) > 10 else results['val_accs'][-1:]
    
    # Plot histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].hist(start_losses, bins=20, alpha=0.7, color='blue')
    axes[0, 0].set_title('Start Loss Distribution')
    axes[0, 0].set_xlabel('Validation Loss')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(end_losses, bins=20, alpha=0.7, color='red')
    axes[0, 1].set_title('End Loss Distribution')
    axes[0, 1].set_xlabel('Validation Loss')
    axes[0, 1].set_ylabel('Frequency')
    
    axes[1, 0].hist(start_accs, bins=20, alpha=0.7, color='blue')
    axes[1, 0].set_title('Start Accuracy Distribution')
    axes[1, 0].set_xlabel('Validation Accuracy')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(end_accs, bins=20, alpha=0.7, color='red')
    axes[1, 1].set_title('End Accuracy Distribution')
    axes[1, 1].set_xlabel('Validation Accuracy')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    hist_path = os.path.join(output_dir, f'{prefix}_start_loss_hist.png')
    plt.savefig(hist_path)
    print(f"[saved] {hist_path}")
    
    # Save individual histograms
    for i, (data, title, filename) in enumerate([
        (start_losses, 'Start Loss Distribution', f'{prefix}_start_loss_hist.png'),
        (end_losses, 'End Loss Distribution', f'{prefix}_end_loss_hist.png'),
        (start_accs, 'Start Accuracy Distribution', f'{prefix}_start_acc_hist.png'),
        (end_accs, 'End Accuracy Distribution', f'{prefix}_end_acc_hist.png')
    ]):
        plt.figure(figsize=(6, 4))
        plt.hist(data, bins=20, alpha=0.7)
        plt.title(title)
        plt.xlabel(title.split()[1])
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_files', required=True, help='Comma-separated list of weight files')
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--n_samples', type=int, default=100)
    args = parser.parse_args()

    device = args.device
    
    # Load validation data
    train_path = 'src/data/shakespeare_train.pt'
    val_path = 'src/data/shakespeare_val.pt'
    meta_path = 'src/data/shakespeare_meta.pt'
    
    _, _, _, val_loader = load_shakespeare_data(train_path, val_path, batch_size=32)
    
    weights_files = args.weights_files.split(',')
    
    for weights_file in weights_files:
        weights_file = weights_file.strip()
        if not os.path.exists(weights_file):
            print(f"Warning: {weights_file} does not exist, skipping")
            continue
            
        # Extract prefix from filename for output naming
        basename = os.path.basename(weights_file)
        prefix = basename.replace('.pt', '')
        
        # Create output directory
        output_dir = os.path.join(args.results_dir, f'val_analysis_{prefix}', 'val_loss_distribution')
        
        # Evaluate weights
        results = load_weights_and_evaluate(weights_file, meta_path, val_loader, device, args.n_samples)
        
        # Plot distributions
        plot_distributions(results, output_dir, prefix)
        
        # Save metrics as CSV
        df = pd.DataFrame(results)
        start_metrics_path = os.path.join(os.path.dirname(output_dir), f'{prefix}_start_val_metrics.csv')
        end_metrics_path = os.path.join(os.path.dirname(output_dir), f'{prefix}_end_val_metrics.csv')
        
        # Save start and end metrics
        start_df = df.head(10) if len(df) > 10 else df.head(1)
        end_df = df.tail(10) if len(df) > 10 else df.tail(1)
        
        start_df.to_csv(start_metrics_path, index=False)
        end_df.to_csv(end_metrics_path, index=False)
        
        # Save summary statistics
        summary = {
            'start_loss_mean': float(np.mean(start_df['val_losses'])),
            'start_loss_std': float(np.std(start_df['val_losses'])),
            'end_loss_mean': float(np.mean(end_df['val_losses'])),
            'end_loss_std': float(np.std(end_df['val_losses'])),
            'start_acc_mean': float(np.mean(start_df['val_accs'])),
            'start_acc_std': float(np.std(start_df['val_accs'])),
            'end_acc_mean': float(np.mean(end_df['val_accs'])),
            'end_acc_std': float(np.std(end_df['val_accs'])),
        }
        
        summary_path = os.path.join(os.path.dirname(output_dir), f'{prefix}_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[saved] {start_metrics_path}")
        print(f"[saved] {end_metrics_path}")
        print(f"[saved] {summary_path}")
    
    print("[done] Validation loss distribution analysis completed")


if __name__ == '__main__':
    main()

