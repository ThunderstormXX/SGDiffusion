#!/usr/bin/env python3
import os, sys, argparse, json
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm  # ← NEW
import matplotlib.pyplot as plt

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

def load_data_fullbatch(dataset_train: str, batch_size: int, sample_size: int):
    if dataset_train == "mnist":
        train_dataset, test_dataset, train_loader, val_loader = MNIST(batch_size=batch_size, sample_size=sample_size)
    else:
        train_dataset, test_dataset, train_loader, val_loader = load_similar_mnist_data(batch_size=batch_size, sample_size=sample_size)

    all_images, all_labels = [], []
    for images, labels in train_loader:
        all_images.append(images); all_labels.append(labels)
    full_batch_images = torch.cat(all_images, dim=0)
    full_batch_labels = torch.cat(all_labels, dim=0)

    full_ds = TensorDataset(full_batch_images, full_batch_labels)
    full_loader = DataLoader(full_ds, batch_size=len(full_ds), shuffle=False)
    return (train_dataset, test_dataset, full_loader, val_loader, full_batch_images, full_batch_labels)

def create_model(model_type):
    if model_type == 'mlp':
        return FlexibleMLP(hidden_dim=8, num_hidden_layers=1, input_downsample=6)
    else:
        return FlexibleCNN(in_channels=1, conv_channels=[12], conv_kernels=[3], conv_strides=[1],
                           conv_use_relu_list=[True], conv_dropouts=[0.0], conv_use_bn=True,
                           pool_after=[False], gap_size=1, mlp_hidden_dims=[11],
                           mlp_use_relu_list=[True], mlp_dropouts=[0.0], output_dim=10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train', choices=['mnist', 'mnist_similar'], required=True)
    parser.add_argument('--dataset_val', default='mnist')
    parser.add_argument('--model', choices=['mlp', 'cnn'], required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--checkpoint_in', required=True)
    parser.add_argument('--results_dir', default='src/scripts/exp_results')
    parser.add_argument('--sample_size', type=int, default=6000)
    args = parser.parse_args()
    
    device = get_device()
    torch.manual_seed(args.seed)
    
    train_ds, val_ds, train_loader_full, val_loader, full_X, full_y = load_data_fullbatch(
        args.dataset_train, args.batch_size, args.sample_size
    )
    full_X, full_y = full_X.to(device), full_y.to(device)

    model = create_model(args.model).to(device)
    model.load_state_dict(torch.load(os.path.join(args.results_dir, args.checkpoint_in), map_location=device))
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GD: {args.model}, {args.dataset_train}->{args.dataset_val}, train_size={len(train_ds)}, val_size={len(val_ds)}, lr={args.lr}, epochs={args.epochs}, batch_size=full, seed={args.seed}, params={n_params}")
    
    train_losses, val_losses, val_accs = [], [], []
    
    for epoch in tqdm(range(args.epochs), desc="Epochs (GD)"):
        model.train()
        for data, target in train_loader_full:  # один полный батч
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        vloss, correct = 0.0, 0
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validate", leave=False):
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
    with open(os.path.join(args.results_dir, "logs_gd.json"), 'w') as f:
        json.dump(logs, f)
    
    with open(os.path.join(args.results_dir, "logs_sgd.json"), 'r') as f:
        sgd_logs = json.load(f)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(sgd_logs['train_losses'] + train_losses, label='Train Loss')
    plt.plot(sgd_logs['val_losses'] + val_losses, label='Val Loss')
    plt.axvline(len(sgd_logs['train_losses']), color='red', linestyle='--', alpha=0.5)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(sgd_logs['val_accs'] + val_accs, label='Val Acc')
    plt.axvline(len(sgd_logs['val_accs']), color='red', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "loss_acc_sgd_then_gd.png"))

if __name__ == '__main__':
    main()
