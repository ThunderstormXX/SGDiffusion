import os
import sys
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

# ======== Локальные модули ========
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import MNIST
from src.model import FlexibleMLP

def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--max-epochs', type=int, default=60000, help='Maximum epochs')
    parser.add_argument('--save-dir', type=str, default='data/checkpoints/exp16', help='Save directory')
    parser.add_argument('--seed', type=int, default=228, help='Random seed')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # ======== Фиксация случайности ========
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # ======== Настройки ========
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAMPLE_SIZE = 386
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ======== Загрузка данных ========
    train_dataset, test_dataset, train_loader, test_loader = MNIST(
        batch_size=args.batch_size, sample_size=SAMPLE_SIZE
    )
    
    # ======== Модель ========
    model = FlexibleMLP(
        hidden_dim=8,
        num_hidden_layers=1,
        input_downsample=6
    ).to(DEVICE)
    
    # ======== Информация о модели и гиперпараметрах ========
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Конфигурация эксперимента:")
    print(f"   Device: {DEVICE}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Sample size: {SAMPLE_SIZE}")
    print(f"   Max epochs: {args.max_epochs}")
    print(f"   Random seed: {args.seed}")
    print(f"\n🏗️  Архитектура модели:")
    print(f"   Hidden dim: 8")
    print(f"   Hidden layers: 1")
    print(f"   Input downsample: 6")
    print(f"   Total parameters: {num_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # ======== Обучение ========
    print(f"\n🚀 Начинаем обучение...")
    model.train()
    
    losses_per_epoch = []
    pbar = tqdm(range(args.max_epochs), desc="Training")
    
    for epoch in pbar:
        epoch_losses = []
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses_per_epoch.append(avg_loss)
        pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        # Обновляем график каждые 1000 эпох
        if (epoch + 1) % 1000 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(losses_per_epoch)
            plt.title(f'Training Loss (lr={args.lr}, epoch {epoch+1})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plot_path = os.path.join(args.save_dir, f'loss_progress_lr{args.lr}.png')
            plt.savefig(plot_path)
            plt.close()
    
    # ======== Сохранение модели ========
    model_path = os.path.join(args.save_dir, f"plateau_model_lr{args.lr}.pth")
    torch.save(model.state_dict(), model_path)
    
    # ======== Сохранение метаданных ========
    metadata = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'final_epoch': len(losses_per_epoch),
        'final_loss': losses_per_epoch[-1],
        'losses_per_epoch': losses_per_epoch,
        'config': {
            'hidden_dim': 8,
            'num_hidden_layers': 1,
            'input_downsample': 6,
            'sample_size': SAMPLE_SIZE
        }
    }
    
    metadata_path = os.path.join(args.save_dir, f"plateau_metadata_lr{args.lr}.npy")
    np.save(metadata_path, metadata)
    
    # ======== Финальный график ========
    plt.figure(figsize=(12, 6))
    plt.plot(losses_per_epoch)
    plt.title(f'Final Training Loss (lr={args.lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plot_path = os.path.join(args.save_dir, f'loss_progress_lr{args.lr}.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\n💾 Модель сохранена: {model_path}")
    print(f"💾 Метаданные сохранены: {metadata_path}")
    print(f"📈 График сохранен: {plot_path}")

if __name__ == "__main__":
    main()