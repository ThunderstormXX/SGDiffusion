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
    parser = argparse.ArgumentParser(description='Train model to plateau - Experiment 17')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--max-epochs', type=int, default=60000, help='Maximum epochs')
    parser.add_argument('--save-dir', type=str, default='data/checkpoints/exp17', help='Save directory')
    parser.add_argument('--seed', type=int, default=228, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda, cpu, or auto)')
    return parser.parse_args()

def evaluate_model(model, test_loader, criterion, device):
    """Оценка модели на валидационной выборке"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main():
    args = parse_args()
    
    # ======== Фиксация случайности ========
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # ======== Настройки ========
    if args.device == 'auto':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device.isdigit():
        DEVICE = f'cuda:{args.device}'
    else:
        DEVICE = args.device
    SAMPLE_SIZE = 1000  # Увеличенный размер выборки
    
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
    
    print(f"\n📊 Конфигурация эксперимента 17:")
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
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    pbar = tqdm(range(args.max_epochs), 
                desc=f"Training (lr={args.lr})",
                ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    
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
        
        avg_train_loss = np.mean(epoch_losses)
        
        train_losses.append(avg_train_loss)
        
        # Валидация каждые 100 эпох
        if (epoch + 1) % 100 == 0:
            val_loss, val_acc = evaluate_model(model, test_loader, criterion, DEVICE)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Отслеживание улучшений
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                improvement_indicator = "📈"
            else:
                epochs_without_improvement += 100
                improvement_indicator = "📉" if epochs_without_improvement > 1000 else "➡️"
            
            pbar.set_postfix({
                'TrLoss': f'{avg_train_loss:.4f}',
                'ValLoss': f'{val_loss:.4f}',
                'ValAcc': f'{val_acc:.1f}%',
                'Best': f'{best_val_loss:.4f}',
                'NoImpr': f'{epochs_without_improvement}',
                'Status': improvement_indicator
            })
        else:
            # Показываем только train loss между валидациями
            pbar.set_postfix({
                'TrLoss': f'{avg_train_loss:.4f}',
                'ValLoss': f'{val_losses[-1]:.4f}' if val_losses else 'N/A',
                'ValAcc': f'{val_accuracies[-1]:.1f}%' if val_accuracies else 'N/A',
                'Best': f'{best_val_loss:.4f}' if best_val_loss != float('inf') else 'N/A'
            })
        
        # Обновляем график каждые 1000 эпох
        if (epoch + 1) % 1000 == 0:
            plt.figure(figsize=(15, 5))
            
            # График потерь
            plt.subplot(1, 3, 1)
            plt.plot(train_losses, label='Train Loss')
            if val_losses:
                val_epochs = list(range(99, len(train_losses), 100))[:len(val_losses)]
                plt.plot(val_epochs, val_losses, label='Val Loss', marker='o')
            plt.title(f'Loss (lr={args.lr})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # График точности
            plt.subplot(1, 3, 2)
            if val_accuracies:
                val_epochs = list(range(99, len(train_losses), 100))[:len(val_accuracies)]
                plt.plot(val_epochs, val_accuracies, label='Val Accuracy', marker='o', color='green')
            plt.title(f'Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)
            
            # Логарифмический график потерь
            plt.subplot(1, 3, 3)
            plt.semilogy(train_losses, label='Train Loss')
            if val_losses:
                plt.semilogy(val_epochs, val_losses, label='Val Loss', marker='o')
            plt.title(f'Loss (log scale)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (log)')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(args.save_dir, f'progress_lr{args.lr}.png')
            plt.savefig(plot_path)
            plt.close()
    
    # ======== Финальная валидация ========
    final_val_loss, final_val_acc = evaluate_model(model, test_loader, criterion, DEVICE)
    
    # ======== Сохранение модели ========
    model_path = os.path.join(args.save_dir, f"plateau_model_lr{args.lr}.pth")
    torch.save(model.state_dict(), model_path)
    
    # ======== Сохранение метаданных ========
    metadata = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'final_epoch': len(train_losses),
        'final_train_loss': train_losses[-1],
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'config': {
            'hidden_dim': 8,
            'num_hidden_layers': 1,
            'input_downsample': 6,
            'sample_size': SAMPLE_SIZE
        }
    }
    
    metadata_path = os.path.join(args.save_dir, f"plateau_metadata_lr{args.lr}.npy")
    np.save(metadata_path, metadata)
    
    # ======== Финальные графики ========
    plt.figure(figsize=(15, 5))
    
    # График потерь
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        val_epochs = list(range(99, len(train_losses), 100))[:len(val_losses)]
        plt.plot(val_epochs, val_losses, label='Val Loss', marker='o')
    plt.title(f'Final Loss (lr={args.lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # График точности
    plt.subplot(1, 3, 2)
    if val_accuracies:
        val_epochs = list(range(99, len(train_losses), 100))[:len(val_accuracies)]
        plt.plot(val_epochs, val_accuracies, label='Val Accuracy', marker='o', color='green')
    plt.title(f'Final Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Логарифмический график потерь
    plt.subplot(1, 3, 3)
    plt.semilogy(train_losses, label='Train Loss')
    if val_losses:
        plt.semilogy(val_epochs, val_losses, label='Val Loss', marker='o')
    plt.title(f'Final Loss (log scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, f'final_lr{args.lr}.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\n💾 Модель сохранена: {model_path}")
    print(f"💾 Метаданные сохранены: {metadata_path}")
    print(f"📈 График сохранен: {plot_path}")
    print(f"🎯 Финальная точность на валидации: {final_val_acc:.2f}%")

if __name__ == "__main__":
    main()