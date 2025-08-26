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
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'gd'], default='sgd', help='Optimizer type')
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
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    print(f"   Optimizer: {args.optimizer.upper()}")
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
    
    # Для GD создаем полный батч
    if args.optimizer == 'gd':
        all_images = []
        all_labels = []
        for images, labels in train_loader:
            all_images.append(images)
            all_labels.append(labels)
        full_batch_images = torch.cat(all_images, dim=0).to(DEVICE)
        full_batch_labels = torch.cat(all_labels, dim=0).to(DEVICE)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # ======== Обучение ========
    print(f"\n🚀 Начинаем обучение с {args.optimizer.upper()}...")
    model.train()
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    pbar = tqdm(range(args.max_epochs), 
                desc=f"Training ({args.optimizer.upper()}, lr={args.lr})",
                ncols=100)
    
    step_count = 0
    
    for epoch in pbar:
        if args.optimizer == 'sgd':
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                step_count += 1
        else:  # GD
            optimizer.zero_grad()
            outputs = model(full_batch_images)
            loss = criterion(outputs, full_batch_labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            step_count += 1
        
        # Валидация каждые 100 эпох
        if (epoch + 1) % 100 == 0:
            val_loss, val_acc = evaluate_model(model, test_loader, criterion, DEVICE)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            recent_loss = np.mean(train_losses[-len(train_loader):]) if len(train_losses) >= len(train_loader) else train_losses[-1]
            pbar.set_postfix({
                'train_loss': f'{recent_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_acc': f'{val_acc:.1f}%'
            })
        else:
            recent_loss = train_losses[-1] if train_losses else 0
            pbar.set_postfix({'train_loss': f'{recent_loss:.4f}'})
        
        # Обновляем график каждые 500 эпох
        if (epoch + 1) % 500 == 0:
            plt.figure(figsize=(15, 5))
            
            # График потерь
            plt.subplot(1, 3, 1)
            plt.plot(train_losses, label='Train Loss', alpha=0.7)
            if val_losses:
                # Позиции валидационных точек в шагах
                val_steps = []
                for i in range(len(val_losses)):
                    val_epoch = (i + 1) * 100
                    if args.optimizer == 'sgd':
                        val_step = val_epoch * len(train_loader)
                    else:  # GD
                        val_step = val_epoch
                    val_steps.append(val_step)
                plt.plot(val_steps, val_losses, label='Val Loss', marker='o')
            plt.title(f'Loss ({args.optimizer.upper()}, lr={args.lr}, epoch {epoch+1})')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # График точности
            plt.subplot(1, 3, 2)
            if val_accuracies:
                plt.plot(val_steps[:len(val_accuracies)], val_accuracies, label='Val Accuracy', marker='o', color='green')
            plt.title(f'Validation Accuracy ({args.optimizer.upper()})')
            plt.xlabel('Step')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)
            
            # Логарифмический график потерь
            plt.subplot(1, 3, 3)
            plt.semilogy(train_losses, label='Train Loss', alpha=0.7)
            if val_losses:
                plt.semilogy(val_steps[:len(val_losses)], val_losses, label='Val Loss', marker='o')
            plt.title(f'Loss (log scale)')
            plt.xlabel('Step')
            plt.ylabel('Loss (log)')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(args.save_dir, f'progress_{args.optimizer}_lr{args.lr}.png')
            plt.savefig(plot_path)
            plt.close()
    
    # ======== Финальная валидация ========
    final_val_loss, final_val_acc = evaluate_model(model, test_loader, criterion, DEVICE)
    
    # ======== Сохранение модели ========
    model_path = os.path.join(args.save_dir, f"plateau_model_{args.optimizer}_lr{args.lr}.pth")
    torch.save(model.state_dict(), model_path)
    
    # ======== Сохранение метаданных ========
    metadata = {
        'optimizer': args.optimizer,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'final_epoch': len(train_losses),
        'final_train_loss': train_losses[-1],
        'total_steps': len(train_losses),
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'batches_per_epoch': len(train_loader),
        'config': {
            'hidden_dim': 8,
            'num_hidden_layers': 1,
            'input_downsample': 6,
            'sample_size': SAMPLE_SIZE
        }
    }
    
    metadata_path = os.path.join(args.save_dir, f"plateau_metadata_{args.optimizer}_lr{args.lr}.npy")
    np.save(metadata_path, metadata)
    
    # ======== Финальные графики ========
    plt.figure(figsize=(15, 5))
    
    # График потерь
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    if val_losses:
        val_steps = []
        for i in range(len(val_losses)):
            val_epoch = (i + 1) * 100
            if args.optimizer == 'sgd':
                val_step = val_epoch * len(train_loader)
            else:  # GD
                val_step = val_epoch
            val_steps.append(val_step)
        plt.plot(val_steps, val_losses, label='Val Loss', marker='o')
    plt.title(f'Final Loss ({args.optimizer.upper()}, lr={args.lr})')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # График точности
    plt.subplot(1, 3, 2)
    if val_accuracies:
        plt.plot(val_steps[:len(val_accuracies)], val_accuracies, label='Val Accuracy', marker='o', color='green')
    plt.title(f'Final Validation Accuracy ({args.optimizer.upper()})')
    plt.xlabel('Step')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Логарифмический график потерь
    plt.subplot(1, 3, 3)
    plt.semilogy(train_losses, label='Train Loss', alpha=0.7)
    if val_losses:
        plt.semilogy(val_steps, val_losses, label='Val Loss', marker='o')
    plt.title(f'Final Loss (log scale)')
    plt.xlabel('Step')
    plt.ylabel('Loss (log)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    final_plot_path = os.path.join(args.save_dir, f'final_{args.optimizer}_lr{args.lr}.png')
    plt.savefig(final_plot_path)
    plt.close()
    
    print(f"\n💾 Модель сохранена: {model_path}")
    print(f"💾 Метаданные сохранены: {metadata_path}")
    print(f"📈 Финальный график сохранен: {final_plot_path}")
    print(f"🎯 Финальная точность на валидации: {final_val_acc:.2f}%")

if __name__ == "__main__":
    main()