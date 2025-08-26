import os
import sys
import torch
import numpy as np
import random
import argparse
import pickle
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

# ======== Локальные модули ========
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import MNIST
from src.model import FlexibleMLP

def parse_args():
    parser = argparse.ArgumentParser(description='Continue training and log hessians - Experiment 17')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--post-plateau-steps', type=int, default=500, help='Steps after plateau')
    parser.add_argument('--save-dir', type=str, default='data/checkpoints/exp17', help='Save directory')
    parser.add_argument('--seed', type=int, default=228, help='Random seed')
    parser.add_argument('--source-optimizer', type=str, choices=['sgd', 'gd'], required=True, help='Source optimizer that trained the model')
    return parser.parse_args()

def compute_hessian(model, images, labels, criterion, device):
    """Вычисляет гессиан для текущего батча"""
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    
    model.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    
    hessian = torch.zeros(n_params, n_params, device=device)
    param_idx = 0
    
    for grad in grads:
        grad_flat = grad.contiguous().view(-1)
        for j, g in enumerate(grad_flat):
            if g.requires_grad:
                grad2 = torch.autograd.grad(g, params, retain_graph=True, allow_unused=True)
                col_idx = 0
                for k, g2 in enumerate(grad2):
                    if g2 is not None:
                        g2_flat = g2.contiguous().view(-1)
                        hessian[param_idx + j, col_idx:col_idx + len(g2_flat)] = g2_flat
                        col_idx += len(g2_flat)
                    else:
                        col_idx += params[k].numel()
        param_idx += grad.numel()
    
    return hessian.cpu().numpy()

def get_model_params_vector(model):
    """Получает развернутый вектор параметров модели"""
    params = []
    for p in model.parameters():
        params.append(p.data.view(-1))
    return torch.cat(params).cpu().numpy()

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
    
    # ======== Загрузка данных ========
    train_dataset, test_dataset, train_loader, test_loader = MNIST(
        batch_size=args.batch_size, sample_size=SAMPLE_SIZE
    )
    
    # ======== Загрузка модели из точки плато ========
    model = FlexibleMLP(
        hidden_dim=8,
        num_hidden_layers=1,
        input_downsample=6
    ).to(DEVICE)
    
    print(f"\n📊 Конфигурация эксперимента 17:")
    print(f"   Device: {DEVICE}")
    print(f"   Source optimizer: {args.source_optimizer.upper()}")
    print(f"   Continue with: SGD")
    print(f"   Learning rate: {args.lr}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Sample size: {SAMPLE_SIZE}")
    print(f"   Post-plateau steps: {args.post_plateau_steps}")
    
    # Загружаем модель, обученную указанным оптимизатором
    plateau_model_path = os.path.join(args.save_dir, f"plateau_model_{args.source_optimizer}_lr{args.lr}.pth")
    if not os.path.exists(plateau_model_path):
        raise FileNotFoundError(f"Модель не найдена: {plateau_model_path}")
    
    model.load_state_dict(torch.load(plateau_model_path, map_location=DEVICE))
    print(f"✅ Загружена модель: {plateau_model_path}")
    
    # Загружаем метаданные исходного обучения
    plateau_metadata_path = os.path.join(args.save_dir, f"plateau_metadata_{args.source_optimizer}_lr{args.lr}.npy")
    if os.path.exists(plateau_metadata_path):
        plateau_metadata = np.load(plateau_metadata_path, allow_pickle=True).item()
        initial_train_losses = plateau_metadata['train_losses']
        initial_val_losses = plateau_metadata['val_losses']
        initial_val_accuracies = plateau_metadata['val_accuracies']
        print(f"✅ Загружены метаданные исходного обучения ({len(initial_train_losses)} эпох)")
    else:
        initial_train_losses = []
        initial_val_losses = []
        initial_val_accuracies = []
        print("⚠️ Метаданные исходного обучения не найдены")
    
    # Продолжаем только с SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # ======== Продолжение обучения с логированием ========
    print(f"\n🚀 Продолжаем обучение на {args.post_plateau_steps} шагов с SGD...")
    
    params_vectors = []
    hessians = []
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    data_iter = iter(train_loader)
    
    pbar = tqdm(range(args.post_plateau_steps), 
                desc=f"Hessian tracking (from {args.source_optimizer.upper()}, lr={args.lr})",
                ncols=100)
    
    for step in pbar:
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Сохраняем параметры модели до шага
        params_vector = get_model_params_vector(model)
        params_vectors.append(params_vector)
        
        # Вычисляем loss на текущем батче
        model.train()
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_losses.append(loss.item())
        
        # Вычисляем гессиан
        hessian = compute_hessian(model, images, labels, criterion, DEVICE)
        hessians.append(hessian)
        
        # Валидация каждые 50 шагов
        if (step + 1) % 50 == 0:
            val_loss, val_acc = evaluate_model(model, test_loader, criterion, DEVICE)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            pbar.set_postfix({
                'train_loss': f'{loss.item():.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_acc': f'{val_acc:.1f}%'
            })
        else:
            pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})
        
        # Делаем шаг оптимизации
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # ======== Обновление существующих графиков ========
    # Обновляем progress график
    progress_plot_path = os.path.join(args.save_dir, f'progress_{args.source_optimizer}_lr{args.lr}.png')
    final_plot_path = os.path.join(args.save_dir, f'final_{args.source_optimizer}_lr{args.lr}.png')
    
    def update_plot(save_path, title_suffix=""):
        plt.figure(figsize=(15, 5))
        
        # График потерь с вертикальной линией
        plt.subplot(1, 3, 1)
        if initial_train_losses:
            plt.plot(range(len(initial_train_losses)), initial_train_losses, 
                    label=f'Initial ({args.source_optimizer.upper()})', alpha=0.7)
            transition_point = len(initial_train_losses)
            plt.axvline(x=transition_point, color='red', linestyle='--', alpha=0.8, 
                       label='Switch to SGD')
            # Продолжение с SGD
            sgd_steps = range(transition_point, transition_point + len(train_losses))
            plt.plot(sgd_steps, train_losses, label='SGD continuation', color='orange')
        else:
            plt.plot(train_losses, label='SGD', color='orange')
        
        plt.title(f'Training Loss ({args.source_optimizer.upper()} → SGD, lr={args.lr}){title_suffix}')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # График валидационных потерь
        plt.subplot(1, 3, 2)
        if initial_val_losses:
            # Позиции валидационных точек для исходного обучения
            if 'batches_per_epoch' in plateau_metadata:
                batches_per_epoch = plateau_metadata['batches_per_epoch']
                if args.source_optimizer == 'sgd':
                    val_steps_initial = [(i+1)*100*batches_per_epoch - 1 for i in range(len(initial_val_losses))]
                else:  # GD
                    val_steps_initial = [(i+1)*100 - 1 for i in range(len(initial_val_losses))]
            else:
                val_steps_initial = list(range(99, len(initial_train_losses), 100))[:len(initial_val_losses)]
            
            plt.plot(val_steps_initial, initial_val_losses, 
                    label=f'Initial ({args.source_optimizer.upper()})', marker='o', alpha=0.7)
            transition_point = len(initial_train_losses)
            plt.axvline(x=transition_point, color='red', linestyle='--', alpha=0.8)
        
        if val_losses:
            val_steps_sgd = [len(initial_train_losses) + (i+1)*50 for i in range(len(val_losses))]
            plt.plot(val_steps_sgd, val_losses, label='SGD continuation', 
                    marker='s', color='orange')
        
        plt.title('Validation Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # График валидационной точности
        plt.subplot(1, 3, 3)
        if initial_val_accuracies:
            plt.plot(val_steps_initial[:len(initial_val_accuracies)], initial_val_accuracies, 
                    label=f'Initial ({args.source_optimizer.upper()})', marker='o', alpha=0.7)
            plt.axvline(x=transition_point, color='red', linestyle='--', alpha=0.8)
        
        if val_accuracies:
            plt.plot(val_steps_sgd[:len(val_accuracies)], val_accuracies, label='SGD continuation', 
                    marker='s', color='orange')
        
        plt.title('Validation Accuracy')
        plt.xlabel('Step')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    # Обновляем оба графика
    if os.path.exists(progress_plot_path):
        update_plot(progress_plot_path, " (Updated)")
    if os.path.exists(final_plot_path):
        update_plot(final_plot_path, " (Final + SGD)")
    
    # Создаем новый объединенный график
    combined_plot_path = os.path.join(args.save_dir, f'combined_trajectory_{args.source_optimizer}_lr{args.lr}.png')
    update_plot(combined_plot_path)
    
    # ======== Сохранение тензоров ========
    params_tensor = np.array(params_vectors)
    hessians_tensor = np.array(hessians)
    
    params_path = os.path.join(args.save_dir, f"params_tensor_{args.source_optimizer}_lr{args.lr}.pkl")
    hessians_path = os.path.join(args.save_dir, f"hessians_tensor_{args.source_optimizer}_lr{args.lr}.pkl")
    
    with open(params_path, 'wb') as f:
        pickle.dump(params_tensor, f)
    
    with open(hessians_path, 'wb') as f:
        pickle.dump(hessians_tensor, f)
    
    # ======== Сохранение метаданных ========
    metadata = {
        'source_optimizer': args.source_optimizer,
        'continue_optimizer': 'sgd',
        'lr': args.lr,
        'batch_size': args.batch_size,
        'post_plateau_steps': args.post_plateau_steps,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'initial_train_losses': initial_train_losses,
        'initial_val_losses': initial_val_losses,
        'initial_val_accuracies': initial_val_accuracies,
        'params_shape': params_tensor.shape,
        'hessians_shape': hessians_tensor.shape,
        'seed': args.seed,
        'sample_size': SAMPLE_SIZE
    }
    
    metadata_path = os.path.join(args.save_dir, f"metadata_{args.source_optimizer}_lr{args.lr}.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n💾 Параметры сохранены: {params_path}")
    print(f"💾 Гессианы сохранены: {hessians_path}")
    print(f"💾 Метаданные сохранены: {metadata_path}")
    print(f"📈 Графики обновлены: progress, final, combined")
    
    if val_accuracies:
        print(f"🎯 Финальная точность на валидации: {val_accuracies[-1]:.2f}%")

if __name__ == "__main__":
    main()