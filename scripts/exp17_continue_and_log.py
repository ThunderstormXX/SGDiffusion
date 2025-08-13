import os
import sys
import torch
import numpy as np
import random
import argparse
import pickle
from tqdm import tqdm
import torch.nn as nn

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
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda, cpu, or auto)')
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
    if args.device == 'auto':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device.isdigit():
        DEVICE = f'cuda:{args.device}'
    else:
        DEVICE = args.device
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
    
    # ======== Информация о модели и гиперпараметрах ========
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Конфигурация эксперимента 17:")
    print(f"   Device: {DEVICE}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Sample size: {SAMPLE_SIZE}")
    print(f"   Post-plateau steps: {args.post_plateau_steps}")
    print(f"   Random seed: {args.seed}")
    print(f"\n🏗️  Архитектура модели:")
    print(f"   Hidden dim: 8")
    print(f"   Hidden layers: 1")
    print(f"   Input downsample: 6")
    print(f"   Total parameters: {num_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    plateau_model_path = os.path.join(args.save_dir, f"plateau_model_lr{args.lr}.pth")
    if not os.path.exists(plateau_model_path):
        raise FileNotFoundError(f"Модель в точке плато не найдена: {plateau_model_path}")
    
    model.load_state_dict(torch.load(plateau_model_path, map_location=DEVICE))
    print(f"\n✅ Загружена модель из точки плато: {plateau_model_path}")
    

    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # ======== Продолжение обучения с логированием ========
    print(f"\n🚀 Продолжаем обучение на {args.post_plateau_steps} шагов...")
    
    params_vectors = []
    hessians = []
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    data_iter = iter(train_loader)
    
    # Получаем начальные метрики для сравнения
    initial_val_loss, initial_val_acc = evaluate_model(model, test_loader, criterion, DEVICE)
    print(f"📊 Начальные метрики: Val Loss = {initial_val_loss:.4f}, Val Acc = {initial_val_acc:.2f}%")
    
    pbar = tqdm(range(args.post_plateau_steps), 
                desc=f"Hessian tracking (lr={args.lr})",
                ncols=130,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    
    for step in pbar:
        # Получаем данные
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
            
            # Обновляем прогресс-бар с метриками
            val_change = val_loss - initial_val_loss
            acc_change = val_acc - initial_val_acc
            
            # Вычисляем статистики гессиана
            current_hessian = hessians[-1]
            eigenvals = np.linalg.eigvals(current_hessian)
            max_eigenval = np.max(np.real(eigenvals))
            min_eigenval = np.min(np.real(eigenvals))
            condition_number = max_eigenval / max(abs(min_eigenval), 1e-10)
            
            pbar.set_postfix({
                'TrLoss': f'{loss.item():.4f}',
                'ValLoss': f'{val_loss:.4f}',
                'ValAcc': f'{val_acc:.1f}%',
                'ΔValLoss': f'{val_change:+.4f}',
                'ΔValAcc': f'{acc_change:+.1f}%',
                'MaxEig': f'{max_eigenval:.2e}',
                'CondNum': f'{condition_number:.1e}'
            })
        else:
            # Обновляем только тренировочные метрики
            if len(hessians) > 0:
                current_hessian = hessians[-1]
                eigenvals = np.linalg.eigvals(current_hessian)
                max_eigenval = np.max(np.real(eigenvals))
                
                pbar.set_postfix({
                    'TrLoss': f'{loss.item():.4f}',
                    'ValLoss': f'{val_losses[-1]:.4f}' if val_losses else f'{initial_val_loss:.4f}',
                    'ValAcc': f'{val_accuracies[-1]:.1f}%' if val_accuracies else f'{initial_val_acc:.1f}%',
                    'MaxEig': f'{max_eigenval:.2e}',
                    'Step': f'{step+1}/{args.post_plateau_steps}'
                })
        
        # Делаем шаг оптимизации
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Финальная валидация
    final_val_loss, final_val_acc = evaluate_model(model, test_loader, criterion, DEVICE)
    print(f"\n🏁 Финальные метрики:")
    print(f"   Val Loss: {initial_val_loss:.4f} → {final_val_loss:.4f} (изменение: {final_val_loss - initial_val_loss:+.4f})")
    print(f"   Val Acc:  {initial_val_acc:.2f}% → {final_val_acc:.2f}% (изменение: {final_val_acc - initial_val_acc:+.2f}%)")
    
    # ======== Сохранение тензоров ========
    params_tensor = np.array(params_vectors)  # shape: (steps, n_params)
    hessians_tensor = np.array(hessians)      # shape: (steps, n_params, n_params)
    
    params_path = os.path.join(args.save_dir, f"params_tensor_lr{args.lr}.pkl")
    hessians_path = os.path.join(args.save_dir, f"hessians_tensor_lr{args.lr}.pkl")
    
    with open(params_path, 'wb') as f:
        pickle.dump(params_tensor, f)
    
    with open(hessians_path, 'wb') as f:
        pickle.dump(hessians_tensor, f)
    
    # ======== Сохранение метаданных ========
    metadata = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'post_plateau_steps': args.post_plateau_steps,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_steps': list(range(49, args.post_plateau_steps, 50))[:len(val_losses)],
        'initial_val_loss': initial_val_loss,
        'initial_val_acc': initial_val_acc,
        'final_val_loss': final_val_loss,
        'final_val_acc': final_val_acc,
        'params_shape': params_tensor.shape,
        'hessians_shape': hessians_tensor.shape,
        'seed': args.seed,
        'sample_size': SAMPLE_SIZE
    }
    
    metadata_path = os.path.join(args.save_dir, f"metadata_lr{args.lr}.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n💾 Параметры сохранены: {params_path} (shape: {params_tensor.shape})")
    print(f"💾 Гессианы сохранены: {hessians_path} (shape: {hessians_tensor.shape})")
    print(f"💾 Метаданные сохранены: {metadata_path}")
    
    # Статистика гессианов
    if hessians:
        all_eigenvals = []
        for h in hessians[::50]:  # Каждый 50-й гессиан
            eigenvals = np.linalg.eigvals(h)
            all_eigenvals.extend(np.real(eigenvals))
        
        all_eigenvals = np.array(all_eigenvals)
        significant_eigenvals = all_eigenvals[np.abs(all_eigenvals) > 1e-6]
        
        print(f"\n📊 Статистика гессианов:")
        print(f"   Всего собственных значений: {len(all_eigenvals)}")
        print(f"   Значимых (|λ| > 1e-6): {len(significant_eigenvals)}")
        if len(significant_eigenvals) > 0:
            print(f"   Максимальное: {np.max(significant_eigenvals):.4e}")
            print(f"   Минимальное: {np.min(significant_eigenvals):.4e}")
            print(f"   Среднее: {np.mean(significant_eigenvals):.4e}")
    
    if val_accuracies:
        print(f"\n🎯 Финальная точность на валидации: {val_accuracies[-1]:.2f}%")

if __name__ == "__main__":
    main()