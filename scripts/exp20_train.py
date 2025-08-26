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
    parser = argparse.ArgumentParser(description='Common start then branch SGD - Experiment 20')
    parser.add_argument('--initial-lr', type=float, default=0.1, help='Initial learning rate for common path')
    parser.add_argument('--branch-lrs', type=str, default='0.001,0.01,0.1,0.5', help='Branch learning rates (comma-separated)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--sgd-epochs', type=int, default=10, help='Initial SGD epochs')
    parser.add_argument('--gd-epochs', type=int, default=10, help='GD epochs to minimum')
    parser.add_argument('--branch-epochs', type=int, default=16, help='Branch SGD epochs with logging')
    parser.add_argument('--save-dir', type=str, default='data/checkpoints/exp20', help='Save directory')
    parser.add_argument('--seed', type=int, default=228, help='Random seed')
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
    
    # Парсим branch learning rates
    branch_lrs = [float(lr) for lr in args.branch_lrs.split(',')]
    
    # ======== Фиксация случайности ========
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # ======== Настройки ========
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAMPLE_SIZE = 1000
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ======== Загрузка данных ========
    train_dataset, test_dataset, train_loader, test_loader = MNIST(
        batch_size=args.batch_size, sample_size=SAMPLE_SIZE
    )
    
    # Создаем полный батч для GD
    all_images = []
    all_labels = []
    for images, labels in train_loader:
        all_images.append(images)
        all_labels.append(labels)
    full_batch_images = torch.cat(all_images, dim=0).to(DEVICE)
    full_batch_labels = torch.cat(all_labels, dim=0).to(DEVICE)
    
    # ======== Модель ========
    model = FlexibleMLP(
        hidden_dim=8,
        num_hidden_layers=1,
        input_downsample=6
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n📊 Конфигурация эксперимента 20:")
    print(f"   Device: {DEVICE}")
    print(f"   Initial learning rate: {args.initial_lr}")
    print(f"   Branch learning rates: {branch_lrs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Sample size: {SAMPLE_SIZE}")
    print(f"   SGD epochs: {args.sgd_epochs}")
    print(f"   GD epochs: {args.gd_epochs}")
    print(f"   Branch epochs: {args.branch_epochs}")
    
    # Хранилища для общего пути
    common_train_losses = []
    common_val_losses = []
    common_val_accuracies = []
    
    # ======== ОБЩИЙ ПУТЬ: SGD + GD ========
    print(f"\n🚀 ОБЩИЙ ПУТЬ: SGD ({args.sgd_epochs} эпох) + GD ({args.gd_epochs} эпох)...")
    
    # SGD этап
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=args.initial_lr)
    model.train()
    
    pbar = tqdm(range(args.sgd_epochs), desc="Common SGD", ncols=100)
    for epoch in pbar:
        epoch_losses = []
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer_sgd.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_sgd.step()
            
            epoch_losses.append(loss.item())
        
        epoch_avg_loss = np.mean(epoch_losses)
        common_train_losses.append(epoch_avg_loss)
        
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, DEVICE)
        common_val_losses.append(val_loss)
        common_val_accuracies.append(val_acc)
        
        pbar.set_postfix({
            'train_loss': f'{epoch_avg_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.1f}%'
        })
    
    sgd_end = len(common_train_losses)
    
    # GD этап
    optimizer_gd = torch.optim.SGD(model.parameters(), lr=args.initial_lr)
    model.train()
    
    pbar = tqdm(range(args.gd_epochs), desc="Common GD", ncols=100)
    for epoch in pbar:
        optimizer_gd.zero_grad()
        outputs = model(full_batch_images)
        loss = criterion(outputs, full_batch_labels)
        loss.backward()
        optimizer_gd.step()
        
        common_train_losses.append(loss.item())
        
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, DEVICE)
        common_val_losses.append(val_loss)
        common_val_accuracies.append(val_acc)
        
        pbar.set_postfix({
            'train_loss': f'{loss.item():.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.1f}%'
        })
    
    gd_end = len(common_train_losses)
    
    # Сохраняем состояние модели для разветвления
    branch_start_state = model.state_dict().copy()
    
    print(f"✅ Общий путь завершен. SGD: {sgd_end} эпох, GD: {gd_end} эпох")
    
    # ======== РАЗВЕТВЛЕНИЕ ДЛЯ РАЗНЫХ LR ========
    all_branch_data = {}
    
    for branch_lr in branch_lrs:
        print(f"\n🌿 ВЕТКА lr={branch_lr}: SGD с логированием ({args.branch_epochs} эпох)...")
        
        # Восстанавливаем состояние модели
        model.load_state_dict(branch_start_state)
        
        optimizer_branch = torch.optim.SGD(model.parameters(), lr=branch_lr)
        
        branch_train_losses = []
        branch_val_losses = []
        branch_val_accuracies = []
        params_vectors = []
        hessians = []
        
        pbar = tqdm(range(args.branch_epochs), desc=f"Branch lr={branch_lr}", ncols=100)
        
        for epoch in pbar:
            epoch_losses = []
            epoch_hessians = []
            epoch_params = []
            
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                # Сохраняем параметры до шага
                params_vector = get_model_params_vector(model)
                epoch_params.append(params_vector)
                
                # Вычисляем loss
                model.train()
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_losses.append(loss.item())
                
                # Вычисляем гессиан
                hessian = compute_hessian(model, images, labels, criterion, DEVICE)
                epoch_hessians.append(hessian)
                
                # Делаем шаг оптимизации
                optimizer_branch.zero_grad()
                loss.backward()
                optimizer_branch.step()
            
            # Сохраняем все батчи эпохи
            params_vectors.extend(epoch_params)
            hessians.extend(epoch_hessians)
            
            # Логируем средний лосс за эпоху
            epoch_avg_loss = np.mean(epoch_losses)
            branch_train_losses.append(epoch_avg_loss)
            
            # Валидация каждую эпоху
            val_loss, val_acc = evaluate_model(model, test_loader, criterion, DEVICE)
            branch_val_losses.append(val_loss)
            branch_val_accuracies.append(val_acc)
            
            pbar.set_postfix({
                'train_loss': f'{epoch_avg_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_acc': f'{val_acc:.1f}%'
            })
        
        # Сохраняем данные ветки
        all_branch_data[branch_lr] = {
            'train_losses': branch_train_losses,
            'val_losses': branch_val_losses,
            'val_accuracies': branch_val_accuracies,
            'params_vectors': params_vectors,
            'hessians': hessians
        }
        
        print(f"✅ Ветка lr={branch_lr} завершена")
    
    # ======== Создание объединенного графика ========
    plt.figure(figsize=(20, 10))
    
    # График потерь
    plt.subplot(2, 3, 1)
    # Общий путь
    plt.plot(range(len(common_train_losses)), common_train_losses, 'k-', linewidth=2, label='Common path')
    plt.axvline(x=sgd_end, color='red', linestyle='--', alpha=0.8, label='SGD → GD')
    plt.axvline(x=gd_end, color='blue', linestyle='--', alpha=0.8, label='GD → Branch')
    
    # Ветки
    colors = ['red', 'green', 'blue', 'orange']
    for i, (branch_lr, data) in enumerate(all_branch_data.items()):
        branch_x = range(gd_end, gd_end + len(data['train_losses']))
        plt.plot(branch_x, data['train_losses'], color=colors[i % len(colors)], 
                label=f'SGD lr={branch_lr}', alpha=0.8)
    
    plt.title('Training Loss - All Branches')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Логарифмический график
    plt.subplot(2, 3, 2)
    plt.semilogy(range(len(common_train_losses)), common_train_losses, 'k-', linewidth=2, label='Common path')
    plt.axvline(x=sgd_end, color='red', linestyle='--', alpha=0.8)
    plt.axvline(x=gd_end, color='blue', linestyle='--', alpha=0.8)
    
    for i, (branch_lr, data) in enumerate(all_branch_data.items()):
        branch_x = range(gd_end, gd_end + len(data['train_losses']))
        plt.semilogy(branch_x, data['train_losses'], color=colors[i % len(colors)], 
                    label=f'SGD lr={branch_lr}', alpha=0.8)
    
    plt.title('Training Loss (log scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.legend()
    plt.grid(True)
    
    # Валидационные потери
    plt.subplot(2, 3, 3)
    plt.plot(range(len(common_val_losses)), common_val_losses, 'k-', linewidth=2, marker='o', label='Common path')
    plt.axvline(x=sgd_end, color='red', linestyle='--', alpha=0.8)
    plt.axvline(x=gd_end, color='blue', linestyle='--', alpha=0.8)
    
    for i, (branch_lr, data) in enumerate(all_branch_data.items()):
        branch_x = range(gd_end, gd_end + len(data['val_losses']))
        plt.plot(branch_x, data['val_losses'], color=colors[i % len(colors)], 
                marker='s', label=f'SGD lr={branch_lr}', alpha=0.8)
    
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Валидационная точность
    plt.subplot(2, 3, 4)
    plt.plot(range(len(common_val_accuracies)), common_val_accuracies, 'k-', linewidth=2, marker='o', label='Common path')
    plt.axvline(x=sgd_end, color='red', linestyle='--', alpha=0.8)
    plt.axvline(x=gd_end, color='blue', linestyle='--', alpha=0.8)
    
    for i, (branch_lr, data) in enumerate(all_branch_data.items()):
        branch_x = range(gd_end, gd_end + len(data['val_accuracies']))
        plt.plot(branch_x, data['val_accuracies'], color=colors[i % len(colors)], 
                marker='s', label=f'SGD lr={branch_lr}', alpha=0.8)
    
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Только ветки - потери
    plt.subplot(2, 3, 5)
    for i, (branch_lr, data) in enumerate(all_branch_data.items()):
        plt.plot(data['train_losses'], color=colors[i % len(colors)], 
                label=f'lr={branch_lr}', alpha=0.8)
    
    plt.title('Branch Training Losses Only')
    plt.xlabel('Branch Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Только ветки - точность
    plt.subplot(2, 3, 6)
    for i, (branch_lr, data) in enumerate(all_branch_data.items()):
        plt.plot(data['val_accuracies'], color=colors[i % len(colors)], 
                marker='o', label=f'lr={branch_lr}', alpha=0.8)
    
    plt.title('Branch Validation Accuracy Only')
    plt.xlabel('Branch Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, f'exp20_branches.png')
    plt.savefig(plot_path)
    plt.close()
    
    # ======== Сохранение результатов ========
    for branch_lr, data in all_branch_data.items():
        # Параметры и гессианы
        params_tensor = np.array(data['params_vectors'])
        hessians_tensor = np.array(data['hessians'])
        
        params_path = os.path.join(args.save_dir, f"params_lr{branch_lr}.pkl")
        hessians_path = os.path.join(args.save_dir, f"hessians_lr{branch_lr}.pkl")
        
        with open(params_path, 'wb') as f:
            pickle.dump(params_tensor, f)
        
        with open(hessians_path, 'wb') as f:
            pickle.dump(hessians_tensor, f)
        
        print(f"\n💾 Ветка lr={branch_lr}:")
        print(f"   Параметры: {params_path}")
        print(f"   📏 Размер: {params_tensor.shape} ({params_tensor.nbytes / 1024 / 1024:.1f} MB)")
        print(f"   Гессианы: {hessians_path}")
        print(f"   📏 Размер: {hessians_tensor.shape} ({hessians_tensor.nbytes / 1024 / 1024:.1f} MB)")
    
    # Общие метаданные с данными для графиков
    metadata = {
        'initial_lr': args.initial_lr,
        'branch_lrs': branch_lrs,
        'batch_size': args.batch_size,
        'sgd_epochs': args.sgd_epochs,
        'gd_epochs': args.gd_epochs,
        'branch_epochs': args.branch_epochs,
        'sgd_end': sgd_end,
        'gd_end': gd_end,
        'common_train_losses': common_train_losses,
        'common_val_losses': common_val_losses,
        'common_val_accuracies': common_val_accuracies,
        'branch_data': {lr: {
            'train_losses': data['train_losses'],
            'val_losses': data['val_losses'], 
            'val_accuracies': data['val_accuracies']
        } for lr, data in all_branch_data.items()},
        'seed': args.seed,
        'sample_size': SAMPLE_SIZE
    }
    
    metadata_path = os.path.join(args.save_dir, f"metadata_exp20.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n💾 Общие метаданные: {metadata_path}")
    print(f"📈 График: {plot_path}")
    print(f"\n📊 Сводка:")
    print(f"   🔢 Веток: {len(branch_lrs)}")
    print(f"   🔢 Общих эпох: {len(common_train_losses)}")
    print(f"   🔢 Батчей на ветку: {len(all_branch_data[branch_lrs[0]]['params_vectors'])}")

if __name__ == "__main__":
    main()