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
from src.model import FlexibleMLP

def parse_args():
    parser = argparse.ArgumentParser(description='Three-stage training: SGD → GD → SGD - Experiment 19 (Similar MNIST)')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--sgd-epochs', type=int, default=10, help='Initial SGD epochs')
    parser.add_argument('--gd-epochs', type=int, default=10, help='GD epochs to minimum')
    parser.add_argument('--sgd-log-epochs', type=int, default=16, help='Final SGD epochs with logging (~1000 batch iterations)')
    parser.add_argument('--save-dir', type=str, default='data/checkpoints/exp19', help='Save directory')
    parser.add_argument('--seed', type=int, default=228, help='Random seed')
    return parser.parse_args()

def load_similar_mnist_data(batch_size, sample_size=1000):
    """Загружает данные Similar MNIST из файлов для обучения и классический MNIST для теста"""
    from torchvision import transforms, datasets
    from torch.utils.data import Subset
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # Загружаем похожие данные для обучения
    x_data = torch.load(os.path.join(data_dir, "mnist_similar_x.pt"))
    y_data = torch.load(os.path.join(data_dir, "mnist_similar_y.pt"))
    
    # Балансировка классов
    samples_per_class = sample_size // 10
    balanced_indices = []
    for class_id in range(10):
        class_indices = (y_data == class_id).nonzero(as_tuple=True)[0]
        if len(class_indices) >= samples_per_class:
            selected = class_indices[torch.randperm(len(class_indices))[:samples_per_class]]
        else:
            selected = class_indices
        balanced_indices.extend(selected.tolist())
    
    balanced_indices = torch.tensor(balanced_indices)
    x_data = x_data[balanced_indices]
    y_data = y_data[balanced_indices]
    
    # Создаем обучающий датасет из похожих данных
    train_dataset = torch.utils.data.TensorDataset(x_data, y_data)
    
    # Загружаем классический MNIST для теста
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Создаем загрузчики
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataset, test_dataset, train_loader, test_loader

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
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ======== Загрузка данных ========
    SAMPLE_SIZE = 1000
    train_dataset, test_dataset, train_loader, test_loader = load_similar_mnist_data(args.batch_size, SAMPLE_SIZE)
    
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
    
    print(f"\n📊 Конфигурация эксперимента 19:")
    print(f"   Device: {DEVICE}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Dataset size: {len(train_dataset)} train, {len(test_dataset)} test")
    print(f"   SGD epochs: {args.sgd_epochs}")
    print(f"   GD epochs: {args.gd_epochs}")
    print(f"   SGD logging epochs: {args.sgd_log_epochs} (~{args.sgd_log_epochs * len(train_loader)} batch iterations)")
    
    # Хранилища для всех этапов
    all_train_losses = []
    all_val_losses = []
    all_val_accuracies = []
    stage_transitions = []
    
    # ======== ЭТАП 1: SGD ========
    print(f"\n🚀 ЭТАП 1: SGD обучение ({args.sgd_epochs} эпох)...")
    
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=args.lr)
    model.train()
    
    pbar = tqdm(range(args.sgd_epochs), desc="Stage 1: SGD", ncols=100)
    
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
        
        # Логируем средний лосс за эпоху
        epoch_avg_loss = np.mean(epoch_losses)
        all_train_losses.append(epoch_avg_loss)
        
        # Валидация каждую эпоху
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, DEVICE)
        all_val_losses.append(val_loss)
        all_val_accuracies.append(val_acc)
        
        pbar.set_postfix({
            'train_loss': f'{epoch_avg_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.1f}%'
        })
    
    stage_transitions.append(len(all_train_losses))
    print(f"✅ ЭТАП 1 завершен. Переход в точке: {stage_transitions[0]} эпох")
    
    # ======== ЭТАП 2: GD ========
    print(f"\n🎯 ЭТАП 2: GD к минимуму ({args.gd_epochs} эпох)...")
    
    optimizer_gd = torch.optim.SGD(model.parameters(), lr=args.lr)
    model.train()
    
    pbar = tqdm(range(args.gd_epochs), desc="Stage 2: GD", ncols=100)
    
    for epoch in pbar:
        optimizer_gd.zero_grad()
        outputs = model(full_batch_images)
        loss = criterion(outputs, full_batch_labels)
        loss.backward()
        optimizer_gd.step()
        
        all_train_losses.append(loss.item())
        
        # Валидация каждую эпоху
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, DEVICE)
        all_val_losses.append(val_loss)
        all_val_accuracies.append(val_acc)
        
        pbar.set_postfix({
            'train_loss': f'{loss.item():.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.1f}%'
        })
    
    stage_transitions.append(len(all_train_losses))
    print(f"✅ ЭТАП 2 завершен. Переход в точке: {stage_transitions[1]} эпох")
    
    # ======== ЭТАП 3: SGD с логированием ========
    print(f"\n🔍 ЭТАП 3: SGD с логированием ({args.sgd_log_epochs} эпох)...")
    
    optimizer_sgd_log = torch.optim.SGD(model.parameters(), lr=args.lr)
    
    params_vectors = []
    hessians = []
    
    pbar = tqdm(range(args.sgd_log_epochs), desc="Stage 3: SGD + Logging", ncols=100)
    
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
            optimizer_sgd_log.zero_grad()
            loss.backward()
            optimizer_sgd_log.step()
        
        # Сохраняем все батчи эпохи
        params_vectors.extend(epoch_params)
        hessians.extend(epoch_hessians)
        
        # Логируем средний лосс за эпоху
        epoch_avg_loss = np.mean(epoch_losses)
        all_train_losses.append(epoch_avg_loss)
        
        # Валидация каждую эпоху
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, DEVICE)
        all_val_losses.append(val_loss)
        all_val_accuracies.append(val_acc)
        
        pbar.set_postfix({
            'train_loss': f'{epoch_avg_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.1f}%'
        })
    
    stage_transitions.append(len(all_train_losses))
    print(f"✅ ЭТАП 3 завершен. Финальная точка: {stage_transitions[2]} эпох")
    
    # ======== Создание графиков ========
    plt.figure(figsize=(15, 10))
    
    # График потерь с переходами
    plt.subplot(2, 2, 1)
    plt.plot(all_train_losses, alpha=0.7, color='blue')
    plt.axvline(x=stage_transitions[0], color='red', linestyle='--', alpha=0.8, label='SGD → GD')
    plt.axvline(x=stage_transitions[1], color='green', linestyle='--', alpha=0.8, label='GD → SGD+Log')
    plt.title(f'Training Loss (lr={args.lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Логарифмический график потерь
    plt.subplot(2, 2, 2)
    plt.semilogy(all_train_losses, alpha=0.7, color='blue')
    plt.axvline(x=stage_transitions[0], color='red', linestyle='--', alpha=0.8, label='SGD → GD')
    plt.axvline(x=stage_transitions[1], color='green', linestyle='--', alpha=0.8, label='GD → SGD+Log')
    plt.title(f'Training Loss (log scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.legend()
    plt.grid(True)
    
    # График валидационных потерь
    plt.subplot(2, 2, 3)
    if all_val_losses:
        plt.plot(range(len(all_val_losses)), all_val_losses, marker='o', alpha=0.7)
        plt.axvline(x=stage_transitions[0], color='red', linestyle='--', alpha=0.8)
        plt.axvline(x=stage_transitions[1], color='green', linestyle='--', alpha=0.8)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # График валидационной точности
    plt.subplot(2, 2, 4)
    if all_val_accuracies:
        plt.plot(range(len(all_val_accuracies)), all_val_accuracies, marker='o', color='green', alpha=0.7)
        plt.axvline(x=stage_transitions[0], color='red', linestyle='--', alpha=0.8)
        plt.axvline(x=stage_transitions[1], color='green', linestyle='--', alpha=0.8)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, f'exp19_lr{args.lr}.png')
    plt.savefig(plot_path)
    plt.close()
    
    # ======== Сохранение результатов ========
    # Модель
    model_path = os.path.join(args.save_dir, f"model_lr{args.lr}.pth")
    torch.save(model.state_dict(), model_path)
    
    # Параметры и гессианы
    params_tensor = np.array(params_vectors)
    hessians_tensor = np.array(hessians)
    
    params_path = os.path.join(args.save_dir, f"params_lr{args.lr}.pkl")
    hessians_path = os.path.join(args.save_dir, f"hessians_lr{args.lr}.pkl")
    
    with open(params_path, 'wb') as f:
        pickle.dump(params_tensor, f)
    
    with open(hessians_path, 'wb') as f:
        pickle.dump(hessians_tensor, f)
    
    # Метаданные
    metadata = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'sgd_epochs': args.sgd_epochs,
        'gd_epochs': args.gd_epochs,
        'sgd_log_epochs': args.sgd_log_epochs,
        'stage_transitions': stage_transitions,
        'all_train_losses': all_train_losses,
        'all_val_losses': all_val_losses,
        'all_val_accuracies': all_val_accuracies,
        'params_shape': params_tensor.shape,
        'hessians_shape': hessians_tensor.shape,
        'seed': args.seed,
        'sample_size': SAMPLE_SIZE,
        'dataset': 'similar_mnist'
    }
    
    metadata_path = os.path.join(args.save_dir, f"metadata_lr{args.lr}.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n💾 Модель сохранена: {model_path}")
    print(f"💾 Параметры сохранены: {params_path}")
    print(f"   📏 Размер: {params_tensor.shape} ({params_tensor.nbytes / 1024 / 1024:.1f} MB)")
    print(f"💾 Гессианы сохранены: {hessians_path}")
    print(f"   📏 Размер: {hessians_tensor.shape} ({hessians_tensor.nbytes / 1024 / 1024:.1f} MB)")
    print(f"💾 Метаданные сохранены: {metadata_path}")
    print(f"📈 График сохранен: {plot_path}")
    
    print(f"\n📊 Сводка сохраненных данных:")
    print(f"   🔢 Батчей с гессианами: {len(hessians)}")
    print(f"   🔢 Батчей с параметрами: {len(params_vectors)}")
    print(f"   📐 Размерность гессиана: {hessians_tensor.shape[1]}×{hessians_tensor.shape[2]}")
    print(f"   📐 Размерность параметров: {params_tensor.shape[1]}")
    
    if all_val_accuracies:
        print(f"🎯 Финальная точность: {all_val_accuracies[-1]:.2f}%")

if __name__ == "__main__":
    main()