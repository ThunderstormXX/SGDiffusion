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
    parser = argparse.ArgumentParser(description='Continue training and log hessians')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--post-plateau-steps', type=int, default=500, help='Steps after plateau')
    parser.add_argument('--save-dir', type=str, default='data/checkpoints/exp16', help='Save directory')
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

def main():
    args = parse_args()
    
    # ======== Фиксация случайности ========
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # ======== Настройки ========
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAMPLE_SIZE = 386
    
    # ======== Загрузка данных ========
    train_dataset, _, train_loader, _ = MNIST(
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
    
    print(f"\n📊 Конфигурация эксперимента:")
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
    losses = []
    data_iter = iter(train_loader)
    
    for step in tqdm(range(args.post_plateau_steps), desc="Post-plateau steps"):
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
        losses.append(loss.item())
        
        # Вычисляем гессиан
        hessian = compute_hessian(model, images, labels, criterion, DEVICE)
        hessians.append(hessian)
        
        # Делаем шаг оптимизации
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
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
        'losses': losses,
        'params_shape': params_tensor.shape,
        'hessians_shape': hessians_tensor.shape,
        'seed': args.seed
    }
    
    metadata_path = os.path.join(args.save_dir, f"metadata_lr{args.lr}.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n💾 Параметры сохранены: {params_path} (shape: {params_tensor.shape})")
    print(f"💾 Гессианы сохранены: {hessians_path} (shape: {hessians_tensor.shape})")
    print(f"💾 Метаданные сохранены: {metadata_path}")

if __name__ == "__main__":
    main()