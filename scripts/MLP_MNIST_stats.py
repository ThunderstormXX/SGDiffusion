import sys
import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import FlexibleMLP
from src.utils import MNIST

DEVICE = 'mps'  # 'cuda' / 'cpu' тоже подойдут
CHECKPOINT_ROOT = "data/checkpoints/exp13"
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

def train_epoch(model, optimizer, train_loader, criterion, device,
                save_dir, step_counter, eval_stats, eval_on_each_step, epoch_num):
    model.train()
    loader = tqdm(train_loader, desc=f"Epoch {epoch_num} - Training", leave=False)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Обновление заголовка tqdm
        loader.set_description(f"Epoch {epoch_num} - Loss: {loss.item():.4f}")

        # Сохранение весов
        torch.save(model.state_dict(), os.path.join(save_dir, f"step_{step_counter[0]}.pt"))

        # Опциональный evaluation
        if eval_on_each_step:
            train_loss, train_acc = evaluate_loss_acc(train_loader, model, criterion, device)
            eval_stats.append((train_loss, train_acc))

        step_counter[0] += 1

def evaluate_loss_acc(data_loader, model, criterion, device):
    model.eval()
    total_loss, total_correct = 0.0, 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples

# Гиперпараметры
num_epochs = 10
batch_size = 32
eval_on_each_step = False  # ⇽ флаг для включения/отключения evaluation после каждого шага
lr_list = np.logspace(-4, 0, 10)

# Фиксация случайности
torch.manual_seed(228)
np.random.seed(228)
random.seed(228)

# Данные
train_dataset, test_dataset, train_loader, test_loader = MNIST(batch_size=batch_size)

for lr in lr_list:
    print(f"\n=== Training with learning rate: {lr:.5f} ===")

    model = FlexibleMLP(
        hidden_dim=8,
        num_hidden_layers=1,
        dropout_list=[0.0],
        use_relu_list=[True]
    ).to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Создание директории для этого LR
    lr_dir = os.path.join(CHECKPOINT_ROOT, f"lr_{lr:.0e}")
    os.makedirs(lr_dir, exist_ok=True)

    step_counter = [0]
    eval_stats = []

    for epoch in range(1, num_epochs + 1):
        train_epoch(
            model, optimizer, train_loader, criterion, DEVICE,
            lr_dir, step_counter, eval_stats, eval_on_each_step, epoch
        )

        # Epoch-wise evaluation
        train_loss, train_acc = evaluate_loss_acc(train_loader, model, criterion, DEVICE)
        val_loss, val_acc = evaluate_loss_acc(test_loader, model, criterion, DEVICE)

        print(f"Epoch [{epoch}/{num_epochs}], "
              f"Loss (train/test): {train_loss:.4f}/{val_loss:.4f}, "
              f"Acc (train/test): {train_acc:.4f}/{val_acc:.4f}")

    # Save evaluation log
    if eval_on_each_step:
        eval_array = np.array(eval_stats)
        np.save(os.path.join(lr_dir, "train_eval_per_step.npy"), eval_array)
