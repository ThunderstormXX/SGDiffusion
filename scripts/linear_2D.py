import torch
import sys
import os
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import MNIST, train, LOGGER
from src.model import CNN
from src.optimizer import NoisySGD

import torch.nn as nn

DEVICE = 'mps'

# Папка для сохранения моделей
CHECKPOINT_DIR = "data/checkpoints/exp12/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

import torch.optim as optim
import pickle

# Фиксируем случайность
torch.manual_seed(42)
np.random.seed(42)

# Генерация данных
n_samples = 1000
sigma = 1.4
class_1 = np.random.normal(loc=[-1, -1], scale=sigma, size=(n_samples, 2))
class_2 = np.random.normal(loc=[1, 1], scale=sigma, size=(n_samples, 2))

data = np.vstack((class_1, class_2))
labels = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

# Определение модели
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(2, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Инициализация модели, функции потерь и оптимизатора
model = SimpleModel()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Параметры обучения
batch_size = 1
epochs = 10
noise_distribution = []
model_snapshots = []

data_loader = torch.utils.data.DataLoader(
    list(zip(data_tensor, labels_tensor)), batch_size=batch_size, shuffle=True
)

# Обучение модели
for epoch in range(epochs):
    # Вычисление полного градиента
    model.zero_grad()
    full_output = model(data_tensor)
    full_loss = criterion(full_output, labels_tensor)
    full_loss.backward()
    full_grad = {name: param.grad.clone() for name, param in model.named_parameters()}
    
    # Обучение с mini-batch SGD
    epoch_loss = 0.0
    for batch_data, batch_labels in data_loader:
        model.zero_grad()
        output = model(batch_data)
        loss = criterion(output, batch_labels)
        loss.backward()
        
        # Сохранение стохастического шума градиента
        noise_snapshot = {}
        for name, param in model.named_parameters():
            noise_snapshot[name] = param.grad.clone() - full_grad[name]
        noise_distribution.append(noise_snapshot)
        
        # Обновление параметров модели
        optimizer.step()
        epoch_loss += loss.item()
    
    # Сохранение параметров модели
    model_snapshots.append({name: param.clone() for name, param in model.named_parameters()})
    
    # Вывод train_loss
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/len(data_loader):.4f}")

# Создание директории для сохранения данных
save_path = CHECKPOINT_DIR
os.makedirs(save_path, exist_ok=True)

# Сохранение статистики
with open(os.path.join(save_path, "noise_distribution.pkl"), "wb") as f:
    pickle.dump(noise_distribution, f)

with open(os.path.join(save_path, "model_snapshots.pkl"), "wb") as f:
    pickle.dump(model_snapshots, f)

print("Обучение завершено! Статистика сохранена в data/exp12/")
