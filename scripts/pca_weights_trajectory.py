import os
import sys
import torch
import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt  # добавлено для графика

# ======== Локальные модули ========
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import MNIST
from src.model import FlexibleMLP

# ======== Фиксация случайности ========
torch.manual_seed(228)
np.random.seed(228)
random.seed(228)

# ======== Настройки ========
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
LR = 0.01
NUM_EPOCHS = 30
SAVE_VECTOR_DIR = "data/checkpoints/exp14/vectorized_params"
os.makedirs(SAVE_VECTOR_DIR, exist_ok=True)

# ======== Загрузка данных ========
train_dataset, test_dataset, train_loader, test_loader = MNIST(batch_size=BATCH_SIZE)

# ======== Модель, оптимизатор, лосс ========
model = FlexibleMLP(hidden_dim=16, num_hidden_layers=1).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ======== Обучение ========
model.train()
all_vectors = []
losses_per_step = []

for epoch in range(NUM_EPOCHS):
    loader = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loader.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

        # Векторизация параметров
        param_vector = torch.cat([p.data.view(-1) for p in model.parameters()]).cpu().numpy()
        all_vectors.append(param_vector)

        # Сохраняем текущий лосс
        losses_per_step.append(loss.item())

# ======== Сохранение параметров ========
all_vectors_np = np.stack(all_vectors)
np.save(os.path.join(SAVE_VECTOR_DIR, "params_matrix.npy"), all_vectors_np)
print(f"Saved parameter matrix of shape {all_vectors_np.shape} to {SAVE_VECTOR_DIR}")

# ======== Сохранение графика лосса ========
plt.figure(figsize=(10, 5))
plt.plot(losses_per_step, label='Loss per iteration')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
loss_plot_path = os.path.join(SAVE_VECTOR_DIR, "loss_curve.png")
plt.savefig(loss_plot_path)
plt.close()
print(f"Saved loss curve plot to {loss_plot_path}")

# ======== Evaluation на тесте ========
def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

accuracy = evaluate_accuracy(model, test_loader, DEVICE)
print(f"\n✅ Final test accuracy: {accuracy * 100:.2f}%")
