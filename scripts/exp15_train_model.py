import os
import sys
import torch
import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

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
BATCH_SIZE = 64
LR = 0.1
NUM_EPOCHS = 60000
SAMPLE_SIZE = 386
SAVE_DIR = "data/checkpoints/exp15"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======== Загрузка данных ========
train_dataset, test_dataset, train_loader, test_loader = MNIST(batch_size=BATCH_SIZE, sample_size = SAMPLE_SIZE)

# ======== Конфигурация модели ========
HIDDEN_DIM = 8
NUM_HIDDEN_LAYERS = 1
INPUT_DOWNSAMPLE = 6  # Размерность изображения после сжатия (7x7 = 49)

# ======== Загрузка начальной модели ========
model = FlexibleMLP(
    hidden_dim=HIDDEN_DIM,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    input_downsample=INPUT_DOWNSAMPLE
).to(DEVICE)

initial_model_path = os.path.join(SAVE_DIR, "initial_model.pth")
if os.path.exists(initial_model_path):
    model.load_state_dict(torch.load(initial_model_path))
    print("✅ Загружена начальная модель из:", initial_model_path)
else:
    print("⚠️  Начальная модель не найдена, используется случайная инициализация")

optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ======== Обучение ========
print("\n🚀 Начинаем обучение модели...")
model.train()

losses_per_epoch = []
losses_per_step = []

for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
    
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        epoch_losses.append(current_loss)
        losses_per_step.append(current_loss)

        loader.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {current_loss:.4f}")
    
    avg_epoch_loss = np.mean(epoch_losses)
    losses_per_epoch.append(avg_epoch_loss)
    print(f"📉 Epoch {epoch+1}: Avg Loss = {avg_epoch_loss:.4f}")

# ======== Оценка качества ========
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

train_accuracy = evaluate_accuracy(model, train_loader, DEVICE)
test_accuracy = evaluate_accuracy(model, test_loader, DEVICE)

print("\n✅ Обучение завершено!")
print(f"🎯 Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"🎯 Test Accuracy:  {test_accuracy * 100:.2f}%")
print(f"📉 Final Loss:     {losses_per_step[-1]:.6f}")

# ======== Сохранение обученной модели ========
trained_model_path = os.path.join(SAVE_DIR, "trained_model.pth")
torch.save(model.state_dict(), trained_model_path)

# ======== Сохранение траектории лосса ========
loss_data = {
    'losses_per_step': losses_per_step,
    'losses_per_epoch': losses_per_epoch,
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'config': {
        'hidden_dim': HIDDEN_DIM,
        'num_hidden_layers': NUM_HIDDEN_LAYERS,
        'input_downsample': INPUT_DOWNSAMPLE,
        'lr': LR,
        'batch_size': BATCH_SIZE,
        'epochs': NUM_EPOCHS,
    }
}

loss_path = os.path.join(SAVE_DIR, "training_loss.npy")
np.save(loss_path, loss_data)

# ======== Построение графика лосса ========
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses_per_step)
plt.title('Training Loss per Step')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(losses_per_epoch)
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)

plt.tight_layout()
loss_plot_path = os.path.join(SAVE_DIR, "training_loss_curve.png")
plt.savefig(loss_plot_path)
plt.close()

print(f"\n💾 Обученная модель сохранена:       {trained_model_path}")
print(f"💾 Данные о лоссе сохранены:         {loss_path}")
print(f"📈 График кривой лосса сохранён:     {loss_plot_path}")
