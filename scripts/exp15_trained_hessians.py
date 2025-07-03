import os
import sys
import torch
import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
import pickle

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
BATCH_SIZES = [1, 4, 8, 16, 32, 64]
NUM_ITERATIONS = 1000
SAVE_DIR = "data/checkpoints/exp15"
TRAINED_MODEL_PATH = os.path.join(SAVE_DIR, "trained_model.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

# ======== Загрузка данных ========
train_dataset, test_dataset, _, _ = MNIST(batch_size=64)

# ======== Модель (точно такая же, как при обучении) ========
model = FlexibleMLP(
    hidden_dim=8,
    num_hidden_layers=1,
    input_downsample=6
).to(DEVICE)

# ======== Загрузка обученной модели ========
if os.path.exists(TRAINED_MODEL_PATH):
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE))
    print("✅ Загружена обученная модель:", TRAINED_MODEL_PATH)
else:
    raise FileNotFoundError(f"❌ Обученная модель не найдена по пути: {TRAINED_MODEL_PATH}")

criterion = nn.CrossEntropyLoss()
print(f"Модель содержит {sum(p.numel() for p in model.parameters())} параметров")

# ======== Функция для вычисления гессиана ========
def compute_hessian(model, data_loader, criterion, device, batch_size, num_iterations):
    model.eval()
    hessians = []
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    data_iter = iter(data_loader)

    for iteration in tqdm(range(num_iterations), desc=f"Batch size {batch_size}"):
        try:
            batch_data = []
            batch_labels = []
            for _ in range(batch_size):
                try:
                    images, labels = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    images, labels = next(data_iter)
                batch_data.append(images[0:1])
                batch_labels.append(labels[0:1])

            batch_images = torch.cat(batch_data, dim=0).to(device)
            batch_labels = torch.cat(batch_labels, dim=0).to(device)

            model.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
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

            hessians.append(hessian.cpu().numpy())

        except Exception as e:
            print(f"[!] Ошибка на итерации {iteration}: {e}")
            continue

    return hessians

# ======== Сбор гессианов ========
print("=== Сбор гессианов после обучения ===")

all_hessians = {}

for batch_size in BATCH_SIZES:
    print(f"\n--- Обработка batch_size = {batch_size} ---")

    custom_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,  # Батчи собираем вручную
        shuffle=True
    )

    hessians = compute_hessian(model, custom_loader, criterion, DEVICE, batch_size, NUM_ITERATIONS)
    all_hessians[batch_size] = hessians

    print(f"[✓] Собрано {len(hessians)} гессианов для batch_size = {batch_size}")

# ======== Сохранение гессианов ========
hessians_path = os.path.join(SAVE_DIR, "trained_hessians.pkl")
with open(hessians_path, 'wb') as f:
    pickle.dump(all_hessians, f)

print(f"\n💾 Гессианы после обучения сохранены: {hessians_path}")

# ======== Краткая статистика ========
print("\n=== Статистика по гессианам ===")
for batch_size, hessians in all_hessians.items():
    if hessians:
        print(f"Batch size {batch_size}: {len(hessians)} гессианов, размер: {hessians[0].shape}")
