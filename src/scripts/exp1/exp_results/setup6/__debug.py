#!/usr/bin/env python3
import os, sys
import torch, torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import FlexibleMLP
from src.utils import MNIST, load_similar_mnist_data

# --- конфиг ---
device = torch.device("cpu")
dataset_train = "mnist"
batch_size = 64
sample_size = 6000
seed = 42
checkpoint = "src/scripts/exp1/exp_results/setup6/initial_after_sgd_and_gd.pt"
n_trials = 100000   # сколько Монте-Карло повторов

# --- функции ---
def load_train_loader(dataset_train, batch_size, sample_size, seed):
    if dataset_train == "mnist":
        train_dataset, _, _, _ = MNIST(batch_size=batch_size, sample_size=sample_size)
    else:
        train_dataset, _, _, _ = load_similar_mnist_data(batch_size=batch_size, sample_size=sample_size)
    N = len(train_dataset)
    eff = (N // batch_size) * batch_size
    g = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(train_dataset, replacement=True, num_samples=eff, generator=g)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)
    return DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0), eff, train_dataset

def grad_vector(model, data, target, criterion):
    model.zero_grad(set_to_none=True)
    out = model(data)
    loss = criterion(out, target)
    loss.backward()
    return torch.nn.utils.parameters_to_vector([p.grad for p in model.parameters()]).detach().cpu()

# --- модель ---
model = FlexibleMLP(hidden_dim=8, num_hidden_layers=1, input_downsample=6).to(device)
model.load_state_dict(torch.load(checkpoint, map_location=device))
criterion = nn.CrossEntropyLoss()
import random
from typing import Tuple

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# --- загрузка лоадера/датасета один раз ---
train_loader, eff, train_dataset = load_train_loader(
    dataset_train=dataset_train,
    batch_size=batch_size,
    sample_size=sample_size,
    seed=seed
)

# Чтобы убрать стохастику дропаута/BN и сравнивать именно градиенты эмпирического риска
model.eval()

@torch.no_grad()
def _stack_dataset(ds) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(x)
        # y может быть int или тензор
        if torch.is_tensor(y):
            ys.append(int(y.item()))
        else:
            ys.append(int(y))
    X = torch.stack(xs, dim=0).to(device)
    Y = torch.tensor(ys, dtype=torch.long, device=device)
    return X, Y

def grad_vector(model, data, target, criterion):
    # как у тебя, но безопасно обрабатываем None-градиенты
    model.zero_grad(set_to_none=True)
    out = model(data)
    loss = criterion(out, target)
    loss.backward()
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p))
        else:
            grads.append(p.grad)
    return torch.nn.utils.parameters_to_vector(grads).detach().cpu()

# --- 1) Полный градиент GD одним большим батчем ---
full_X, full_y = _stack_dataset(train_dataset)
g_full = grad_vector(model, full_X, full_y, criterion)  # градиент среднего лосса по всему датасету

# --- 2) Усреднение стох. градиентов по многим батчам из одного и того же лоадера ---
sum_grad = torch.zeros_like(g_full)
loader_iter = iter(train_loader)

for t in range(n_trials):
    try:
        data, target = next(loader_iter)
    except StopIteration:
        # НЕ пересоздаём самплер; просто начинаем новый проход по уже созданному лоадеру
        loader_iter = iter(train_loader)
        data, target = next(loader_iter)

    data = data.to(device)
    target = target.to(device)
    g = grad_vector(model, data, target, criterion)
    sum_grad += g

g_mean = sum_grad / float(n_trials)

# --- 3) Сравнение норм ---
diff = g_mean - g_full
l2 = diff.norm().item()
rel = l2 / (g_full.norm().item() + 1e-12)
cos = torch.dot(g_mean, g_full) / (g_mean.norm() * g_full.norm() + 1e-12)

print(f"||g_mean - g_full||_2 = {l2:.6e}  (relative: {rel:.6%})")
print(f"cosine(g_mean, g_full) = {cos.item():.6f}")
print(f"||g_full||_2 = {g_full.norm().item():.6e},  ||g_mean||_2 = {g_mean.norm().item():.6e}")
