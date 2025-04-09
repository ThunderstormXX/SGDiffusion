

import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import math


def init_data(n_samples=1000, loc1 = [-3, -1], loc2 = [1, 3] ,sigma1=4, sigma2 = 3, seed=42, batch_size=1, shuffle=True, device='cpu', full_loader=False):
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Генерация данных
    class_1 = np.random.normal(loc=loc1, scale=sigma1, size=(n_samples, 2))
    class_2 = np.random.normal(loc=loc2, scale=sigma2, size=(n_samples, 2))

    data = np.vstack((class_1, class_2))
    labels = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    # DataLoader
    dataset = TensorDataset(data_tensor, labels_tensor)
    # === Полный DataLoader (для градиентов) и батчевый (для обучения) ===

    batch_size = 1
    full_loader = DataLoader(TensorDataset(data_tensor, labels_tensor), batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(TensorDataset(data_tensor, labels_tensor), batch_size=batch_size, shuffle=True)
    
    return dataset, full_loader, train_loader, data, data_tensor, labels


def loss_visualise(loss_trajectory):
    # === График сходимости ===
    plt.figure(figsize=(8, 5))
    plt.plot(loss_trajectory, label="Loss")
    plt.xlabel("SGD Step")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Loss Convergence During Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def prediction_visualise(data , data_tensor, model, device):
    # === Визуализация предсказаний модели ===
    model.eval()
    with torch.no_grad():
        all_preds = model(data_tensor.to(device)).cpu().numpy().flatten()

    # Предсказанные метки
    predicted_labels = (all_preds >= 0.5).astype(int)

    # Отображение точек с цветом по предсказанному классу
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='coolwarm', alpha=0.6, edgecolors='k')
    plt.title("Model Predictions After Training")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualise_simple_model(data, data_tensor, labels, model, device):
    """
    Визуализация предсказаний модели, разделяющей плоскости и расчет accuracy.
    
    data: numpy array, исходные данные (для отображения точек)
    data_tensor: torch.Tensor, данные в формате тензора для модели
    labels: numpy array или torch.Tensor, истинные метки классов (для расчета accuracy)
    model: обученная модель
    device: устройство (CPU или GPU)
    """
    model.eval()
    with torch.no_grad():
        # Предсказания модели
        all_preds = model(data_tensor.to(device)).cpu().numpy().flatten()

    # Предсказанные метки
    predicted_labels = (all_preds >= 0.5).astype(int)

    # Вычисление точности
    true_labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    accuracy = (predicted_labels == true_labels).mean()
    # print(f"Accuracy: {accuracy * 100:.2f}%")

    # === Построение разделяющей линии ===
    # Получение весов и смещения из линейного слоя
    weights = model.linear.weight.data.cpu().numpy().flatten()  # w1, w2
    bias = model.linear.bias.data.cpu().numpy()[0]  # b

    # Уравнение разделяющей линии: x2 = -(w1/w2) * x1 - (b/w2)
    x1_min, x1_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    x2_min, x2_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    x1 = np.linspace(x1_min, x1_max, 100)
    if weights[1] != 0:  # Чтобы избежать деления на ноль
        x2 = -(weights[0] / weights[1]) * x1 - (bias / weights[1])
    else:
        x2 = np.full_like(x1, -bias / weights[0])  # Вертикальная линия

    # === Визуализация данных и разделяющей линии ===
    plt.figure(figsize=(8, 6))
    # Данные с истинными метками
    plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap='coolwarm', alpha=0.6, edgecolors='k', label='True Labels')
    # Разделяющая линия
    plt.plot(x1, x2, color='black', linestyle='--', label='Decision Boundary')
    plt.title(f"Model Predictions: accuracy = {accuracy}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_weight_and_bias_trajectory_with_gradient_stats(base_folder, num_steps):
    param_trajectories = {}  # имя параметра -> список весов и смещений (тензоры)
    grad_means = {}          # имя параметра -> список средних градиентов
    grad_stds = {}           # имя параметра -> список std градиентов

    for step in range(num_steps):
        folder = os.path.join(base_folder, f"trajectory_step_{step}")
        model_path = os.path.join(folder, "model.pt")
        grad_files = sorted([f for f in os.listdir(folder) if f.startswith("gradients_") and f.endswith(".pt")])

        if not os.path.exists(model_path) or not grad_files:
            continue

        # Загружаем веса модели
        state_dict = torch.load(model_path)
        for name, weight in state_dict.items():
            weight_flat = weight.cpu().flatten()
            param_trajectories.setdefault(name, []).append(weight_flat)

        # Собираем градиенты
        grad_accum = {}
        for gf in grad_files:
            grads = torch.load(os.path.join(folder, gf))
            for name, g in grads.items():
                g_flat = g.cpu().flatten()
                grad_accum.setdefault(name, []).append(g_flat)

        for name in grad_accum:
            grads_tensor = torch.stack(grad_accum[name])  # (num_batches, num_params)
            grad_mean = grads_tensor.mean(dim=0)
            grad_std = grads_tensor.std(dim=0)
            grad_means.setdefault(name, []).append(grad_mean)
            grad_stds.setdefault(name, []).append(grad_std)

    # Визуализация
    plt.figure(figsize=(12, 6))
    for name in param_trajectories:
        weights = torch.stack(param_trajectories[name])  # (steps, num_params)
        means = torch.stack(grad_means.get(name, []))
        stds = torch.stack(grad_stds.get(name, []))

        for i in range(weights.shape[1]):  # для каждого веса и bias
            y = weights[:, i].numpy()
            x = list(range(len(y)))

            # Строим график для каждого веса и смещения
            plt.plot(x, y, label=f"{name}[{i}]")

            # Добавляем "трубки" стандартного отклонения градиентов
            if means.shape[0] > i:
                mean_g = means[:, i].numpy()
                std_g = stds[:, i].numpy()

                plt.errorbar(
                    x, y,
                    yerr=std_g,
                    fmt='o', markersize=2,
                    ecolor='gray', alpha=0.3,
                    capsize=3
                )

    # Настройка графика
    plt.title("Trajectory of Weights and Biases with Gradient Stats")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def visualize_gradient_histograms_at_steps(base_folder, num_steps, batch_size=1):
    grad_histograms = {}  # имя параметра -> гистограмма градиентов на разных шагах

    # Собираем статистику
    for step in range(num_steps):
        folder = os.path.join(base_folder, f"trajectory_step_{step}")
        model_path = os.path.join(folder, "model.pt")
        grad_files = sorted([f for f in os.listdir(folder) if f.startswith("gradients_") and f.endswith(".pt")])

        if not os.path.exists(model_path) or not grad_files:
            continue

        # Загружаем градиенты
        grad_accum = {}
        for gf in grad_files:
            grads = torch.load(os.path.join(folder, gf))
            for name, g in grads.items():
                g_flat = g.cpu().flatten()
                grad_accum.setdefault(name, []).append(g_flat)

        for name in grad_accum:
            grads_tensor = torch.stack(grad_accum[name])  # (num_batches, num_params)

            # Усреднение по батчам
            grad_means = []
            for i in range(0, grads_tensor.shape[0], batch_size):
                batch_grads = grads_tensor[i:i+batch_size]
                batch_mean = batch_grads.mean(dim=0)
                grad_means.append(batch_mean)
            
            # Собираем усредненные градиенты
            grad_histograms.setdefault(name, []).append(torch.stack(grad_means))

    # Визуализация: для каждого веса/смещения строим гистограмму
    for name, grad_data in grad_histograms.items():
        for i in range(grad_data[0].shape[1]):  # Для каждого веса или смещения
            # Гистограммы на 0%, 25%, 50%, 75%, 100% итерациях
            steps = [0, int(0.25 * num_steps), int(0.5 * num_steps), int(0.75 * num_steps), num_steps - 1]
            grads_at_steps = [grad_data[step][:, i].cpu().numpy() for step in steps]

            plt.figure(figsize=(8, 6))
            plt.title(f"Gradient Histogram for {name}[{i}]")
            plt.xlabel("Gradient Value")
            plt.ylabel("Frequency")

            for j, step in enumerate(steps):
                plt.hist(grads_at_steps[j], bins=20, alpha=0.6, label=f"Step {step} ({int(100 * (step / (num_steps - 1)))}%)")
            
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

def alpha_estimator(m, X):
    """
    Оценка параметра α согласно Corollary 2.4 из Mohammadi (2014).
    """
    N = len(X)
    n = N // m  # Количество групп
    Y = torch.sum(X.view(n, m, -1), dim=1)  # Группировка и суммирование

    eps = np.finfo(float).eps  # Маленькое число для защиты от логарифма 0
    Y_log_norm = torch.log(Y.norm(dim=1) + eps).mean()
    X_log_norm = torch.log(X.norm(dim=1) + eps).mean()
    
    diff = (Y_log_norm - X_log_norm) / math.log(m)
    
    return float(np.clip(1 / diff, a_min = 0 , a_max = 2))

def alpha_estimator2(m, k, X):
    """
    Оценка параметра α согласно Corollary 2.2 из Mohammadi (2014).
    """
    N = len(X)
    n = int(N / m)  # Размер группы
    Y = torch.sum(X.view(n, m, -1), 1)
    eps = np.spacing(1)
    
    Y_log_norm = torch.log(Y.norm(dim=1) + eps)
    X_log_norm = torch.log(X.norm(dim=1) + eps)

    # Выбираем k-й элемент после сортировки
    Yk = torch.sort(Y_log_norm)[0][k - 1]
    Xk = torch.sort(X_log_norm)[0][m * k - 1]
    
    diff = (Yk - Xk) / math.log(m)
    
    return float(np.clip(1 / diff, a_min = 0 , a_max = 2))

def visualize_alpha_over_steps(base_folder, num_steps, batch_size=1, estimate_alpha_fn=None):
    assert estimate_alpha_fn is not None, "Нужно передать функцию оценки альфы!"

    alpha_values = {}  # name -> [alpha_t0, alpha_t1, ..., alpha_tN]
    grad_values = {}

    for step in range(num_steps):
        folder = os.path.join(base_folder, f"trajectory_step_{step}")
        model_path = os.path.join(folder, "model.pt")
        grad_files = sorted([f for f in os.listdir(folder) if f.startswith("gradients_") and f.endswith(".pt")])

        if not os.path.exists(model_path) or not grad_files:
            continue

        # Собираем градиенты
        grad_accum = {}
        for gf in grad_files:
            grads = torch.load(os.path.join(folder, gf))
            for name, g in grads.items():
                g_flat = g.cpu().flatten()
                grad_accum.setdefault(name, []).append(g_flat)

        # Учет батчей и оценка альф
        for name, grads_list in grad_accum.items():
            grads_tensor = torch.stack(grads_list)  # (num_batches, num_params)

            # Группируем по батчам
            grad_means = []
            for i in range(0, grads_tensor.shape[0], batch_size):
                batch = grads_tensor[i:i+batch_size]
                batch_mean = batch.mean(dim=0)
                grad_means.append(batch_mean)

            grad_means_tensor = torch.stack(grad_means)  # (num_effective_batches, num_params)

            # Оценка альфы по каждому параметру отдельно
            for i in range(grad_means_tensor.shape[1]):
                grad_series = grad_means_tensor[:, i].numpy()
                alpha = estimate_alpha_fn(grad_series)
                alpha_values.setdefault((name, i), []).append(alpha)
                grad_values.setdefault((name, i), []).append(grad_series)

    # Визуализация
    for (name, idx), alpha_series in alpha_values.items():
        plt.figure(figsize=(8, 4))
        plt.plot(alpha_series, marker='o')
        plt.ylim(0.5, 2.0)
        plt.title(f"Estimated alpha for {name}[{idx}]")
        plt.xlabel("Training Step")
        plt.ylabel("Estimated α")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    return alpha_values, grad_values