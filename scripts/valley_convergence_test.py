import os
import sys
import torch
import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.decomposition import PCA

# ======== Локальные модули ========
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import MNIST
from src.model import FlexibleMLP

# ======== Фиксация случайности ========
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ======== Настройки ========
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
LR = 0.01
INIT_EPOCHS = 5  # Эпохи для начальной инициализации
CONVERGENCE_EPOCHS = 15  # Эпохи для проверки сходимости
PCA_COMPONENTS = 50  # Число компонент PCA для ускорения вычислений
SAVE_DIR = "data/checkpoints/valley_test"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======== Загрузка данных ========
train_dataset, test_dataset, train_loader, test_loader = MNIST(batch_size=BATCH_SIZE)

# ======== Функция для обучения модели и сбора траектории ========
def train_and_collect_trajectory(model, optimizer, epochs, name_prefix):
    model.train()
    trajectory = []
    losses = []
    
    for epoch in range(epochs):
        loader = tqdm(train_loader, desc=f"{name_prefix} Epoch {epoch+1}", leave=False)
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loader.set_description(f"{name_prefix} Epoch {epoch+1} | Loss: {loss.item():.4f}")
            
            # Векторизация параметров
            param_vector = torch.cat([p.data.view(-1) for p in model.parameters()]).cpu().numpy()
            trajectory.append(param_vector)
            losses.append(loss.item())
    
    return np.stack(trajectory), np.array(losses)

# ======== Функция для вычисления пересечения многомерных гауссиан ========
def gaussian_overlap(mean1, mean2, std1, std2):
    """
    Вычисляет коэффициент перекрытия двух многомерных гауссиан
    с диагональными ковариационными матрицами (только стандартные отклонения).
    """
    # Вычисляем расстояние Бхаттачария между двумя гауссианами
    # Для диагональных ковариационных матриц формула упрощается
    
    # Вычисляем средние стандартные отклонения
    var_avg = (std1**2 + std2**2) / 2
    
    # Вычисляем первый член расстояния Бхаттачария
    # Для диагональных матриц можно использовать поэлементные операции
    term1 = 1/8 * np.sum(((mean1 - mean2)**2) / var_avg)
    
    # Вычисляем второй член
    term2 = 0.5 * np.sum(np.log(var_avg) - 0.5 * (np.log(std1**2) + np.log(std2**2)))
    
    # Расстояние Бхаттачария
    db = term1 + term2
    
    # Коэффициент Бхаттачария (мера перекрытия)
    bc = np.exp(-db)
    
    # Возвращаем коэффициент перекрытия (0 - нет перекрытия, 1 - полное перекрытие)
    return bc

# ======== Функция для 3D визуализации гауссиан с помощью PCA ========
def visualize_gaussians_3d(trajectory1, trajectory2, save_path):
    """
    Визуализирует две траектории в 3D с помощью PCA
    """
    # Объединяем траектории для PCA
    combined_data = np.vstack([trajectory1, trajectory2])
    
    # Применяем PCA для снижения размерности до 3
    pca = PCA(n_components=3)
    data_3d = pca.fit_transform(combined_data)
    
    # Разделяем данные обратно на две траектории
    traj1_3d = data_3d[:len(trajectory1)]
    traj2_3d = data_3d[len(trajectory1):]
    
    # Создаем 3D график
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Визуализируем траектории
    ax.scatter(traj1_3d[:, 0], traj1_3d[:, 1], traj1_3d[:, 2], c='blue', alpha=0.6, label='Model 1')
    ax.scatter(traj2_3d[:, 0], traj2_3d[:, 1], traj2_3d[:, 2], c='red', alpha=0.6, label='Model 2')
    
    # Добавляем подписи
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D PCA визуализация траекторий')
    ax.legend()
    
    # Сохраняем график
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Возвращаем объясненную дисперсию
    return pca.explained_variance_ratio_

# ======== Основная функция ========
def main():
    # Создаем две модели с разными инициализациями
    model1 = FlexibleMLP(hidden_dim=16, num_hidden_layers=1).to(DEVICE)
    model2 = FlexibleMLP(hidden_dim=16, num_hidden_layers=1).to(DEVICE)
    
    # Разные инициализации
    torch.manual_seed(SEED)
    model1.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    torch.manual_seed(SEED + 100)
    model2.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    
    # Оптимизаторы и функция потерь
    global criterion
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=LR)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=LR)
    
    # Начальное обучение (5 эпох)
    print("Начальное обучение моделей (5 эпох)...")
    trajectory1_init, losses1_init = train_and_collect_trajectory(model1, optimizer1, INIT_EPOCHS, "Model1")
    trajectory2_init, losses2_init = train_and_collect_trajectory(model2, optimizer2, INIT_EPOCHS, "Model2")
    
    # Сохраняем состояния моделей после 5 эпох
    model1_state = {k: v.clone() for k, v in model1.state_dict().items()}
    model2_state = {k: v.clone() for k, v in model2.state_dict().items()}
    
    # Продолжаем обучение для проверки сходимости
    print("\nПроверка сходимости моделей...")
    trajectory1_conv, losses1_conv = train_and_collect_trajectory(model1, optimizer1, CONVERGENCE_EPOCHS, "Model1")
    trajectory2_conv, losses2_conv = train_and_collect_trajectory(model2, optimizer2, CONVERGENCE_EPOCHS, "Model2")
    
    # Вычисляем статистики траекторий
    mean1 = np.mean(trajectory1_conv, axis=0)
    mean2 = np.mean(trajectory2_conv, axis=0)
    std1 = np.std(trajectory1_conv, axis=0)
    std2 = np.std(trajectory2_conv, axis=0)
    
    # Вычисляем пересечение гауссиан
    print("\nВычисление пересечения гауссиан...")
    overlap = gaussian_overlap(mean1, mean2, std1, std2)
    print(f"Объем пересечения гауссиан: {overlap:.4f}")
    
    # Интерпретация результата
    if overlap > 0.5:
        print("Модели сходятся к одной долине (высокое пересечение гауссиан)")
    else:
        print("Модели сходятся к разным долинам (низкое пересечение гауссиан)")
    
    # Визуализация траекторий в 3D с помощью PCA
    print("\nСоздание 3D визуализации траекторий...")
    
    # Визуализация траекторий сходимости
    var_explained = visualize_gaussians_3d(
        trajectory1_conv, 
        trajectory2_conv, 
        os.path.join(SAVE_DIR, "pca_3d_convergence.png")
    )
    print(f"Объясненная дисперсия первых 3 компонент: {var_explained}")
    
    # Визуализация всех траекторий (инициализация + сходимость)
    full_traj1 = np.concatenate([trajectory1_init, trajectory1_conv])
    full_traj2 = np.concatenate([trajectory2_init, trajectory2_conv])
    visualize_gaussians_3d(
        full_traj1, 
        full_traj2, 
        os.path.join(SAVE_DIR, "pca_3d_full.png")
    )
    
    # Визуализация потерь с выделением фаз обучения
    plt.figure(figsize=(12, 6))
    
    # Объединяем потери для непрерывности графика
    losses1 = np.concatenate([losses1_init, losses1_conv])
    losses2 = np.concatenate([losses2_init, losses2_conv])
    
    # Создаем массив шагов
    steps = np.arange(len(losses1))
    
    # Разделяем на фазы
    init_steps = steps[:len(losses1_init)]
    conv_steps = steps[len(losses1_init):]
    
    # Рисуем фазу инициализации
    plt.plot(init_steps, losses1[:len(losses1_init)], 'b-', label='Model 1 (Инициализация)')
    plt.plot(init_steps, losses2[:len(losses2_init)], 'r-', label='Model 2 (Инициализация)')
    
    # Рисуем фазу сходимости
    plt.plot(conv_steps, losses1[len(losses1_init):], 'b--', label='Model 1 (Сходимость)')
    plt.plot(conv_steps, losses2[len(losses2_init):], 'r--', label='Model 2 (Сходимость)')
    
    # Добавляем заливку для фазы сходимости
    plt.axvspan(len(losses1_init), len(losses1), alpha=0.2, color='gray', label='Фаза сходимости')
    
    # Добавляем вертикальную линию разделения
    plt.axvline(x=len(losses1_init), color='r', linestyle='--', label='Контрольная точка (5 эпох)')
    
    plt.xlabel('Шаги')
    plt.ylabel('Потери')
    plt.title('Кривые потерь для двух моделей')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "loss_comparison.png"))
    plt.close()
    
    # Визуализация расстояния между моделями
    distances = []
    for i in range(len(trajectory1_conv)):
        dist = np.linalg.norm(trajectory1_conv[i] - trajectory2_conv[i])
        distances.append(dist)
    
    plt.figure(figsize=(10, 5))
    plt.plot(distances)
    plt.xlabel('Шаги')
    plt.ylabel('Евклидово расстояние')
    plt.title('Расстояние между параметрами моделей')
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "parameter_distance.png"))
    plt.close()
    
    # Сохраняем результаты
    np.save(os.path.join(SAVE_DIR, "trajectory1.npy"), np.concatenate([trajectory1_init, trajectory1_conv]))
    np.save(os.path.join(SAVE_DIR, "trajectory2.npy"), np.concatenate([trajectory2_init, trajectory2_conv]))
    np.save(os.path.join(SAVE_DIR, "losses1.npy"), losses1)
    np.save(os.path.join(SAVE_DIR, "losses2.npy"), losses2)
    
    # Оценка точности моделей
    accuracy1 = evaluate_accuracy(model1, test_loader, DEVICE)
    accuracy2 = evaluate_accuracy(model2, test_loader, DEVICE)
    
    print(f"\nТочность Model1: {accuracy1 * 100:.2f}%")
    print(f"Точность Model2: {accuracy2 * 100:.2f}%")

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

if __name__ == "__main__":
    main()