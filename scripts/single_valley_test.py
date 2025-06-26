import os
import sys
import torch
import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
ADAM_LR = 0.001  # Learning rate для Adam
SGD_LR = 0.0001  # Learning rate для SGD (в 10 раз меньше)
ADAM_EPOCHS = 10  # Эпохи для обучения с Adam до достижения долины
SGD_INIT_EPOCHS = 5  # Эпохи для начальной дивергенции с SGD
CONVERGENCE_EPOCHS = 15  # Эпохи для проверки сходимости с SGD
SAVE_DIR = "data/checkpoints/single_valley_test"
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
    # Вычисляем средние стандартные отклонения
    var_avg = (std1**2 + std2**2) / 2
    
    # Вычисляем первый член расстояния Бхаттачария
    term1 = 1/8 * np.sum(((mean1 - mean2)**2) / var_avg)
    
    # Вычисляем второй член
    term2 = 0.5 * np.sum(np.log(var_avg) - 0.5 * (np.log(std1**2) + np.log(std2**2)))
    
    # Расстояние Бхаттачария
    db = term1 + term2
    
    # Коэффициент Бхаттачария (мера перекрытия)
    bc = np.exp(-db)
    
    return bc

# ======== Функция для 3D визуализации гауссиан с помощью PCA ========
def visualize_gaussians_3d(trajectory1, trajectory2, save_path, title="3D PCA визуализация траекторий"):
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
    ax.scatter(traj1_3d[:, 0], traj1_3d[:, 1], traj1_3d[:, 2], c='blue', alpha=0.6, label='SGD 1')
    ax.scatter(traj2_3d[:, 0], traj2_3d[:, 1], traj2_3d[:, 2], c='red', alpha=0.6, label='SGD 2')
    
    # Добавляем подписи
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(title)
    ax.legend()
    
    # Сохраняем график
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Возвращаем объясненную дисперсию
    return pca.explained_variance_ratio_

# ======== Основная функция ========
def main():
    # Создаем базовую модель
    base_model = FlexibleMLP(hidden_dim=16, num_hidden_layers=1).to(DEVICE)
    
    # Инициализация
    torch.manual_seed(SEED)
    base_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    
    # Оптимизатор Adam для обучения до долины
    global criterion
    criterion = nn.CrossEntropyLoss()
    optimizer_adam = torch.optim.Adam(base_model.parameters(), lr=ADAM_LR)
    
    # Обучение с Adam до достижения долины
    print(f"Обучение базовой модели с Adam ({ADAM_EPOCHS} эпох)...")
    trajectory_adam, losses_adam = train_and_collect_trajectory(base_model, optimizer_adam, ADAM_EPOCHS, "Base (Adam)")
    
    # Сохраняем состояние модели после Adam
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    
    # Создаем две копии модели для SGD
    model1 = FlexibleMLP(hidden_dim=16, num_hidden_layers=1).to(DEVICE)
    model2 = FlexibleMLP(hidden_dim=16, num_hidden_layers=1).to(DEVICE)
    
    # Загружаем состояние базовой модели в обе копии
    model1.load_state_dict(base_state)
    model2.load_state_dict(base_state)
    
    # Создаем оптимизаторы SGD с разными seed для стохастичности
    torch.manual_seed(SEED)
    optimizer1_sgd = torch.optim.SGD(model1.parameters(), lr=SGD_LR)
    torch.manual_seed(SEED + 100)  # Разный seed для второго оптимизатора
    optimizer2_sgd = torch.optim.SGD(model2.parameters(), lr=SGD_LR)
    
    # Запускаем два SGD процесса из одной точки для начальной дивергенции
    print(f"\nЗапуск двух SGD процессов из одной точки ({SGD_INIT_EPOCHS} эпох)...")
    trajectory1_init, losses1_init = train_and_collect_trajectory(model1, optimizer1_sgd, SGD_INIT_EPOCHS, "SGD1 Init")
    trajectory2_init, losses2_init = train_and_collect_trajectory(model2, optimizer2_sgd, SGD_INIT_EPOCHS, "SGD2 Init")
    
    # Сохраняем состояния моделей после начальной дивергенции
    model1_state = {k: v.clone() for k, v in model1.state_dict().items()}
    model2_state = {k: v.clone() for k, v in model2.state_dict().items()}
    
    # Продолжаем обучение с SGD для проверки сходимости
    print(f"\nПроверка сходимости моделей с SGD ({CONVERGENCE_EPOCHS} эпох)...")
    trajectory1_conv, losses1_conv = train_and_collect_trajectory(model1, optimizer1_sgd, CONVERGENCE_EPOCHS, "SGD1 Conv")
    trajectory2_conv, losses2_conv = train_and_collect_trajectory(model2, optimizer2_sgd, CONVERGENCE_EPOCHS, "SGD2 Conv")
    
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
        print("Модели остались в одной долине (высокое пересечение гауссиан)")
    else:
        print("Модели разошлись в разные долины (низкое пересечение гауссиан)")
    
    # Визуализация траекторий в 3D с помощью PCA
    print("\nСоздание 3D визуализации траекторий...")
    
    # Визуализация траекторий сходимости
    var_explained = visualize_gaussians_3d(
        trajectory1_conv, 
        trajectory2_conv, 
        os.path.join(SAVE_DIR, "pca_3d_convergence.png"),
        "3D PCA визуализация траекторий сходимости"
    )
    print(f"Объясненная дисперсия первых 3 компонент: {var_explained}")
    
    # Визуализация всех SGD траекторий (инициализация + сходимость)
    full_traj1 = np.concatenate([trajectory1_init, trajectory1_conv])
    full_traj2 = np.concatenate([trajectory2_init, trajectory2_conv])
    visualize_gaussians_3d(
        full_traj1, 
        full_traj2, 
        os.path.join(SAVE_DIR, "pca_3d_full_sgd.png"),
        "3D PCA визуализация полных SGD траекторий"
    )
    
    # Визуализация всего процесса (Adam + SGD)
    # Добавляем последнюю точку Adam к началу обеих SGD траекторий
    adam_endpoint = trajectory_adam[-1].reshape(1, -1)
    full_process1 = np.vstack([adam_endpoint, full_traj1])
    full_process2 = np.vstack([adam_endpoint, full_traj2])
    visualize_gaussians_3d(
        full_process1,
        full_process2,
        os.path.join(SAVE_DIR, "pca_3d_full_process.png"),
        "3D PCA визуализация полного процесса (Adam → SGD)"
    )
    
    # Визуализация потерь
    plt.figure(figsize=(12, 6))
    
    # Создаем массив шагов
    steps_adam = np.arange(len(losses_adam))
    steps_sgd1 = np.arange(len(losses_adam), len(losses_adam) + len(losses1_init) + len(losses1_conv))
    steps_sgd2 = np.arange(len(losses_adam), len(losses_adam) + len(losses2_init) + len(losses2_conv))
    
    # Разделяем SGD на фазы
    steps_sgd1_init = steps_sgd1[:len(losses1_init)]
    steps_sgd1_conv = steps_sgd1[len(losses1_init):]
    steps_sgd2_init = steps_sgd2[:len(losses2_init)]
    steps_sgd2_conv = steps_sgd2[len(losses2_init):]
    
    # Рисуем фазу Adam
    plt.plot(steps_adam, losses_adam, 'g-', label='Adam (базовая модель)')
    
    # Рисуем фазу начальной дивергенции с SGD
    plt.plot(steps_sgd1_init, losses1_init, 'b--', label='SGD1 (дивергенция)')
    plt.plot(steps_sgd2_init, losses2_init, 'r--', label='SGD2 (дивергенция)')
    
    # Рисуем фазу сходимости с SGD
    plt.plot(steps_sgd1_conv, losses1_conv, 'b-', label='SGD1 (сходимость)')
    plt.plot(steps_sgd2_conv, losses2_conv, 'r-', label='SGD2 (сходимость)')
    
    # Добавляем заливку для разных фаз
    plt.axvspan(0, len(losses_adam), alpha=0.1, color='green', label='Фаза Adam')
    plt.axvspan(len(losses_adam), len(losses_adam) + len(losses1_init), alpha=0.1, color='orange', label='Фаза дивергенции SGD')
    plt.axvspan(len(losses_adam) + len(losses1_init), len(losses_adam) + len(losses1_init) + len(losses1_conv), 
                alpha=0.1, color='blue', label='Фаза сходимости SGD')
    
    # Добавляем вертикальные линии разделения
    plt.axvline(x=len(losses_adam), color='g', linestyle='--', label='Adam → SGD')
    plt.axvline(x=len(losses_adam) + len(losses1_init), color='orange', linestyle='--', label='Дивергенция → Сходимость')
    
    plt.xlabel('Шаги')
    plt.ylabel('Потери')
    plt.title('Кривые потерь для всего процесса')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "loss_comparison.png"))
    plt.close()
    
    # Визуализация расстояния между моделями во время SGD
    distances_init = []
    for i in range(len(trajectory1_init)):
        dist = np.linalg.norm(trajectory1_init[i] - trajectory2_init[i])
        distances_init.append(dist)
        
    distances_conv = []
    for i in range(len(trajectory1_conv)):
        dist = np.linalg.norm(trajectory1_conv[i] - trajectory2_conv[i])
        distances_conv.append(dist)
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(distances_init)), distances_init, 'r-', label='Фаза дивергенции')
    plt.plot(np.arange(len(distances_init), len(distances_init) + len(distances_conv)), distances_conv, 'b-', label='Фаза сходимости')
    plt.axvline(x=len(distances_init), color='orange', linestyle='--', label='Дивергенция → Сходимость')
    plt.xlabel('Шаги SGD')
    plt.ylabel('Евклидово расстояние')
    plt.title('Расстояние между параметрами моделей')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "parameter_distance.png"))
    plt.close()
    
    # Сохраняем результаты
    np.save(os.path.join(SAVE_DIR, "trajectory_adam.npy"), trajectory_adam)
    np.save(os.path.join(SAVE_DIR, "trajectory1_init.npy"), trajectory1_init)
    np.save(os.path.join(SAVE_DIR, "trajectory2_init.npy"), trajectory2_init)
    np.save(os.path.join(SAVE_DIR, "trajectory1_conv.npy"), trajectory1_conv)
    np.save(os.path.join(SAVE_DIR, "trajectory2_conv.npy"), trajectory2_conv)
    np.save(os.path.join(SAVE_DIR, "losses_adam.npy"), losses_adam)
    np.save(os.path.join(SAVE_DIR, "losses1_init.npy"), losses1_init)
    np.save(os.path.join(SAVE_DIR, "losses2_init.npy"), losses2_init)
    np.save(os.path.join(SAVE_DIR, "losses1_conv.npy"), losses1_conv)
    np.save(os.path.join(SAVE_DIR, "losses2_conv.npy"), losses2_conv)
    
    # Оценка точности моделей
    accuracy1 = evaluate_accuracy(model1, test_loader, DEVICE)
    accuracy2 = evaluate_accuracy(model2, test_loader, DEVICE)
    
    print(f"\nТочность SGD1: {accuracy1 * 100:.2f}%")
    print(f"Точность SGD2: {accuracy2 * 100:.2f}%")

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