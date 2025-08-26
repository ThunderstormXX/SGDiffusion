#!/usr/bin/env python3
"""
Визуализация траекторий весов модели из сохраненных данных экспериментов
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize weight trajectories from saved experiments')
    parser.add_argument('--exp-dir', type=str, required=True, help='Experiment directory (e.g., data/checkpoints/exp20)')
    parser.add_argument('--lrs', type=str, default='0.001,0.01,0.1,0.5', help='Learning rates to visualize (comma-separated)')
    parser.add_argument('--max-weights', type=int, default=20, help='Maximum number of weights to show trajectories')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save plots (default: same as exp-dir)')
    parser.add_argument('--pca-ref-lr', type=str, default=None, help='Reference LR for PCA (default: first available)')
    return parser.parse_args()

def load_params(exp_dir, lr):
    """Загружает параметры для указанного learning rate"""
    params_path = os.path.join(exp_dir, f"params_lr{lr}.pkl")
    print(f"   Ищем файл: {params_path}")
    if not os.path.exists(params_path):
        print(f"⚠️ Файл не найден: {params_path}")
        return None
    
    try:
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
        return params
    except Exception as e:
        print(f"❌ Ошибка загрузки {params_path}: {e}")
        return None

def plot_weight_trajectories(all_params, lrs, max_weights, save_dir):
    """Строит траектории весов"""
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    
    # Определяем количество весов для отображения
    min_weights = min(params.shape[1] for params in all_params.values() if params is not None)
    n_weights = min(max_weights, min_weights)
    
    plt.figure(figsize=(20, 15))
    
    # 1. Траектории отдельных весов
    plt.subplot(3, 3, 1)
    
    # Анализ константных весов
    const_info = []
    for lr, params in all_params.items():
        if params is None:
            continue
        # Проверяем какие веса константны (стандартное отклонение < 1e-8)
        weight_stds = np.std(params, axis=0)
        const_weights = np.sum(weight_stds < 1e-8)
        non_const_weights = params.shape[1] - const_weights
        const_info.append(f'lr={lr}: {const_weights} const, {non_const_weights} var')
    
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        # Показываем первые несколько весов
        for w_idx in range(min(5, n_weights)):
            alpha = 0.7 if w_idx == 0 else 0.3
            linewidth = 2 if w_idx == 0 else 1
            plt.plot(params[:, w_idx], color=color, alpha=alpha, linewidth=linewidth,
                    label=f'lr={lr}' if w_idx == 0 else '')
    
    plt.title('Individual Weight Trajectories')
    plt.xlabel('Batch Iteration')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Добавляем информацию о константных весах
    info_text = '\n'.join(const_info)
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=8, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Среднее значение весов
    plt.subplot(3, 3, 2)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        mean_weights = np.mean(params, axis=1)
        plt.plot(mean_weights, color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('Mean Weight Value')
    plt.xlabel('Batch Iteration')
    plt.ylabel('Mean Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Стандартное отклонение весов
    plt.subplot(3, 3, 3)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        std_weights = np.std(params, axis=1)
        plt.plot(std_weights, color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('Weight Standard Deviation')
    plt.xlabel('Batch Iteration')
    plt.ylabel('Std Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Норма весов (L2)
    plt.subplot(3, 3, 4)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        l2_norm = np.linalg.norm(params, axis=1)
        plt.plot(l2_norm, color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('L2 Norm of Weights')
    plt.xlabel('Batch Iteration')
    plt.ylabel('L2 Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Изменения весов (градиенты)
    plt.subplot(3, 3, 5)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        weight_changes = np.diff(params, axis=0)
        mean_change = np.mean(np.abs(weight_changes), axis=1)
        plt.plot(mean_change, color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('Mean Absolute Weight Change')
    plt.xlabel('Batch Iteration')
    plt.ylabel('Mean |ΔWeight|')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Распределение весов (гистограмма для последней итерации)
    plt.subplot(3, 3, 6)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        final_weights = params[-1, :]
        plt.hist(final_weights, bins=30, alpha=0.5, color=color, label=f'lr={lr}', density=True)
    
    plt.title('Final Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Траектории в 2D (первые два веса)
    plt.subplot(3, 3, 7)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None or params.shape[1] < 2:
            continue
        color = colors[i % len(colors)]
        plt.plot(params[:, 0], params[:, 1], color=color, alpha=0.7, linewidth=2, label=f'lr={lr}')
        plt.scatter(params[0, 0], params[0, 1], color=color, s=100, marker='o', edgecolor='black')  # Start
        plt.scatter(params[-1, 0], params[-1, 1], color=color, s=100, marker='s', edgecolor='black')  # End
    
    plt.title('2D Weight Trajectory (w0 vs w1)')
    plt.xlabel('Weight 0')
    plt.ylabel('Weight 1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Максимальное изменение весов
    plt.subplot(3, 3, 8)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        weight_changes = np.diff(params, axis=0)
        max_change = np.max(np.abs(weight_changes), axis=1)
        plt.plot(max_change, color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('Max Absolute Weight Change')
    plt.xlabel('Batch Iteration')
    plt.ylabel('Max |ΔWeight|')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Кумулятивное изменение весов
    plt.subplot(3, 3, 9)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        weight_changes = np.diff(params, axis=0)
        cumulative_change = np.cumsum(np.mean(np.abs(weight_changes), axis=1))
        plt.plot(cumulative_change, color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('Cumulative Weight Change')
    plt.xlabel('Batch Iteration')
    plt.ylabel('Cumulative |ΔWeight|')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохранение
    save_path = os.path.join(save_dir, 'weight_trajectories.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def plot_weight_statistics(all_params, lrs, save_dir):
    """Строит статистики весов"""
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    
    plt.figure(figsize=(15, 10))
    
    # 1. Статистики по времени
    plt.subplot(2, 3, 1)
    stats = {}
    for lr, params in all_params.items():
        if params is None:
            continue
        stats[lr] = {
            'mean': np.mean(params, axis=1),
            'std': np.std(params, axis=1),
            'min': np.min(params, axis=1),
            'max': np.max(params, axis=1),
            'median': np.median(params, axis=1)
        }
    
    for i, (lr, stat) in enumerate(stats.items()):
        color = colors[i % len(colors)]
        plt.fill_between(range(len(stat['mean'])), 
                        stat['mean'] - stat['std'], 
                        stat['mean'] + stat['std'], 
                        alpha=0.3, color=color)
        plt.plot(stat['mean'], color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('Weight Statistics Over Time')
    plt.xlabel('Batch Iteration')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Финальные распределения (box plot)
    plt.subplot(2, 3, 2)
    final_weights_data = []
    labels = []
    for lr, params in all_params.items():
        if params is None:
            continue
        final_weights_data.append(params[-1, :])
        labels.append(f'lr={lr}')
    
    plt.boxplot(final_weights_data, labels=labels)
    plt.title('Final Weight Distributions')
    plt.ylabel('Weight Value')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 3. Корреляция между весами
    plt.subplot(2, 3, 3)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None or params.shape[1] < 2:
            continue
        color = colors[i % len(colors)]
        # Корреляция между первыми двумя весами
        corr = np.corrcoef(params[:, 0], params[:, 1])[0, 1]
        plt.bar(i, corr, color=color, alpha=0.7, label=f'lr={lr}')
    
    plt.title('Weight Correlation (w0 vs w1)')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Энтропия весов
    plt.subplot(2, 3, 4)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        entropies = []
        for t in range(params.shape[0]):
            weights = params[t, :]
            # Нормализуем веса для вычисления энтропии
            weights_norm = np.abs(weights) / (np.sum(np.abs(weights)) + 1e-8)
            entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-8))
            entropies.append(entropy)
        plt.plot(entropies, color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('Weight Entropy Over Time')
    plt.xlabel('Batch Iteration')
    plt.ylabel('Entropy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Спектр весов (собственные значения)
    plt.subplot(2, 3, 5)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        # Ковариационная матрица весов
        cov_matrix = np.cov(params.T)
        eigenvals = np.linalg.eigvals(cov_matrix)
        eigenvals = np.sort(eigenvals)[::-1]  # Сортируем по убыванию
        plt.plot(eigenvals[:min(10, len(eigenvals))], 'o-', color=color, label=f'lr={lr}')
    
    plt.title('Weight Covariance Eigenvalues')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Расстояние от начальной точки
    plt.subplot(2, 3, 6)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        initial_weights = params[0, :]
        distances = [np.linalg.norm(params[t, :] - initial_weights) for t in range(params.shape[0])]
        plt.plot(distances, color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('Distance from Initial Weights')
    plt.xlabel('Batch Iteration')
    plt.ylabel('L2 Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохранение
    save_path = os.path.join(save_dir, 'weight_statistics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def plot_pca_trajectories(all_params, lrs, save_dir):
    """Строит PCA траектории весов"""
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    
    # Выбираем референсную траекторию (первую доступную)
    ref_lr = list(all_params.keys())[0]
    ref_params = all_params[ref_lr]
    
    # Обучаем PCA на референсной траектории
    pca = PCA(n_components=min(10, ref_params.shape[1]))
    pca.fit(ref_params)
    
    print(f"\n📊 PCA анализ (референс: lr={ref_lr})")
    print(f"   Объясненная дисперсия первых 5 компонент: {pca.explained_variance_ratio_[:5]}")
    
    plt.figure(figsize=(20, 15))
    
    # 1. Объясненная дисперсия
    plt.subplot(3, 4, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_[:10]), 'o-')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('PC Component')
    plt.ylabel('Cumulative Variance Ratio')
    plt.grid(True, alpha=0.3)
    
    # 2. Траектории в PC1-PC2
    plt.subplot(3, 4, 2)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        params_pca = pca.transform(params)
        plt.plot(params_pca[:, 0], params_pca[:, 1], color=color, alpha=0.7, linewidth=2, label=f'lr={lr}')
        plt.scatter(params_pca[0, 0], params_pca[0, 1], color=color, s=100, marker='o', edgecolor='black')
        plt.scatter(params_pca[-1, 0], params_pca[-1, 1], color=color, s=100, marker='s', edgecolor='black')
    
    plt.title('PC1 vs PC2 Trajectories')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Траектории в PC1-PC3
    plt.subplot(3, 4, 3)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        params_pca = pca.transform(params)
        plt.plot(params_pca[:, 0], params_pca[:, 2], color=color, alpha=0.7, linewidth=2, label=f'lr={lr}')
        plt.scatter(params_pca[0, 0], params_pca[0, 2], color=color, s=100, marker='o', edgecolor='black')
        plt.scatter(params_pca[-1, 0], params_pca[-1, 2], color=color, s=100, marker='s', edgecolor='black')
    
    plt.title('PC1 vs PC3 Trajectories')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Траектории в PC2-PC3
    plt.subplot(3, 4, 4)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        params_pca = pca.transform(params)
        plt.plot(params_pca[:, 1], params_pca[:, 2], color=color, alpha=0.7, linewidth=2, label=f'lr={lr}')
        plt.scatter(params_pca[0, 1], params_pca[0, 2], color=color, s=100, marker='o', edgecolor='black')
        plt.scatter(params_pca[-1, 1], params_pca[-1, 2], color=color, s=100, marker='s', edgecolor='black')
    
    plt.title('PC2 vs PC3 Trajectories')
    plt.xlabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. PC1 по времени
    plt.subplot(3, 4, 5)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        params_pca = pca.transform(params)
        plt.plot(params_pca[:, 0], color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('PC1 Over Time')
    plt.xlabel('Batch Iteration')
    plt.ylabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. PC2 по времени
    plt.subplot(3, 4, 6)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        params_pca = pca.transform(params)
        plt.plot(params_pca[:, 1], color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('PC2 Over Time')
    plt.xlabel('Batch Iteration')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. PC3 по времени
    plt.subplot(3, 4, 7)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        params_pca = pca.transform(params)
        plt.plot(params_pca[:, 2], color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('PC3 Over Time')
    plt.xlabel('Batch Iteration')
    plt.ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Расстояние в PCA пространстве
    plt.subplot(3, 4, 8)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        params_pca = pca.transform(params)
        distances = [np.linalg.norm(params_pca[t, :3]) for t in range(params_pca.shape[0])]
        plt.plot(distances, color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('Distance in PCA Space (PC1-3)')
    plt.xlabel('Batch Iteration')
    plt.ylabel('L2 Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Проекция на первые 3 компоненты (3D scatter)
    ax = plt.subplot(3, 4, 9, projection='3d')
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        params_pca = pca.transform(params)
        ax.plot(params_pca[:, 0], params_pca[:, 1], params_pca[:, 2], color=color, alpha=0.7, linewidth=2, label=f'lr={lr}')
        ax.scatter(params_pca[0, 0], params_pca[0, 1], params_pca[0, 2], color=color, s=100, marker='o')
        ax.scatter(params_pca[-1, 0], params_pca[-1, 1], params_pca[-1, 2], color=color, s=100, marker='s')
    
    ax.set_title('3D PCA Trajectories')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.3f})')
    ax.legend()
    
    # 10. Скорость в PCA пространстве
    plt.subplot(3, 4, 10)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        params_pca = pca.transform(params)
        velocities = np.diff(params_pca[:, :3], axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        plt.plot(speeds, color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('Speed in PCA Space')
    plt.xlabel('Batch Iteration')
    plt.ylabel('Speed (L2 norm of velocity)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 11. Угол траектории в PC1-PC2
    plt.subplot(3, 4, 11)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        params_pca = pca.transform(params)
        velocities = np.diff(params_pca[:, :2], axis=0)
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        plt.plot(angles, color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('Trajectory Angle in PC1-PC2')
    plt.xlabel('Batch Iteration')
    plt.ylabel('Angle (radians)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 12. Кривизна траектории
    plt.subplot(3, 4, 12)
    for i, (lr, params) in enumerate(all_params.items()):
        if params is None:
            continue
        color = colors[i % len(colors)]
        params_pca = pca.transform(params)
        # Вычисляем кривизну как изменение направления
        velocities = np.diff(params_pca[:, :2], axis=0)
        accelerations = np.diff(velocities, axis=0)
        curvatures = []
        for t in range(len(accelerations)):
            v = velocities[t]
            a = accelerations[t]
            curvature = np.abs(np.cross(v, a)) / (np.linalg.norm(v)**3 + 1e-8)
            curvatures.append(curvature)
        plt.plot(curvatures, color=color, linewidth=2, label=f'lr={lr}')
    
    plt.title('Trajectory Curvature')
    plt.xlabel('Batch Iteration')
    plt.ylabel('Curvature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохранение
    save_path = os.path.join(save_dir, 'pca_trajectories.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path, pca

def main():
    args = parse_args()
    
    # Парсим learning rates
    lrs = [float(lr) for lr in args.lrs.split(',')]
    
    # Определяем директорию для сохранения
    save_dir = args.save_dir if args.save_dir else args.exp_dir
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"📊 Визуализация траекторий весов")
    print(f"📁 Директория эксперимента: {args.exp_dir}")
    print(f"📊 Learning rates: {lrs}")
    print(f"💾 Сохранение в: {save_dir}")
    
    # Загружаем параметры для всех learning rates
    all_params = {}
    for lr in lrs:
        print(f"\n📥 Загрузка параметров для lr={lr}...")
        params = load_params(args.exp_dir, lr)
        if params is not None:
            print(f"   ✅ Загружено: {params.shape} (итераций × параметров)")
            all_params[lr] = params
        else:
            print(f"   ❌ Не удалось загрузить")
    
    if not all_params:
        print("❌ Не удалось загрузить ни одного файла с параметрами")
        return
    
    print(f"\n📈 Создание графиков траекторий...")
    traj_path = plot_weight_trajectories(all_params, lrs, args.max_weights, save_dir)
    print(f"   ✅ Траектории: {traj_path}")
    
    # Анализ константных весов
    print(f"\n🔍 Анализ константных весов:")
    for lr, params in all_params.items():
        if params is None:
            continue
        weight_stds = np.std(params, axis=0)
        const_weights = np.sum(weight_stds < 1e-8)
        non_const_weights = params.shape[1] - const_weights
        const_ratio = const_weights / params.shape[1] * 100
        print(f"   lr={lr}: {const_weights} константных, {non_const_weights} вариабельных ({const_ratio:.1f}% конст.)")
    
    print(f"\n📊 Создание графиков статистик...")
    stats_path = plot_weight_statistics(all_params, lrs, save_dir)
    print(f"   ✅ Статистики: {stats_path}")
    
    print(f"\n🔬 Создание PCA анализа...")
    pca_path, pca_model = plot_pca_trajectories(all_params, lrs, save_dir)
    print(f"   ✅ PCA траектории: {pca_path}")
    
    print(f"\n🎉 Визуализация завершена!")
    print(f"📁 Результаты сохранены в: {save_dir}")
    print(f"📊 PCA объясненная дисперсия (первые 3): {pca_model.explained_variance_ratio_[:3]}")
    
    # Итоговая статистика константных весов
    print(f"\n📊 Итоговая статистика константных весов:")
    for lr, params in all_params.items():
        if params is None:
            continue
        weight_stds = np.std(params, axis=0)
        const_weights = np.sum(weight_stds < 1e-8)
        non_const_weights = params.shape[1] - const_weights
        const_ratio = const_weights / params.shape[1] * 100
        print(f"   lr={lr}: {const_weights} константных, {non_const_weights} вариабельных ({const_ratio:.1f}% конст.)")

if __name__ == "__main__":
    main()