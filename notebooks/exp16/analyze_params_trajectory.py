#!/usr/bin/env python3
"""
Анализ траекторий параметров для exp16 в стиле PCA из exp14
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import os
import sys

# Добавляем корневую директорию в путь
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_params_data(lr_folder):
    """Загружает данные параметров для указанного learning rate"""
    params_path = os.path.join(lr_folder, f"params_tensor_lr{lr_folder.split('_')[-1]}.pkl")
    metadata_path = os.path.join(lr_folder, f"metadata_lr{lr_folder.split('_')[-1]}.pkl")
    
    if not os.path.exists(params_path) or not os.path.exists(metadata_path):
        return None, None
    
    with open(params_path, 'rb') as f:
        params_tensor = pickle.load(f)
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return params_tensor, metadata

def analyze_trajectory_pca(params_tensor, lr_value, save_dir):
    """Анализирует траекторию параметров с помощью PCA"""
    print(f"Анализ траектории для lr={lr_value}")
    print(f"Размер тензора параметров: {params_tensor.shape}")
    
    # Центрируем данные
    params_centered = params_tensor - params_tensor.mean(axis=0)
    
    # Применяем PCA
    n_components = min(20, params_tensor.shape[0] - 1)
    pca = PCA(n_components=n_components)
    params_pca = pca.fit_transform(params_centered)
    
    print(f"Объясненная дисперсия (первые 10 компонент):")
    for i, var in enumerate(pca.explained_variance_ratio_[:10]):
        print(f"Компонента {i+1}: {var:.4f}")
    
    # График объясненной дисперсии
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel("Количество компонент")
    plt.ylabel("Кумулятивная объясненная дисперсия")
    plt.title(f"PCA: Объясненная дисперсия (lr={lr_value})")
    plt.grid(True)
    
    # Траектория в первых двух главных компонентах
    plt.subplot(2, 2, 2)
    plt.plot(params_pca[:, 0], params_pca[:, 1], 'b-', alpha=0.7, linewidth=1)
    plt.scatter(params_pca[0, 0], params_pca[0, 1], c='green', s=100, marker='o', label='Начало')
    plt.scatter(params_pca[-1, 0], params_pca[-1, 1], c='red', s=100, marker='s', label='Конец')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Траектория в пространстве PC1-PC2 (lr={lr_value})")
    plt.legend()
    plt.grid(True)
    
    # Эволюция первых трех компонент во времени
    plt.subplot(2, 2, 3)
    steps = np.arange(len(params_pca))
    plt.plot(steps, params_pca[:, 0], label='PC1', alpha=0.8)
    plt.plot(steps, params_pca[:, 1], label='PC2', alpha=0.8)
    if params_pca.shape[1] > 2:
        plt.plot(steps, params_pca[:, 2], label='PC3', alpha=0.8)
    plt.xlabel("Шаг")
    plt.ylabel("Значение компоненты")
    plt.title(f"Эволюция главных компонент (lr={lr_value})")
    plt.legend()
    plt.grid(True)
    
    # Распределение значений первой компоненты
    plt.subplot(2, 2, 4)
    plt.hist(params_pca[:, 0], bins=30, alpha=0.7, density=True)
    plt.xlabel("PC1")
    plt.ylabel("Плотность")
    plt.title(f"Распределение PC1 (lr={lr_value})")
    plt.grid(True)
    
    plt.tight_layout()
    
    # Сохраняем график
    output_path = os.path.join(save_dir, f"params_trajectory_pca_lr{lr_value}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {output_path}")
    plt.close()
    
    return pca, params_pca

def main():
    # Путь к данным exp16
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'checkpoints', 'exp16')
    
    if not os.path.exists(base_dir):
        print(f"Директория {base_dir} не найдена!")
        return
    
    # Создаем директорию для результатов
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Анализируем каждый learning rate
    lr_folders = [d for d in os.listdir(base_dir) if d.startswith('lr_')]
    
    for lr_folder in sorted(lr_folders):
        lr_path = os.path.join(base_dir, lr_folder)
        if not os.path.isdir(lr_path):
            continue
            
        lr_value = lr_folder.split('_')[-1]
        print(f"\n{'='*50}")
        print(f"Обработка learning rate: {lr_value}")
        print(f"{'='*50}")
        
        # Загружаем данные
        params_tensor, metadata = load_params_data(lr_path)
        
        if params_tensor is None:
            print(f"Данные для lr={lr_value} не найдены, пропускаем...")
            continue
        
        # Анализируем траекторию
        try:
            pca, params_pca = analyze_trajectory_pca(params_tensor, lr_value, results_dir)
            print(f"Анализ для lr={lr_value} завершен успешно")
        except Exception as e:
            print(f"Ошибка при анализе lr={lr_value}: {e}")
            continue
    
    print(f"\nВсе результаты сохранены в: {results_dir}")

if __name__ == "__main__":
    main()