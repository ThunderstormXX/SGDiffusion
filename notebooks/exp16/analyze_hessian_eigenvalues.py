#!/usr/bin/env python3
"""
Анализ собственных чисел гессианов для exp16
Строит графики с распределением собственных чисел (mean и дисперсия)
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

# Добавляем корневую директорию в путь
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_hessian_data(lr_folder):
    """Загружает данные гессианов для указанного learning rate"""
    hessians_path = os.path.join(lr_folder, f"hessians_tensor_lr{lr_folder.split('_')[-1]}.pkl")
    metadata_path = os.path.join(lr_folder, f"metadata_lr{lr_folder.split('_')[-1]}.pkl")
    
    if not os.path.exists(hessians_path) or not os.path.exists(metadata_path):
        return None, None
    
    with open(hessians_path, 'rb') as f:
        hessians_tensor = pickle.load(f)
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return hessians_tensor, metadata

def compute_eigenvalues(hessians_tensor):
    """Вычисляет собственные числа для каждого гессиана"""
    print(f"Вычисление собственных чисел для тензора размера: {hessians_tensor.shape}")
    
    eigenvalues_list = []
    
    for i, hessian in enumerate(hessians_tensor):
        if i % 10 == 0:
            print(f"Обработка гессиана {i+1}/{len(hessians_tensor)}")
        
        try:
            # Вычисляем собственные числа
            eigenvals = np.linalg.eigvals(hessian)
            # Сортируем по убыванию
            eigenvals = np.sort(eigenvals)[::-1]
            eigenvalues_list.append(eigenvals)
        except Exception as e:
            print(f"Ошибка при вычислении собственных чисел для гессиана {i}: {e}")
            continue
    
    return np.array(eigenvalues_list)

def analyze_eigenvalues(eigenvalues, lr_value, save_dir):
    """Анализирует собственные числа гессианов"""
    print(f"Анализ собственных чисел для lr={lr_value}")
    print(f"Размер массива собственных чисел: {eigenvalues.shape}")
    
    # Статистики по собственным числам
    mean_eigenvals = np.mean(eigenvalues, axis=0)
    std_eigenvals = np.std(eigenvalues, axis=0)
    max_eigenvals = np.max(eigenvalues, axis=0)
    min_eigenvals = np.min(eigenvalues, axis=0)
    
    print(f"Максимальное собственное число: {np.max(eigenvalues):.6f}")
    print(f"Минимальное собственное число: {np.min(eigenvalues):.6f}")
    print(f"Среднее максимальное собственное число: {np.mean(max_eigenvals):.6f}")
    
    # Создаем графики
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Распределение собственных чисел с доверительным интервалом
    ax1 = axes[0, 0]
    indices = np.arange(len(mean_eigenvals))
    ax1.plot(indices, mean_eigenvals, 'b-', linewidth=2, label='Среднее')
    ax1.fill_between(indices, 
                     mean_eigenvals - std_eigenvals, 
                     mean_eigenvals + std_eigenvals, 
                     alpha=0.3, color='blue', label='±1σ')
    ax1.set_xlabel('Индекс собственного числа')
    ax1.set_ylabel('Значение')
    ax1.set_title(f'Собственные числа гессианов (lr={lr_value})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('symlog')
    
    # 2. Распределение максимальных собственных чисел
    ax2 = axes[0, 1]
    max_eigenvals_per_step = np.max(eigenvalues, axis=1)
    ax2.hist(max_eigenvals_per_step, bins=30, alpha=0.7, density=True, color='red')
    ax2.axvline(np.mean(max_eigenvals_per_step), color='black', linestyle='--', 
                label=f'Среднее: {np.mean(max_eigenvals_per_step):.4f}')
    ax2.set_xlabel('Максимальное собственное число')
    ax2.set_ylabel('Плотность')
    ax2.set_title(f'Распределение максимальных собственных чисел (lr={lr_value})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Эволюция максимального собственного числа во времени
    ax3 = axes[1, 0]
    steps = np.arange(len(max_eigenvals_per_step))
    ax3.plot(steps, max_eigenvals_per_step, 'g-', alpha=0.8, linewidth=1)
    ax3.set_xlabel('Шаг')
    ax3.set_ylabel('Максимальное собственное число')
    ax3.set_title(f'Эволюция максимального собственного числа (lr={lr_value})')
    ax3.grid(True, alpha=0.3)
    
    # 4. Спектр собственных чисел (первые 50)
    ax4 = axes[1, 1]
    n_show = min(50, eigenvalues.shape[1])
    for i in range(0, len(eigenvalues), max(1, len(eigenvalues)//10)):
        ax4.semilogy(range(n_show), eigenvalues[i, :n_show], alpha=0.3, color='purple')
    
    # Показываем среднее
    ax4.semilogy(range(n_show), mean_eigenvals[:n_show], 'k-', linewidth=3, label='Среднее')
    ax4.set_xlabel('Индекс собственного числа')
    ax4.set_ylabel('Значение (log scale)')
    ax4.set_title(f'Спектр собственных чисел (первые {n_show}) (lr={lr_value})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохраняем график
    output_path = os.path.join(save_dir, f"hessian_eigenvalues_lr{lr_value}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {output_path}")
    plt.close()
    
    # Дополнительный график: детальная статистика
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Статистики по времени
    ax1 = axes[0, 0]
    mean_per_step = np.mean(eigenvalues, axis=1)
    std_per_step = np.std(eigenvalues, axis=1)
    steps = np.arange(len(mean_per_step))
    
    ax1.plot(steps, mean_per_step, 'b-', label='Среднее')
    ax1.fill_between(steps, mean_per_step - std_per_step, mean_per_step + std_per_step, 
                     alpha=0.3, color='blue')
    ax1.set_xlabel('Шаг')
    ax1.set_ylabel('Среднее собственное число')
    ax1.set_title(f'Эволюция среднего собственного числа (lr={lr_value})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Отношение максимального к минимальному
    ax2 = axes[0, 1]
    min_eigenvals_per_step = np.min(eigenvalues, axis=1)
    condition_numbers = max_eigenvals_per_step / np.abs(min_eigenvals_per_step)
    ax2.semilogy(steps, condition_numbers, 'r-', alpha=0.8)
    ax2.set_xlabel('Шаг')
    ax2.set_ylabel('Число обусловленности (log scale)')
    ax2.set_title(f'Число обусловленности гессиана (lr={lr_value})')
    ax2.grid(True, alpha=0.3)
    
    # Количество положительных/отрицательных собственных чисел
    ax3 = axes[1, 0]
    positive_count = np.sum(eigenvalues > 0, axis=1)
    negative_count = np.sum(eigenvalues < 0, axis=1)
    zero_count = np.sum(np.abs(eigenvalues) < 1e-10, axis=1)
    
    ax3.plot(steps, positive_count, 'g-', label='Положительные', linewidth=2)
    ax3.plot(steps, negative_count, 'r-', label='Отрицательные', linewidth=2)
    ax3.plot(steps, zero_count, 'k-', label='≈ Нулевые', linewidth=2)
    ax3.set_xlabel('Шаг')
    ax3.set_ylabel('Количество')
    ax3.set_title(f'Знаки собственных чисел (lr={lr_value})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Распределение всех собственных чисел
    ax4 = axes[1, 1]
    all_eigenvals = eigenvalues.flatten()
    ax4.hist(all_eigenvals, bins=50, alpha=0.7, density=True, color='orange')
    ax4.axvline(0, color='black', linestyle='--', label='Ноль')
    ax4.axvline(np.mean(all_eigenvals), color='red', linestyle='--', 
                label=f'Среднее: {np.mean(all_eigenvals):.4f}')
    ax4.set_xlabel('Собственное число')
    ax4.set_ylabel('Плотность')
    ax4.set_title(f'Распределение всех собственных чисел (lr={lr_value})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохраняем дополнительный график
    output_path_stats = os.path.join(save_dir, f"hessian_eigenvalues_stats_lr{lr_value}.png")
    plt.savefig(output_path_stats, dpi=300, bbox_inches='tight')
    print(f"График статистики сохранен: {output_path_stats}")
    plt.close()
    
    return mean_eigenvals, std_eigenvals

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
        hessians_tensor, metadata = load_hessian_data(lr_path)
        
        if hessians_tensor is None:
            print(f"Данные гессианов для lr={lr_value} не найдены, пропускаем...")
            continue
        
        # Вычисляем собственные числа
        try:
            eigenvalues = compute_eigenvalues(hessians_tensor)
            if len(eigenvalues) == 0:
                print(f"Не удалось вычислить собственные числа для lr={lr_value}")
                continue
                
            # Анализируем собственные числа
            mean_eigenvals, std_eigenvals = analyze_eigenvalues(eigenvalues, lr_value, results_dir)
            print(f"Анализ собственных чисел для lr={lr_value} завершен успешно")
            
        except Exception as e:
            print(f"Ошибка при анализе собственных чисел lr={lr_value}: {e}")
            continue
    
    print(f"\nВсе результаты сохранены в: {results_dir}")

if __name__ == "__main__":
    main()