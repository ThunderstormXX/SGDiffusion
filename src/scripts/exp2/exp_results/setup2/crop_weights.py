#!/usr/bin/env python3
"""
Создаёт прореженную версию файла весов, оставляя только каждую N-ю итерацию.
Использует mmap для экономии памяти при работе с большими файлами.
"""
import torch
import os
import sys

def main():
    input_path = "sgd_weights_lr0.001.pt"
    output_path = "sgd_weights_lr0.001_cropped_per_10iters.pt"
    step = 10  # каждые 10 итераций
    
    print(f"[1/4] Загружаем {input_path} с mmap...")
    print(f"      Размер файла: {os.path.getsize(input_path) / 1e9:.2f} GB")
    
    # mmap=True позволяет не загружать весь файл в RAM
    # weights_only=True для безопасности и скорости
    data = torch.load(input_path, map_location="cpu", mmap=True, weights_only=True)
    
    print(f"[2/4] Исходный shape: {data.shape}")
    print(f"      dtype: {data.dtype}")
    
    # data shape: [N, T, D]
    # Берём каждый step-й timestep, начиная с 0: [0, 10, 20, ...]
    N, T, D = data.shape
    indices = list(range(0, T, step))
    print(f"[3/4] Оставляем итерации: 0, {step}, {2*step}, ... (всего {len(indices)} из {T})")
    
    # Делаем срез - это создаст копию только нужных данных
    cropped = data[:, ::step, :].clone()
    
    print(f"      Новый shape: {cropped.shape}")
    
    # Освобождаем память от mmap
    del data
    
    print(f"[4/4] Сохраняем в {output_path}...")
    torch.save(cropped, output_path)
    
    final_size = os.path.getsize(output_path) / 1e9
    print(f"      Готово! Размер: {final_size:.2f} GB")

if __name__ == "__main__":
    main()
