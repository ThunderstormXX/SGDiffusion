#!/usr/bin/env python3
"""
Эксперимент 20: Общий путь SGD+GD, затем разветвление SGD с разными lr
1) Общий путь: SGD (lr=0.1) + GD (lr=0.1)
2) Разветвление: SGD с lr=[1e-3, 1e-2, 1e-1, 5e-1] + логирование гессианов
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Запускает команду и выводит результат"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Успешно выполнено")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка выполнения: {e}")
        return False

def main():
    # Конфигурация эксперимента
    initial_lr = 0.5  # 1e-1 для общего пути
    branch_lrs = "0.001,0.01,0.1,0.5"  # 1e-3, 1e-2, 1e-1, 5e-1
    batch_size = 64
    sgd_epochs = 5000
    gd_epochs = 1000
    branch_epochs = 68  # ~1000 batch iterations
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(base_dir, "exp20_train.py")
    
    print("🔬 ЭКСПЕРИМЕНТ 20: Общий путь + разветвление SGD")
    print(f"📊 Начальный lr: {initial_lr}")
    print(f"📊 Ветки lr: {branch_lrs}")
    print(f"📦 Batch size: {batch_size}")
    print("\n📋 План эксперимента:")
    print(f"1) Общий SGD: {sgd_epochs} эпох (lr={initial_lr})")
    print(f"2) Общий GD: {gd_epochs} эпох (lr={initial_lr})")
    print(f"3) Разветвление SGD: {branch_epochs} эпох для каждого lr")
    print(f"4) Логирование гессианов для каждой ветки")
    
    train_cmd = [
        sys.executable, train_script,
        "--initial-lr", str(initial_lr),
        "--branch-lrs", branch_lrs,
        "--batch-size", str(batch_size),
        "--sgd-epochs", str(sgd_epochs),
        "--gd-epochs", str(gd_epochs),
        "--branch-epochs", str(branch_epochs),
        "--save-dir", "data/checkpoints/exp20"
    ]
    
    if run_command(train_cmd, "Эксперимент с разветвлением"):
        print("\n🎉 Эксперимент 20 выполнен успешно!")
        print("\n📈 Созданные файлы:")
        print("   🏆 Общий график всех веток")
        print("   📊 Гессианы для каждой ветки")
        print("   📈 Траектории параметров для каждой ветки")
        print("   💾 Общие метаданные эксперимента")
        print("\n🔍 Структура эксперимента:")
        print("   1️⃣ Общий путь: SGD → GD (одинаковый для всех)")
        print("   2️⃣ Разветвление: SGD с разными lr")
        print("   3️⃣ Логирование: гессианы и параметры по батчам")
        print("   4️⃣ Графики: общая траектория + ветки")
    else:
        print("❌ Эксперимент завершился с ошибкой")
    
    print(f"\n📁 Результаты сохранены в: data/checkpoints/exp20/")
    print("🔬 Готово для анализа разветвления SGD из общей точки!")

if __name__ == "__main__":
    main()