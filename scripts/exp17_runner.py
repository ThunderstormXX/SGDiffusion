#!/usr/bin/env python3
"""
Эксперимент 17: SGD vs GD с отслеживанием гессианов
1) Обучить 4 модели: SGD+GD × lr(0.1, 0.01) до сходимости
2) Для каждой точки продолжить обучение только SGD с логированием гессианов
3) Графики с вертикальной линией перехода к SGD
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
    # Конфигурация экспериментов
    learning_rates = [0.1, 0.01]  # 1e-1, 1e-2
    optimizers = ['sgd', 'gd']
    batch_size = 64
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(base_dir, "exp17_train_to_plateau.py")
    continue_script = os.path.join(base_dir, "exp17_continue_and_log.py")
    
    print("🔬 ЭКСПЕРИМЕНТ 17: SGD vs GD с отслеживанием гессианов")
    print(f"📊 Learning rates: {learning_rates}")
    print(f"🔧 Optimizers: {optimizers}")
    print(f"📦 Batch size: {batch_size}")
    print("\n📋 План:")
    print("1) Обучить 4 модели до сходимости (SGD+GD × 2 lr)")
    print("2) Для каждой точки продолжить с SGD + логирование гессианов")
    print("3) Создать объединенные графики с линией перехода")
    
    success_count = 0
    total_tasks = len(learning_rates) * len(optimizers) * 2  # 4 обучения + 4 продолжения
    
    # Этап 1: Обучение до плато всех 4 моделей
    print(f"\n{'='*60}")
    print("📚 ЭТАП 1: Обучение до плато")
    print(f"{'='*60}")
    
    for lr in learning_rates:
        for optimizer in optimizers:
            experiment_name = f"{optimizer.upper()}_lr{lr}"
            
            train_cmd = [
                sys.executable, train_script,
                "--lr", str(lr),
                "--batch-size", str(batch_size),
                "--optimizer", optimizer,
                "--max-epochs", "1000",
                "--save-dir", "data/checkpoints/exp17"
            ]
            
            if run_command(train_cmd, f"Обучение до плато: {experiment_name}"):
                success_count += 1
                print(f"✅ {experiment_name}: Модель обучена")
            else:
                print(f"❌ {experiment_name}: Ошибка обучения")
    
    # Этап 2: Продолжение с SGD для каждой точки
    print(f"\n{'='*60}")
    print("🔍 ЭТАП 2: Продолжение с SGD + логирование гессианов")
    print(f"{'='*60}")
    
    for lr in learning_rates:
        for source_optimizer in optimizers:
            experiment_name = f"{source_optimizer.upper()}_lr{lr} → SGD"
            
            continue_cmd = [
                sys.executable, continue_script,
                "--lr", str(lr),
                "--batch-size", str(batch_size),
                "--source-optimizer", source_optimizer,
                "--post-plateau-steps", "500",
                "--save-dir", "data/checkpoints/exp17"
            ]
            
            if run_command(continue_cmd, f"Отслеживание гессианов: {experiment_name}"):
                success_count += 1
                print(f"✅ {experiment_name}: Гессианы записаны")
            else:
                print(f"❌ {experiment_name}: Ошибка отслеживания")
    
    # Итоговый отчет
    print(f"\n{'='*80}")
    print("📊 ИТОГОВЫЙ ОТЧЕТ ЭКСПЕРИМЕНТА 17")
    print(f"{'='*80}")
    print(f"✅ Успешно выполнено: {success_count}/{total_tasks}")
    print(f"❌ Неудачных попыток: {total_tasks - success_count}")
    
    if success_count == total_tasks:
        print("\n🎉 Все эксперименты выполнены успешно!")
        print("\n📈 Созданные файлы:")
        print("   🏆 4 обученные модели (SGD+GD × 2 lr)")
        print("   📊 4 набора гессианов (все → SGD)")
        print("   📈 4 объединенных графика с линией перехода")
        print("   💾 Метаданные и траектории параметров")
    else:
        print(f"\n⚠️ {total_tasks - success_count} задач завершились с ошибками")
    
    print(f"\n📁 Результаты сохранены в: data/checkpoints/exp17/")
    print("🔬 Готово для анализа поведения оптимизаторов на плато!")

if __name__ == "__main__":
    main()