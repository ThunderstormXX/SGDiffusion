#!/usr/bin/env python3
"""
Эксперимент 19: SGD → GD → SGD с логированием (Similar MNIST)
1) SGD обучение (50к итераций)
2) GD к минимуму (10к итераций)  
3) SGD с логированием гессианов (500 итераций)

Для lr = 0.1 и 0.5
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
    learning_rates = [0.01, 0.1, 0.5]  # 1e-1, 5e-1
    batch_size = 64
    sgd_epochs = 5000
    gd_epochs = 1000
    sgd_log_epochs = 68  # ~1000 batch iterations
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(base_dir, "exp19_train.py")
    
    print("🔬 ЭКСПЕРИМЕНТ 19: SGD → GD → SGD с логированием (Similar MNIST)")
    print(f"📊 Learning rates: {learning_rates}")
    print(f"📦 Batch size: {batch_size}")
    print("\n📋 План для каждого lr:")
    print(f"1) SGD обучение: {sgd_epochs} эпох")
    print(f"2) GD к минимуму: {gd_epochs} эпох")
    print(f"3) SGD + логирование: {sgd_log_epochs} эпох (~1000 батчей)")
    
    success_count = 0
    total_tasks = len(learning_rates)
    
    # Запуск экспериментов для каждого lr
    for lr in learning_rates:
        experiment_name = f"lr={lr}"
        
        train_cmd = [
            sys.executable, train_script,
            "--lr", str(lr),
            "--batch-size", str(batch_size),
            "--sgd-epochs", str(sgd_epochs),
            "--gd-epochs", str(gd_epochs),
            "--sgd-log-epochs", str(sgd_log_epochs),
            "--save-dir", "data/checkpoints/exp19"
        ]
        
        if run_command(train_cmd, f"Трехэтапное обучение: {experiment_name}"):
            success_count += 1
            print(f"✅ {experiment_name}: Эксперимент завершен")
        else:
            print(f"❌ {experiment_name}: Ошибка выполнения")
    
    # Итоговый отчет
    print(f"\n{'='*80}")
    print("📊 ИТОГОВЫЙ ОТЧЕТ ЭКСПЕРИМЕНТА 19")
    print(f"{'='*80}")
    print(f"✅ Успешно выполнено: {success_count}/{total_tasks}")
    print(f"❌ Неудачных попыток: {total_tasks - success_count}")
    
    if success_count == total_tasks:
        print("\n🎉 Все эксперименты выполнены успешно!")
        print("\n📈 Созданные файлы для каждого lr:")
        print("   🏆 Финальная модель после 3 этапов")
        print("   📊 Гессианы из этапа 3 (SGD в минимуме)")
        print("   📈 Траектории параметров этапа 3")
        print("   📉 Объединенный график всех 3 этапов")
        print("   💾 Полные метаданные эксперимента")
        print("\n🔍 Этапы эксперимента:")
        print("   1️⃣ SGD: Обучение до хорошей точки")
        print("   2️⃣ GD: Доползание до настоящего минимума")
        print("   3️⃣ SGD: Изучение шума в минимуме")
    else:
        print(f"\n⚠️ {total_tasks - success_count} экспериментов завершились с ошибками")
    
    print(f"\n📁 Результаты сохранены в: data/checkpoints/exp19/")
    print("🔬 Готово для анализа поведения SGD в минимуме на похожих данных!")

if __name__ == "__main__":
    main()