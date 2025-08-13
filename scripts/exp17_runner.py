#!/usr/bin/env python3
"""
Эксперимент 17: Обучение с отслеживанием гессианов
Основан на эксперименте 16 с улучшениями:
- Увеличенный размер выборки (1000 вместо 386)
- Отслеживание валидационных метрик
- Learning rates: 1e-1, 1e-2
- Batch size: 64
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Запускает команду и выводит результат"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Команда: {' '.join(cmd)}")
    
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
    batch_size = 64
    device = sys.argv[1] if len(sys.argv) > 1 else 'auto'  # Получаем device из аргументов
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(base_dir, "exp17_train_to_plateau.py")
    continue_script = os.path.join(base_dir, "exp17_continue_and_log.py")
    
    print("🔬 ЗАПУСК ЭКСПЕРИМЕНТА 17: Обучение с отслеживанием гессианов")
    print(f"📊 Learning rates: {learning_rates}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🖥️  Device: {device}")
    
    success_count = 0
    total_experiments = len(learning_rates) * 2  # 2 этапа на каждый lr
    
    for lr in learning_rates:
        experiment_name = f"lr{lr}"
        
        # Этап 1: Обучение до плато
        train_cmd = [
            sys.executable, train_script,
            "--lr", str(lr),
            "--batch-size", str(batch_size),
            "--max-epochs", "10",
            "--save-dir", f"data/checkpoints/exp17",
            "--device", device
        ]
        
        if run_command(train_cmd, f"Обучение до плато: {experiment_name}"):
            success_count += 1
            
            # Этап 2: Продолжение с отслеживанием гессианов
            continue_cmd = [
                sys.executable, continue_script,
                "--lr", str(lr),
                "--batch-size", str(batch_size),
                "--post-plateau-steps", "10",
                "--save-dir", f"data/checkpoints/exp17",
                "--device", device
            ]
            
            if run_command(continue_cmd, f"Отслеживание гессианов: {experiment_name}"):
                success_count += 1
            else:
                print(f"⚠️  Не удалось выполнить отслеживание гессианов для {experiment_name}")
        else:
            print(f"⚠️  Не удалось обучить модель для {experiment_name}")
    
    # Итоговый отчет
    print(f"\n{'='*60}")
    print(f"📊 ИТОГОВЫЙ ОТЧЕТ ЭКСПЕРИМЕНТА 17")
    print(f"{'='*60}")
    print(f"✅ Успешно выполнено: {success_count}/{total_experiments}")
    print(f"❌ Неудачных попыток: {total_experiments - success_count}")
    
    if success_count == total_experiments:
        print("🎉 Все эксперименты выполнены успешно!")
    else:
        print("⚠️  Некоторые эксперименты завершились с ошибками")
    
    print(f"\n📁 Результаты сохранены в: data/checkpoints/exp17/")
    print("📈 Для анализа результатов используйте соответствующие notebook'и")

if __name__ == "__main__":
    main()