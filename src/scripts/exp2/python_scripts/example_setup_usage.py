#!/usr/bin/env python3
"""
Пример использования скрипта average_hessians.py с параметром --setup
"""
import os
import subprocess
import sys

def run_command(cmd):
    """Запустить команду и показать результат"""
    print(f"\n{'='*50}")
    print(f"Выполняется: {cmd}")
    print('='*50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Return code: {result.returncode}")
    except Exception as e:
        print(f"Ошибка выполнения команды: {e}")

def main():
    print("Примеры использования average_hessians.py с параметром --setup")
    
    # Базовая директория для примеров
    base_dir = "src/scripts/exp2/exp_results"
    
    examples = [
        # Основные примеры
        "python average_hessians.py --setup 1",
        "python average_hessians.py --setup 2 --output_suffix _mean",
        
        # С кастомной директорией
        f"python average_hessians.py --setup 1 --results_dir {base_dir}",
        
        # Показать help
        "python average_hessians.py --help",
    ]
    
    print("Доступные команды:")
    for i, cmd in enumerate(examples, 1):
        print(f"{i}. {cmd}")
    
    print("\nВыберите номер команды для выполнения (или 'all' для всех, 'q' для выхода):")
    
    while True:
        choice = input("> ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 'all':
            for cmd in examples:
                run_command(cmd)
            break
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(examples):
                    run_command(examples[idx])
                else:
                    print(f"Неверный номер. Выберите от 1 до {len(examples)}")
            except ValueError:
                print("Введите число, 'all' или 'q'")

if __name__ == '__main__':
    main()
