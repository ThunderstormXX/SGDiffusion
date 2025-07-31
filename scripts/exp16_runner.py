import os
import sys
import subprocess
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Run full exp16 cycle for multiple learning rates')
    parser.add_argument('--lrs', nargs='+', type=float, default=[1e-3, 1e-4, 1e-5], 
                       help='Learning rates to test')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--max-epochs', type=int, default=60000, help='Maximum epochs')
    parser.add_argument('--post-plateau-steps', type=int, default=500, help='Steps after plateau')
    parser.add_argument('--base-save-dir', type=str, default='data/checkpoints/exp16', 
                       help='Base save directory')
    parser.add_argument('--seed', type=int, default=228, help='Random seed')
    return parser.parse_args()

def run_command(cmd, description):
    """Запускает команду и выводит результат"""
    print(f"\n🔄 {description}")
    print(f"Команда: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} - успешно завершено")
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])  # Последние 500 символов
    else:
        print(f"❌ {description} - ошибка")
        print("STDERR:", result.stderr)
        return False
    
    return True

def main():
    args = parse_args()
    
    print("=" * 60)
    print("🚀 Запуск полного цикла эксперимента exp16")
    print(f"📅 Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Learning rates: {args.lrs}")
    print(f"📦 Batch size: {args.batch_size}")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for lr in args.lrs:
        print(f"\n{'='*40}")
        print(f"🎯 Обработка learning rate: {lr}")
        print(f"{'='*40}")
        
        # Создаем подкаталог для текущего lr
        save_dir = os.path.join(args.base_save_dir, f"lr_{lr}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Шаг 1: Обучение модели
        train_cmd = [
            'python', os.path.join(script_dir, 'exp16_train_to_plateau.py'),
            '--lr', str(lr),
            '--batch-size', str(args.batch_size),
            '--max-epochs', str(args.max_epochs),
            '--save-dir', save_dir,
            '--seed', str(args.seed)
        ]
        
        if not run_command(train_cmd, f"Обучение модели (lr={lr})"):
            print(f"❌ Пропускаем lr={lr} из-за ошибки в обучении")
            continue
        
        # Шаг 2: Продолжение обучения с логированием
        continue_cmd = [
            'python', os.path.join(script_dir, 'exp16_continue_and_log.py'),
            '--lr', str(lr),
            '--batch-size', str(args.batch_size),
            '--post-plateau-steps', str(args.post_plateau_steps),
            '--save-dir', save_dir,
            '--seed', str(args.seed)
        ]
        
        if not run_command(continue_cmd, f"Продолжение обучения (lr={lr})"):
            print(f"⚠️  Ошибка в продолжении обучения для lr={lr}")
        
        print(f"✅ Завершена обработка lr={lr}")
    
    print("\n" + "=" * 60)
    print("🎉 Полный цикл эксперимента exp16 завершен!")
    print(f"📅 Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Результаты сохранены в: {args.base_save_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()