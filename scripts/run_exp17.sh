#!/bin/bash

# Эксперимент 17: SGD vs GD с отслеживанием гессианов
# Основан на эксперименте 16 с улучшениями

echo "🔬 Запуск эксперимента 17: Обучение с отслеживанием гессианов"
echo "📊 Конфигурация:"
echo "   - Learning rates: 0.1, 0.01"
echo "   - Batch size: 64"
echo "   - Sample size: 1000 (увеличено с 386)"
echo "   - Отслеживание валидационных метрик"
echo ""

# Переходим в директорию скриптов
cd "$(dirname "$0")"

# Активируем виртуальное окружение если есть
if [ -f "../.venv/bin/activate" ]; then
    echo "🔧 Активируем виртуальное окружение..."
    source ../.venv/bin/activate
fi

# Создаем директорию для результатов
mkdir -p ../data/checkpoints/exp17

# Запускаем runner
echo "🚀 Запускаем все эксперименты..."
python exp17_runner.py

echo ""
echo "✅ Эксперимент 17 завершен!"
echo "📁 Результаты сохранены в: ../data/checkpoints/exp17/"
echo ""
echo "📋 Созданные файлы:"
echo "   - plateau_model_lr0.1.pth, plateau_model_lr0.01.pth"
echo "   - params_tensor_lr*.pkl, hessians_tensor_lr*.pkl"
echo "   - metadata_lr*.pkl, plateau_metadata_lr*.npy"
echo "   - Графики прогресса обучения"