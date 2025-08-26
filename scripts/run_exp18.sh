#!/bin/bash

# Эксперимент 18: SGD → GD → SGD с логированием
# Автоматический запуск для lr = 0.1 и 0.5

echo "🔬 Запуск эксперимента 18: SGD → GD → SGD"
echo "📊 Learning rates: 0.1, 0.5"
echo "📋 Этапы:"
echo "   1) SGD: 50,000 шагов"
echo "   2) GD: 10,000 шагов (к минимуму)"
echo "   3) SGD: 500 шагов (с логированием)"
echo ""

# Переход в директорию скриптов
cd "$(dirname "$0")"

# Создание директории для результатов
mkdir -p data/checkpoints/exp18

# Запуск runner скрипта
python exp18_runner.py

echo ""
echo "✅ Эксперимент 18 завершен!"
echo "📁 Результаты в: data/checkpoints/exp18/"