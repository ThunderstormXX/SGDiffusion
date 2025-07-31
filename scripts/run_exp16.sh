#!/bin/bash

# Эксперимент exp16: Обучение и логирование гессианов
# Запуск для разных learning rates

echo "🚀 Запуск эксперимента exp16"
echo "Время начала: $(date)"

# Параметры
LRS=(1e-1)
BATCH_SIZE=64
MAX_EPOCHS=60000
POST_PLATEAU_STEPS=500
BASE_SAVE_DIR="data/checkpoints/exp16"
SEED=228

# Создаем базовую директорию
mkdir -p "$BASE_SAVE_DIR"

# Запуск для каждого learning rate
for lr in "${LRS[@]}"; do
    echo ""
    echo "========================================="
    echo "🎯 Обработка learning rate: $lr"
    echo "========================================="
    
    # Создаем подкаталог для lr
    SAVE_DIR="$BASE_SAVE_DIR/lr_$lr"
    mkdir -p "$SAVE_DIR"
    
    echo "📁 Сохранение в: $SAVE_DIR"
    
    # Шаг 1: Обучение модели
    echo "🔄 Шаг 1: Обучение модели..."
    python scripts/exp16_train_to_plateau.py \
        --lr "$lr" \
        --batch-size "$BATCH_SIZE" \
        --max-epochs "$MAX_EPOCHS" \
        --save-dir "$SAVE_DIR" \
        --seed "$SEED"
    
    if [ $? -ne 0 ]; then
        echo "❌ Ошибка в обучении для lr=$lr"
        continue
    fi
    
    # Шаг 2: Продолжение обучения с логированием
    echo "🔄 Шаг 2: Продолжение обучения с логированием..."
    python scripts/exp16_continue_and_log.py \
        --lr "$lr" \
        --batch-size "$BATCH_SIZE" \
        --post-plateau-steps "$POST_PLATEAU_STEPS" \
        --save-dir "$SAVE_DIR" \
        --seed "$SEED"
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Ошибка в продолжении обучения для lr=$lr"
    else
        echo "✅ Завершена обработка lr=$lr"
    fi
done

echo ""
echo "========================================="
echo "🎉 Эксперимент exp16 завершен!"
echo "Время завершения: $(date)"
echo "📁 Результаты в: $BASE_SAVE_DIR"
echo "========================================="