# Эксперимент exp16: Обучение до плато и логирование гессианов

## Описание
Эксперимент exp16 исследует поведение модели после достижения плато лосса. Использует ту же архитектуру и гиперпараметры, что и exp15.

## Структура эксперимента

### 1. `exp16_train_to_plateau.py`
Обучает модель до достижения плато лосса:
- Критерий плато: среднее изменение лосса за окно эпох < порога
- Сохраняет модель в точке плато

### 2. `exp16_continue_and_log.py`
Продолжает обучение после плато:
- На каждом шаге вычисляет гессиан по текущему батчу
- Сохраняет параметры модели и метаданные
- Логирует траекторию обучения

### 3. `exp16_runner.py`
Запускает полный цикл для нескольких learning rates:
- По умолчанию: 1e-3, 1e-4, 1e-5
- Создает отдельные подкаталоги для каждого lr

## Запуск

### Полный цикл (рекомендуется):
```bash
python scripts/exp16_runner.py
```

### Отдельные этапы:
```bash
# Обучение до плато
python scripts/exp16_train_to_plateau.py --lr 0.001

# Продолжение с логированием
python scripts/exp16_continue_and_log.py --lr 0.001
```

## Параметры по умолчанию
- Batch size: 16 (как в exp15)
- Архитектура: FlexibleMLP (8 hidden, 1 layer, downsample=6)
- Плато: окно 5 эпох, порог 1e-4
- Post-plateau шаги: 500
- Seed: 228

## Выходные файлы
- `plateau_model_lr{lr}.pth` - модель в точке плато
- `plateau_metadata_lr{lr}.npy` - метаданные обучения до плато
- `post_plateau_data_lr{lr}.pkl` - гессианы и параметры после плато