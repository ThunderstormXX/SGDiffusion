# Подробное описание экспериментального сетапа SGDiffusion

## Обзор

Проект SGDiffusion содержит систематические эксперименты по изучению поведения Stochastic Gradient Descent (SGD) и Gradient Descent (GD) на различных конфигурациях моделей и данных. Эксперименты организованы в модульную структуру с несколькими сетапами для каждого эксперимента.

## Структура экспериментов

### Эксперимент 1 (exp1) - Анализ MNIST

Основной эксперимент, изучающий поведение SGD/GD на данных MNIST с использованием MLP и CNN моделей.

#### Основные скрипты запуска

##### 1. `src/scripts/exp1/bash/run_full_setup.sh`

**Назначение**: Главный скрипт для полного запуска всех этапов эксперимента.

**Использование**: 
```bash
./run_full_setup.sh <SETUP_NUM>
```

**Что происходит**:
1. Принимает номер сетапа (по умолчанию 1)
2. Последовательно выполняет 5 этапов эксперимента:
   - Этап 1: SGD тренировка
   - Этап 2: GD тренировка 
   - Этап 3: Анализ траекторий гессианов
   - Этап 4: Множественные запуски SGD
   - Этап 5: Множественные запуски GD

##### 2. `src/scripts/exp1/bash/run_partial_setup.sh`

**Назначение**: Выполняет отдельный этап эксперимента.

**Использование**:
```bash
./run_partial_setup.sh <SETUP_NUM> <SCRIPT_NUM>
```

**Детальное описание этапов**:

**Этап 1 - SGD Тренировка**:
- Запускает `train_sgd.py`
- Тренирует модель с заданными параметрами
- Сохраняет checkpoint как `initial_after_sgd.pt`
- Логирует loss траектории

**Этап 2 - GD Тренировка**:  
- Запускает `train_gd.py`
- Загружает checkpoint после SGD
- Продолжает тренировку методом GD
- Сохраняет финальный checkpoint как `initial_after_sgd_and_gd.pt`

**Этап 3 - Анализ гессианов**:
- Запускает `sgd_hessians.py`  
- Вычисляет гессианы в процессе SGD тренировки
- Анализирует собственные значения гессианов
- Изучает траектории для разных learning rates

**Этап 4 - Множественные запуски SGD**:
- Запускает `sgd_many_runs.py`
- Выполняет много независимых запусков SGD
- Собирает статистику по разным инициализациям
- Анализирует распределение траекторий

**Этап 5 - Множественные запуски GD**:
- Запускает `gd_many_runs.py`
- Выполняет детерминированные запуски GD
- Сравнивает поведение с SGD
- Применяет различные learning rate scaling

##### 3. Дополнительные скрипты

**`run_gd_many_runs.sh`**:
- Отдельный запуск только 4-го этапа (множественные GD запуски)
- Полезен для дополнительного анализа

**`run_val_loss_distribution.sh`**:
- Анализ распределения validation loss
- Запускает `val_loss_distribution.py` для каждого learning rate
- Генерирует визуализации распределений

### Конфигурационные файлы

Каждый сетап имеет свой конфигурационный файл `config.sh` со специфичными параметрами:

#### Setup 1 (`setup1/config.sh`):
```bash
DATASET_TRAIN="mnist"
DATASET_VAL="mnist"  
MODEL="mlp"
BATCH_SIZE=64
SAMPLE_SIZE=6400
LR=0.1
EPOCHS_SGD=10000
EPOCHS_GD=10000
SEED=42
DATALOADER="default"
LRS_LIST="0.05,0.01,0.005,0.001"
DTYPE="float64"
DEVICE="cpu"
```

#### Setup 2 (`setup2/config.sh`):
```bash
DATASET_TRAIN="mnist"
DATASET_VAL="mnist"
MODEL="mlp" 
BATCH_SIZE=64
SAMPLE_SIZE=6400
LR=0.1
EPOCHS_SGD=10000
EPOCHS_GD=10000
SEED=42
DATALOADER="replacement"    # ключевое отличие
LRS_LIST="0.05,0.01,0.005,0.001"
DTYPE="float64" 
DEVICE="cpu"
```

#### Setup 3 (`setup3/config.sh`):
```bash
DATASET_TRAIN="mnist"
DATASET_VAL="mnist"
MODEL="cnn"                 # CNN вместо MLP
BATCH_SIZE=64
SAMPLE_SIZE=6400
LR=0.1
EPOCHS_SGD=1000            # Меньше эпох
EPOCHS_GD=1000
SEED=42
LRS_LIST="0.1,0.01"        # Меньше learning rates
```

#### Setup 4 (`setup4/config.sh`):
```bash
DATASET_TRAIN="mnist_similar"  # Модифицированные данные
DATASET_VAL="mnist"
MODEL="cnn"
BATCH_SIZE=64
SAMPLE_SIZE=6400
LR=0.1
EPOCHS_SGD=1000
EPOCHS_GD=1000
SEED=42
DATALOADER="replacement"
LRS_LIST="0.1,0.01"
```

#### Setup 6 (`setup6/config.sh`):
```bash
DATASET_TRAIN="mnist"
DATASET_VAL="mnist"
MODEL="mlp"
BATCH_SIZE=64
SAMPLE_SIZE=6400
LR=0.1
EPOCHS_SGD=1000
EPOCHS_GD=1000
SEED=42
DATALOADER="replacement"
MANY_RUNS_SAMPLES=1        # Только 1 семпл для GD (детерминирован)
LRS_LIST="0.1,0.01"
```

#### Setup 7 (`setup7/config.sh`):
```bash
DATASET_TRAIN="mnist"
DATASET_VAL="mnist"
MODEL="mlp"
BATCH_SIZE=64
SAMPLE_SIZE=6400
LR=0.1
EPOCHS_SGD=10000
EPOCHS_GD=10000
SEED=42
DATALOADER="replacement"
LRS_LIST="0.1,0.01"
DTYPE="float32"           # Пониженная точность
```

#### Setup 8 (`setup8/config.sh`):
```bash
DATASET_TRAIN="mnist"
DATASET_VAL="mnist"
MODEL="mlp"
BATCH_SIZE=64
SAMPLE_SIZE=6400
LR=0.1
EPOCHS_SGD=10000
EPOCHS_GD=10000
SEED=42
DATALOADER="default"
LRS_LIST="0.1,0.01"
DTYPE="float32"          # float32 с default dataloader
```

## Эксперимент 2 (exp2) - Анализ Shakespeare

Аналогичная экспериментальная структура, но для текстовых данных и модели NanoGPT. Изучает поведение SGD/GD на задаче language modeling.

### Датасет Shakespeare

**Источник данных**: 
- URL: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- Классический датасет произведений Шекспира для character-level language modeling

**Препроцессинг данных**:

1. **Ограничение размера**: Берутся первые 6000 символов из текста
2. **Словарь**: Ограничивается 25 символами (a-y + пробел)
   - Все остальные символы заменяются на пробел
   - Все символы приводятся к нижнему регистру
3. **Токенизация**: Character-level (каждый символ = отдельный токен)
4. **Создание последовательностей**: 
   - Размер блока: 16 символов (block_size=16)
   - Каждая последовательность: input[i:i+16] → target[i+1:i+17]
   - Классическое autoregressive language modeling
5. **Train/Val split**: 80% train, 20% validation

**Статистики датасета**:
```
Исходный текст: ~6000 символов
Словарь: 25 символов ['abcdefghijklmnopqrstuvwxy ']
Последовательности: ~5984 примера (длиной 16)
Train: ~4787 последовательностей
Val: ~1197 последовательностей
```

**Пример обработанного текста**:
```
Оригинал: "First Citizen: Before we proceed any further, hear me speak."
После обработки: "first citizen  before we proceed any further  hear me speak "
```

### Конфигурации exp2

#### Setup 1:
```bash
DATASET_TRAIN="shakespeare"
DATASET_VAL="shakespeare" 
MODEL="nanogpt"
BATCH_SIZE=64
LR=0.01
EPOCHS_SGD=10000
EPOCHS_GD=10000
DATALOADER="default"          # Стандартный sampling
MANY_RUNS_SAMPLES=10          # Меньше runs для экономии времени
LRS_LIST="0.001"              # Один learning rate
```

#### Setup 2:
```bash
DATASET_TRAIN="shakespeare"
DATASET_VAL="shakespeare"
MODEL="nanogpt" 
BATCH_SIZE=64
LR=0.01
EPOCHS_SGD=10000
EPOCHS_GD=10000
DATALOADER="replacement"      # Sampling с возвратом
MANY_RUNS_SAMPLES=1000        # Больше runs для статистики
LRS_LIST="0.01"               # Другой learning rate
```

### Особенности эксперимента 2

1. **Задача**: Character-level language modeling (предсказание следующего символа)
2. **Метрика**: Cross-entropy loss на последовательностях
3. **Архитектура**: Causal transformer с attention механизмом
4. **Размер модели**: ~1000 параметров (vs ~100-200 в MLP/CNN из exp1)
5. **Данные**: Последовательная структура vs независимые примеры в MNIST
6. **Complexity**: Изучение attention patterns и position embeddings

## Архитектура моделей

### MLP (Multi-Layer Perceptron):
```python
FlexibleMLP(
    hidden_dim=8, 
    num_hidden_layers=1, 
    input_downsample=6
)
```

### CNN (Convolutional Neural Network):
```python
FlexibleCNN(
    in_channels=1, 
    conv_channels=[12], 
    conv_kernels=[3], 
    conv_strides=[1],
    conv_use_relu_list=[True], 
    conv_dropouts=[0.0], 
    conv_use_bn=True,
    pool_after=[False], 
    gap_size=1, 
    mlp_hidden_dims=[11],
    mlp_use_relu_list=[True], 
    mlp_dropouts=[0.0], 
    output_dim=10
)
```

### NanoGPT (Simplified Transformer):

**Основная архитектура**:
```python
NanoGPT(
    vocab_size=25,      # Размер словаря (символы a-y + пробел)
    n_embd=8,           # Размерность эмбеддингов
    n_head=1,           # Количество attention голов
    n_layer=1,          # Количество transformer блоков
    block_size=16,      # Максимальная длина последовательности
    mlp_ratio=1         # Коэффициент расширения MLP
)
```

**Детали архитектуры**:

1. **Token и Position Embeddings**:
   - `wte`: Embedding для токенов (`vocab_size × n_embd`)
   - `wpe`: Embedding для позиций (`block_size × n_embd`)

2. **Transformer блоки** (повторяются `n_layer` раз):
   - **Layer Normalization** перед каждым подблоком
   - **Causal Self-Attention**:
     - Вычисляет Q, K, V через одну линейную проекцию
     - Применяет каузальную маску (треугольная матрица)
     - Scaled dot-product attention
     - Выходная проекция
   - **Feed-Forward Network (MLP)**:
     - Линейный слой с расширением (`n_embd → mlp_ratio * n_embd`)
     - GELU активация
     - Проекция обратно (`mlp_ratio * n_embd → n_embd`)
   - **Residual connections** вокруг attention и MLP

3. **Language Modeling Head**:
   - Финальная Layer Normalization
   - Линейная проекция в словарь (`n_embd → vocab_size`)

**Целевое количество параметров**: ~1000 параметров
- При конфигурации по умолчанию (n_layer=1, n_head=1, n_embd=8, vocab_size=25, block_size=16)

**Инициализация весов**:
- Линейные слои: Normal(0, 0.02)
- Embeddings: Normal(0, 0.02)
- Bias: Zeros (где применимо)

**Основные возможности**:
- **Обучение**: Стандартная language modeling задача с cross-entropy loss
- **Генерация**: Autoregressive генерация с температурным семплированием
- **Causal Masking**: Обеспечивает автогрессивное поведение

**Особенности реализации**:
- Упрощенная версия GPT для исследовательских целей
- Нет dropout'а для детерминированности экспериментов
- Оптимизирована для малого количества параметров
- Поддержка различных длин контекста (до `block_size`)
- Совместима с SGD/GD оптимизацией из основных экспериментов

## Ключевые параметры экспериментов

### Общие параметры:
- **BATCH_SIZE**: Размер батча (обычно 64)
- **SAMPLE_SIZE**: Размер выборки (обычно 6400) 
- **SEED**: Фиксированный seed для воспроизводимости (42)
- **EPOCHS_SGD/GD**: Количество эпох тренировки
- **HESSIAN_STEPS**: Шаги для анализа гессианов (1000)
- **MANY_RUNS_STEPS**: Шаги для множественных запусков (1111)
- **MANY_RUNS_SAMPLES**: Количество независимых запусков

### Специфические параметры:
- **DATALOADER**: 
  - `"default"` - стандартный PyTorch DataLoader
  - `"replacement"` - семплирование с возвратом
- **GD_SCALING**: Масштабирование learning rate для GD
- **LRS_LIST**: Список learning rates для анализа
- **DTYPE**: Точность вычислений (float32/float64)
- **DEVICE**: Устройство вычислений (cpu/cuda/mps)

## Результаты и вывод

Результаты сохраняются в структуре:
```
src/scripts/exp1/exp_results/setup{N}/
├── initial_after_sgd.pt              # Checkpoint после SGD
├── initial_after_sgd_and_gd.pt       # Checkpoint после SGD+GD  
├── sgd_weights_lr{X}.pt               # Веса для каждого LR
├── sgd_losses_lr{X}.json              # Траектории loss
├── hessian_eigenvalues_lr{X}.json     # Собственные значения гессианов
├── many_runs_sgd_results_lr{X}.json   # Результаты множественных запусков SGD
├── many_runs_gd_results_lr{X}.json    # Результаты множественных запусков GD
└── val_analysis_lr{X}/                # Анализ validation loss
```

## Запуск экспериментов

### Полный эксперимент:
```bash
cd src/scripts/exp1/bash
./run_full_setup.sh 2  # Запуск setup2
```

### Отдельный этап:
```bash
cd src/scripts/exp1/bash  
./run_partial_setup.sh 2 3  # Setup2, этап 3 (гессианы)
```

### Анализ validation loss:
```bash
cd src/scripts/exp1/bash
./run_val_loss_distribution.sh 2  # Анализ для setup2
```

## Сравнение экспериментов

| Аспект | Эксперимент 1 (MNIST) | Эксперимент 2 (Shakespeare) |
|--------|------------------------|------------------------------|
| **Данные** | Изображения 28×28, классификация | Текстовые последовательности, генерация |
| **Размер данных** | 6400 примеров MNIST | 6000 символов → ~5984 последовательности |
| **Задача** | Многоклассовая классификация (10 классов) | Character-level language modeling |
| **Архитектуры** | MLP (8 hidden) и CNN (12 channels) | NanoGPT (8 embd, 1 head, 1 layer) |
| **Параметры** | ~100-200 параметров | ~1000 параметров |
| **Loss функция** | CrossEntropy для классификации | CrossEntropy для sequence prediction |
| **Структура данных** | Независимые примеры (i.i.d.) | Последовательные зависимые токены |
| **Learning rates** | 0.001-0.1 (широкий диапазон) | 0.001-0.01 (узкий диапазон) |
| **Сложность** | Простые feed-forward сети | Attention mechanism + embeddings |

## Научные цели

### Общие для обоих экспериментов:

1. **Сравнение SGD vs GD**: Различия в поведении и конвергенции
2. **Анализ гессианов**: Эволюция собственных значений в процессе обучения  
3. **Влияние гиперпараметров**: Batch size, learning rate, архитектура
4. **Стохастичность**: Роль шума в SGD оптимизации
5. **Scaling поведение**: Как параметры влияют на траектории обучения
6. **Репликируемость**: Статистика по множественным независимым запускам

### Специфичные для exp1 (MNIST):
- **Архитектурные различия**: MLP vs CNN поведение
- **Данные**: влияние `mnist_similar` vs обычного MNIST
- **DataLoader**: `default` vs `replacement` семплирование
- **Точность**: влияние float32 vs float64

### Специфичные для exp2 (Shakespeare):
- **Attention patterns**: как эволюционируют веса attention
- **Position embeddings**: влияние позиционной информации
- **Sequential dependencies**: роль временных зависимостей в оптимизации
- **Language modeling**: специфика задач генерации vs классификации
- **Token embeddings**: динамика обучения представлений символов

### Сравнительный анализ:
- **Landscape сложность**: Различия в loss поверхности между CV и NLP задачами
- **Размер модели**: Влияние количества параметров на SGD/GD поведение
- **Структура данных**: i.i.d. vs sequential влияние на оптимизацию
- **Generalization**: Как разные архитектуры влияют на обобщающую способность

Вся система спроектирована для систематического и воспроизводимого изучения фундаментальных аспектов оптимизации в глубоком обучении на разнообразных задачах и архитектурах.
