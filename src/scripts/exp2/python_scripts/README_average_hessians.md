# Average Hessians Script

Скрипт `average_hessians.py` предназначен для усреднения гессианов по траектории обучения.

## Описание

Скрипт загружает сохраненные гессианы размерности `(n_steps, n_params, n_params)` и вычисляет их среднее по первой размерности (траектории), получая усредненный гессиан размерности `(n_params, n_params)`.

## Использование

### Обработка по номеру setup (НОВАЯ ФУНКЦИЯ!)

```bash
# Обработать все гессианы для setup1
python average_hessians.py --setup 1

# Обработать все гессианы для setup5 с кастомным суффиксом
python average_hessians.py --setup 5 --output_suffix _mean

# Обработать setup3 в кастомной базовой директории
python average_hessians.py --setup 3 --results_dir path/to/base/directory
```

### Обработка всех файлов в директории

```bash
python average_hessians.py --results_dir src/scripts/exp2/exp_results
```

### Обработка одного файла

```bash
python average_hessians.py --results_dir src/scripts/exp2/exp_results --single_file hessians_traj_lr0.1.pt
```

### Кастомный суффикс для выходных файлов

```bash
python average_hessians.py --results_dir src/scripts/exp2/exp_results --output_suffix _mean
```

## Параметры

- `--results_dir`: Директория с файлами гессианов или базовая директория для поиска setup (по умолчанию: `src/scripts/exp2/exp_results`)
- `--setup`: Номер setup для обработки (например, 1 для setup1) - автоматически найдет все директории setup{номер}
- `--single_file`: Имя одного файла для обработки (опциональный)
- `--output_suffix`: Суффикс для выходных файлов (по умолчанию: `_averaged`)

## Входные файлы

Скрипт ожидает файлы с именами вида `hessians_traj_lr{lr}.pt`, содержащие тензоры размерности `(n_steps, n_params, n_params)`.

## Выходные файлы

Создает файлы с именами вида `hessians_traj_lr{lr}_averaged.pt`, содержащие усредненные гессианы размерности `(n_params, n_params)`.

## Логика работы с setup

При использовании параметра `--setup`, скрипт:

1. Ищет все директории, содержащие `setup{номер}` в базовой директории
2. В каждой найденной директории ищет файлы `hessians_traj_lr*.pt` (исключая уже усредненные)
3. Обрабатывает все найденные файлы, создавая усредненные версии в тех же директориях

Например, для `--setup 1` будут найдены и обработаны:
- `setup1/hessians_traj_lr0.1.pt` → `setup1/hessians_traj_lr0.1_averaged.pt`
- `setup1_run2/hessians_traj_lr0.01.pt` → `setup1_run2/hessians_traj_lr0.01_averaged.pt`
- `my_setup1_test/hessians_traj_lr0.001.pt` → `my_setup1_test/hessians_traj_lr0.001_averaged.pt`

## Дополнительная информация

Скрипт также выводит статистики собственных значений для каждого усредненного гессиана:
- Максимальное собственное значение
- Минимальное собственное значение  
- Количество положительных собственных значений
- Количество отрицательных собственных значений
