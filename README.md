# Amptune

Библиотека для обработки сейсмических данных с возможностью амплификации и выравнивания амплитуд в заданных окнах.

## Функционал

### Основные возможности

- **Масштабирование амплитуд** (`scale`) - увеличение или уменьшение амплитуд в заданном окне
- **Выравнивание амплитуд** (`align`) - приведение амплитуд в окне к уровню окружающих данных
- **Плавные переходы** - создание градиентных масок для избежания резких скачков амплитуд

### Режимы переходов

- `outside` - плавный переход от границ окна наружу
- `inside` - плавный переход от границ окна внутрь

## Использование

```python
from amplify import amplify_seismic_window

# Масштабирование
amplified_data, mask, window = amplify_seismic_window(
    seismic_data=data,
    dt_ms=4.0,
    target_window=[(trace, time_ms), ...],
    mode='scale',
    scale_factor=2.0,
    transition_width_traces=10,
    transition_width_time_ms=40.0
)

# Выравнивание
aligned_data, mask, window = amplify_seismic_window(
    seismic_data=data,
    dt_ms=4.0,
    target_window=[(trace, time_ms), ...],
    mode='align',
    align_width_traces=20,
    align_width_time_ms=80.0
)
```

## Файлы

- `amplify.py` - основная библиотека с функциями обработки
- `viz_test.py` - тестовый скрипт с визуализацией результатов

## Требования

- numpy
- matplotlib (для визуализации)
