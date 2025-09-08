import numpy as np

def create_transition_mask(
    seismic_data_shape: tuple[int, int],
    window_indices: np.ndarray,
    transition_width_traces: int,
    transition_width_time_ms: float,
    dt_ms: float,
    transition_mode: str = 'outside'
) -> np.ndarray:
    """
    Создает двумерную маску с линейным градиентом для переходной зоны.

    Args:
        seismic_data_shape (tuple): Размерность сейсмических данных (n_traces, n_time_samples).
        window_indices (np.ndarray): Булева маска, где True - внутри окна.
        transition_width_traces (int): Ширина переходной зоны по трассам.
        transition_width_time_ms (float): Ширина переходной зоны по времени в мс.
        dt_ms (float): Интервал дискретизации в миллисекундах.
        transition_mode (str): Режим перехода ('outside' или 'inside').

    Returns:
        np.ndarray: Маска с плавным переходом.
    """
    if transition_mode not in ['outside', 'inside']:
        raise ValueError("Режим перехода должен быть 'outside' или 'inside'.")

    n_traces, n_time_samples = seismic_data_shape
    mask = window_indices.astype(float)
    
    transition_width_samples = transition_width_time_ms / dt_ms
    if transition_width_samples <= 0 or transition_width_traces <= 0:
        return mask

    # Находим координаты всех точек на границе окна
    boundary_mask = np.logical_xor(window_indices, np.roll(window_indices, 1, axis=0)) | \
                    np.logical_xor(window_indices, np.roll(window_indices, 1, axis=1)) | \
                    np.logical_xor(window_indices, np.roll(window_indices, 1, axis=0)) | \
                    np.logical_xor(window_indices, np.roll(window_indices, 1, axis=1))

    boundary_coords = np.argwhere(boundary_mask)

    if boundary_coords.size > 0:
        distances = np.full(seismic_data_shape, np.inf)
        
        # Определяем, какие точки нам нужно обработать в зависимости от режима
        if transition_mode == 'outside':
            target_indices = np.argwhere(~window_indices)
        else: # 'inside'
            target_indices = np.argwhere(window_indices)
        
        # Вычисление взвешенного расстояния для каждой точки
        # (Опять же, этот цикл неэффективен, но нагляден)
        for i, j in target_indices:
            dists_sq = (
                ((i - boundary_coords[:, 0]) / transition_width_traces)**2 + 
                ((j - boundary_coords[:, 1]) / transition_width_samples)**2
            )
            distances[i, j] = np.min(np.sqrt(dists_sq))

        if transition_mode == 'outside':
            # Градиент: 1 внутри, 0 вне, и плавный переход
            transition_factor = np.clip(1 - distances, 0, 1)
            mask[~window_indices] = transition_factor[~window_indices]
        else: # 'inside'
            # Градиент: 0 в центре, 1 на границе, и плавный переход
            # Используем min_dist_to_center
            # Для простоты, мы будем использовать расстояние до границы
            # и инвертируем его.
            transition_factor = np.clip(distances, 0, 1)
            mask[window_indices] = transition_factor[window_indices]
            # Остальная часть маски остается 0, т.к. вне окна нет перехода
            
    return mask

def amplify_seismic_window(
    seismic_data: np.ndarray,
    dt_ms: float,
    target_window: list[tuple[int, float]],
    mode: str,
    scale_factor: float = 1.0,
    transition_width_traces: int = 5,
    transition_width_time_ms: float = 20.0,
    transition_mode: str = 'inside', 
    align_width_traces: int = 5,
    align_width_time_ms: float = 50.0
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Амплифицирует или выравнивает амплитуды сейсмических данных в заданном окне.
    """
    if mode not in ['scale', 'align']:
        raise ValueError("Режим работы должен быть 'scale' или 'align'.")

    n_traces, n_time_samples = seismic_data.shape

    window_indices = np.zeros_like(seismic_data, dtype=bool)
    window_coords_indices = [(t, int(s / dt_ms)) for t, s in target_window]
    
    min_trace = min(c[0] for c in window_coords_indices)
    max_trace = max(c[0] for c in window_coords_indices)
    
    for trace_idx in range(min_trace, max_trace + 1):
        trace_points = sorted([c[1] for c in window_coords_indices if c[0] == trace_idx])
        if trace_points:
            min_time = trace_points[0]
            max_time = trace_points[-1]
            window_indices[trace_idx, min_time:max_time+1] = True
            
    # Передаем новый аргумент в функцию создания маски
    transition_mask = create_transition_mask(
        seismic_data.shape, window_indices, transition_width_traces, transition_width_time_ms, dt_ms, transition_mode
    )
    
    # ... (Остальная часть функции остается без изменений)
    # Код для режимов 'scale' и 'align' не меняется, т.к. он уже работает с градиентной маской,
    # и теперь эта маска просто будет создаваться по-другому в зависимости от transition_mode.
    
    output_data = seismic_data.copy()
    
    amplification_factor = 1.0
    if mode == 'scale':
        amplification_factor = 1.0 + transition_mask * (scale_factor - 1.0)
    elif mode == 'align':
        # Вычисление RMS внутри окна
        rms_in_window = np.sqrt(np.mean(seismic_data[window_indices]**2))
        
        # Вычисление RMS в окрестности
        align_width_time_samples = int(align_width_time_ms / dt_ms)
        min_t_window, max_t_window = min(c[0] for c in window_coords_indices), max(c[0] for c in window_coords_indices)
        min_s_window, max_s_window = min(c[1] for c in window_coords_indices), max(c[1] for c in window_coords_indices)
        min_t_surr = max(0, min_t_window - align_width_traces)
        max_t_surr = min(n_traces - 1, max_t_window + align_width_traces)
        min_s_surr = max(0, min_s_window - align_width_time_samples)
        max_s_surr = min(n_time_samples - 1, max_s_window + align_width_time_samples)

        surrounding_mask = np.zeros_like(seismic_data, dtype=bool)
        surrounding_mask[min_t_surr:max_t_surr+1, min_s_surr:max_s_surr+1] = True
        surrounding_mask[window_indices] = False
        
        if np.any(surrounding_mask):
            rms_surrounding = np.sqrt(np.mean(seismic_data[surrounding_mask]**2))
        else:
            rms_surrounding = 1.0

        if rms_in_window > 0:
            align_factor = rms_surrounding / rms_in_window
        else:
            align_factor = 1.0
        
        amplification_factor = 1.0 + transition_mask * (align_factor - 1.0)
    
    output_data = output_data * amplification_factor
    return output_data, transition_mask, window_indices