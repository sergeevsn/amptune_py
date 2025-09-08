import numpy as np
import matplotlib.pyplot as plt
from amplify import amplify_seismic_window



def plot_results(
    original_data: np.ndarray,
    amplified_data: np.ndarray,
    mask: np.ndarray,
    title: str,
    vmin: float,
    vmax: float
):
    """Вспомогательная функция для построения графиков."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1, ax2, ax3 = axes
    
    # 1. Исходные данные
    ax1.imshow(original_data, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
    ax1.set_title(f'1. Исходные данные ({title})')
    ax1.set_xlabel('Сэмплы времени')
    ax1.set_ylabel('Трассы')
    ax1.set_xlim(0, original_data.shape[1])
    ax1.set_ylim(original_data.shape[0], 0)
    
    # 2. Градиентная маска
    im = ax2.imshow(mask, aspect='auto', cmap='gray', vmin=0, vmax=1)
    ax2.set_title('2. Двумерная градиентная маска')
    ax2.set_xlabel('Сэмплы времени')
    ax2.set_xlim(0, original_data.shape[1])
    ax2.set_ylim(original_data.shape[0], 0)
    
    # 3. Результат
    ax3.imshow(amplified_data, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
    ax3.set_title(f'3. Результат ({title})')
    ax3.set_xlabel('Сэмплы времени')
    ax3.set_xlim(0, original_data.shape[1])
    ax3.set_ylim(original_data.shape[0], 0)

    fig.colorbar(im, ax=ax2, orientation='vertical', label='Коэффициент перехода (0-1)')
    plt.tight_layout()
    plt.show()

    

if __name__ == '__main__':
    # Параметры синтетических данных
    n_traces = 100
    n_time_samples = 300
    dt_ms = 4.0
    
    # Определение окна
    target_window = []
    for t in range(35, 66):
        target_window.append((t, 100 * dt_ms))
        target_window.append((t, 200 * dt_ms))
    for s_ms in np.arange(100 * dt_ms, 201 * dt_ms, dt_ms):
        target_window.append((35, s_ms))
        target_window.append((65, s_ms))
    
    # --- Тест режима 'scale' ---
    print("--- Тестирование режима 'scale' ---")
    seismic_data_scale = np.random.randn(n_traces, n_time_samples) * 0.1
    seismic_data_scale[35:65, 100:200] += np.sin(np.linspace(0, 20, 100)) * np.cos(np.linspace(0, 10, 30))[:, np.newaxis] * 2.0
    scale_factor = 3.0
    transition_width_traces = 10
    transition_width_time_ms = 40.0
    
    amplified_data_scale, transition_mask_scale, _ = amplify_seismic_window(
        seismic_data=seismic_data_scale,
        dt_ms=dt_ms,
        target_window=target_window,
        mode='scale',
        scale_factor=scale_factor,
        transition_width_traces=transition_width_traces,
        transition_width_time_ms=transition_width_time_ms
    )
    plot_results(seismic_data_scale, amplified_data_scale, transition_mask_scale, 'Масштабирование', -3, 3)

    # --- Тест режима 'align' ---
    print("\n--- Тестирование режима 'align' ---")
    # Создаем данные для теста: низкий фон и очень высокий сигнал в окне
    seismic_data_align = np.random.randn(n_traces, n_time_samples) * 0.2  # Низкий фон
    seismic_data_align[35:65, 100:200] += np.sin(np.linspace(0, 20, 100)) * np.cos(np.linspace(0, 10, 30))[:, np.newaxis] * 5.0 # Высокий сигнал
    
    align_width_traces = 20
    align_width_time_ms = 80.0
    
    amplified_data_align, transition_mask_align, _ = amplify_seismic_window(
        seismic_data=seismic_data_align,
        dt_ms=dt_ms,
        target_window=target_window,
        mode='align',
        transition_width_traces=transition_width_traces,
        transition_width_time_ms=transition_width_time_ms,
        align_width_traces=align_width_traces,
        align_width_time_ms=align_width_time_ms
    )
    
    # Выводим RMS для проверки
    rms_in_window_original = np.sqrt(np.mean(seismic_data_align[35:65, 100:200]**2))
    rms_surrounding_original = np.sqrt(np.mean(seismic_data_align[seismic_data_align < 3]**2))
    rms_in_window_aligned = np.sqrt(np.mean(amplified_data_align[35:65, 100:200]**2))
    
    print(f"Оригинальный RMS в окне: {rms_in_window_original:.2f}")
    print(f"Оригинальный RMS в окрестности: {rms_surrounding_original:.2f}")
    print(f"RMS в окне после выравнивания: {rms_in_window_aligned:.2f}")

    plot_results(seismic_data_align, amplified_data_align, transition_mask_align, 'Выравнивание', -1, 1)

     # --- Тест режима 'inside' ---
    print("\n--- Тестирование режима 'inside' ---")
    seismic_data_inside = np.random.randn(n_traces, n_time_samples) * 0.1
    seismic_data_inside[35:65, 100:200] += np.sin(np.linspace(0, 20, 100)) * np.cos(np.linspace(0, 10, 30))[:, np.newaxis] * 5.0

    amplified_data_inside, transition_mask_inside, _ = amplify_seismic_window(
        seismic_data=seismic_data_inside,
        dt_ms=dt_ms,
        target_window=target_window,
        mode='scale',
        scale_factor=0.2, # Уменьшаем амплитуду в центре
        transition_width_traces=10,
        transition_width_time_ms=40.0,
        transition_mode='inside' # Включаем новый режим
    )
    
    plot_results(seismic_data_inside, amplified_data_inside, transition_mask_inside, 'Переход внутри', -5, 5)