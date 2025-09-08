import numpy as np
from scipy.ndimage import distance_transform_edt

def create_transition_mask(
    seismic_data_shape: tuple[int, int],
    window_indices: np.ndarray,
    transition_width_traces: int,
    transition_width_time_ms: float,
    dt_ms: float,
    transition_mode: str = 'outside'
) -> np.ndarray:
    """
    Creates a weight mask (from 0.0 to 1.0) for smooth amplification.
    """
    if transition_width_traces <= 0 or transition_width_time_ms <= 0:
        return window_indices.astype(float)

    transition_width_samples = transition_width_time_ms / dt_ms
    sampling = [1 / transition_width_traces, 1 / transition_width_samples]

    if transition_mode == 'outside':
        distances = distance_transform_edt(~window_indices, sampling=sampling)
        transition_factor = np.clip(1.0 - distances, 0, 1)
        mask = np.where(window_indices, 1.0, transition_factor)
    else:  # 'inside'
        distances = distance_transform_edt(window_indices, sampling=sampling)
        max_dist_inside = np.max(distances)
        if max_dist_inside == 0:
            return window_indices.astype(float)
        mask = distances / max_dist_inside
        mask[~window_indices] = 0.0
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
    align_width_traces: int = 10,
    align_width_time_ms: float = 50.0
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Amplifies or aligns seismic data amplitudes in the specified window.
    """
    if mode not in ['scale', 'align']:
        raise ValueError("Mode must be 'scale' or 'align'.")

    n_traces, n_time_samples = seismic_data.shape

    # Create binary mask for selected area
    window_indices = np.zeros_like(seismic_data, dtype=bool)
    if not target_window:
        return seismic_data, np.zeros_like(seismic_data), window_indices
        
    window_coords_indices = [(int(t), int(s / dt_ms)) for t, s in target_window]
    
    # "Fill" area by traces
    # Group points by traces for efficiency
    traces_map = {}
    for trace, sample in window_coords_indices:
        if trace not in traces_map:
            traces_map[trace] = []
        traces_map[trace].append(sample)
    
    for trace_idx, samples in traces_map.items():
        if samples:
            min_time, max_time = min(samples), max(samples)
            window_indices[trace_idx, min_time:max_time + 1] = True

    if not np.any(window_indices):
        return seismic_data, np.zeros_like(seismic_data), window_indices

    # Create weight mask with smooth transition
    blending_mask = create_transition_mask(
        seismic_data.shape, window_indices, transition_width_traces,
        transition_width_time_ms, dt_ms, transition_mode
    )
    
    # Determine target amplification coefficient
    target_amplification = 1.0
    if mode == 'scale':
        target_amplification = scale_factor
        
    elif mode == 'align':
        
        # Calculate RMS inside window
        rms_in_window = np.sqrt(np.mean(seismic_data[window_indices]**2))
        
        # Find boundaries by actual mask `window_indices`, not by vertices.
        selected_traces, selected_samples = np.where(window_indices)
        min_t_window, max_t_window = np.min(selected_traces), np.max(selected_traces)
        min_s_window, max_s_window = np.min(selected_samples), np.max(selected_samples)

        # Expand these boundaries to define surrounding area
        align_width_time_samples = int(align_width_time_ms / dt_ms)
        min_t_surr = max(0, min_t_window - align_width_traces)
        max_t_surr = min(n_traces, max_t_window + align_width_traces + 1)
        min_s_surr = max(0, min_s_window - align_width_time_samples)
        max_s_surr = min(n_time_samples, max_s_window + align_width_time_samples + 1)

        # Create mask for surrounding area and "cut out" inner area from it
        surrounding_mask = np.zeros_like(seismic_data, dtype=bool)
        surrounding_mask[min_t_surr:max_t_surr, min_s_surr:max_s_surr] = True
        surrounding_mask[window_indices] = False
        
        if np.any(surrounding_mask):
            rms_surrounding = np.sqrt(np.mean(seismic_data[surrounding_mask]**2))
        else:
            # If surrounding area is empty, don't change anything
            rms_surrounding = rms_in_window

        # Avoid division by zero if window is silent
        if rms_in_window > 1e-9:
            target_amplification = rms_surrounding / rms_in_window
        else:
            target_amplification = 1.0

    # Create final multiplier mask and apply
    final_multiplier_mask = 1.0 + blending_mask * (target_amplification - 1.0)
    output_data = seismic_data * final_multiplier_mask
    
    return output_data, final_multiplier_mask, window_indices