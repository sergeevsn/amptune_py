# Amptune

Library for seismic data processing with amplitude amplification and alignment capabilities in specified windows.

## Features

### Main Capabilities

- **Amplitude scaling** (`scale`) - increase or decrease amplitudes in a specified window
- **Amplitude alignment** (`align`) - adjust amplitudes in the window to match surrounding data levels
- **Smooth transitions** - create gradient masks to avoid sharp amplitude jumps

### Transition Modes

- `outside` - smooth transition from window boundaries outward
- `inside` - smooth transition from window boundaries inward

## Usage

```python
from amplify import amplify_seismic_window

# Scaling
amplified_data, mask, window = amplify_seismic_window(
    seismic_data=data,
    dt_ms=4.0,
    target_window=[(trace, time_ms), ...],
    mode='scale',
    scale_factor=2.0,
    transition_width_traces=10,
    transition_width_time_ms=40.0
)

# Alignment
aligned_data, mask, window = amplify_seismic_window(
    seismic_data=data,
    dt_ms=4.0,
    target_window=[(trace, time_ms), ...],
    mode='align',
    align_width_traces=20,
    align_width_time_ms=80.0
)
```

## Files

- `amplify.py` - main library with processing functions
- `viz_test.py` - test script with result visualization

## Requirements

- numpy
- matplotlib (for visualization)
