# Amptune

Interactive seismic data processing application with amplitude amplification and alignment capabilities in specified windows.

## Features

### Main Capabilities

- **Amplitude scaling** (`scale`) - increase or decrease amplitudes in a specified window
- **Amplitude alignment** (`align`) - adjust amplitudes in the window to match surrounding data levels
- **Smooth transitions** - create gradient masks to avoid sharp amplitude jumps
- **Interactive GUI** - PyQt5 application for interactive seismic data processing

### Transition Modes

- `outside` - smooth transition from window boundaries outward
- `inside` - smooth transition from window boundaries inward

## Interactive Application

### Launching the Application

```bash
python seisamptune.py
```

### GUI Features

1. **Data Loading**: Load SEG-Y files through the "Load SEG-Y File" button
2. **Visualization**: Display seismic data in grayscale colormap
3. **Interactive Area Selection**:
   - **Point by Point**: Left mouse button - add points, right mouse button - finalize area
   - **Rectangle**: Left mouse drag to create rectangular area
4. **Real-time Processing**: When area is finalized, `amplify_seismic_window` is automatically applied
5. **Parameter Control Panel**:
   - Mode (scale/align)
   - Scale Factor (for scale mode)
   - Align Width (Traces/Time) (for align mode)
   - Transition Traces
   - Transition Time
   - Transition Mode (inside/outside)
   - Apply Changes button
6. **Undo-Redo System**: Ability to undo and redo changes for result comparison
7. **Save Results**: Save processed data in SEG-Y format

### Usage

1. Load a SEG-Y file using the "Load SEG-Y File" button
2. Select selection mode (Point by Point or Rectangle)
3. Select an area on the seismic section
4. Configure processing parameters in the right panel
5. Click "Apply Changes" to process the selected area
6. Use "Undo" and "Redo" buttons to revert and repeat changes
7. Save the result using "Save Processed Data" button

### History Management

- **Undo**: Undo the last action
- **Redo**: Redo the undone action
- **Reset**: Reset all changes to original state
- **Clear Selection**: Clear current selection
- History saves up to 20 recent states for quick result comparison

## Programmatic Usage

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
    transition_width_time_ms=40.0,
    transition_mode='inside'
)

# Alignment
aligned_data, mask, window = amplify_seismic_window(
    seismic_data=data,
    dt_ms=4.0,
    target_window=[(trace, time_ms), ...],
    mode='align',
    align_width_traces=20,
    align_width_time_ms=80.0,
    transition_width_traces=5,
    transition_width_time_ms=20.0
)
```

## File Structure

- `seisamptune.py` - main interactive PyQt5 application
- `amplify.py` - core library with processing functions
- `ioutils.py` - utilities for SEG-Y file I/O operations
- `requirements.txt` - Python dependencies

## Requirements

- numpy>=1.21.0
- matplotlib>=3.5.0
- PyQt5>=5.15.0
- segyio>=1.9.0
- scipy (for distance transforms)

## Installation

```bash
pip install -r requirements.txt
```

## Key Features

### Interactive Selection
- **Point-by-point selection**: Click to add points, right-click to finalize polygon
- **Rectangle selection**: Drag to create rectangular selection area
- **Visual feedback**: Selected areas are highlighted with red lines/rectangles

### Processing Modes
- **Scale mode**: Multiply amplitudes by a specified factor
- **Align mode**: Adjust amplitudes to match surrounding data RMS levels

### Smooth Transitions
- Configurable transition zones to avoid sharp amplitude boundaries
- Inside/outside transition modes for different blending effects
- Adjustable transition width in both trace and time dimensions

### Data Management
- Full undo/redo history with up to 20 states
- Reset to original data functionality
- Save processed data while preserving original trace headers