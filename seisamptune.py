import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QComboBox, QDoubleSpinBox, QSpinBox, QGroupBox,
                             QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# These modules are assumed to be in the same directory or installed
import ioutils
from amplify import amplify_seismic_window

class SeismicCanvas(FigureCanvas):
    """Interactive canvas"""

    window_selected = pyqtSignal(list)
    SELECTION_TAG = "seismic_selection_gid"

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Trace Number')
        self.ax.set_ylabel('Time (ms)')
        self.ax.set_title('Seismic Data')
        self.data = None
        self.processed_data = None
        self.sample_interval = None
        self.vmin = None
        self.vmax = None
        self.selection_mode = 'point'
        self.points = []
        self.rect_start = None
        self.current_rect_artist = None
        self.dragging = False
        self.mpl_connect('button_press_event', self.on_click)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)

    def set_data(self, data, sample_interval):
        self.data = data
        self.processed_data = data.copy()
        self.sample_interval = sample_interval
        self.vmin, self.vmax = np.percentile(data, [1, 99])
        self.plot_data()

    def plot_data(self):
        if self.data is None: return
        self.ax.clear()
        n_traces, n_samples = self.data.shape
        time_ms = np.arange(n_samples) * self.sample_interval * 1000
        self.im = self.ax.imshow(self.processed_data.T, aspect='auto', cmap='gray',
                                vmin=self.vmin, vmax=self.vmax,
                                extent=[0, n_traces, time_ms[-1], time_ms[0]])
        self.ax.set_xlabel('Trace Number')
        self.ax.set_ylabel('Time (ms)')
        self.ax.set_title('Seismic Data')
        if not hasattr(self, 'colorbar') or self.colorbar.ax.get_figure() != self.fig:
            self.colorbar = self.fig.colorbar(self.im, ax=self.ax, label='Amplitude')
        else:
            self.colorbar.update_normal(self.im)
        self.draw()

    def update_processed_data(self, new_data):
        self.processed_data = new_data
        if hasattr(self, 'im') and self.im is not None:
            self.im.set_array(new_data.T)
            self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
            if hasattr(self, 'colorbar'):
                self.colorbar.update_normal(self.im)
            self.draw()
        else:
            self.plot_data()

    def set_selection_mode(self, mode):
        self.selection_mode = mode
        self.clear_selection()

    def clear_selection(self):
        self.points = []
        self.rect_start = None
        self.dragging = False
        for artist in self.ax.lines[:]:
            if artist.get_gid() == self.SELECTION_TAG:
                artist.remove()
        for artist in self.ax.patches[:]:
            if artist.get_gid() == self.SELECTION_TAG:
                artist.remove()
        if self.current_rect_artist:
            self.current_rect_artist.remove()
            self.current_rect_artist = None
        self.draw()

    def on_click(self, event):
        if event.inaxes != self.ax or self.data is None: return

        if event.button == 1:  # Left click
            if self.selection_mode == 'point':
                if not self.points:
                    self.clear_selection()

                trace_idx = int(event.xdata)
                time_ms = event.ydata
                self.points.append((trace_idx, time_ms))
                self.ax.plot(trace_idx, time_ms, 'ro', markersize=6, gid=self.SELECTION_TAG)
                if len(self.points) > 1:
                    p1, p2 = self.points[-2:]
                    self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=1.5, gid=self.SELECTION_TAG)
                self.draw()

            elif self.selection_mode == 'rectangle':
                # For rectangle, each click starts a new selection, so we always clear
                self.clear_selection()
                self.rect_start = (event.xdata, event.ydata)
                self.dragging = True

        elif event.button == 3 and self.selection_mode == 'point':  # Right click
            if len(self.points) >= 2:
                self.finalize_selection()
            else:
                self.clear_selection()
    
    def on_release(self, event):
        if event.button == 1 and self.dragging and self.selection_mode == 'rectangle':
            self.dragging = False
            self.finalize_rectangle(event.xdata, event.ydata)

    def on_motion(self, event):
        if not self.dragging or not self.rect_start or event.inaxes != self.ax: return
        if self.current_rect_artist: self.current_rect_artist.remove()
        width = event.xdata - self.rect_start[0]
        height = event.ydata - self.rect_start[1]
        self.current_rect_artist = Rectangle(self.rect_start, width, height,
                                           linewidth=1.5, edgecolor='red',
                                           facecolor='none', linestyle='--')
        self.ax.add_patch(self.current_rect_artist)
        self.draw()

    def finalize_rectangle(self, x_end, y_end):
        if not self.rect_start: return
        x_min, x_max = sorted((self.rect_start[0], x_end))
        y_min, y_max = sorted((self.rect_start[1], y_end))
        self.points = [(x_min, y_min), (x_max, y_max)]
        self.finalize_selection()

    def finalize_selection(self):
        if len(self.points) < 2:
            self.clear_selection()
            return
        if self.current_rect_artist:
            self.current_rect_artist.remove()
            self.current_rect_artist = None
        rasterized_points = []
        if self.selection_mode == 'point':
            vertices = np.array(self.points)
            min_trace, max_trace = int(vertices[:, 0].min()), int(vertices[:, 0].max())
            closed_vertices = np.vstack([vertices, vertices[0]])
            for trace_idx in range(min_trace, max_trace + 1):
                intersections = []
                for i in range(len(closed_vertices) - 1):
                    p1, p2 = closed_vertices[i], closed_vertices[i+1]
                    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
                    if x1 > x2: x1, y1, x2, y2 = x2, y2, x1, y1
                    if trace_idx >= x1 and trace_idx < x2 and x2 - x1 != 0:
                        y_intersect = y1 + (trace_idx - x1) * (y2 - y1) / (x2 - x1)
                        intersections.append(y_intersect)
                intersections.sort()
                for i in range(0, len(intersections), 2):
                    if i + 1 < len(intersections):
                        rasterized_points.append((trace_idx, intersections[i]))
                        rasterized_points.append((trace_idx, intersections[i+1]))
            self.ax.plot(closed_vertices[:, 0], closed_vertices[:, 1], 'r--', linewidth=2, gid=self.SELECTION_TAG)
        elif self.selection_mode == 'rectangle':
            x_min, y_min = self.points[0]
            x_max, y_max = self.points[1]
            for trace_idx in range(int(x_min), int(x_max) + 1):
                rasterized_points.append((trace_idx, y_min))
                rasterized_points.append((trace_idx, y_max))
            width, height = x_max - x_min, y_max - y_min
            final_rect = Rectangle((x_min, y_min), width, height, linewidth=2,
                                   edgecolor='red', facecolor='none', gid=self.SELECTION_TAG)
            self.ax.add_patch(final_rect)
        self.draw()
        if rasterized_points:
            self.window_selected.emit(rasterized_points)

class SeismicApp(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Seismic Data Amplification Tuning Tool')
        self.setGeometry(100, 100, 1400, 800)

        self.original_data = None
        self.current_data = None
        self.sample_interval = None
        self.original_file_path = None
        self.history = []
        self.history_index = -1
        self.max_history_size = 20
        self.last_selected_points = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        left_panel = QVBoxLayout()
        control_layout = QHBoxLayout()
        self.load_btn = QPushButton('Load SEG-Y File')
        self.save_btn = QPushButton('Save Processed Data')
        self.reset_btn = QPushButton('Reset')
        self.clear_selection_btn = QPushButton('Clear Selection')
        self.undo_btn = QPushButton('Undo')
        self.redo_btn = QPushButton('Redo')
        self.load_btn.clicked.connect(self.load_file)
        self.save_btn.clicked.connect(self.save_file)
        self.reset_btn.clicked.connect(self.reset_data)
        self.clear_selection_btn.clicked.connect(self.clear_current_selection)
        self.undo_btn.clicked.connect(self.undo_action)
        self.redo_btn.clicked.connect(self.redo_action)
        for btn in [self.save_btn, self.reset_btn, self.clear_selection_btn, self.undo_btn, self.redo_btn]:
            btn.setEnabled(False)
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.save_btn)
        control_layout.addWidget(self.reset_btn)
        control_layout.addWidget(self.clear_selection_btn)
        control_layout.addWidget(self.undo_btn)
        control_layout.addWidget(self.redo_btn)
        control_layout.addStretch()
        left_panel.addLayout(control_layout)
        self.canvas = SeismicCanvas()
        self.canvas.window_selected.connect(self.on_window_selected)
        left_panel.addWidget(self.canvas)
        main_layout.addLayout(left_panel, 3)
        right_panel = self.create_control_panel()
        main_layout.addLayout(right_panel, 1)
        self._on_processing_mode_changed(self.mode_combo.currentText())

    def create_control_panel(self):
        """Create control panel with Apply button."""
        panel = QVBoxLayout()
        
        selection_group = QGroupBox("Selection Mode")
        selection_layout = QVBoxLayout()
        self.selection_mode_combo = QComboBox()
        self.selection_mode_combo.addItems(['Point by Point', 'Rectangle'])
        self.selection_mode_combo.currentTextChanged.connect(self.on_selection_mode_changed)
        selection_layout.addWidget(QLabel("Mode:"))
        selection_layout.addWidget(self.selection_mode_combo)
        selection_group.setLayout(selection_layout)
        panel.addWidget(selection_group)
        
        params_group = QGroupBox("Amplification Parameters")
        params_layout = QVBoxLayout()
        
        params_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['scale', 'align'])
        self.mode_combo.currentTextChanged.connect(self._on_processing_mode_changed)
        params_layout.addWidget(self.mode_combo)
        self.scale_factor_label = QLabel("Scale Factor:")
        params_layout.addWidget(self.scale_factor_label)
        self.scale_factor_spin = QDoubleSpinBox()
        self.scale_factor_spin.setRange(0.1, 20.0)
        self.scale_factor_spin.setValue(2.0)
        self.scale_factor_spin.setSingleStep(0.1)
        params_layout.addWidget(self.scale_factor_spin)
        self.align_traces_label = QLabel("Align Width (Traces):")
        params_layout.addWidget(self.align_traces_label)
        self.align_traces_spin = QSpinBox()
        self.align_traces_spin.setRange(1, 200)
        self.align_traces_spin.setValue(10)
        params_layout.addWidget(self.align_traces_spin)
        self.align_time_label = QLabel("Align Width (Time, ms):")
        params_layout.addWidget(self.align_time_label)
        self.align_time_spin = QDoubleSpinBox()
        self.align_time_spin.setRange(1.0, 2000.0)
        self.align_time_spin.setValue(50.0)
        self.align_time_spin.setSingleStep(10.0)
        params_layout.addWidget(self.align_time_spin)
        params_layout.addWidget(QLabel("Transition Traces:"))
        self.transition_traces_spin = QSpinBox()
        self.transition_traces_spin.setRange(0, 100)
        self.transition_traces_spin.setValue(5)
        params_layout.addWidget(self.transition_traces_spin)
        params_layout.addWidget(QLabel("Transition Time (ms):"))
        self.transition_time_spin = QDoubleSpinBox()
        self.transition_time_spin.setRange(0.0, 1000.0)
        self.transition_time_spin.setValue(20.0)
        self.transition_time_spin.setSingleStep(5.0)
        params_layout.addWidget(self.transition_time_spin)
        params_layout.addWidget(QLabel("Transition Mode:"))
        self.transition_mode_combo = QComboBox()
        self.transition_mode_combo.addItems(['inside', 'outside'])
        params_layout.addWidget(self.transition_mode_combo)
        
        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.setEnabled(False)  # Will be inactive until something is selected
        self.apply_button.clicked.connect(self._reprocess_last_selection)
        params_layout.addWidget(self.apply_button)

        params_group.setLayout(params_layout)
        panel.addWidget(params_group)
        
        info_group = QGroupBox("Data Info")
        info_layout = QVBoxLayout()
        self.data_info_label = QLabel("No data loaded")
        self.data_info_label.setWordWrap(True)
        info_layout.addWidget(self.data_info_label)
        info_group.setLayout(info_layout)
        panel.addWidget(info_group)
        history_group = QGroupBox("History")
        history_layout = QVBoxLayout()
        self.history_info_label = QLabel("No history")
        self.history_info_label.setWordWrap(True)
        history_layout.addWidget(self.history_info_label)
        history_group.setLayout(history_layout)
        panel.addWidget(history_group)
        panel.addStretch()

        
        return panel
        
    def on_window_selected(self, points):
        """Handle window selection. Adds new entry to history."""
        self.last_selected_points = points
        self.apply_button.setEnabled(True)  # Activate Apply button
        self._process_window(points, add_to_history=True)
    
    def _reprocess_last_selection(self):
        """Recalculates last selection with current parameters. Does not add to history."""
        if not self.last_selected_points:
            return
            
        base_data_index = max(0, self.history_index - 1)
        if base_data_index < len(self.history):
             base_data = self.history[base_data_index]['data']
             self._process_window(self.last_selected_points, add_to_history=False, base_data=base_data)

    def clear_current_selection(self):
        self.canvas.clear_selection()
        self.last_selected_points = None
        self.apply_button.setEnabled(False)  # Deactivate button
        if self.history_index > 0:
             self.undo_action()
    
    def reset_data(self):
        if self.original_data is not None:
            self.last_selected_points = None
            self.apply_button.setEnabled(False)
            self.history = []
            self.history_index = -1
            self.save_to_history(self.original_data, "Data reset to original")
            self.current_data = self.original_data.copy()
            self.canvas.set_data(self.original_data, self.sample_interval)

    def undo_action(self):
        if self.history_index > 0:
            self.last_selected_points = None
            self.apply_button.setEnabled(False)
            self.canvas.clear_selection()

            self.history_index -= 1
            state = self.history[self.history_index]
            self.current_data = state['data'].copy()
            self.canvas.update_processed_data(self.current_data)
            self.update_undo_redo_buttons()

    def _on_processing_mode_changed(self, mode):
        is_scale_mode = (mode == 'scale')
        self.scale_factor_label.setVisible(is_scale_mode)
        self.scale_factor_spin.setVisible(is_scale_mode)
        self.align_traces_label.setVisible(not is_scale_mode)
        self.align_traces_spin.setVisible(not is_scale_mode)
        self.align_time_label.setVisible(not is_scale_mode)
        self.align_time_spin.setVisible(not is_scale_mode)

    def _process_window(self, points, add_to_history, base_data=None):
        if base_data is None: base_data = self.current_data
        if base_data is None: return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            dt_ms = self.sample_interval * 1000
            processed_data, _, _ = amplify_seismic_window(
                seismic_data=base_data, dt_ms=dt_ms, target_window=points,
                mode=self.mode_combo.currentText(), scale_factor=self.scale_factor_spin.value(),
                transition_width_traces=self.transition_traces_spin.value(),
                transition_width_time_ms=self.transition_time_spin.value(),
                transition_mode=self.transition_mode_combo.currentText(),
                align_width_traces=self.align_traces_spin.value(),
                align_width_time_ms=self.align_time_spin.value()
            )
            self.current_data = processed_data
            self.canvas.update_processed_data(processed_data)
            description = f"Amplify: {self.mode_combo.currentText()}"
            if add_to_history:
                self.save_to_history(processed_data, description)
            else:
                if self.history_index >= 0:
                    self.history[self.history_index]['data'] = processed_data.copy()
                    self.history[self.history_index]['description'] = description
                    self.update_undo_redo_buttons()
        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"An error occurred during processing:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    def on_selection_mode_changed(self, mode_text):
        mode = 'point' if mode_text == 'Point by Point' else 'rectangle'
        self.canvas.set_selection_mode(mode)
        
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load SEG-Y File", "", "SEG-Y Files (*.sgy *.segy)")
        if not file_path: return
        try:
            self.last_selected_points = None
            self.apply_button.setEnabled(False)
            data, si = ioutils.read_segy(file_path)
            if data is None:
                QMessageBox.critical(self, "Error", "Failed to load data from SEG-Y file.")
                return
            self.original_data = data
            self.current_data = data.copy()
            self.sample_interval = si
            self.original_file_path = file_path
            self.history = []
            self.history_index = -1
            self.save_to_history(data, "Original data loaded")
            self.canvas.set_data(data, self.sample_interval)
            n_traces, n_samples = data.shape
            info_text = f"File: {file_path.split('/')[-1]}\nTraces: {n_traces}\nSamples: {n_samples}\nInterval: {si*1000:.2f} ms"
            self.data_info_label.setText(info_text)
            for btn in [self.save_btn, self.reset_btn, self.clear_selection_btn]:
                btn.setEnabled(True)
            self.update_undo_redo_buttons()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while loading the file:\n{e}")

    def save_file(self):
        if self.current_data is None: return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Processed SEG-Y File", "", "SEG-Y Files (*.sgy *.segy)")
        if file_path:
            try:
                ioutils.write_segy(file_path, self.original_file_path, self.current_data, self.sample_interval)
                QMessageBox.information(self, "Success", "File saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save file:\n{e}")

    def save_to_history(self, data, description=""):
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append({'data': data.copy(), 'description': description})
        if len(self.history) > self.max_history_size:
            self.history.pop(0)
        self.history_index = len(self.history) - 1
        self.update_undo_redo_buttons()

    def redo_action(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            state = self.history[self.history_index]
            self.current_data = state['data'].copy()
            self.canvas.update_processed_data(self.current_data)
            self.update_undo_redo_buttons()
            self.last_selected_points = None
            self.apply_button.setEnabled(False)
            self.canvas.clear_selection()

    def update_undo_redo_buttons(self):
        self.undo_btn.setEnabled(self.history_index > 0)
        self.redo_btn.setEnabled(self.history_index < len(self.history) - 1)
        if self.history:
            current_desc = self.history[self.history_index]['description']
            history_text = f"Current: {current_desc}\nHistory: {self.history_index + 1}/{len(self.history)}"
            self.history_info_label.setText(history_text)
        else:
            self.history_info_label.setText("No history")

def main():
    app = QApplication(sys.argv)
    window = SeismicApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()