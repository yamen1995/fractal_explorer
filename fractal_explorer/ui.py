import numpy as np
import ast
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog,
    QSlider, QComboBox, QHBoxLayout, QProgressBar, QSizePolicy, QLineEdit, QCheckBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint, QTimer, QRectF

from fractal_explorer.fractal_math import (
    compute_fractal, compute_blended_fractal,
    blend_fractals_mask, blend_fractals_alternating, JULIA_PRESETS
)
from .colormaps import apply_colormap, blend_colormaps, COLORMAPS

# --- Worker Thread for Fractal Calculation ---
class FractalWorker(QThread):
    image_ready = pyqtSignal(np.ndarray, tuple)
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(
        self, min_x, max_x, min_y, max_y, width, height, maxiter, fractal_type, power,
        julia_c=0j, colormap_name='plasma',
        blend_enabled=False, colormap_2_name='viridis', blend_factor=0.5, blend_mode='linear', nonlinear_power=2.0, segment_point=0.5,
        fractal_blend_enabled=False, fractal2_type=0, fractal2_power=2.0, fractal2_iter=500, fractal_blend_mode='mask', fractal_blend_factor=0.5
    ):
        super().__init__()
        # Fractal params
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.width = width
        self.height = height
        self.maxiter = maxiter
        self.fractal_type = fractal_type
        self.julia_c = julia_c
        self.colormap_name = colormap_name
        self.power = power
        self._abort = False
        # Colormap blending params
        self.blend_enabled = blend_enabled
        self.colormap_2_name = colormap_2_name
        self.blend_factor = blend_factor
        self.blend_mode = blend_mode
        self.nonlinear_power = nonlinear_power
        self.segment_point = segment_point
        # Fractal blending params
        self.fractal_blend_enabled = fractal_blend_enabled
        self.fractal2_type = fractal2_type
        self.fractal2_power = fractal2_power
        self.fractal2_iter = fractal2_iter
        self.fractal_blend_mode = fractal_blend_mode
        self.fractal_blend_factor = fractal_blend_factor

    def run(self):
        def progress_callback(percent):
            self.progress.emit(percent)
        if self.fractal_blend_enabled:
            pixels1, pixels2 = compute_blended_fractal(
                self.min_x, self.max_x, self.min_y, self.max_y,
                self.width, self.height,
                self.maxiter, self.fractal_type, self.power, self.julia_c,
                self.fractal2_iter, self.fractal2_type, self.fractal2_power, self.julia_c,
                progress_callback
            )
            if self.fractal_blend_mode == 'mask':
                pixels = blend_fractals_mask(pixels1, pixels2, self.fractal_blend_factor)
            else:
                pixels = blend_fractals_alternating(pixels1, pixels2, mode='checker')
        else:
            pixels = compute_fractal(
                self.min_x, self.max_x, self.min_y, self.max_y,
                self.width, self.height, self.maxiter,
                self.fractal_type, self.power, self.julia_c,
                progress_callback
            )
        if self._abort:
            return
        if self.blend_enabled:
            colored = blend_colormaps(
                pixels,
                self.colormap_name,
                self.colormap_2_name,
                self.blend_factor,
                self.blend_mode,
                self.nonlinear_power,
                self.segment_point
            )
        else:
            colored = apply_colormap(pixels, self.colormap_name)
        self.image_ready.emit(colored, (self.min_x, self.max_x, self.min_y, self.max_y))
        self.progress.emit(100)
        self.finished_signal.emit()

    def abort(self):
        self._abort = True

# --- Custom QLabel for Selection Rectangle ---
class FractalImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection_rect = None

    def set_selection_rect(self, rect):
        self.selection_rect = rect
        self.update()

    def clear_selection_rect(self):
        self.selection_rect = None
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selection_rect and not self.selection_rect.isNull():
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 255, 0), 1, Qt.DashLine))
            painter.drawRect(self.selection_rect.normalized())

# --- Main Application Widget ---
class FractalExplorer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fractal Explorer")
        self.setMinimumSize(800, 600)
        self._init_state()
        self._setup_ui()
        QTimer.singleShot(100, self.start_render)

    def _init_state(self):
        # Fractal parameters
        self.min_x, self.max_x = -2.0, 1.0
        self.min_y, self.max_y = -1.5, 1.5
        self.maxiter = 500
        self.colormap_name = 'plasma'
        self.zoom_factor = 1.0
        self.fractal_type = 0
        self.julia_c = complex(-0.7, 0.27015)
        self.selection_active = False
        self.current_image = None
        self.last_render_params = None
        self.pixmap_offset = QPoint(0, 0)
        self.pixmap_size = QPoint(0, 0)
        self.worker = None
        self.last_pos = None

    def _setup_ui(self):
        # --- Controls ---
        self.image_label = FractalImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.status_label = QLabel("Ready")
        self.coord_label = QLabel("")
        self.zoom_label = QLabel(f"Zoom: 1x")
        self.iter_label = QLabel(f"Iterations: {self.maxiter}")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.render_button = QPushButton("Render")
        self.render_button.clicked.connect(self.start_render)
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.reset_button = QPushButton("Reset View")
        self.reset_button.clicked.connect(self.reset_view)
        self.iter_slider = QSlider(Qt.Horizontal)
        self.iter_slider.setRange(100, 50000)
        self.iter_slider.setValue(self.maxiter)
        self.iter_slider.valueChanged.connect(self.set_iterations)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(list(COLORMAPS.keys()))
        self.cmap_combo.setCurrentText(self.colormap_name)
        self.cmap_combo.currentTextChanged.connect(self.set_colormap)
        self.fractal_combo = QComboBox()
        self.fractal_combo.addItems(["Mandelbrot", "Julia", "Burning Ship", "Tricorn", "Celtic Mandelbrot", "Buffalo"])
        self.fractal_combo.currentIndexChanged.connect(self.fractal_set_changed)
        self.julia_real_input = QLineEdit("-0.7")
        self.julia_real_input.setFixedWidth(60)
        self.julia_imag_input = QLineEdit("0.27015")
        self.julia_imag_input.setFixedWidth(60)
        self.julia_real_input.editingFinished.connect(self.start_render)
        self.julia_imag_input.editingFinished.connect(self.start_render)
        self.julia_combo = QComboBox()
        self.exponent = QLineEdit("2")
        self.exponent.setFixedWidth(80)
        self.exponent.editingFinished.connect(self.start_render)
        self.complex_mode = QCheckBox("Complex")
        self.complex_mode.stateChanged.connect(self.start_render)
        for name, _ in JULIA_PRESETS:
            self.julia_combo.addItem(name)
        self.julia_combo.currentIndexChanged.connect(self.handle_julia_combo)
        # --- Layouts ---
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Iterations:"))
        control_layout.addWidget(self.iter_slider)
        control_layout.addWidget(self.iter_label)
        control_layout.addWidget(QLabel("Colormap:"))
        control_layout.addWidget(self.cmap_combo)
        control_layout.addWidget(self.render_button)
        control_layout.addWidget(self.save_button)
        control_layout.addWidget(self.reset_button)
        fractal_layout = QHBoxLayout()
        fractal_layout.addWidget(QLabel("Fractal Set:"))
        fractal_layout.addWidget(self.fractal_combo)
        fractal_layout.addWidget(QLabel("Julia c:"))
        fractal_layout.addWidget(self.julia_combo)
        fractal_layout.addWidget(self.julia_real_input)
        fractal_layout.addWidget(QLabel("+"))
        fractal_layout.addWidget(self.julia_imag_input)
        fractal_layout.addWidget(QLabel("i"))
        fractal_layout.addStretch(1)
        fractal_layout.addWidget(QLabel("Exponent:"))
        fractal_layout.addWidget(self.exponent)
        fractal_layout.addWidget(self.complex_mode)
        # --- Colormap blending controls ---
        self.blend_checkbox = QCheckBox("Blend Colormaps")
        self.blend_checkbox.stateChanged.connect(self.start_render)
        self.cmap2_combo = QComboBox()
        self.cmap2_combo.addItems(list(COLORMAPS.keys()))
        self.cmap2_combo.setCurrentText('viridis')
        self.cmap2_combo.currentTextChanged.connect(self.start_render)
        self.blend_factor_slider = QSlider(Qt.Horizontal)
        self.blend_factor_slider.setRange(0, 100)
        self.blend_factor_slider.setValue(50)
        self.blend_factor_slider.valueChanged.connect(self.update_blend_params)
        self.blend_mode_combo = QComboBox()
        self.blend_mode_combo.addItems(['linear', 'nonlinear', 'segment'])
        self.blend_mode_combo.currentTextChanged.connect(self.update_blend_params)
        self.nonlinear_power_input = QLineEdit("2.0")
        self.nonlinear_power_input.setFixedWidth(50)
        self.nonlinear_power_input.editingFinished.connect(self.update_blend_params)
        self.segment_point_input = QLineEdit("0.5")
        self.segment_point_input.setFixedWidth(50)
        self.segment_point_input.editingFinished.connect(self.update_blend_params)
        blend_layout = QHBoxLayout()
        blend_layout.addWidget(self.blend_checkbox)
        blend_layout.addWidget(QLabel("Colormap 2:"))
        blend_layout.addWidget(self.cmap2_combo)
        blend_layout.addWidget(QLabel("Blend Factor:"))
        blend_layout.addWidget(self.blend_factor_slider)
        blend_layout.addWidget(QLabel("Blend Mode:"))
        blend_layout.addWidget(self.blend_mode_combo)
        blend_layout.addWidget(QLabel("Power:"))
        blend_layout.addWidget(self.nonlinear_power_input)
        blend_layout.addWidget(QLabel("Segment:"))
        blend_layout.addWidget(self.segment_point_input)
        # --- Fractal blending controls ---
        self.fractal_blend_checkbox = QCheckBox("Blend Two Fractals")
        self.fractal_blend_checkbox.stateChanged.connect(self.start_render)
        self.fractal_blend_mode_combo = QComboBox()
        self.fractal_blend_mode_combo.addItems(['mask', 'alternating'])
        self.fractal_blend_mode_combo.currentTextChanged.connect(self.start_render)
        self.fractal2_combo = QComboBox()
        self.fractal2_combo.addItems(["Mandelbrot", "Julia", "Burning Ship", "Tricorn", "Celtic Mandelbrot", "Buffalo"])
        self.fractal2_combo.setCurrentIndex(0)
        self.fractal2_combo.currentIndexChanged.connect(self.update_fractal_blend_params)
        self.fractal2_power = QLineEdit("2")
        self.fractal2_power.setFixedWidth(60)
        self.fractal2_power.editingFinished.connect(self.update_fractal_blend_params)
        self.fractal2_iter = QLineEdit("500")
        self.fractal2_iter.setFixedWidth(60)
        self.fractal2_iter.editingFinished.connect(self.update_fractal_blend_params)
        self.fractal_blend_factor_slider = QSlider(Qt.Horizontal)
        self.fractal_blend_factor_slider.setRange(0, 100)
        self.fractal_blend_factor_slider.setValue(50)
        self.fractal_blend_factor_slider.valueChanged.connect(self.update_fractal_blend_params)
        fractal_blend_layout = QHBoxLayout()
        fractal_blend_layout.addWidget(self.fractal_blend_checkbox)
        fractal_blend_layout.addWidget(QLabel("Fractal 2:"))
        fractal_blend_layout.addWidget(self.fractal2_combo)
        fractal_blend_layout.addWidget(QLabel("Power:"))
        fractal_blend_layout.addWidget(self.fractal2_power)
        fractal_blend_layout.addWidget(QLabel("Iterations:"))
        fractal_blend_layout.addWidget(self.fractal2_iter)
        fractal_blend_layout.addWidget(QLabel("Blend Mode:"))
        fractal_blend_layout.addWidget(self.fractal_blend_mode_combo)
        fractal_blend_layout.addWidget(QLabel("Blend Factor:"))
        fractal_blend_layout.addWidget(self.fractal_blend_factor_slider)
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.coord_label, 1, Qt.AlignRight)
        status_layout.addWidget(self.zoom_label, 0, Qt.AlignRight)
        credit_label = QLabel("Developed by: Yamen Tahseen")
        credit_label.setAlignment(Qt.AlignCenter)
        credit_label.setStyleSheet("color: gray; font-size: 10pt;")
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label, 1)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(fractal_layout)
        main_layout.addLayout(blend_layout)
        main_layout.addLayout(fractal_blend_layout)
        main_layout.addLayout(status_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(credit_label)
        self.setLayout(main_layout)
        # Mouse tracking and events
        self.image_label.setMouseTracking(True)
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event
        self._set_styles()

    def _set_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 11pt;
                background: #282828;
                color: #f0f0f0;
            }
            QPushButton {
                background-color: #2d89ef;
                color: white;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1e5fa3;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444;
                background: #444;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2d89ef;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -4px 0;
                border-radius: 9px;
            }
            QComboBox, QLineEdit {
                background: #222;
                color: #f0f0f0;
                border-radius: 4px;
                padding: 2px 6px;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 5px;
                text-align: center;
                background: #222;
            }
            QProgressBar::chunk {
                background-color: #2d89ef;
                width: 20px;
            }
        """)

    # --- UI Event Handlers ---
    def fractal_set_changed(self, idx):
        self.fractal_type = idx
        self.update_julia_visibility()
        self.start_render()

    def update_julia_visibility(self):
        is_julia = self.fractal_combo.currentIndex() == 1
        self.julia_real_input.setEnabled(is_julia and self.julia_combo.currentIndex() == 0)
        self.julia_imag_input.setEnabled(is_julia and self.julia_combo.currentIndex() == 0)
        self.julia_combo.setEnabled(is_julia)

    def handle_julia_combo(self):
        idx = self.julia_combo.currentIndex()
        preset = JULIA_PRESETS[idx][1]
        if preset is not None:
            real, imag = preset
            self.julia_real_input.setText(f"{real}")
            self.julia_imag_input.setText(f"{imag}")
            self.julia_real_input.setEnabled(False)
            self.julia_imag_input.setEnabled(False)
        else:
            self.julia_real_input.setEnabled(True)
            self.julia_imag_input.setEnabled(True)
        self.start_render()

    def set_colormap(self, cmap_name):
        self.colormap_name = cmap_name
        if self.current_image is not None:
            self.update_image_display()

    def set_iterations(self, value):
        self.maxiter = value
        self.iter_label.setText(f"Iterations: {value}")

    # --- Rendering Pipeline ---
    def start_render(self):
        if self.worker and self.worker.isRunning():
            self.worker.abort()
            self.worker.wait()
        julia_c = self.get_julia_c()
        width = self.image_label.width()
        height = self.image_label.height()
        power = self.get_exponent()
        if width < 50 or height < 50:
            return
        # Gather blending parameters
        blend_enabled = self.blend_checkbox.isChecked()
        colormap_2_name = self.cmap2_combo.currentText()
        blend_factor = self.blend_factor_slider.value() / 100.0
        blend_mode = self.blend_mode_combo.currentText()
        nonlinear_power = self._safe_float(self.nonlinear_power_input.text(), 2.0)
        segment_point = self._safe_float(self.segment_point_input.text(), 0.5)
        # Gather fractal blending parameters
        fractal_blend_enabled = self.fractal_blend_checkbox.isChecked()
        fractal2_type = self.fractal2_combo.currentIndex()
        fractal2_power = self._safe_float(self.fractal2_power.text(), 2.0)
        fractal2_iter = self._safe_int(self.fractal2_iter.text(), 500)
        fractal_blend_mode = self.fractal_blend_mode_combo.currentText()
        fractal_blend_factor = self.fractal_blend_factor_slider.value() / 100.0
        self.status_label.setText("Rendering...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.worker = FractalWorker(
            self.min_x, self.max_x, self.min_y, self.max_y,
            width, height, self.maxiter,
            fractal_type=self.fractal_type,
            julia_c=julia_c,
            colormap_name=self.colormap_name,
            power=power,
            blend_enabled=blend_enabled,
            colormap_2_name=colormap_2_name,
            blend_factor=blend_factor,
            blend_mode=blend_mode,
            nonlinear_power=nonlinear_power,
            segment_point=segment_point,
            fractal_blend_enabled=fractal_blend_enabled,
            fractal2_type=fractal2_type,
            fractal2_power=fractal2_power,
            fractal2_iter=fractal2_iter,
            fractal_blend_mode=fractal_blend_mode,
            fractal_blend_factor=fractal_blend_factor
        )
        self.worker.image_ready.connect(self.handle_image_ready)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self.render_finished)
        self.worker.start()

    def _safe_float(self, text, default):
        try:
            return float(text)
        except Exception:
            return default

    def _safe_int(self, text, default):
        try:
            return int(text)
        except Exception:
            return default

    def get_exponent(self):
        text = self.exponent.text()
        if self.complex_mode.isChecked():
            try:
                value = ast.literal_eval(text)
                if isinstance(value, (int, float, complex)):
                    return complex(value)
                else:
                    return 2 + 0j
            except Exception:
                return 2 + 0j
        else:
            try:
                value = ast.literal_eval(text)
                if isinstance(value, (int, float)):
                    return float(value)
                else:
                    return 2.0
            except Exception:
                return 2.0

    def get_julia_c(self):
        try:
            real_part = float(self.julia_real_input.text())
            imag_part = float(self.julia_imag_input.text())
            return complex(real_part, imag_part)
        except Exception:
            return complex(-0.7, 0.27015)

    def handle_image_ready(self, image_array, params):
        self.current_image = image_array
        self.last_render_params = params
        self.update_image_display()

    def update_image_display(self):
        if self.current_image is None:
            return
        h, w, _ = self.current_image.shape
        bytes_per_line = 3 * w
        qimage = QImage(
            self.current_image.data,
            w, h,
            bytes_per_line,
            QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.pixmap_offset = QPoint(
            (self.image_label.width() - scaled_pixmap.width()) // 2,
            (self.image_label.height() - scaled_pixmap.height()) // 2
        )
        self.pixmap_size = QPoint(scaled_pixmap.width(), scaled_pixmap.height())

    def render_finished(self):
        self.status_label.setText("Ready")
        self.progress_bar.setVisible(False)
        self.zoom_label.setText(f"Zoom: {self.zoom_factor:.1f}x")

    def save_image(self):
        if self.current_image is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Fractal Image", "",
            "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)"
        )
        if not filename:
            return
        h, w, _ = self.current_image.shape
        bytes_per_line = 3 * w
        qimage = QImage(
            self.current_image.data,
            w, h,
            bytes_per_line,
            QImage.Format_RGB888
        )
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename += '.png'
        qimage.save(filename)

    def reset_view(self):
        self.min_x, self.max_x = -2.0, 1.0
        self.min_y, self.max_y = -1.5, 1.5
        self.zoom_factor = 1.0
        self.start_render()

    def zoom(self, factor, center_x, center_y):
        if not self.pixmap_size.x() or not self.pixmap_size.y():
            return
        center_x -= self.pixmap_offset.x()
        center_y -= self.pixmap_offset.y()
        center_real = self.min_x + center_x * (self.max_x - self.min_x) / self.pixmap_size.x()
        center_imag = self.min_y + center_y * (self.max_y - self.min_y) / self.pixmap_size.y()
        range_x = (self.max_x - self.min_x) / factor
        range_y = (self.max_y - self.min_y) / factor
        self.min_x = center_real - range_x / 2
        self.max_x = center_real + range_x / 2
        self.min_y = center_imag - range_y / 2
        self.max_y = center_imag + range_y / 2
        self.zoom_factor *= factor
        self.start_render()

    def wheelEvent(self, event):
        zoom_factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        pos = event.pos()
        self.zoom(zoom_factor, pos.x(), pos.y())

    def mouse_move_event(self, event):
        if self.current_image is None:
            return
        x = event.pos().x() - self.pixmap_offset.x()
        y = event.pos().y() - self.pixmap_offset.y()
        if (0 <= x < self.pixmap_size.x() and
            0 <= y < self.pixmap_size.y() and
            self.pixmap_size.x() > 0 and self.pixmap_size.y() > 0):
            real = self.min_x + x * (self.max_x - self.min_x) / self.pixmap_size.x()
            imag = self.min_y + y * (self.max_y - self.min_y) / self.pixmap_size.y()
            self.coord_label.setText(f"({real:.8f}, {imag:.8f})")
        else:
            self.coord_label.setText("")
        if self.selection_active and self.last_pos:
            rect = QRectF(self.last_pos, event.pos()).normalized()
            self.image_label.set_selection_rect(rect)

    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            self.last_pos = event.pos()
            self.selection_active = True
        elif event.button() == Qt.RightButton:
            self.zoom(1.5, self.image_label.width()//2, self.image_label.height()//2)
            self.image_label.clear_selection_rect()

    def mouse_release_event(self, event):
        if event.button() == Qt.LeftButton and self.selection_active:
            self.selection_active = False
            rect = self.image_label.selection_rect
            if rect and rect.width() > 5 and rect.height() > 5:
                pixmap_rect = QRectF(self.pixmap_offset, self.pixmap_size)
                adj_rect = rect.intersected(pixmap_rect)
                adj_rect = QRect(
                    int(adj_rect.left() - self.pixmap_offset.x()),
                    int(adj_rect.top() - self.pixmap_offset.y()),
                    int(adj_rect.width()),
                    int(adj_rect.height())
                )
                if adj_rect.width() > 5 and adj_rect.height() > 5:
                    x1 = adj_rect.left() * (self.max_x - self.min_x) / self.pixmap_size.x() + self.min_x
                    x2 = adj_rect.right() * (self.max_x - self.min_x) / self.pixmap_size.x() + self.min_x
                    y1 = adj_rect.top() * (self.max_y - self.min_y) / self.pixmap_size.y() + self.min_y
                    y2 = adj_rect.bottom() * (self.max_y - self.min_y) / self.pixmap_size.y() + self.min_y
                    self.min_x, self.max_x = min(x1, x2), max(x1, x2)
                    self.min_y, self.max_y = min(y1, y2), max(y1, y2)
                    self.zoom_factor *= (self.pixmap_size.x() / adj_rect.width())
                    self.start_render()
            self.image_label.clear_selection_rect()

    def resizeEvent(self, event):
        self.start_render()
        super().resizeEvent(event)
    def update_blend_params(self, *args):
    # Just update UI state, do NOT render
        pass

    def update_fractal_blend_params(self, *args):
    # Just update UI state, do NOT render
        pass

def set_dark_palette(app):
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(40, 40, 40))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(50, 50, 50))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)