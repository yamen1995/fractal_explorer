import numpy as np
import ast
import imageio
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
from .colormaps import apply_colormap, blend_colormaps, COLORMAPS, apply_lyapunov_colormap

# --- Worker Thread for Fractal Calculation ---
class FractalWorker(QThread):
    image_ready = pyqtSignal(np.ndarray, tuple)
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(
        self, min_x, max_x, min_y, max_y, width, height, maxiter, fractal_type, power_or_sequence,
        julia_c=0j, colormap_name='plasma',
        lyapunov_seq="AB", lyapunov_warmup=100,
        blend_enabled=False, colormap_2_name='viridis', blend_factor=0.5, blend_mode='linear', nonlinear_power=2.0, segment_point=0.5,
        fractal_blend_enabled=False, fractal2_type=0, fractal2_power_or_sequence=2.0, fractal2_iter=500,
        fractal_blend_mode='mask', fractal_blend_factor=0.5,
        lyapunov_seq2="AB"
    ):
        super().__init__()
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
        self.power_or_sequence = power_or_sequence
        self.lyapunov_seq = lyapunov_seq
        self.lyapunov_warmup = lyapunov_warmup
        self._abort = False
        self.blend_enabled = blend_enabled
        self.colormap_2_name = colormap_2_name
        self.blend_factor = blend_factor
        self.blend_mode = blend_mode
        self.nonlinear_power = nonlinear_power
        self.segment_point = segment_point
        self.fractal_blend_enabled = fractal_blend_enabled
        self.fractal2_type = fractal2_type
        self.fractal2_power_or_sequence = fractal2_power_or_sequence
        self.fractal2_iter = fractal2_iter
        self.fractal_blend_mode = fractal_blend_mode
        self.fractal_blend_factor = fractal_blend_factor
        self.lyapunov_seq2 = lyapunov_seq2

    def run(self):
        def progress_callback(percent):
            self.progress.emit(percent)
        if self.fractal_blend_enabled:
            pixels1, pixels2 = compute_blended_fractal(
                self.min_x, self.max_x, self.min_y, self.max_y,
                self.width, self.height,
                self.maxiter, self.fractal_type, self.power_or_sequence, self.julia_c,
                self.fractal2_iter, self.fractal2_type, self.fractal2_power_or_sequence, self.julia_c,
                lyapunov_seq1=self.lyapunov_seq if self.fractal_type == 6 else "AB",
                lyapunov_seq2=self.lyapunov_seq2 if self.fractal2_type == 6 else "AB",
                lyapunov_warmup=self.lyapunov_warmup,
                progress_callback=progress_callback
            )
            if self.fractal_blend_mode == 'mask':
                pixels = blend_fractals_mask(pixels1, pixels2, self.fractal_blend_factor)
            else:
                pixels = blend_fractals_alternating(pixels1, pixels2, mode='checker')
        else:
            pixels = compute_fractal(
                self.min_x, self.max_x, self.min_y, self.max_y,
                self.width, self.height, self.maxiter,
                self.fractal_type, self.power_or_sequence, self.julia_c,
                lyapunov_seq=self.lyapunov_seq, lyapunov_warmup=self.lyapunov_warmup,
                progress_callback=progress_callback
            )
        if self._abort:
            return

        effective_fractal_type = self.fractal_type if not self.fractal_blend_enabled else -1

        if effective_fractal_type == 6:
            if self.blend_enabled:
                rgb1 = apply_lyapunov_colormap(pixels)
                rgb2 = apply_colormap(np.copy(pixels), self.colormap_2_name)
                colored = rgb1 * (1 - self.blend_factor) + rgb2 * self.blend_factor
                colored = np.clip(colored, 0, 255).astype(np.uint8)
            else:
                colored = apply_lyapunov_colormap(pixels)
        else:
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

        if self._abort:
            return

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
        self.setFocusPolicy(Qt.StrongFocus)
        self._init_state()
        self._setup_ui()
        QTimer.singleShot(100, self.start_render)

    def _init_state(self):
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
        self.animation_width = None
        self.animation_height = None

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
        self.fractal_combo.addItems([
            "Mandelbrot", "Julia", "Burning Ship", "Tricorn", "Celtic Mandelbrot",
            "Buffalo", "Lyapunov", "Mandelbar", "Perpendicular Burning Ship", "Perpendicular Buffalo"
        ])
        self.fractal_combo.currentIndexChanged.connect(self.fractal_set_changed)

        # Julia parameters
        self.julia_label = QLabel("Julia c:")
        self.julia_real_input = QLineEdit("-0.7")
        self.julia_real_input.setFixedWidth(60)
        self.julia_imag_input = QLineEdit("0.27015")
        self.julia_imag_input.setFixedWidth(60)
        self.julia_plus_label = QLabel("+")
        self.julia_i_label = QLabel("i")
        self.julia_real_input.editingFinished.connect(self.start_render)
        self.julia_imag_input.editingFinished.connect(self.start_render)
        self.julia_combo = QComboBox()
        for name, _ in JULIA_PRESETS:
            self.julia_combo.addItem(name)
        self.julia_combo.currentIndexChanged.connect(self.handle_julia_combo)

        # Exponent parameters (for complex fractals)
        self.exponent_label = QLabel("Exponent:")
        self.exponent_input = QLineEdit("2") # Renamed from self.exponent
        self.exponent_input.setFixedWidth(80)
        self.exponent_input.editingFinished.connect(self.start_render)
        self.complex_mode_checkbox = QCheckBox("Complex") # Renamed from self.complex_mode
        self.complex_mode_checkbox.stateChanged.connect(self.start_render)

        # Lyapunov parameters
        self.lyapunov_sequence_label = QLabel("Lyapunov Seq (AB):")
        self.lyapunov_sequence_input = QLineEdit("AB")
        self.lyapunov_sequence_input.setFixedWidth(100)
        self.lyapunov_sequence_input.editingFinished.connect(self.start_render)

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
        # Julia controls (will be hidden/shown)
        fractal_layout.addWidget(self.julia_label)
        fractal_layout.addWidget(self.julia_combo)
        fractal_layout.addWidget(self.julia_real_input)
        fractal_layout.addWidget(self.julia_plus_label)
        fractal_layout.addWidget(self.julia_imag_input)
        fractal_layout.addWidget(self.julia_i_label)
        # Lyapunov controls (will be hidden/shown)
        fractal_layout.addWidget(self.lyapunov_sequence_label)
        fractal_layout.addWidget(self.lyapunov_sequence_input)
        fractal_layout.addStretch(1)
        # Exponent controls (will be hidden/shown)
        fractal_layout.addWidget(self.exponent_label)
        fractal_layout.addWidget(self.exponent_input)
        fractal_layout.addWidget(self.complex_mode_checkbox)

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
        self.fractal2_combo.addItems([
            "Mandelbrot", "Julia", "Burning Ship", "Tricorn", "Celtic Mandelbrot",
            "Buffalo", "Lyapunov", "Mandelbar", "Perpendicular Burning Ship", "Perpendicular Buffalo"
        ])
        self.fractal2_combo.setCurrentIndex(0)
        self.fractal2_combo.currentIndexChanged.connect(self.update_fractal_blend_params)

        self.fractal2_power_label = QLabel("Power/Seq:") # Label for power/sequence input
        self.fractal2_power_input = QLineEdit("2") # Renamed from fractal2_power
        self.fractal2_power_input.setFixedWidth(80)
        self.fractal2_power_input.editingFinished.connect(self.update_fractal_blend_params)

        self.fractal2_iter_input = QLineEdit("500") # Renamed from fractal2_iter
        self.fractal2_iter_input.setFixedWidth(60)
        self.fractal2_iter_input.editingFinished.connect(self.update_fractal_blend_params)

        self.fractal_blend_factor_slider = QSlider(Qt.Horizontal)
        self.fractal_blend_factor_slider.setRange(0, 100)
        self.fractal_blend_factor_slider.setValue(50)
        self.fractal_blend_factor_slider.valueChanged.connect(self.update_fractal_blend_params)

        # Lyapunov specific controls for fractal 2 in blending (initially hidden)
        self.fractal2_lyapunov_seq_label = QLabel("Lyapunov Seq 2 (AB):")
        self.fractal2_lyapunov_seq_input = QLineEdit("AB")
        self.fractal2_lyapunov_seq_input.setFixedWidth(100)
        self.fractal2_lyapunov_seq_input.editingFinished.connect(self.update_fractal_blend_params)

        fractal_blend_layout = QHBoxLayout()
        fractal_blend_layout.addWidget(self.fractal_blend_checkbox)
        fractal_blend_layout.addWidget(QLabel("Fractal 2:"))
        fractal_blend_layout.addWidget(self.fractal2_combo)
        fractal_blend_layout.addWidget(self.fractal2_power_label) # Use new label
        fractal_blend_layout.addWidget(self.fractal2_power_input) # Use new input
        # Add Lyapunov specific input for fractal 2, initially hidden
        fractal_blend_layout.addWidget(self.fractal2_lyapunov_seq_label)
        fractal_blend_layout.addWidget(self.fractal2_lyapunov_seq_input)
        fractal_blend_layout.addWidget(QLabel("Iterations:"))
        fractal_blend_layout.addWidget(self.fractal2_iter_input) # Use new input
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

        # Coordinate input for direct navigation
        self.center_x_input = QLineEdit(str((self.min_x + self.max_x) / 2))
        self.center_x_input.setFixedWidth(120)
        self.center_y_input = QLineEdit(str((self.min_y + self.max_y) / 2))
        self.center_y_input.setFixedWidth(120)
        self.goto_button = QPushButton("Go to (x, y)")
        self.goto_button.clicked.connect(self.goto_coordinates)

        coord_input_layout = QHBoxLayout()
        coord_input_layout.addWidget(QLabel("Center X:"))
        coord_input_layout.addWidget(self.center_x_input)
        coord_input_layout.addWidget(QLabel("Center Y:"))
        coord_input_layout.addWidget(self.center_y_input)
        coord_input_layout.addWidget(self.goto_button)

        main_layout.addLayout(coord_input_layout)
        
        # --- Animation Controls ---
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animation_step)
        self.current_animation_step = 0
        self.total_animation_steps = 0
        self.animation_param_values = []
        self.animation_frames = [] # To store frames for export

        animation_group_layout = QHBoxLayout()
        self.animate_variable_combo = QComboBox()
        self.animate_variable_combo.addItems([
            "Julia C Real", "Julia C Imag", "Exponent Real", "Exponent Imag", "Iterations"
        ])
        self.animate_start_input = QLineEdit("0")
        self.animate_start_input.setFixedWidth(70)
        self.animate_end_input = QLineEdit("1")
        self.animate_end_input.setFixedWidth(70)
        self.animate_steps_input = QLineEdit("100")
        self.animate_steps_input.setFixedWidth(50)
        self.animate_fps_slider = QSlider(Qt.Horizontal)
        self.animate_fps_slider.setRange(1, 60) # FPS
        self.animate_fps_slider.setValue(10)
        self.animate_fps_slider.setFixedWidth(100)
        self.animate_fps_label = QLabel("10 FPS")
        self.animate_fps_slider.valueChanged.connect(
            lambda val: self.animate_fps_label.setText(f"{val} FPS")
        )
        self.start_animation_button = QPushButton("Start Animation")
        self.start_animation_button.clicked.connect(self.start_animation)
        self.stop_animation_button = QPushButton("Stop Animation")
        self.stop_animation_button.clicked.connect(self.stop_animation)
        self.stop_animation_button.setEnabled(False)
        self.export_animation_button = QPushButton("Export Animation")
        self.export_animation_button.clicked.connect(self.export_animation)
        self.export_animation_button.setEnabled(False) # Enabled when frames are available

        animation_group_layout.addWidget(QLabel("Animate:"))
        animation_group_layout.addWidget(self.animate_variable_combo)
        animation_group_layout.addWidget(QLabel("Start:"))
        animation_group_layout.addWidget(self.animate_start_input)
        animation_group_layout.addWidget(QLabel("End:"))
        animation_group_layout.addWidget(self.animate_end_input)
        animation_group_layout.addWidget(QLabel("Steps:"))
        animation_group_layout.addWidget(self.animate_steps_input)
        animation_group_layout.addWidget(QLabel("Speed:"))
        animation_group_layout.addWidget(self.animate_fps_slider)
        animation_group_layout.addWidget(self.animate_fps_label)
        animation_group_layout.addWidget(self.start_animation_button)
        animation_group_layout.addWidget(self.stop_animation_button)
        animation_group_layout.addWidget(self.export_animation_button)
        animation_group_layout.addStretch(1)

        main_layout.addLayout(animation_group_layout) # Add animation controls to main layout

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
        self.update_fractal_controls_visibility() # Initial call to set visibility

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
        if self.fractal_type == 6:  # Lyapunov
            self.min_x, self.max_x = 2.0, 4.0
            self.min_y, self.max_y = 2.0, 4.0
            self.zoom_factor = 1.0
        else:
            # Reset to default Mandelbrot-like view if not Lyapunov,
            # or if switching from Lyapunov to something else.
            # A more advanced approach might store last view per fractal type.
            is_julia = (self.fractal_type == 1)
            if not is_julia: # Julia has its own C constant, view might be specific
                self.min_x, self.max_x = -2.0, 1.0
                self.min_y, self.max_y = -1.5, 1.5
                self.zoom_factor = 1.0
            # If it's Julia, we let the existing view parameters remain,
            # as they are often specific to the chosen Julia constant.

        self.update_fractal_controls_visibility()
        self.start_render()

    def update_fractal_controls_visibility(self):
        fractal_idx = self.fractal_combo.currentIndex()
        is_julia = (fractal_idx == 1) # Julia
        is_lyapunov = (fractal_idx == 6) # Lyapunov
        is_complex_fractal = not is_lyapunov # All others are complex plane fractals

        # Julia controls
        self.julia_label.setVisible(is_julia)
        self.julia_combo.setVisible(is_julia)
        self.julia_real_input.setVisible(is_julia)
        self.julia_plus_label.setVisible(is_julia)
        self.julia_imag_input.setVisible(is_julia)
        self.julia_i_label.setVisible(is_julia)
        if is_julia:
            is_custom_julia = (self.julia_combo.currentIndex() == 0)
            self.julia_real_input.setEnabled(is_custom_julia)
            self.julia_imag_input.setEnabled(is_custom_julia)
        else:
            self.julia_real_input.setEnabled(False)
            self.julia_imag_input.setEnabled(False)

        # Lyapunov controls
        self.lyapunov_sequence_label.setVisible(is_lyapunov)
        self.lyapunov_sequence_input.setVisible(is_lyapunov)

        # Exponent and Complex mode (for non-Lyapunov fractals)
        self.exponent_label.setVisible(is_complex_fractal)
        self.exponent_input.setVisible(is_complex_fractal)
        self.complex_mode_checkbox.setVisible(is_complex_fractal)

        # Fractal Blending - Fractal 2 controls
        fractal2_idx = self.fractal2_combo.currentIndex()
        is_fractal2_lyapunov = (fractal2_idx == 6) # Lyapunov for fractal 2
        self.fractal2_lyapunov_seq_label.setVisible(self.fractal_blend_checkbox.isChecked() and is_fractal2_lyapunov)
        self.fractal2_lyapunov_seq_input.setVisible(self.fractal_blend_checkbox.isChecked() and is_fractal2_lyapunov)
        self.fractal2_power_label.setVisible(self.fractal_blend_checkbox.isChecked())
        self.fractal2_power_input.setVisible(self.fractal_blend_checkbox.isChecked() and not is_fractal2_lyapunov)
        if self.fractal_blend_checkbox.isChecked():
            if is_fractal2_lyapunov:
                self.fractal2_power_label.setText("Seq 2:")
            else:
                self.fractal2_power_label.setText("Power 2:")


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
            self.update_image_display() # Consider re-rendering if colormap depends on raw data range

    def set_iterations(self, value):
        self.maxiter = value
        self.iter_label.setText(f"Iterations: {value}")
        # No immediate re-render, user will click "Render" or it happens on other param change

    # --- Rendering Pipeline ---
    def start_render(self):
        if self.worker and self.worker.isRunning():
            self.worker.abort()
            self.worker.wait()

        # Use locked size during animation, else use current widget size
        if self.animation_width and self.animation_height and getattr(self, "animation_timer", None) and self.animation_timer.isActive():
            width = self.animation_width
            height = self.animation_height
        else:
            width = self.image_label.width()
            height = self.image_label.height()
        if width < 50 or height < 50:
            return

        julia_c = self.get_julia_c()
        current_fractal_type = self.fractal_combo.currentIndex()
        lyapunov_sequence = "AB"
        power_val = 2.0

        if current_fractal_type == 6:
            power_or_sequence = self.lyapunov_sequence_input.text().upper()
            if not power_or_sequence or not all(c in 'AB' for c in power_or_sequence):
                power_or_sequence = "AB"
                self.lyapunov_sequence_input.setText(power_or_sequence)
            lyapunov_sequence = power_or_sequence
        else:
            power_or_sequence = self.get_exponent()
            power_val = power_or_sequence

        blend_enabled = self.blend_checkbox.isChecked()
        colormap_2_name = self.cmap2_combo.currentText()
        blend_factor = self.blend_factor_slider.value() / 100.0
        blend_mode = self.blend_mode_combo.currentText()
        nonlinear_power = self._safe_float(self.nonlinear_power_input.text(), 2.0)
        segment_point = self._safe_float(self.segment_point_input.text(), 0.5)

        fractal_blend_enabled = self.fractal_blend_checkbox.isChecked()
        fractal2_type = self.fractal2_combo.currentIndex()
        fractal2_iter_val = self._safe_int(self.fractal2_iter_input.text(), 500)
        fractal_blend_mode = self.fractal_blend_mode_combo.currentText()
        fractal_blend_factor_val = self.fractal_blend_factor_slider.value() / 100.0

        lyapunov_seq2_val = "AB"
        power2_val = 2.0

        if fractal2_type == 6:
            fractal2_power_or_sequence = self.fractal2_lyapunov_seq_input.text().upper()
            if not fractal2_power_or_sequence or not all(c in 'AB' for c in fractal2_power_or_sequence):
                fractal2_power_or_sequence = "AB"
                self.fractal2_lyapunov_seq_input.setText(fractal2_power_or_sequence)
            lyapunov_seq2_val = fractal2_power_or_sequence
        else:
            try:
                value = ast.literal_eval(self.fractal2_power_input.text())
                if isinstance(value, (int, float, complex)):
                    fractal2_power_or_sequence = complex(value)
                else:
                    fractal2_power_or_sequence = 2.0 + 0j
            except (ValueError, SyntaxError, TypeError):
                fractal2_power_or_sequence = 2.0 + 0j
            power2_val = fractal2_power_or_sequence

        self.status_label.setText("Rendering...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = FractalWorker(
            self.min_x, self.max_x, self.min_y, self.max_y,
            width, height, self.maxiter,
            fractal_type=current_fractal_type,
            power_or_sequence=power_or_sequence,
            julia_c=julia_c,
            colormap_name=self.colormap_name,
            lyapunov_seq=lyapunov_sequence,
            blend_enabled=blend_enabled,
            colormap_2_name=colormap_2_name,
            blend_factor=blend_factor,
            blend_mode=blend_mode,
            nonlinear_power=nonlinear_power,
            segment_point=segment_point,
            fractal_blend_enabled=fractal_blend_enabled,
            fractal2_type=fractal2_type,
            fractal2_power_or_sequence=fractal2_power_or_sequence,
            fractal2_iter=fractal2_iter_val,
            lyapunov_seq2=lyapunov_seq2_val,
            fractal_blend_mode=fractal_blend_mode,
            fractal_blend_factor=fractal_blend_factor_val
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

    def get_exponent(self): # This is for the main fractal's exponent
        text = self.exponent_input.text() # Use renamed input
        if self.complex_mode_checkbox.isChecked(): # Use renamed checkbox
            try:
                value = ast.literal_eval(text)
                if isinstance(value, (int, float, complex)):
                    return complex(value)
                else:
                    return 2 + 0j # Default complex exponent
            except Exception:
                return 2 + 0j # Default complex exponent on error
        else: # Real exponent
            try:
                value = ast.literal_eval(text)
                if isinstance(value, (int, float)):
                    return float(value)
                else:
                    return 2.0 # Default real exponent
            except Exception:
                return 2.0 # Default real exponent on error

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

        if self.animation_timer.isActive():
            # Check if we are still expecting frames for the current animation sequence
            if len(self.animation_frames) < self.total_animation_steps:
                self.animation_frames.append(np.copy(image_array))
                self.status_label.setText(f"Animation: Frame {len(self.animation_frames)}/{self.total_animation_steps} captured.")

            # Increment step counter *after* processing the frame for the *previous* step number
            # This means self.current_animation_step is the *next* step to be prepared by animation_step()
            self.current_animation_step += 1

            if self.current_animation_step >= self.total_animation_steps:
                # This check is after incrementing, so if current_step == total_steps, all frames are done.
                # This will be caught by animation_step's initial check or stop_animation called from render_finished
                pass # Let animation_step or render_finished handle the actual stop

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

        center_x = (self.min_x + self.max_x) / 2
        center_y = (self.min_y + self.max_y) / 2
        self.center_x_input.setText(f"{center_x:.8f}")
        self.center_y_input.setText(f"{center_y:.8f}")

        if self.animation_timer.isActive() and \
           self.current_animation_step >= self.total_animation_steps and \
           len(self.animation_frames) == self.total_animation_steps:
            # Animation was active, all steps have been prepared by animation_step,
            # and all frames have been captured by handle_image_ready.
            self.stop_animation()

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
        current_fractal_type = self.fractal_combo.currentIndex()
        if current_fractal_type == 6: # Lyapunov
            self.min_x, self.max_x = 2.0, 4.0
            self.min_y, self.max_y = 2.0, 4.0
        else: # Default for Mandelbrot and other complex fractals
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
        # This method is connected to colormap blend controls
        # For now, it does nothing, but could update a preview or status
        # Re-render is triggered by start_render on main controls or "Render" button
        self.update_fractal_controls_visibility() # Ensure visibility is correct
        # No direct re-render here to avoid too frequent updates from sliders for example
        pass

    def update_fractal_blend_params(self, *args):
        # This method is connected to fractal blend controls
        self.update_fractal_controls_visibility() # Update visibility of fractal 2 params
        # No direct re-render here
        pass

    # --- Animation Methods ---
    def start_animation(self):
        try:
            start_val = float(self.animate_start_input.text())
            end_val = float(self.animate_end_input.text())
            steps = int(self.animate_steps_input.text())
            if steps <= 0:
                self.status_label.setText("Animation steps must be positive.")
                return
        except ValueError:
            self.status_label.setText("Invalid animation parameters.")
            return

        self.animation_param_values = np.linspace(start_val, end_val, steps)
        self.animation_frames = [] # Clear previous frames
        self.current_animation_step = 0
        self.total_animation_steps = steps

        # Lock the animation frame size at the start
        self.animation_width = self.image_label.width()
        self.animation_height = self.image_label.height()

        self.start_animation_button.setEnabled(False)
        self.stop_animation_button.setEnabled(True)
        self.export_animation_button.setEnabled(False) # Disable during animation
        self.set_animation_controls_enabled(False)

        fps = self.animate_fps_slider.value()
        self.animation_timer.start(int(1000 / fps)) # Timer interval in milliseconds
        self.status_label.setText("Animation running...")

    def stop_animation(self):
        self.animation_timer.stop()
        self.start_animation_button.setEnabled(True)
        self.stop_animation_button.setEnabled(False)
        if self.animation_frames: # Enable export if frames were captured
            self.export_animation_button.setEnabled(True)
        self.set_animation_controls_enabled(True)
        self.status_label.setText("Animation stopped." + (f" {len(self.animation_frames)} frames captured." if self.animation_frames else ""))

    def animation_step(self):
        if self.worker and self.worker.isRunning():
            # self.status_label.setText(f"Animation: Waiting for render of step {self.current_animation_step + 1}...")
            return # Wait for current render to finish before starting next step's logic

        if self.current_animation_step >= self.total_animation_steps:
            # This condition should ideally be primarily handled in handle_image_ready or render_finished
            # after the last frame is processed.
            # If reached here, it means timer fired after last step was initiated.
            # This will be caught by stop_animation if all frames are done
            return

        self.status_label.setText(f"Animation: Preparing step {self.current_animation_step + 1}/{self.total_animation_steps}")
        current_val = self.animation_param_values[self.current_animation_step]
        param_to_animate = self.animate_variable_combo.currentText()

        # Update parameters based on current_val
        if param_to_animate == "Julia C Real":
            self.julia_real_input.setText(f"{current_val:.8f}")
            if self.fractal_combo.currentIndex() == 1: self.julia_combo.setCurrentIndex(0)
        elif param_to_animate == "Julia C Imag":
            self.julia_imag_input.setText(f"{current_val:.8f}")
            if self.fractal_combo.currentIndex() == 1: self.julia_combo.setCurrentIndex(0)
        elif param_to_animate == "Exponent Real":
            current_exp = self.get_exponent()
            if isinstance(current_exp, complex): self.exponent_input.setText(f"{current_val:.4f}{current_exp.imag:+.4f}j")
            else: self.exponent_input.setText(f"{current_val:.4f}")
        elif param_to_animate == "Exponent Imag":
            self.complex_mode_checkbox.setChecked(True)
            current_exp = self.get_exponent()
            if isinstance(current_exp, complex): self.exponent_input.setText(f"{current_exp.real:.4f}{current_val:+.4f}j")
            else: self.exponent_input.setText(f"{float(current_exp):.4f}{current_val:+.4f}j")
        elif param_to_animate == "Iterations":
            self.iter_slider.setValue(int(current_val))

        self.start_render() # This will render the frame for self.current_animation_step
        # self.current_animation_step is incremented in handle_image_ready AFTER frame is stored.

    def set_animation_controls_enabled(self, enabled):
        """Enable/disable controls that should not be changed during animation."""
        self.iter_slider.setEnabled(enabled)
        self.fractal_combo.setEnabled(enabled)
        self.cmap_combo.setEnabled(enabled)
        self.render_button.setEnabled(enabled)
        # Julia controls
        self.julia_combo.setEnabled(enabled and self.fractal_combo.currentIndex() == 1)
        is_custom_julia = (self.julia_combo.currentIndex() == 0)
        self.julia_real_input.setEnabled(enabled and self.fractal_combo.currentIndex() == 1 and is_custom_julia)
        self.julia_imag_input.setEnabled(enabled and self.fractal_combo.currentIndex() == 1 and is_custom_julia)
        # Exponent
        self.exponent_input.setEnabled(enabled)
        self.complex_mode_checkbox.setEnabled(enabled)
        # Lyapunov
        self.lyapunov_sequence_input.setEnabled(enabled)
        # Blending controls could also be disabled if they interfere
        self.blend_checkbox.setEnabled(enabled)
        self.fractal_blend_checkbox.setEnabled(enabled)
        # Animation parameter inputs
        self.animate_variable_combo.setEnabled(enabled)
        self.animate_start_input.setEnabled(enabled)
        self.animate_end_input.setEnabled(enabled)
        self.animate_steps_input.setEnabled(enabled)
        self.animate_fps_slider.setEnabled(enabled)

    def export_animation(self):
        if not self.animation_frames:
            self.status_label.setText("No animation frames to export.")
            return

        # Ensure all frames have the same shape
        shapes = [frame.shape for frame in self.animation_frames]
        if len(set(shapes)) > 1:
            self.status_label.setText("Error: Not all animation frames have the same shape.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Animation", "",
            "GIF Files (*.gif);;MP4 Video Files (*.mp4);;All Files (*)"
        )
        if not filename:
            return

        self.status_label.setText(f"Exporting animation to {filename}...")
        QApplication.processEvents()

        fps = self.animate_fps_slider.value()
        try:
            if filename.lower().endswith('.gif'):
                imageio.mimsave(filename, self.animation_frames, fps=fps, subrectangles=True)
            elif filename.lower().endswith('.mp4'):
                imageio.mimsave(filename, self.animation_frames, fps=fps, macro_block_size=1)
            else:
                self.status_label.setText("Unsupported file type. Please use .gif or .mp4")
                return

            QTimer.singleShot(2000, lambda: self.status_label.setText(f"Animation exported to {filename} (simulated)."))
            print(f"Simulated export: {len(self.animation_frames)} frames, FPS: {fps} to {filename}")
            return
        except Exception as e:
            self.status_label.setText(f"Error exporting animation: {e}")
            print(f"Error exporting animation: {e}")

        self.export_animation_button.setEnabled(True)

    def keyPressEvent(self, event):
        pan_fraction = 0.1  # Move 10% of current view per keypress
        dx = (self.max_x - self.min_x) * pan_fraction
        dy = (self.max_y - self.min_y) * pan_fraction
        moved = False

        if event.key() == Qt.Key_Left:
            self.min_x -= dx
            self.max_x -= dx
            moved = True
        elif event.key() == Qt.Key_Right:
            self.min_x += dx
            self.max_x += dx
            moved = True
        elif event.key() == Qt.Key_Up:
            self.min_y -= dy
            self.max_y -= dy
            moved = True
        elif event.key() == Qt.Key_Down:
            self.min_y += dy
            self.max_y += dy
            moved = True

        if moved:
            self.start_render()

    def goto_coordinates(self):
        try:
            center_x = float(self.center_x_input.text())
            center_y = float(self.center_y_input.text())
            width = self.max_x - self.min_x
            height = self.max_y - self.min_y
            self.min_x = center_x - width / 2
            self.max_x = center_x + width / 2
            self.min_y = center_y - height / 2
            self.max_y = center_y + height / 2
            self.start_render()
        except Exception:
            self.status_label.setText("Invalid coordinates.")

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