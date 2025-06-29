import numpy as np
import ast
import imageio
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog,
    QSlider, QComboBox, QHBoxLayout, QProgressBar, QSizePolicy, QLineEdit,
    QCheckBox, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPalette, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint, QTimer, QRectF, QSize

from fractal_explorer.fractal_math import (
    compute_fractal, compute_blended_fractal,
    blend_fractals_mask, blend_fractals_alternating, JULIA_PRESETS
)
from .colormaps import apply_colormap, blend_colormaps, COLORMAPS, apply_lyapunov_colormap, BlendMode

# --- Constants ---
DEFAULT_MIN_X, DEFAULT_MAX_X = -2.0, 1.0
DEFAULT_MIN_Y, DEFAULT_MAX_Y = -1.5, 1.5
LYAPUNOV_MIN_X, LYAPUNOV_MAX_X = 2.0, 4.0
LYAPUNOV_MIN_Y, LYAPUNOV_MAX_Y = 2.0, 4.0
DEFAULT_MAX_ITER = 500
DEFAULT_JULIA_C = complex(-0.7, 0.27015)
MIN_RENDER_WIDTH_HEIGHT = 50

# --- Worker Thread for Fractal Calculation ---
class FractalWorker(QThread):
    """
    QThread worker for performing fractal calculations asynchronously.
    Emits signals for progress updates, image readiness, and completion.
    """
    image_ready = pyqtSignal(np.ndarray, tuple) # image_array, render_params
    progress = pyqtSignal(int) # percentage
    finished_signal = pyqtSignal()

    def __init__(
        self, min_x: float, max_x: float, min_y: float, max_y: float,
        width: int, height: int, maxiter: int,
        fractal_type: int, power_or_sequence: any, # Union[float, complex, str]
        julia_c: complex = DEFAULT_JULIA_C, colormap_name: str = 'plasma',
        lyapunov_seq: str = "AB", lyapunov_warmup: int = 100,
        blend_enabled: bool = False, colormap_2_name: str = 'viridis',
        blend_factor: float = 0.5, blend_mode: BlendMode = 'linear', # type: ignore
        nonlinear_power: float = 2.0, segment_point: float = 0.5,
        fractal_blend_enabled: bool = False, fractal2_type: int = 0,
        fractal2_power_or_sequence: any = 2.0, # Union[float, complex, str]
        fractal2_iter: int = 500, fractal_blend_mode: str = 'mask', # Literal['mask', 'alternating']
        fractal_blend_factor: float = 0.5, lyapunov_seq2: str = "AB"
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

        # Colormap blending parameters
        self.blend_enabled = blend_enabled
        self.colormap_2_name = colormap_2_name
        self.blend_factor = blend_factor
        self.blend_mode: BlendMode = blend_mode
        self.nonlinear_power = nonlinear_power
        self.segment_point = segment_point

        # Fractal blending parameters
        self.fractal_blend_enabled = fractal_blend_enabled
        self.fractal2_type = fractal2_type
        self.fractal2_power_or_sequence = fractal2_power_or_sequence
        self.fractal2_iter = fractal2_iter
        self.fractal_blend_mode = fractal_blend_mode
        self.fractal_blend_factor = fractal_blend_factor
        self.lyapunov_seq2 = lyapunov_seq2


    def run(self):
        """Main execution method for the worker thread."""
        try:
            def progress_callback(percent: int):
                if self._abort: return
                self.progress.emit(percent)

            if self.fractal_blend_enabled:
                # Determine sequences for Lyapunov if applicable
                l_seq1 = self.lyapunov_seq if self.fractal_type == 6 else "AB"
                l_seq2 = self.lyapunov_seq2 if self.fractal2_type == 6 else "AB"

                pixels1, pixels2 = compute_blended_fractal(
                    self.min_x, self.max_x, self.min_y, self.max_y,
                    self.width, self.height,
                    self.maxiter, self.fractal_type, self.power_or_sequence, self.julia_c,
                    self.fractal2_iter, self.fractal2_type, self.fractal2_power_or_sequence, self.julia_c, # Julia C is shared for now
                    lyapunov_seq1=l_seq1, lyapunov_seq2=l_seq2,
                    lyapunov_warmup=self.lyapunov_warmup,
                    progress_callback=progress_callback
                )
                if self._abort: return

                if self.fractal_blend_mode == 'mask':
                    pixels = blend_fractals_mask(pixels1, pixels2, self.fractal_blend_factor)
                else: # 'alternating'
                    pixels = blend_fractals_alternating(pixels1, pixels2, mode='checker') # Default checker for alternating
            else:
                pixels = compute_fractal(
                    self.min_x, self.max_x, self.min_y, self.max_y,
                    self.width, self.height, self.maxiter,
                    self.fractal_type, self.power_or_sequence, self.julia_c,
                    lyapunov_seq=self.lyapunov_seq, lyapunov_warmup=self.lyapunov_warmup,
                    progress_callback=progress_callback
                )
            if self._abort: return

            # Determine which fractal type's colormapping rules to apply
            # If blending fractals, the colormap is applied to the result of fractal math blending.
            # If not blending fractals, it's the primary fractal type.
            # Lyapunov has special colormapping.
            effective_fractal_type_for_coloring = self.fractal_type
            if self.fractal_blend_enabled:
                # If blending, and one of them is Lyapunov, how should coloring work?
                # Current logic: if primary is Lyapunov, use Lyapunov coloring.
                # This might need refinement if, e.g., blending Mandelbrot with Lyapunov.
                # For now, let's assume the `pixels` array from blending is treated as "generic"
                # unless the primary fractal dictates Lyapunov coloring.
                # If fractal_blend_enabled, the `pixels` is already a mix.
                # Let's simplify: if colormap blending is also on, it uses standard colormaps.
                # If only fractal blending is on, it uses standard colormaps on the blended raw data.
                # This means Lyapunov specific coloring is only for non-fractal-blended Lyapunov.
                # The old code had `effective_fractal_type = self.fractal_type if not self.fractal_blend_enabled else -1`
                # This `-1` would make it fall into the `else` block for standard colormapping.
                # This seems reasonable: blended raw data gets standard colormaps.
                 effective_fractal_type_for_coloring = -1 # Generic data for standard colormapping


            if effective_fractal_type_for_coloring == 6: # Lyapunov
                if self.blend_enabled: # Colormap blending for Lyapunov
                    # This blends Lyapunov's special map with a standard one.
                    rgb1 = apply_lyapunov_colormap(pixels)
                    # Create a copy for the second colormap if pixels might be modified by apply_colormap
                    rgb2 = apply_colormap(np.copy(pixels), self.colormap_2_name)
                    # Simple linear blend for this specific Lyapunov + standard map case
                    colored = rgb1.astype(np.float32) * (1 - self.blend_factor) + rgb2.astype(np.float32) * self.blend_factor
                    colored = np.clip(colored, 0, 255).astype(np.uint8)
                else: # Single Lyapunov colormap
                    colored = apply_lyapunov_colormap(pixels)
            else: # Standard fractals or blended fractal data
                if self.blend_enabled: # Colormap blending
                    colored = blend_colormaps(
                        pixels,
                        self.colormap_name,
                        self.colormap_2_name,
                        self.blend_factor,
                        self.blend_mode,
                        self.nonlinear_power,
                        self.segment_point
                    )
                else: # Single standard colormap
                    colored = apply_colormap(pixels, self.colormap_name)

            if self._abort: return

            self.image_ready.emit(colored, (self.min_x, self.max_x, self.min_y, self.max_y))
            self.progress.emit(100)
        except Exception as e:
            # TODO: Emit an error signal to be handled by the main thread
            print(f"Error in FractalWorker: {e}") # Basic error logging
        finally:
            self.finished_signal.emit()


    def abort(self):
        """Signals the worker to abort its current task."""
        self._abort = True

# --- Custom QLabel for Selection Rectangle ---
class FractalImageLabel(QLabel):
    """Custom QLabel that can draw a selection rectangle."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.selection_rect: Optional[QRectF] = None

    def set_selection_rect(self, rect: QRectF):
        """Sets the selection rectangle and triggers a repaint."""
        self.selection_rect = rect
        self.update() # Schedule a paint event

    def clear_selection_rect(self):
        """Clears the selection rectangle and triggers a repaint."""
        self.selection_rect = None
        self.update()

    def paintEvent(self, event: QPaintEvent): # type: ignore
        """Overrides paintEvent to draw the selection rectangle if active."""
        super().paintEvent(event)
        if self.selection_rect and not self.selection_rect.isNull():
            painter = QPainter(self)
            pen = QPen(QColor(255, 255, 0), 1, Qt.DashLine) # Yellow dashed line
            painter.setPen(pen)
            painter.drawRect(self.selection_rect.normalized())


# --- Main Application Widget ---
class FractalExplorer(QWidget):
    """Main application widget for exploring fractals."""
    def __init__(self):
        super().__init__()
        self._init_app_window()
        self._init_state()
        self._setup_ui_layouts() # Main layout structure
        self._connect_signals()
        self._set_styles() # Apply custom stylesheet
        self.update_fractal_controls_visibility() # Initial UI state based on defaults
        QTimer.singleShot(100, self.start_render) # Initial render after UI is shown


    def _init_app_window(self):
        """Initializes main window properties."""
        self.setWindowTitle("Fractal Explorer")
        try:
            # Attempt to load an application icon (replace with your actual icon path)
            app_icon = QIcon("fractal_explorer/resources/icon.ico") # Or .png
            if not app_icon.isNull():
                 self.setWindowIcon(app_icon)
            else:
                print("Warning: Application icon not found or invalid.")
        except Exception as e:
            print(f"Warning: Could not load application icon: {e}")

        self.setMinimumSize(800, 600)
        self.setFocusPolicy(Qt.StrongFocus) # For keyboard events like panning


    def _init_state(self):
        """Initializes the application's state variables."""
        self.min_x, self.max_x = DEFAULT_MIN_X, DEFAULT_MAX_X
        self.min_y, self.max_y = DEFAULT_MIN_Y, DEFAULT_MAX_Y
        self.maxiter: int = DEFAULT_MAX_ITER
        self.colormap_name: str = 'plasma'
        self.zoom_factor: float = 1.0
        self.fractal_type: int = 0 # Default to Mandelbrot
        self.julia_c: complex = DEFAULT_JULIA_C

        self.selection_active: bool = False
        self.current_image: Optional[np.ndarray] = None
        self.last_render_params: Optional[tuple] = None # Stores (min_x, max_x, min_y, max_y) of last render

        self.pixmap_offset = QPoint(0, 0) # Offset of scaled pixmap within the label
        self.pixmap_size = QPoint(0, 0)   # Actual size of scaled pixmap

        self.worker: Optional[FractalWorker] = None
        self.last_pos: Optional[QPoint] = None # For mouse drag selection

        # Animation state
        self.animation_width: Optional[int] = None
        self.animation_height: Optional[int] = None
        self.animation_timer = QTimer(self)
        self.current_animation_step: int = 0
        self.total_animation_steps: int = 0
        self.animation_param_values: np.ndarray = np.array([])
        self.animation_frames: list[np.ndarray] = []

    # --- UI Setup Sub-methods ---
    def _setup_main_controls(self) -> QHBoxLayout:
        """Sets up the main control row (iterations, colormap, render, save, reset)."""
        self.iter_slider = QSlider(Qt.Horizontal)
        self.iter_slider.setRange(100, 50000) # Min/Max iterations
        self.iter_slider.setValue(self.maxiter)
        self.iter_slider.setToolTip("Adjust maximum iterations for fractal calculation.")
        self.iter_label = QLabel(f"Iterations: {self.maxiter}")

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(list(COLORMAPS.keys()))
        self.cmap_combo.setCurrentText(self.colormap_name)
        self.cmap_combo.setToolTip("Select the colormap for the fractal.")

        self.render_button = QPushButton("Render")
        self.render_button.setToolTip("Re-render the fractal with current settings.")
        self.save_button = QPushButton("Save Image")
        self.save_button.setToolTip("Save the current fractal image to a file.")
        self.reset_button = QPushButton("Reset View")
        self.reset_button.setToolTip("Reset zoom and position to the default view.")

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Iterations:"))
        layout.addWidget(self.iter_slider, 1) # Slider takes more space
        layout.addWidget(self.iter_label)
        layout.addWidget(QLabel("Colormap:"))
        layout.addWidget(self.cmap_combo)
        layout.addWidget(self.render_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.reset_button)
        return layout

    def _setup_fractal_params_controls(self) -> QHBoxLayout:
        """Sets up controls for fractal type and its specific parameters."""
        self.fractal_combo = QComboBox()
        self.fractal_combo.addItems([
            "Mandelbrot", "Julia", "Burning Ship", "Tricorn", "Celtic Mandelbrot",
            "Buffalo", "Lyapunov", "Mandelbar", "Perpendicular Burning Ship", "Perpendicular Buffalo"
        ])
        self.fractal_combo.setToolTip("Select the type of fractal to generate.")

        # Julia parameters
        self.julia_label = QLabel("Julia C:")
        self.julia_real_input = QLineEdit(str(DEFAULT_JULIA_C.real))
        self.julia_real_input.setFixedWidth(60)
        self.julia_imag_input = QLineEdit(str(DEFAULT_JULIA_C.imag))
        self.julia_imag_input.setFixedWidth(60)
        self.julia_plus_label = QLabel("+")
        self.julia_i_label = QLabel("i")
        self.julia_combo = QComboBox()
        for name, _ in JULIA_PRESETS: self.julia_combo.addItem(name)
        self.julia_combo.setToolTip("Select a preset Julia constant or choose 'Custom'.")

        # Exponent parameters
        self.exponent_label = QLabel("Exponent:")
        self.exponent_input = QLineEdit("2")
        self.exponent_input.setFixedWidth(80)
        self.exponent_input.setToolTip("Set the exponent for the fractal formula (e.g., 2 for z^2+c). Can be complex.")
        self.complex_mode_checkbox = QCheckBox("Complex Exp.")
        self.complex_mode_checkbox.setToolTip("Allow complex numbers for the exponent.")

        # Lyapunov parameters
        self.lyapunov_sequence_label = QLabel("Lyapunov Seq (AB):")
        self.lyapunov_sequence_input = QLineEdit("AB")
        self.lyapunov_sequence_input.setFixedWidth(100)
        self.lyapunov_sequence_input.setToolTip("Sequence of 'A' and 'B' for Lyapunov fractal (e.g., AAB).")

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Fractal Set:"))
        layout.addWidget(self.fractal_combo)
        # Julia (conditionally visible)
        layout.addWidget(self.julia_label)
        layout.addWidget(self.julia_combo)
        layout.addWidget(self.julia_real_input)
        layout.addWidget(self.julia_plus_label)
        layout.addWidget(self.julia_imag_input)
        layout.addWidget(self.julia_i_label)
        # Lyapunov (conditionally visible)
        layout.addWidget(self.lyapunov_sequence_label)
        layout.addWidget(self.lyapunov_sequence_input)
        layout.addStretch(1)
        # Exponent (conditionally visible)
        layout.addWidget(self.exponent_label)
        layout.addWidget(self.exponent_input)
        layout.addWidget(self.complex_mode_checkbox)
        return layout

    def _setup_colormap_blending_controls(self) -> QHBoxLayout:
        """Sets up controls for blending two colormaps."""
        self.blend_checkbox = QCheckBox("Blend Colormaps")
        self.blend_checkbox.setToolTip("Enable blending of two colormaps.")
        self.cmap2_combo = QComboBox()
        self.cmap2_combo.addItems(list(COLORMAPS.keys()))
        self.cmap2_combo.setCurrentText('viridis')
        self.cmap2_combo.setToolTip("Select the second colormap for blending.")

        self.blend_factor_slider = QSlider(Qt.Horizontal)
        self.blend_factor_slider.setRange(0, 100) # Represents 0.0 to 1.0
        self.blend_factor_slider.setValue(50)
        self.blend_factor_slider.setToolTip("Adjust the mixing proportion between the two colormaps.")

        self.blend_mode_combo = QComboBox()
        self.blend_mode_combo.addItems(['linear', 'nonlinear', 'segment'])
        self.blend_mode_combo.setToolTip("Choose the algorithm for blending colormaps.")

        self.nonlinear_power_input = QLineEdit("2.0")
        self.nonlinear_power_input.setFixedWidth(50)
        self.nonlinear_power_input.setToolTip("Exponent for 'nonlinear' blend mode.")
        self.segment_point_input = QLineEdit("0.5")
        self.segment_point_input.setFixedWidth(50)
        self.segment_point_input.setToolTip("Threshold for 'segment' blend mode (0.0-1.0).")

        layout = QHBoxLayout()
        layout.addWidget(self.blend_checkbox)
        layout.addWidget(QLabel("Colormap 2:"))
        layout.addWidget(self.cmap2_combo)
        layout.addWidget(QLabel("Blend Factor:"))
        layout.addWidget(self.blend_factor_slider)
        layout.addWidget(QLabel("Mode:"))
        layout.addWidget(self.blend_mode_combo)
        layout.addWidget(QLabel("Power:"))
        layout.addWidget(self.nonlinear_power_input)
        layout.addWidget(QLabel("Segment:"))
        layout.addWidget(self.segment_point_input)
        layout.addStretch(1)
        return layout

    def _setup_fractal_blending_controls(self) -> QHBoxLayout:
        """Sets up controls for blending two different fractals."""
        self.fractal_blend_checkbox = QCheckBox("Blend Two Fractals")
        self.fractal_blend_checkbox.setToolTip("Enable blending of two different fractal calculations.")

        self.fractal2_combo = QComboBox()
        self.fractal2_combo.addItems([ # Same list as primary fractal_combo
            "Mandelbrot", "Julia", "Burning Ship", "Tricorn", "Celtic Mandelbrot",
            "Buffalo", "Lyapunov", "Mandelbar", "Perpendicular Burning Ship", "Perpendicular Buffalo"
        ])
        self.fractal2_combo.setCurrentIndex(0) # Default to Mandelbrot
        self.fractal2_combo.setToolTip("Select the second fractal type for blending.")

        self.fractal2_power_label = QLabel("Power/Seq 2:")
        self.fractal2_power_input = QLineEdit("2")
        self.fractal2_power_input.setFixedWidth(80)
        self.fractal2_power_input.setToolTip("Exponent or sequence for the second fractal.")

        self.fractal2_lyapunov_seq_label = QLabel("Lyapunov Seq 2:")
        self.fractal2_lyapunov_seq_input = QLineEdit("AB")
        self.fractal2_lyapunov_seq_input.setFixedWidth(100)
        self.fractal2_lyapunov_seq_input.setToolTip("Sequence for the second fractal if it's Lyapunov type.")

        self.fractal2_iter_input = QLineEdit("500")
        self.fractal2_iter_input.setFixedWidth(60)
        self.fractal2_iter_input.setToolTip("Maximum iterations for the second fractal.")

        self.fractal_blend_mode_combo = QComboBox()
        self.fractal_blend_mode_combo.addItems(['mask', 'alternating']) # mask=weighted avg, alternating=pattern
        self.fractal_blend_mode_combo.setToolTip("Method for blending the raw data of two fractals.")

        self.fractal_blend_factor_slider = QSlider(Qt.Horizontal)
        self.fractal_blend_factor_slider.setRange(0, 100)
        self.fractal_blend_factor_slider.setValue(50)
        self.fractal_blend_factor_slider.setToolTip("Mixing proportion for 'mask' blend mode.")

        layout = QHBoxLayout()
        layout.addWidget(self.fractal_blend_checkbox)
        layout.addWidget(QLabel("Fractal 2:"))
        layout.addWidget(self.fractal2_combo)
        layout.addWidget(self.fractal2_power_label)
        layout.addWidget(self.fractal2_power_input)
        layout.addWidget(self.fractal2_lyapunov_seq_label) # Conditionally visible
        layout.addWidget(self.fractal2_lyapunov_seq_input) # Conditionally visible
        layout.addWidget(QLabel("Iter 2:"))
        layout.addWidget(self.fractal2_iter_input)
        layout.addWidget(QLabel("Blend Mode:"))
        layout.addWidget(self.fractal_blend_mode_combo)
        layout.addWidget(QLabel("Factor:"))
        layout.addWidget(self.fractal_blend_factor_slider)
        layout.addStretch(1)
        return layout

    def _setup_navigation_controls(self) -> QHBoxLayout:
        """Sets up controls for direct coordinate navigation."""
        self.center_x_input = QLineEdit(f"{(self.min_x + self.max_x) / 2:.4f}")
        self.center_x_input.setFixedWidth(120)
        self.center_x_input.setToolTip("Enter the desired X-coordinate for the center of the view.")
        self.center_y_input = QLineEdit(f"{(self.min_y + self.max_y) / 2:.4f}")
        self.center_y_input.setFixedWidth(120)
        self.center_y_input.setToolTip("Enter the desired Y-coordinate for the center of the view.")
        self.goto_button = QPushButton("Go to (x, y)")
        self.goto_button.setToolTip("Navigate to the specified center coordinates.")

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Center X:"))
        layout.addWidget(self.center_x_input)
        layout.addWidget(QLabel("Center Y:"))
        layout.addWidget(self.center_y_input)
        layout.addWidget(self.goto_button)
        layout.addStretch(1)
        return layout

    def _setup_animation_controls(self) -> QHBoxLayout:
        """Sets up controls for creating animations."""
        self.animate_variable_combo = QComboBox()
        self.animate_variable_combo.addItems([
            "Julia C Real", "Julia C Imag", "Exponent Real", "Exponent Imag", "Iterations"
        ])
        self.animate_variable_combo.setToolTip("Select the parameter to animate.")

        self.animate_start_input = QLineEdit("0")
        self.animate_start_input.setFixedWidth(70)
        self.animate_start_input.setToolTip("Starting value for the animated parameter.")
        self.animate_end_input = QLineEdit("1")
        self.animate_end_input.setFixedWidth(70)
        self.animate_end_input.setToolTip("Ending value for the animated parameter.")
        self.animate_steps_input = QLineEdit("100")
        self.animate_steps_input.setFixedWidth(50)
        self.animate_steps_input.setToolTip("Number of frames in the animation.")

        self.animate_fps_slider = QSlider(Qt.Horizontal)
        self.animate_fps_slider.setRange(1, 60) # FPS
        self.animate_fps_slider.setValue(10)
        self.animate_fps_slider.setFixedWidth(100)
        self.animate_fps_slider.setToolTip("Frames per second for the animation playback/export.")
        self.animate_fps_label = QLabel(f"{self.animate_fps_slider.value()} FPS")

        self.start_animation_button = QPushButton("Start Animation")
        self.start_animation_button.setToolTip("Begin generating animation frames.")
        self.stop_animation_button = QPushButton("Stop Animation")
        self.stop_animation_button.setEnabled(False)
        self.stop_animation_button.setToolTip("Stop the current animation generation.")
        self.export_animation_button = QPushButton("Export Animation")
        self.export_animation_button.setEnabled(False)
        self.export_animation_button.setToolTip("Export captured animation frames to GIF or MP4.")

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Animate:"))
        layout.addWidget(self.animate_variable_combo)
        layout.addWidget(QLabel("Start:"))
        layout.addWidget(self.animate_start_input)
        layout.addWidget(QLabel("End:"))
        layout.addWidget(self.animate_end_input)
        layout.addWidget(QLabel("Steps:"))
        layout.addWidget(self.animate_steps_input)
        layout.addWidget(QLabel("FPS:"))
        layout.addWidget(self.animate_fps_slider)
        layout.addWidget(self.animate_fps_label)
        layout.addWidget(self.start_animation_button)
        layout.addWidget(self.stop_animation_button)
        layout.addWidget(self.export_animation_button)
        layout.addStretch(1)
        return layout

    def _setup_ui_layouts(self):
        """Creates and arranges all UI elements and layouts."""
        self.image_label = FractalImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(QSize(300,200)) # Ensure it's not too small initially

        self.status_label = QLabel("Ready")
        self.coord_label = QLabel("") # Displays cursor coordinates over fractal
        self.zoom_label = QLabel(f"Zoom: {self.zoom_factor:.1f}x")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False) # Show only during rendering

        # --- Create individual control group layouts ---
        main_controls_layout = self._setup_main_controls()
        fractal_params_layout = self._setup_fractal_params_controls()
        colormap_blend_layout = self._setup_colormap_blending_controls()
        fractal_blend_layout = self._setup_fractal_blending_controls()
        nav_layout = self._setup_navigation_controls()
        animation_layout = self._setup_animation_controls()

        # --- Status and Credits ---
        status_bar_layout = QHBoxLayout()
        status_bar_layout.addWidget(self.status_label)
        status_bar_layout.addWidget(self.coord_label, 1, Qt.AlignRight) # Stretch coord_label
        status_bar_layout.addWidget(self.zoom_label, 0, Qt.AlignRight)

        credit_label = QLabel("Developed by: Yamen Tahseen") # As in original
        credit_label.setAlignment(Qt.AlignCenter)
        credit_label.setStyleSheet("color: gray; font-size: 10pt;")

        # --- Main Vertical Layout ---
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label, 1) # Image label takes most space
        main_layout.addLayout(main_controls_layout)
        main_layout.addLayout(fractal_params_layout)
        main_layout.addLayout(colormap_blend_layout)
        main_layout.addLayout(fractal_blend_layout)
        main_layout.addLayout(nav_layout)
        main_layout.addLayout(animation_layout)
        main_layout.addLayout(status_bar_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(credit_label)

        self.setLayout(main_layout)

    def _connect_signals(self):
        """Connects Qt signals to their respective slots (handlers)."""
        # Main controls
        self.render_button.clicked.connect(self.start_render)
        self.save_button.clicked.connect(self.save_image)
        self.reset_button.clicked.connect(self.reset_view)
        self.iter_slider.valueChanged.connect(self.set_iterations)
        self.cmap_combo.currentTextChanged.connect(self.set_colormap)

        # Fractal parameters
        self.fractal_combo.currentIndexChanged.connect(self.fractal_set_changed)
        self.julia_real_input.editingFinished.connect(self.start_render_if_julia_custom)
        self.julia_imag_input.editingFinished.connect(self.start_render_if_julia_custom)
        self.julia_combo.currentIndexChanged.connect(self.handle_julia_combo)
        self.exponent_input.editingFinished.connect(self.start_render)
        self.complex_mode_checkbox.stateChanged.connect(self.start_render)
        self.lyapunov_sequence_input.editingFinished.connect(self.start_render)

        # Colormap blending
        self.blend_checkbox.stateChanged.connect(self.start_render) # Re-render on enable/disable
        self.cmap2_combo.currentTextChanged.connect(self.start_render) # Re-render on change
        self.blend_factor_slider.valueChanged.connect(self.update_blend_params_and_render) # Live update
        self.blend_mode_combo.currentTextChanged.connect(self.update_blend_params_and_render)
        self.nonlinear_power_input.editingFinished.connect(self.update_blend_params_and_render)
        self.segment_point_input.editingFinished.connect(self.update_blend_params_and_render)

        # Fractal blending
        self.fractal_blend_checkbox.stateChanged.connect(self.start_render)
        self.fractal2_combo.currentIndexChanged.connect(self.update_fractal_blend_params_and_render)
        self.fractal2_power_input.editingFinished.connect(self.update_fractal_blend_params_and_render)
        self.fractal2_lyapunov_seq_input.editingFinished.connect(self.update_fractal_blend_params_and_render)
        self.fractal2_iter_input.editingFinished.connect(self.update_fractal_blend_params_and_render)
        self.fractal_blend_mode_combo.currentTextChanged.connect(self.update_fractal_blend_params_and_render)
        self.fractal_blend_factor_slider.valueChanged.connect(self.update_fractal_blend_params_and_render)

        # Navigation
        self.goto_button.clicked.connect(self.goto_coordinates)

        # Animation
        self.animation_timer.timeout.connect(self.animation_step)
        self.animate_fps_slider.valueChanged.connect(lambda val: self.animate_fps_label.setText(f"{val} FPS"))
        self.start_animation_button.clicked.connect(self.start_animation)
        self.stop_animation_button.clicked.connect(self.stop_animation)
        self.export_animation_button.clicked.connect(self.export_animation)

        # Mouse interaction with image_label
        self.image_label.setMouseTracking(True) # For mouse_move_event even without button press
        self.image_label.mouseMoveEvent = self.mouse_move_event # type: ignore
        self.image_label.mousePressEvent = self.mouse_press_event # type: ignore
        self.image_label.mouseReleaseEvent = self.mouse_release_event # type: ignore

    def _set_styles(self):
        """Applies a global stylesheet to the application."""
        # This stylesheet is from the original code.
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 11pt;
                background: #282828; /* Dark background */
                color: #f0f0f0;       /* Light text */
            }
            QPushButton {
                background-color: #2d89ef; /* Blue */
                color: white;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1e5fa3; /* Darker blue */
            }
            QPushButton:disabled {
                background-color: #555;
                color: #aaa;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444;
                background: #444;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2d89ef; /* Blue handle */
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -4px 0; /* Center handle */
                border-radius: 9px;
            }
            QComboBox, QLineEdit {
                background: #222; /* Very dark background for inputs */
                color: #f0f0f0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 2px 6px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(none); /* Can use a custom arrow icon if desired */
            }
            QLabel {
                background: transparent; /* Ensure labels don't have their own background */
            }
            QCheckBox {
                background: transparent;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 5px;
                text-align: center;
                background: #222;
                color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #2d89ef; /* Blue progress chunk */
                width: 20px; /* Width of the moving chunk */
            }
        """)

    # --- UI Event Handlers & Logic ---
    def start_render_if_julia_custom(self):
        """Helper to trigger render only if Julia type is active and 'Custom' preset."""
        if self.fractal_combo.currentIndex() == 1 and self.julia_combo.currentIndex() == 0:
            self.start_render()

    def update_blend_params_and_render(self):
        self.update_fractal_controls_visibility() # For dynamic label changes etc.
        self.start_render() # Directly re-render as parameters change

    def update_fractal_blend_params_and_render(self):
        self.update_fractal_controls_visibility()
        self.start_render()

    def fractal_set_changed(self, idx: int):
        """Handles changes to the selected fractal type."""
        self.fractal_type = idx
        if self.fractal_type == 6:  # Lyapunov
            self.min_x, self.max_x = LYAPUNOV_MIN_X, LYAPUNOV_MAX_X
            self.min_y, self.max_y = LYAPUNOV_MIN_Y, LYAPUNOV_MAX_Y
            self.zoom_factor = 1.0
        else: # Non-Lyapunov fractals
            is_julia = (self.fractal_type == 1)
            # Only reset view if not switching to Julia or if it's already a non-Julia type.
            # Julia often has specific views tied to its C constant.
            if not is_julia or self.fractal_combo.itemText(self.fractal_combo.previousIndex()) != "Julia": # type: ignore
                self.min_x, self.max_x = DEFAULT_MIN_X, DEFAULT_MAX_X
                self.min_y, self.max_y = DEFAULT_MIN_Y, DEFAULT_MAX_Y
                self.zoom_factor = 1.0

        self.update_fractal_controls_visibility()
        self.start_render()

    def update_fractal_controls_visibility(self):
        """Updates visibility and enabled state of fractal-specific UI controls."""
        fractal_idx = self.fractal_combo.currentIndex()
        is_julia = (fractal_idx == 1)
        is_lyapunov = (fractal_idx == 6)
        is_complex_fractal = not is_lyapunov # Mandelbrot, Julia, Ship, Tricorn etc.

        # Julia controls
        for widget in [self.julia_label, self.julia_combo, self.julia_real_input,
                       self.julia_plus_label, self.julia_imag_input, self.julia_i_label]:
            widget.setVisible(is_julia)
        if is_julia:
            is_custom_julia = (self.julia_combo.currentIndex() == 0) # "Custom" is at index 0
            self.julia_real_input.setEnabled(is_custom_julia)
            self.julia_imag_input.setEnabled(is_custom_julia)
        else: # Ensure disabled if not Julia, to prevent edits
            self.julia_real_input.setEnabled(False)
            self.julia_imag_input.setEnabled(False)

        # Lyapunov controls for primary fractal
        self.lyapunov_sequence_label.setVisible(is_lyapunov)
        self.lyapunov_sequence_input.setVisible(is_lyapunov)

        # Exponent controls (for non-Lyapunov complex fractals)
        self.exponent_label.setVisible(is_complex_fractal)
        self.exponent_input.setVisible(is_complex_fractal)
        self.complex_mode_checkbox.setVisible(is_complex_fractal)

        # Fractal Blending - Fractal 2 specific controls
        is_fractal_blend_active = self.fractal_blend_checkbox.isChecked()
        fractal2_idx = self.fractal2_combo.currentIndex()
        is_fractal2_lyapunov = (fractal2_idx == 6)

        self.fractal2_lyapunov_seq_label.setVisible(is_fractal_blend_active and is_fractal2_lyapunov)
        self.fractal2_lyapunov_seq_input.setVisible(is_fractal_blend_active and is_fractal2_lyapunov)

        # Show power input for non-Lyapunov fractal 2, hide for Lyapunov fractal 2
        self.fractal2_power_input.setVisible(is_fractal_blend_active and not is_fractal2_lyapunov)

        # Update label for power/sequence input of fractal 2
        self.fractal2_power_label.setVisible(is_fractal_blend_active)
        if is_fractal_blend_active:
            self.fractal2_power_label.setText("Seq 2:" if is_fractal2_lyapunov else "Power 2:")

        # Enable/disable other fractal blend controls based on main checkbox
        for widget in [self.fractal2_combo, self.fractal2_iter_input,
                       self.fractal_blend_mode_combo, self.fractal_blend_factor_slider]:
            widget.setEnabled(is_fractal_blend_active)

        # Colormap blending controls - enable/disable based on checkbox
        is_colormap_blend_active = self.blend_checkbox.isChecked()
        for widget in [self.cmap2_combo, self.blend_factor_slider, self.blend_mode_combo,
                       self.nonlinear_power_input, self.segment_point_input]:
            widget.setEnabled(is_colormap_blend_active)
        # Hide/show Power and Segment inputs based on blend mode for colormaps
        current_cmap_blend_mode = self.blend_mode_combo.currentText()
        self.nonlinear_power_input.setVisible(is_colormap_blend_active and current_cmap_blend_mode == 'nonlinear')
        self.segment_point_input.setVisible(is_colormap_blend_active and current_cmap_blend_mode == 'segment')


    def handle_julia_combo(self):
        """Handles selection changes in the Julia preset combobox."""
        idx = self.julia_combo.currentIndex()
        # JULIA_PRESETS[0] is ("Custom", None)
        if idx > 0 and JULIA_PRESETS[idx][1] is not None:
            real, imag = JULIA_PRESETS[idx][1] # type: ignore
            self.julia_real_input.setText(f"{real}")
            self.julia_imag_input.setText(f"{imag}")
            self.julia_real_input.setEnabled(False)
            self.julia_imag_input.setEnabled(False)
        else: # Custom selected or invalid preset
            self.julia_real_input.setEnabled(True)
            self.julia_imag_input.setEnabled(True)
        self.start_render()

    def set_colormap(self, cmap_name: str):
        """Sets the primary colormap and re-renders if an image is present."""
        self.colormap_name = cmap_name
        # Re-rendering by just changing colormap is fast if raw data is kept.
        # Current implementation re-calculates colors in worker.
        # For instant colormap change without full recalc, would need to store raw 'pixels' array.
        if self.current_image is not None: # Only trigger re-render if there's something to update
             self.start_render() # Or a lighter update if only coloring changes.

    def set_iterations(self, value: int):
        """Sets the maximum iterations and updates the label."""
        self.maxiter = value
        self.iter_label.setText(f"Iterations: {value}")
        # Re-render is typically triggered by "Render" button or other param changes.
        # For live update on slider release: self.iter_slider.sliderReleased.connect(self.start_render)

    # --- Parameter Parsing ---
    def _parse_render_parameters(self) -> dict:
        """Parses all UI control values into a dictionary for the FractalWorker."""
        params = {}
        params['min_x'], params['max_x'] = self.min_x, self.max_x
        params['min_y'], params['max_y'] = self.min_y, self.max_y

        # Determine render dimensions (locked for animation, else current label size)
        if self.animation_timer.isActive() and self.animation_width and self.animation_height:
            params['width'], params['height'] = self.animation_width, self.animation_height
        else:
            params['width'], params['height'] = self.image_label.width(), self.image_label.height()

        if params['width'] < MIN_RENDER_WIDTH_HEIGHT or params['height'] < MIN_RENDER_WIDTH_HEIGHT:
            return {} # Invalid dimensions

        params['maxiter'] = self.maxiter
        params['fractal_type'] = self.fractal_combo.currentIndex()
        params['julia_c'] = self.get_julia_c()
        params['colormap_name'] = self.colormap_name

        # Primary fractal power/sequence
        if params['fractal_type'] == 6: # Lyapunov
            seq_text = self.lyapunov_sequence_input.text().upper()
            if not seq_text or not all(c in 'AB' for c in seq_text):
                seq_text = "AB" # Default/fallback
                self.lyapunov_sequence_input.setText(seq_text)
            params['power_or_sequence'] = seq_text
            params['lyapunov_seq'] = seq_text # Specific arg for worker
        else: # Complex fractals
            params['power_or_sequence'] = self.get_exponent()
            params['lyapunov_seq'] = "AB" # Default, not used by non-Lyapunov

        # Colormap blending parameters
        params['blend_enabled'] = self.blend_checkbox.isChecked()
        params['colormap_2_name'] = self.cmap2_combo.currentText()
        params['blend_factor'] = self.blend_factor_slider.value() / 100.0
        params['blend_mode'] = self.blend_mode_combo.currentText() # type: ignore
        params['nonlinear_power'] = self._safe_float(self.nonlinear_power_input.text(), 2.0)
        params['segment_point'] = self._safe_float(self.segment_point_input.text(), 0.5)

        # Fractal blending parameters
        params['fractal_blend_enabled'] = self.fractal_blend_checkbox.isChecked()
        params['fractal2_type'] = self.fractal2_combo.currentIndex()
        params['fractal2_iter'] = self._safe_int(self.fractal2_iter_input.text(), 500)
        params['fractal_blend_mode'] = self.fractal_blend_mode_combo.currentText()
        params['fractal_blend_factor'] = self.fractal_blend_factor_slider.value() / 100.0

        if params['fractal2_type'] == 6: # Lyapunov for fractal 2
            seq2_text = self.fractal2_lyapunov_seq_input.text().upper()
            if not seq2_text or not all(c in 'AB' for c in seq2_text):
                seq2_text = "AB"
                self.fractal2_lyapunov_seq_input.setText(seq2_text)
            params['fractal2_power_or_sequence'] = seq2_text
            params['lyapunov_seq2'] = seq2_text
        else: # Complex fractal for fractal 2
            try:
                val = ast.literal_eval(self.fractal2_power_input.text())
                params['fractal2_power_or_sequence'] = complex(val) if isinstance(val, (int, float, complex)) else complex(2.0)
            except (ValueError, SyntaxError, TypeError):
                params['fractal2_power_or_sequence'] = complex(2.0)
            params['lyapunov_seq2'] = "AB" # Default, not used

        return params

    # --- Rendering Pipeline ---
    def start_render(self):
        """Initiates the fractal rendering process."""
        if self.worker and self.worker.isRunning():
            self.worker.abort()
            self.worker.wait() # Wait for thread to finish before starting a new one

        render_params = self._parse_render_parameters()
        if not render_params: # Invalid params (e.g., too small dimensions)
            self.status_label.setText("Render cancelled: Invalid dimensions.")
            return

        self.status_label.setText("Rendering...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = FractalWorker(**render_params) # type: ignore
        self.worker.image_ready.connect(self.handle_image_ready)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self.render_finished)
        self.worker.start()

    def _safe_float(self, text: str, default: float) -> float:
        """Safely converts text to float, returning default on failure."""
        try:
            return float(text)
        except ValueError:
            return default

    def _safe_int(self, text: str, default: int) -> int:
        """Safely converts text to int, returning default on failure."""
        try:
            return int(text)
        except ValueError:
            return default

    def get_exponent(self) -> complex | float:
        """Parses the exponent input field, returns complex or float."""
        text = self.exponent_input.text()
        try:
            # ast.literal_eval is safer than eval for simple literals
            value = ast.literal_eval(text)
            if self.complex_mode_checkbox.isChecked():
                return complex(value) # Ensure it's complex if box checked
            elif isinstance(value, (int, float)):
                return float(value) # Return as float if not complex mode
            else: # Parsed, but not a number (e.g. list)
                 return complex(2.0) if self.complex_mode_checkbox.isChecked() else 2.0
        except (ValueError, SyntaxError, TypeError):
            # Fallback to default on any parsing error
            return complex(2.0) if self.complex_mode_checkbox.isChecked() else 2.0


    def get_julia_c(self) -> complex:
        """Parses the Julia constant input fields."""
        try:
            real_part = float(self.julia_real_input.text())
            imag_part = float(self.julia_imag_input.text())
            return complex(real_part, imag_part)
        except ValueError:
            self.status_label.setText("Invalid Julia C value, using default.")
            # Reset to default if parsing fails and update UI
            self.julia_real_input.setText(str(DEFAULT_JULIA_C.real))
            self.julia_imag_input.setText(str(DEFAULT_JULIA_C.imag))
            return DEFAULT_JULIA_C


    def handle_image_ready(self, image_array: np.ndarray, params: tuple):
        """Slot for when the FractalWorker has a new image ready."""
        self.current_image = image_array
        self.last_render_params = params # Save coordinates of this render

        if self.animation_timer.isActive():
            if len(self.animation_frames) < self.total_animation_steps:
                self.animation_frames.append(np.copy(image_array)) # Store a copy
                self.status_label.setText(f"Animation: Frame {len(self.animation_frames)}/{self.total_animation_steps} captured.")

            self.current_animation_step += 1 # Increment for the next frame to be prepared
            # Stop condition is checked in render_finished or animation_step

        self.update_image_display()


    def update_image_display(self):
        """Updates the image_label with the current fractal image."""
        if self.current_image is None or self.current_image.size == 0:
            # Clear pixmap or show a placeholder if no image
            self.image_label.clear()
            return

        h, w, channels = self.current_image.shape
        if channels != 3:
            self.status_label.setText("Error: Image is not RGB.")
            return

        bytes_per_line = channels * w
        qimage = QImage(
            self.current_image.data, # Direct pointer to numpy array
            w, h,
            bytes_per_line,
            QImage.Format_RGB888
        ).copy() # Copy needed because numpy array might go out of scope or change

        pixmap = QPixmap.fromImage(qimage)

        # Scale pixmap to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), # Fit to current label size
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

        # Calculate offset and size for mouse coordinate mapping
        self.pixmap_offset = QPoint(
            (self.image_label.width() - scaled_pixmap.width()) // 2,
            (self.image_label.height() - scaled_pixmap.height()) // 2
        )
        self.pixmap_size = QPoint(scaled_pixmap.width(), scaled_pixmap.height())


    def render_finished(self):
        """Slot for when the FractalWorker finishes its task."""
        if self._abort: # If aborted from worker side due to error, status might be different
            self.status_label.setText("Render Aborted.")
        elif not (self.animation_timer.isActive() and self.current_animation_step < self.total_animation_steps) :
             self.status_label.setText("Ready")

        self.progress_bar.setVisible(False)
        self.zoom_label.setText(f"Zoom: {self.zoom_factor:.2f}x")

        # Update coordinate display inputs
        center_x = (self.min_x + self.max_x) / 2
        center_y = (self.min_y + self.max_y) / 2
        self.center_x_input.setText(f"{center_x:.8f}")
        self.center_y_input.setText(f"{center_y:.8f}")

        # Check if animation sequence has completed
        if self.animation_timer.isActive() and \
           self.current_animation_step >= self.total_animation_steps and \
           len(self.animation_frames) == self.total_animation_steps:
            self.stop_animation() # All frames rendered and captured


    def save_image(self):
        """Saves the current fractal image to a file."""
        if self.current_image is None:
            QMessageBox.information(self, "No Image", "There is no image to save.")
            return

        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Fractal Image", "fractal.png",
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;Bitmap Images (*.bmp)"
        )
        if not filename: # User cancelled
            return

        # Ensure correct QImage format if needed, though RGB888 is common
        h, w, _ = self.current_image.shape
        qimage = QImage(self.current_image.data, w, h, 3 * w, QImage.Format_RGB888).copy()

        # Add extension if missing, based on filter (though Qt usually handles this)
        # Simple check:
        if "." not in filename:
            if "png" in selected_filter.lower(): filename += ".png"
            elif "jpg" in selected_filter.lower(): filename += ".jpg"
            elif "bmp" in selected_filter.lower(): filename += ".bmp"
            else: filename += ".png" # Default

        if qimage.save(filename):
            self.status_label.setText(f"Image saved to {filename}")
        else:
            QMessageBox.warning(self, "Save Error", f"Could not save image to {filename}.")
            self.status_label.setText("Error saving image.")


    def reset_view(self):
        """Resets the view to the default coordinates and zoom for the current fractal type."""
        current_fractal_idx = self.fractal_combo.currentIndex()
        if current_fractal_idx == 6: # Lyapunov
            self.min_x, self.max_x = LYAPUNOV_MIN_X, LYAPUNOV_MAX_X
            self.min_y, self.max_y = LYAPUNOV_MIN_Y, LYAPUNOV_MAX_Y
        else: # Default for Mandelbrot and other complex fractals
            self.min_x, self.max_x = DEFAULT_MIN_X, DEFAULT_MAX_X
            self.min_y, self.max_y = DEFAULT_MIN_Y, DEFAULT_MAX_Y
        self.zoom_factor = 1.0
        self.start_render()


    def zoom(self, factor: float, mouse_x: int, mouse_y: int):
        """Zooms the view by a given factor, centered on mouse coordinates."""
        if not self.pixmap_size.x() or not self.pixmap_size.y(): return # No valid pixmap

        # Convert mouse coordinates from label space to pixmap space
        center_on_pixmap_x = mouse_x - self.pixmap_offset.x()
        center_on_pixmap_y = mouse_y - self.pixmap_offset.y()

        # Convert pixmap coordinates to fractal (real/imaginary) coordinates
        center_real = self.min_x + center_on_pixmap_x * (self.max_x - self.min_x) / self.pixmap_size.x()
        center_imag = self.min_y + center_on_pixmap_y * (self.max_y - self.min_y) / self.pixmap_size.y()

        new_range_x = (self.max_x - self.min_x) / factor
        new_range_y = (self.max_y - self.min_y) / factor

        self.min_x = center_real - new_range_x / 2
        self.max_x = center_real + new_range_x / 2
        self.min_y = center_imag - new_range_y / 2
        self.max_y = center_imag + new_range_y / 2

        self.zoom_factor *= factor
        self.start_render()


    def wheelEvent(self, event: QWheelEvent): # type: ignore
        """Handles mouse wheel events for zooming."""
        zoom_amount = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.zoom(zoom_amount, event.pos().x(), event.pos().y())
        event.accept()


    def mouse_move_event(self, event: QMouseEvent): # type: ignore
        """Handles mouse move events for coordinate display and selection rectangle drawing."""
        if self.current_image is None or not self.pixmap_size.x() or not self.pixmap_size.y():
            self.coord_label.setText("")
            return

        # Convert event position to position on the (potentially scaled) pixmap
        x_on_pixmap = event.pos().x() - self.pixmap_offset.x()
        y_on_pixmap = event.pos().y() - self.pixmap_offset.y()

        if (0 <= x_on_pixmap < self.pixmap_size.x() and 0 <= y_on_pixmap < self.pixmap_size.y()):
            # Map pixmap coordinates to fractal (real/imaginary) coordinates
            real = self.min_x + x_on_pixmap * (self.max_x - self.min_x) / self.pixmap_size.x()
            imag = self.min_y + y_on_pixmap * (self.max_y - self.min_y) / self.pixmap_size.y()
            self.coord_label.setText(f"({real:.8f}, {imag:.8f})")
        else:
            self.coord_label.setText("") # Cursor is outside the displayed fractal image

        if self.selection_active and self.last_pos:
            # Draw selection rectangle relative to the image_label
            rect = QRectF(self.last_pos, event.pos()).normalized()
            self.image_label.set_selection_rect(rect)
        event.accept()


    def mouse_press_event(self, event: QMouseEvent): # type: ignore
        """Handles mouse press events for starting selection or right-click zoom."""
        if event.button() == Qt.LeftButton:
            self.last_pos = event.pos() # Store position relative to image_label
            self.selection_active = True
            self.image_label.set_selection_rect(QRectF(self.last_pos, self.last_pos)) # Start a tiny rect
        elif event.button() == Qt.RightButton: # Simple zoom-in on right click at center
            self.zoom(1.5, self.image_label.width() // 2, self.image_label.height() // 2)
            self.image_label.clear_selection_rect()
        event.accept()


    def mouse_release_event(self, event: QMouseEvent): # type: ignore
        """Handles mouse release events to finalize selection zoom."""
        if event.button() == Qt.LeftButton and self.selection_active:
            self.selection_active = False
            current_selection = self.image_label.selection_rect

            if current_selection and current_selection.width() > 5 and current_selection.height() > 5:
                # Selection rectangle is in image_label coordinates.
                # Need to map it to fractal coordinates.

                # 1. Intersect with the actual pixmap area within the label
                pixmap_rect_in_label = QRectF(self.pixmap_offset, QSizeF(self.pixmap_size.x(), self.pixmap_size.y())) # type: ignore
                valid_selection_in_label = current_selection.intersected(pixmap_rect_in_label)

                if valid_selection_in_label.width() > 5 and valid_selection_in_label.height() > 5:
                    # 2. Convert valid selection from label coordinates to pixmap coordinates
                    sel_left_on_pixmap = valid_selection_in_label.left() - self.pixmap_offset.x()
                    sel_top_on_pixmap = valid_selection_in_label.top() - self.pixmap_offset.y()
                    sel_right_on_pixmap = valid_selection_in_label.right() - self.pixmap_offset.x()
                    sel_bottom_on_pixmap = valid_selection_in_label.bottom() - self.pixmap_offset.y()

                    # 3. Convert pixmap selection coordinates to fractal coordinates
                    x_range = self.max_x - self.min_x
                    y_range = self.max_y - self.min_y

                    new_min_x = self.min_x + (sel_left_on_pixmap / self.pixmap_size.x()) * x_range
                    new_max_x = self.min_x + (sel_right_on_pixmap / self.pixmap_size.x()) * x_range
                    new_min_y = self.min_y + (sel_top_on_pixmap / self.pixmap_size.y()) * y_range
                    new_max_y = self.min_y + (sel_bottom_on_pixmap / self.pixmap_size.y()) * y_range

                    self.min_x, self.max_x = new_min_x, new_max_x
                    self.min_y, self.max_y = new_min_y, new_max_y

                    # Update zoom factor based on the change in x-range (or y-range)
                    self.zoom_factor *= (x_range / (new_max_x - new_min_x)) if (new_max_x - new_min_x) > 1e-12 else 1.0
                    self.start_render()

            self.image_label.clear_selection_rect()
        event.accept()


    def resizeEvent(self, event: QResizeEvent): # type: ignore
        """Handles widget resize events to re-render and update image display."""
        # Re-render on resize to fit new dimensions
        # Debounce this or make it optional if performance is an issue on slow systems
        if self.image_label.width() >= MIN_RENDER_WIDTH_HEIGHT and \
           self.image_label.height() >= MIN_RENDER_WIDTH_HEIGHT:
            self.start_render()
        else: # If too small, just update display if there's an old image
            self.update_image_display()
        super().resizeEvent(event)

    # --- Animation Methods ---
    def start_animation(self):
        """Starts the animation sequence."""
        try:
            start_val = float(self.animate_start_input.text())
            end_val = float(self.animate_end_input.text())
            steps = int(self.animate_steps_input.text())
            if steps <= 1: # Need at least 2 steps for a change
                self.status_label.setText("Animation steps must be greater than 1.")
                return
        except ValueError:
            self.status_label.setText("Invalid animation parameters (must be numbers).")
            return

        self.animation_param_values = np.linspace(start_val, end_val, steps)
        self.animation_frames = [] # Clear previous frames
        self.current_animation_step = 0
        self.total_animation_steps = steps

        # Lock animation frame size to current image label size at start of animation
        self.animation_width = self.image_label.width()
        self.animation_height = self.image_label.height()
        if self.animation_width < MIN_RENDER_WIDTH_HEIGHT or self.animation_height < MIN_RENDER_WIDTH_HEIGHT:
            self.status_label.setText("Animation cancelled: Render area too small.")
            return

        self.set_animation_controls_enabled(False) # Disable UI during animation
        self.start_animation_button.setEnabled(False)
        self.stop_animation_button.setEnabled(True)
        self.export_animation_button.setEnabled(False)

        fps = self.animate_fps_slider.value()
        self.animation_timer.start(int(1000 / fps)) # Timer interval in milliseconds
        self.status_label.setText("Animation running...")
        self.animation_step() # Start the first step immediately


    def stop_animation(self):
        """Stops the current animation sequence."""
        self.animation_timer.stop()
        self.set_animation_controls_enabled(True)
        self.start_animation_button.setEnabled(True)
        self.stop_animation_button.setEnabled(False)
        if self.animation_frames:
            self.export_animation_button.setEnabled(True)

        captured_frames = len(self.animation_frames)
        status_msg = "Animation stopped."
        if captured_frames > 0:
            status_msg += f" {captured_frames} frame(s) captured."
        self.status_label.setText(status_msg)

        # Reset animation lock dimensions
        self.animation_width = None
        self.animation_height = None


    def animation_step(self):
        """Performs one step of the animation, updating parameters and triggering a render."""
        if self.worker and self.worker.isRunning():
            # This can happen if timer fires while previous frame is still rendering.
            # Wait for it to complete; render_finished will call stop_animation or
            # handle_image_ready will increment step and if not done, timer will call animation_step again.
            return

        if self.current_animation_step >= self.total_animation_steps:
            # This means all steps have been initiated.
            # If all frames are also captured, stop_animation would have been called from render_finished.
            # If frames are still pending capture, we wait.
            # If somehow reached here without worker running and not all frames captured, it's an anomaly.
            if len(self.animation_frames) < self.total_animation_steps:
                 self.status_label.setText(f"Animation: Waiting for frame {len(self.animation_frames)+1} capture...")
            # else stop_animation should have been called
            return

        self.status_label.setText(f"Animation: Preparing frame {self.current_animation_step + 1}/{self.total_animation_steps}")
        current_val = self.animation_param_values[self.current_animation_step]
        param_to_animate = self.animate_variable_combo.currentText()

        # Update UI controls based on the animated parameter
        if param_to_animate == "Julia C Real":
            self.julia_real_input.setText(f"{current_val:.8f}")
            if self.fractal_combo.currentIndex() == 1: self.julia_combo.setCurrentIndex(0) # Switch to custom
        elif param_to_animate == "Julia C Imag":
            self.julia_imag_input.setText(f"{current_val:.8f}")
            if self.fractal_combo.currentIndex() == 1: self.julia_combo.setCurrentIndex(0)
        elif param_to_animate == "Exponent Real":
            current_exp = self.get_exponent() # Get current exponent (could be complex)
            imag_part = current_exp.imag if isinstance(current_exp, complex) else 0.0
            self.exponent_input.setText(f"{current_val:.4f}{imag_part:+.4f}j" if self.complex_mode_checkbox.isChecked() else f"{current_val:.4f}")
        elif param_to_animate == "Exponent Imag":
            self.complex_mode_checkbox.setChecked(True) # Force complex mode for imag animation
            current_exp = self.get_exponent()
            real_part = current_exp.real if isinstance(current_exp, complex) else float(current_exp)
            self.exponent_input.setText(f"{real_part:.4f}{current_val:+.4f}j")
        elif param_to_animate == "Iterations":
            self.iter_slider.setValue(int(current_val))
            self.set_iterations(int(current_val)) # Update label too

        self.start_render() # This will render the frame for self.current_animation_step
        # self.current_animation_step is incremented in handle_image_ready after frame is processed.


    def set_animation_controls_enabled(self, enabled: bool):
        """Enables/disables UI controls that should not be changed during animation."""
        # Iteration and main fractal controls
        self.iter_slider.setEnabled(enabled)
        self.fractal_combo.setEnabled(enabled)
        self.cmap_combo.setEnabled(enabled)
        self.render_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled) # Allow saving even if animation controls are active (but not running)

        # Specific fractal parameter inputs
        is_julia_type = (self.fractal_combo.currentIndex() == 1)
        is_custom_julia = (self.julia_combo.currentIndex() == 0)
        self.julia_combo.setEnabled(enabled and is_julia_type)
        self.julia_real_input.setEnabled(enabled and is_julia_type and is_custom_julia)
        self.julia_imag_input.setEnabled(enabled and is_julia_type and is_custom_julia)

        self.exponent_input.setEnabled(enabled)
        self.complex_mode_checkbox.setEnabled(enabled)
        self.lyapunov_sequence_input.setEnabled(enabled)

        # Blending controls
        self.blend_checkbox.setEnabled(enabled)
        # Further enable/disable based on blend_checkbox state by update_fractal_controls_visibility

        self.fractal_blend_checkbox.setEnabled(enabled)
        # Further enable/disable based on fractal_blend_checkbox state by update_fractal_controls_visibility

        # Animation parameter inputs (the ones that define the animation itself)
        self.animate_variable_combo.setEnabled(enabled)
        self.animate_start_input.setEnabled(enabled)
        self.animate_end_input.setEnabled(enabled)
        self.animate_steps_input.setEnabled(enabled)
        self.animate_fps_slider.setEnabled(enabled)

        # Ensure visibility is updated after enabling/disabling parents
        if enabled:
            self.update_fractal_controls_visibility()


    def export_animation(self):
        """Exports captured animation frames to a GIF or MP4 file."""
        if not self.animation_frames:
            QMessageBox.information(self, "No Animation", "No animation frames have been captured to export.")
            return

        # Ensure all frames have consistent shape (important for video codecs)
        first_shape = self.animation_frames[0].shape
        if not all(frame.shape == first_shape for frame in self.animation_frames):
            QMessageBox.warning(self, "Shape Mismatch", "Animation frames have inconsistent dimensions. Cannot export.")
            return

        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Animation", "animation.gif",
            "GIF Files (*.gif);;MP4 Video Files (*.mp4)"
        )
        if not filename: return

        self.status_label.setText(f"Exporting animation to {filename}...")
        QApplication.processEvents() # Update UI

        fps = self.animate_fps_slider.value()
        try:
            # imageio.mimsave arguments:
            # For GIF: subrectangles=True can optimize by only saving changed parts.
            # For MP4: macro_block_size=None (default) or a typical value like 16.
            #          `ffmpeg` plugin is usually needed for MP4.
            if filename.lower().endswith('.gif'):
                imageio.mimsave(filename, self.animation_frames, format='GIF', fps=fps, subrectangles=True)
            elif filename.lower().endswith('.mp4'):
                # Ensure imageio has ffmpeg plugin: imageio.plugins.ffmpeg.download()
                # macro_block_size=1 is unusual; typically None or 16. Let's use default.
                imageio.mimsave(filename, self.animation_frames, format='MP4', fps=fps, quality=8) # quality 0-10
            else:
                # Fallback or try to guess format from extension
                imageio.mimsave(filename, self.animation_frames, fps=fps)

            self.status_label.setText(f"Animation exported to {filename}")
            QMessageBox.information(self, "Export Complete", f"Animation successfully exported to {filename}")

        except Exception as e:
            error_msg = f"Error exporting animation: {e}\n\n"
            if filename.lower().endswith('.mp4'):
                error_msg += "Ensure FFmpeg is installed and accessible to imageio. You might need to run: `imageio.plugins.ffmpeg.download()` in a Python console."
            QMessageBox.critical(self, "Export Error", error_msg)
            self.status_label.setText("Error exporting animation.")
            print(f"Error exporting animation: {e}")


    def keyPressEvent(self, event: QKeyEvent): # type: ignore
        """Handles key press events for panning the fractal view."""
        pan_fraction = 0.1  # Pan by 10% of the current view width/height
        current_x_range = self.max_x - self.min_x
        current_y_range = self.max_y - self.min_y
        dx = current_x_range * pan_fraction
        dy = current_y_range * pan_fraction

        moved = False
        if event.key() == Qt.Key_Left:
            self.min_x -= dx; self.max_x -= dx; moved = True
        elif event.key() == Qt.Key_Right:
            self.min_x += dx; self.max_x += dx; moved = True
        elif event.key() == Qt.Key_Up:
            self.min_y -= dy; self.max_y -= dy; moved = True # Y-axis is often inverted in screen coords vs math
        elif event.key() == Qt.Key_Down:
            self.min_y += dy; self.max_y += dy; moved = True

        if moved:
            self.start_render()
            event.accept()
        else:
            super().keyPressEvent(event)


    def goto_coordinates(self):
        """Navigates the view to the specified center coordinates."""
        try:
            center_x = float(self.center_x_input.text())
            center_y = float(self.center_y_input.text())

            current_width = self.max_x - self.min_x
            current_height = self.max_y - self.min_y

            self.min_x = center_x - current_width / 2
            self.max_x = center_x + current_width / 2
            self.min_y = center_y - current_height / 2
            self.max_y = center_y + current_height / 2

            self.start_render()
        except ValueError:
            QMessageBox.warning(self, "Invalid Coordinates", "Please enter valid numeric coordinates.")
            self.status_label.setText("Invalid coordinates for Go To.")


def set_dark_palette(app: QApplication):
    """Applies a dark color palette to the QApplication."""
    # This is a predefined dark palette. More sophisticated theming can be done.
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(40, 40, 40))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(30, 30, 30)) # Background for text entry widgets
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45)) # Used in some views
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, QColor(40,40,40)) # Dark text on light tooltip
    palette.setColor(QPalette.Text, Qt.white) # General text
    palette.setColor(QPalette.Button, QColor(50, 50, 50))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red) # e.g. for errors or highlights
    palette.setColor(QPalette.Link, QColor(42, 130, 218)) # Standard link blue
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218)) # Selection highlight
    palette.setColor(QPalette.HighlightedText, Qt.black) # Text over selection highlight

    # Disabled states
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(128,128,128))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128,128,128))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128,128,128))
    palette.setColor(QPalette.Disabled, QPalette.Base, QColor(35,35,35))
    palette.setColor(QPalette.Disabled, QPalette.Button, QColor(40,40,40))

    app.setPalette(palette)
    # Note: For the stylesheet to fully integrate with palette changes,
    # it's sometimes better to use palette roles (e.g., `background: palette(window);`)
    # in the QSS, but hardcoded colors in QSS often override palette settings.
    # The provided _set_styles method uses hardcoded colors, which will largely define the look.