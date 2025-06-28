import unittest
from unittest.mock import patch, MagicMock
import sys

# To allow importing from fractal_explorer
sys.path.append('.')

from PyQt5.QtWidgets import QApplication
from fractal_explorer.ui import FractalExplorer, JULIA_PRESETS

# A global QApplication instance is required for PyQt5 widgets, even if not shown.
# We create it once, as creating it multiple times can lead to issues.
app = None

def setUpModule():
    global app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

def tearDownModule():
    global app
    app.quit()
    app = None


class TestFractalExplorer(unittest.TestCase):

    def setUp(self):
        """Set up for each test."""
        self.explorer = FractalExplorer()
        # Mock start_render to prevent actual rendering during tests
        self.explorer.start_render = MagicMock()

    def test_initial_state(self):
        """Test the initial state of FractalExplorer."""
        self.assertEqual(self.explorer.min_x, -2.0)
        self.assertEqual(self.explorer.max_x, 1.0)
        self.assertEqual(self.explorer.min_y, -1.5)
        self.assertEqual(self.explorer.max_y, 1.5)
        self.assertEqual(self.explorer.maxiter, 500)
        self.assertEqual(self.explorer.fractal_type, 0) # Mandelbrot
        self.assertEqual(self.explorer.colormap_name, 'plasma')

    def test_set_iterations(self):
        """Test changing iterations."""
        self.explorer.iter_slider.setValue(1000)
        self.explorer.set_iterations(1000) # Direct call as slider signal might not fire in test
        self.assertEqual(self.explorer.maxiter, 1000)
        self.assertEqual(self.explorer.iter_label.text(), "Iterations: 1000")
        # self.explorer.start_render.assert_not_called() # set_iterations should not auto-render

    def test_set_colormap(self):
        """Test changing colormap."""
        self.explorer.cmap_combo.setCurrentText('viridis')
        # self.explorer.set_colormap('viridis') # Direct call if signal not reliable
        self.assertEqual(self.explorer.colormap_name, 'viridis')
        # In the actual app, update_image_display might be called if current_image exists.
        # Here, start_render is mocked, so we mainly check state.

    def test_fractal_set_changed_to_lyapunov(self):
        """Test changing fractal type to Lyapunov."""
        self.explorer.fractal_combo.setCurrentIndex(6) # Lyapunov
        # The signal should call fractal_set_changed
        self.assertEqual(self.explorer.fractal_type, 6)
        self.assertEqual(self.explorer.min_x, 2.0)
        self.assertEqual(self.explorer.max_x, 4.0)
        self.assertEqual(self.explorer.min_y, 2.0)
        self.assertEqual(self.explorer.max_y, 4.0)
        self.explorer.start_render.assert_called()

    def test_fractal_set_changed_from_lyapunov(self):
        """Test changing fractal type from Lyapunov to Mandelbrot."""
        # First set to Lyapunov
        self.explorer.fractal_combo.setCurrentIndex(6)
        self.explorer.start_render.reset_mock() # Reset mock after initial call

        # Then change to Mandelbrot
        self.explorer.fractal_combo.setCurrentIndex(0) # Mandelbrot
        self.assertEqual(self.explorer.fractal_type, 0)
        self.assertEqual(self.explorer.min_x, -2.0)
        self.assertEqual(self.explorer.max_x, 1.0)
        self.assertEqual(self.explorer.min_y, -1.5)
        self.assertEqual(self.explorer.max_y, 1.5)
        self.explorer.start_render.assert_called()

    def test_reset_view_mandelbrot(self):
        """Test reset_view for Mandelbrot."""
        self.explorer.fractal_combo.setCurrentIndex(0) # Mandelbrot
        self.explorer.min_x, self.explorer.max_x = 0, 1 # Change view
        self.explorer.reset_view()
        self.assertEqual(self.explorer.min_x, -2.0)
        self.assertEqual(self.explorer.max_x, 1.0)
        self.explorer.start_render.assert_called()

    def test_reset_view_lyapunov(self):
        """Test reset_view for Lyapunov."""
        self.explorer.fractal_combo.setCurrentIndex(6) # Lyapunov
        self.explorer.min_x, self.explorer.max_x = 0, 1 # Change view
        self.explorer.reset_view()
        self.assertEqual(self.explorer.min_x, 2.0)
        self.assertEqual(self.explorer.max_x, 4.0)
        self.explorer.start_render.assert_called()

    def test_get_julia_c(self):
        """Test get_julia_c parsing."""
        self.explorer.julia_real_input.setText("0.1")
        self.explorer.julia_imag_input.setText("-0.5")
        self.assertEqual(self.explorer.get_julia_c(), complex(0.1, -0.5))

        self.explorer.julia_real_input.setText("invalid")
        self.explorer.julia_imag_input.setText("0.5")
        default_julia_c = JULIA_PRESETS[1][1] # Classic preset
        self.assertEqual(self.explorer.get_julia_c(), complex(default_julia_c[0], default_julia_c[1]))


    def test_get_exponent_real(self):
        """Test get_exponent for real numbers."""
        self.explorer.exponent_input.setText("3.5")
        self.explorer.complex_mode_checkbox.setChecked(False)
        self.assertEqual(self.explorer.get_exponent(), 3.5)

        self.explorer.exponent_input.setText("invalid")
        self.assertEqual(self.explorer.get_exponent(), 2.0) # Default real

    def test_get_exponent_complex(self):
        """Test get_exponent for complex numbers."""
        self.explorer.exponent_input.setText("1+2j")
        self.explorer.complex_mode_checkbox.setChecked(True)
        self.assertEqual(self.explorer.get_exponent(), complex(1, 2))

        self.explorer.exponent_input.setText("0.5") # Should be 0.5 + 0j
        self.explorer.complex_mode_checkbox.setChecked(True)
        self.assertEqual(self.explorer.get_exponent(), complex(0.5, 0))

        self.explorer.exponent_input.setText("invalid_complex")
        self.explorer.complex_mode_checkbox.setChecked(True)
        self.assertEqual(self.explorer.get_exponent(), complex(2, 0)) # Default complex

    def test_handle_julia_combo_preset(self):
        """Test selecting a Julia preset."""
        self.explorer.fractal_combo.setCurrentIndex(1) # Set to Julia
        self.explorer.start_render.reset_mock()

        classic_julia_preset = JULIA_PRESETS[1][1] # "-0.7 + 0.27015i (Classic)"
        self.explorer.julia_combo.setCurrentIndex(1) # Select classic preset

        self.explorer.handle_julia_combo() # Explicitly call handler
        self.assertEqual(self.explorer.julia_real_input.text(), str(classic_julia_preset[0]))
        self.assertEqual(self.explorer.julia_imag_input.text(), str(classic_julia_preset[1]))
        self.assertFalse(self.explorer.julia_real_input.isEnabled())
        self.explorer.start_render.assert_called()

    def test_handle_julia_combo_custom(self):
        """Test selecting 'Custom' Julia preset."""
        self.explorer.fractal_combo.setCurrentIndex(1) # Set to Julia
        self.explorer.start_render.reset_mock()

        self.explorer.julia_combo.setCurrentIndex(0) # Select "Custom"
        self.explorer.handle_julia_combo() # Explicitly call handler
        self.assertTrue(self.explorer.julia_real_input.isEnabled())
        self.assertTrue(self.explorer.julia_imag_input.isEnabled())
        self.explorer.start_render.assert_called()

    # Test animation related state changes (not the full animation)
    def test_start_animation_state(self):
        self.explorer.animate_start_input.setText("0")
        self.explorer.animate_end_input.setText("1")
        self.explorer.animate_steps_input.setText("10")
        self.explorer.start_animation()

        self.assertFalse(self.explorer.start_animation_button.isEnabled())
        self.assertTrue(self.explorer.stop_animation_button.isEnabled())
        self.assertFalse(self.explorer.export_animation_button.isEnabled()) # No frames yet
        self.assertEqual(len(self.explorer.animation_param_values), 10)
        self.assertEqual(self.explorer.current_animation_step, 0)
        self.assertTrue(self.explorer.animation_timer.isActive())

    def test_stop_animation_state(self):
        # Start it first
        self.explorer.animate_start_input.setText("0")
        self.explorer.animate_end_input.setText("1")
        self.explorer.animate_steps_input.setText("10")
        self.explorer.start_animation()

        # Simulate some frames being added
        self.explorer.animation_frames = [1, 2, 3]

        self.explorer.stop_animation()
        self.assertTrue(self.explorer.start_animation_button.isEnabled())
        self.assertFalse(self.explorer.stop_animation_button.isEnabled())
        self.assertTrue(self.explorer.export_animation_button.isEnabled()) # Frames exist
        self.assertFalse(self.explorer.animation_timer.isActive())

if __name__ == '__main__':
    unittest.main()
