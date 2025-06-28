import unittest
import numpy as np
from fractal_explorer.fractal_math import lyapunov_exponent, fractal_smooth, compute_fractal

class TestFractalMath(unittest.TestCase):

    def test_lyapunov_exponent_basic(self):
        """Test basic Lyapunov exponent calculation."""
        # Test with known stable parameters for ABAB sequence
        # For r=2.5, x converges to (sqrt(r^2-2r+5) - (r-3))/2r = (sqrt(6.25-5+5) - (-0.5))/5 = (2.5+0.5)/5 = 0.6
        # This is a simple case, actual Lyapunov exponent calculation is more complex.
        # For logistic map x_n+1 = r * x_n * (1 - x_n)
        # For r=3.2 (period 2), lyap_exp should be < 0
        # For r=3.8 (chaotic), lyap_exp should be > 0
        # Using sequence "A", a=3.2
        le = lyapunov_exponent("A", 3.2, 0, maxiter=1000, warmup=100)
        self.assertLess(le, 0, "Lyapunov exponent for r=3.2 (period 2) should be negative.")

        # Using sequence "A", a=3.8
        le_chaotic = lyapunov_exponent("A", 3.8, 0, maxiter=1000, warmup=100)
        self.assertGreater(le_chaotic, 0, "Lyapunov exponent for r=3.8 (chaotic) should be positive.")

    def test_fractal_smooth_mandelbrot_inside(self):
        """Test a point known to be inside the Mandelbrot set."""
        # c = 0 is the center of the Mandelbrot set
        val = fractal_smooth(0j, maxiter=100, fractal_type=0, power=2)
        self.assertEqual(val, 100, "Center of Mandelbrot set should reach maxiter.")

    def test_fractal_smooth_mandelbrot_outside(self):
        """Test a point known to be outside the Mandelbrot set."""
        # c = 3 + 0j is well outside the Mandelbrot set
        val = fractal_smooth(3+0j, maxiter=100, fractal_type=0, power=2)
        self.assertLess(val, 100, "Point 3+0j should escape before maxiter.")
        self.assertGreater(val, 0, "Escape value should be positive.")

    def test_fractal_smooth_julia_basic(self):
        """Test a basic Julia set point."""
        # For c = 0, z_0 = 0, Julia set is a circle.
        # If z_0 is inside, it stays inside. If outside, it escapes.
        # Test point z = 0 for Julia with c = -0.7 + 0.27015j (classic)
        # This point (z=0) for this c should be inside the Julia set if c itself is in Mandelbrot set.
        # However, fractal_smooth for Julia uses the input 'c' as the initial 'z'.
        # Let's test z = 0 with a simple c.
        # For c = 0, z = 0 -> z^2 + 0 = 0. Stays at 0.
        val_inside = fractal_smooth(c=0j, maxiter=100, fractal_type=1, power=2, constant_c=0j)
        self.assertEqual(val_inside, 100, "z=0 for Julia with c=0 should reach maxiter.")

        # Test z = 3 for Julia with c = 0j. Should escape.
        val_outside = fractal_smooth(c=3+0j, maxiter=100, fractal_type=1, power=2, constant_c=0j)
        self.assertLess(val_outside, 100, "z=3 for Julia with c=0 should escape.")

    def test_compute_fractal_mandelbrot_output_shape(self):
        """Test the output shape of compute_fractal for Mandelbrot."""
        pixels = compute_fractal(
            min_x=-2.0, max_x=1.0, min_y=-1.5, max_y=1.5,
            width=10, height=10, maxiter=50,
            fractal_type=0, power_or_sequence=2
        )
        self.assertEqual(pixels.shape, (10, 10), "Output pixel array shape mismatch.")

    def test_compute_fractal_lyapunov_output_shape(self):
        """Test the output shape of compute_fractal for Lyapunov."""
        pixels = compute_fractal(
            min_x=2.0, max_x=4.0, min_y=2.0, max_y=4.0,
            width=10, height=10, maxiter=50,
            fractal_type=6, power_or_sequence="AB" # sequence for Lyapunov
        )
        self.assertEqual(pixels.shape, (10, 10), "Output pixel array shape mismatch for Lyapunov.")
        # Check if values are within a reasonable range for Lyapunov exponents (can be negative)
        self.assertTrue(np.all(pixels <= 2.0) and np.all(pixels >= -np.inf))

    def test_fractal_smooth_burning_ship_basic(self):
        """Test a point for Burning Ship."""
        # Point (0,0) for Burning Ship. z_n+1 = (|Re(z_n)| + i|Im(z_n)|)^2 + c
        # z_0 = 0, c = 0 -> z_1 = 0. Should reach maxiter.
        val_inside = fractal_smooth(0j, maxiter=100, fractal_type=2, power=2)
        self.assertEqual(val_inside, 100, "Origin for Burning Ship with c=0 should reach maxiter.")

        # A point that should escape
        val_outside = fractal_smooth(1+1j, maxiter=100, fractal_type=2, power=2)
        self.assertLess(val_outside, 100, "Point 1+1j for Burning Ship should escape.")

    def test_fractal_smooth_tricorn_basic(self):
        """Test a point for Tricorn."""
        # Point (0,0) for Tricorn. z_n+1 = (conj(z_n))^2 + c
        # z_0 = 0, c = 0 -> z_1 = 0. Should reach maxiter.
        val_inside = fractal_smooth(0j, maxiter=100, fractal_type=3, power=2)
        self.assertEqual(val_inside, 100, "Origin for Tricorn with c=0 should reach maxiter.")

        val_outside = fractal_smooth(2+0j, maxiter=100, fractal_type=3, power=2)
        self.assertLess(val_outside, 100, "Point 2+0j for Tricorn should escape.")

    def test_lyapunov_exponent_edge_cases(self):
        """Test Lyapunov exponent with edge cases."""
        # Test with zero iterations
        le_zero_iter = lyapunov_exponent("AB", 2.5, 3.5, maxiter=0, warmup=10)
        self.assertEqual(le_zero_iter, 0.0, "Lyapunov exponent with 0 iterations should be 0.")

        # Test with sequence causing immediate invalid state (e.g., x becomes <=0 or >=1 quickly)
        # If r is too low (e.g., r=0.5), x quickly goes to 0.
        # Derivative term r*(1-2x) at x=0 is r. log(abs(r))
        # If x becomes 0, x = r * x * (1-x) will stay 0.
        # The function should return -np.inf if x hits 0 or 1.
        # For r=0.5, x -> 0. The derivative becomes r. So lyap_sum/maxiter -> log(r).
        le_low_r = lyapunov_exponent("A", 0.5, 0.5, maxiter=500, warmup=100) # Increased iter for convergence
        self.assertAlmostEqual(le_low_r, np.log(0.5), delta=1e-5, msg="Lyapunov exponent for r=0.5 should be log(0.5).")

        le_invalid_high_r = lyapunov_exponent("A", 5.0, 5.0, maxiter=100, warmup=10) # x quickly goes < 0
        self.assertEqual(le_invalid_high_r, -np.inf, "Lyapunov exponent for r=5.0 should lead to -inf.")

    def test_blend_fractals_mask(self):
        """Test blend_fractals_mask."""
        from fractal_explorer.fractal_math import blend_fractals_mask
        pixels1 = np.ones((2, 2)) * 10
        pixels2 = np.ones((2, 2)) * 20

        blended_half = blend_fractals_mask(pixels1, pixels2, blend_factor=0.5)
        expected_half = np.ones((2,2)) * 15
        np.testing.assert_array_almost_equal(blended_half, expected_half, decimal=7, err_msg="Blend factor 0.5 failed.")

        blended_zero = blend_fractals_mask(pixels1, pixels2, blend_factor=0.0)
        np.testing.assert_array_almost_equal(blended_zero, pixels1, decimal=7, err_msg="Blend factor 0.0 failed.")

        blended_one = blend_fractals_mask(pixels1, pixels2, blend_factor=1.0)
        np.testing.assert_array_almost_equal(blended_one, pixels2, decimal=7, err_msg="Blend factor 1.0 failed.")

    def test_blend_fractals_alternating(self):
        """Test blend_fractals_alternating."""
        from fractal_explorer.fractal_math import blend_fractals_alternating
        pixels1 = np.ones((2, 2)) * 1
        pixels2 = np.ones((2, 2)) * 2

        # Checker
        blended_checker = blend_fractals_alternating(pixels1, pixels2, mode='checker')
        expected_checker = np.array([[1, 2], [2, 1]])
        np.testing.assert_array_equal(blended_checker, expected_checker, "Alternating checker blend failed.")

        # Vertical
        blended_vertical = blend_fractals_alternating(pixels1, pixels2, mode='vertical')
        expected_vertical = np.array([[1, 2], [1, 2]])
        np.testing.assert_array_equal(blended_vertical, expected_vertical, "Alternating vertical blend failed.")

        # Horizontal
        blended_horizontal = blend_fractals_alternating(pixels1, pixels2, mode='horizontal')
        expected_horizontal = np.array([[1, 1], [2, 2]])
        np.testing.assert_array_equal(blended_horizontal, expected_horizontal, "Alternating horizontal blend failed.")


if __name__ == '__main__':
    unittest.main()
