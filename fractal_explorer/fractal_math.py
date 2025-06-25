import numpy as np
from numba import njit

# --- Constants and Presets ---
JULIA_PRESETS = [
    ("Custom", None),
    ("-0.7 + 0.27015i (Classic)", (-0.7, 0.27015)),
    ("0.355 + 0.355i (Spiral)", (0.355, 0.355)),
    ("-0.4 + 0.6i (Nebula)", (-0.4, 0.6)),
    ("0.285 + 0.01i (Swirls)", (0.285, 0.01)),
    ("-0.70176 - 0.3842i (Snowflake)", (-0.70176, -0.3842)),
]

# --- Fractal Calculation Functions ---

@njit
def fractal_smooth(c, maxiter, fractal_type, power, constant_c=0j):
    """
    Compute the smooth iteration count for a given point in the complex plane.
    """
    if fractal_type == 1:  # Julia
        z = c
    else:
        z = 0j
    for n in range(maxiter):
        if abs(z) > 2:
            return n + 1 - np.log(np.log2(abs(z)))
        if fractal_type == 0:  # Mandelbrot
            z = z**power + c
        elif fractal_type == 1:  # Julia
            z = z**power + constant_c
        elif fractal_type == 2:  # Burning Ship
            z = complex(abs(z.real), abs(z.imag))
            z = z**power + c
        elif fractal_type == 3:  # Tricorn
            z = np.conj(z)
            z = z**power + c
        elif fractal_type == 4:  # Celtic Mandelbrot
            z = complex(abs(z.real), z.imag)
            z = z**power + c
        elif fractal_type == 5:  # Buffalo
            z = complex(abs(z.real), abs(z.imag))
            z = z**power + c
    return maxiter

def compute_fractal(
    min_x, max_x, min_y, max_y, width, height,
    maxiter, fractal_type, power, constant_c=0j, progress_callback=None
):
    """
    Compute a single fractal array.
    """
    pixels = np.zeros((height, width), dtype=np.float64)
    for x in range(width):
        for y in range(height):
            real = min_x + x * (max_x - min_x) / width
            imag = min_y + y * (max_y - min_y) / height
            c = real + imag * 1j
            pixels[y, x] = fractal_smooth(c, maxiter, fractal_type, power, constant_c)
        if progress_callback and x % max(1, width // 100) == 0:
            percent = int(100 * x / width)
            progress_callback(percent)
    return pixels

def compute_blended_fractal(
    min_x, max_x, min_y, max_y, width, height,
    maxiter1, fractal_type1, power1, constant_c1,
    maxiter2, fractal_type2, power2, constant_c2,
    progress_callback=None
):
    """
    Compute two fractal arrays with different parameters for blending.
    Returns: pixels1, pixels2
    """
    pixels1 = np.zeros((height, width), dtype=np.float64)
    pixels2 = np.zeros((height, width), dtype=np.float64)
    for x in range(width):
        for y in range(height):
            real = min_x + x * (max_x - min_x) / width
            imag = min_y + y * (max_y - min_y) / height
            c = real + imag * 1j
            pixels1[y, x] = fractal_smooth(c, maxiter1, fractal_type1, power1, constant_c1)
            pixels2[y, x] = fractal_smooth(c, maxiter2, fractal_type2, power2, constant_c2)
        if progress_callback and x % max(1, width // 100) == 0:
            percent = int(100 * x / width)
            progress_callback(percent)
    return pixels1, pixels2

def blend_fractals_mask(pixels1, pixels2, blend_factor=0.5):
    """
    Weighted influence: output = (1-blend_factor)*pixels1 + blend_factor*pixels2
    """
    return (1 - blend_factor) * pixels1 + blend_factor * pixels2

def blend_fractals_alternating(pixels1, pixels2, mode='checker'):
    """
    Alternates between pixels1 and pixels2 in a checkerboard or stripe pattern.
    mode: 'checker', 'vertical', or 'horizontal'
    """
    h, w = pixels1.shape
    if mode == 'checker':
        mask = (np.indices((h, w)).sum(axis=0) % 2 == 0)
    elif mode == 'vertical':
        mask = np.zeros((h, w), dtype=bool)
        mask[:, ::2] = True
    elif mode == 'horizontal':
        mask = np.zeros((h, w), dtype=bool)
        mask[::2, :] = True
    else:
        mask = np.ones((h, w), dtype=bool)
    return np.where(mask, pixels1, pixels2)