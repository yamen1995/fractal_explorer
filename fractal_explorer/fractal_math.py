import numpy as np
from numba import njit, prange
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
def fractal_smooth(c, maxiter, fractal_type, constant_c=0j):
    if fractal_type == 1:  # Julia
        z = c
    else:
        z = 0j
    for n in range(maxiter):
        if abs(z) > 2:
            return n + 1 - np.log(np.log2(abs(z)))
        if fractal_type == 0:  # Mandelbrot
            z = z*z + c
        elif fractal_type == 1:  # Julia
            z = z*z + constant_c
        elif fractal_type == 2:  # Burning Ship
            z = complex(abs(z.real), abs(z.imag))
            z = z*z + c
        elif fractal_type == 3:  # Tricorn
            z = np.conj(z)
            z = z*z + c
    return maxiter

@njit(parallel=True)
def compute_fractal(min_x, max_x, min_y, max_y, width, height, maxiter, fractal_type, constant_c=0j):
    pixels = np.zeros((height, width), dtype=np.float64)
    for x in prange(width):
        for y in range(height):
            real = min_x + x * (max_x - min_x) / width
            imag = min_y + y * (max_y - min_y) / height
            c = real + imag * 1j
            pixels[y, x] = fractal_smooth(c, maxiter, fractal_type, constant_c)
    return pixels