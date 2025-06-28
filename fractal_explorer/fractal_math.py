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
def lyapunov_exponent(sequence, a, b, maxiter, warmup=100):
    """
    Calculates the Lyapunov exponent for a given sequence and parameters a, b.
    Sequence should be a string like "ABAB".
    'A' corresponds to parameter 'a', 'B' to 'b'.
    """
    x = 0.5  # Starting value for x, must be in (0,1)
    lyap_sum = 0.0
    seq_len = len(sequence)

    # Warmup iterations to let the system settle
    for i in range(warmup):
        r = a if sequence[i % seq_len] == 'A' else b
        x = r * x * (1.0 - x)
        if x <= 0 or x >= 1: # Bifurcation or invalid state
            return -np.inf # Typically represented as very dark color

    # Main calculation loop
    for i in range(maxiter):
        r = a if sequence[(i + warmup) % seq_len] == 'A' else b
        x = r * x * (1.0 - x)
        if x <= 0 or x >= 1:
             return -np.inf
        derivative = r * (1.0 - 2.0 * x)
        if derivative == 0: # Avoid log(0)
            return -np.inf
        lyap_sum += np.log(abs(derivative))

    if maxiter == 0:
        return 0.0
    return lyap_sum / maxiter

@njit
def fractal_smooth(c, maxiter, fractal_type, power, constant_c=0j, lyapunov_warmup=100):
    """
    Compute the smooth iteration count for a given point in the complex plane.
    """
    if fractal_type == 1:  # Julia
        z = c
    else:
        z = 0j # Mandelbrot, Burning Ship, Tricorn, etc. start with z=0

    for n in range(maxiter):
        if abs(z) > 2: # Escape condition for complex fractals
            absz = abs(z)
            if absz < 1e-12:
                absz = 1e-12
            log2_absz = np.log2(absz)
            if log2_absz < 1e-12:
                log2_absz = 1e-12
            return n + 1 - np.log(log2_absz)

        if fractal_type == 0:  # Mandelbrot
            z = z**power + c
        elif fractal_type == 1:  # Julia
            z = z**power + constant_c
        elif fractal_type == 2:  # Burning Ship
            z_real_abs = abs(z.real)
            z_imag_abs = abs(z.imag)
            z = complex(z_real_abs, z_imag_abs)**power + c
        elif fractal_type == 3:  # Tricorn
            z = np.conj(z)**power + c
        elif fractal_type == 4:  # Celtic Mandelbrot
            z_real_abs = abs(z.real)
            z = complex(z_real_abs, z.imag)**power + c
        elif fractal_type == 5:  # Buffalo
            z_real_abs = abs(z.real)
            z_imag_abs = abs(z.imag)
            z = (complex(z.real, z.imag)**2 - complex(z.real, -z.imag)**2) / 2
            z = complex(abs(z.real), abs(z.imag))
            z = z**power + c
        elif fractal_type == 7:  # Mandelbar
            z = np.conj(z)**power + c
        elif fractal_type == 8:  # Perpendicular Burning Ship
            z_real_abs = abs(z.real)
            z_imag_abs = abs(z.imag)
            z = complex(z_imag_abs, z_real_abs)**power + c
        elif fractal_type == 9:  # Perpendicular Buffalo
            z_real_abs = abs(z.real)
            z_imag_abs = abs(z.imag)
            z = complex(abs(z.imag), abs(z.real))
            z = z**power + c

    return maxiter

def compute_fractal(
    min_x, max_x, min_y, max_y, width, height,
    maxiter, fractal_type, power_or_sequence, constant_c=0j,
    lyapunov_seq="AB", lyapunov_warmup=100,
    progress_callback=None
):
    """
    Compute a single fractal array.
    For Lyapunov, min_x, max_x are 'a' range, min_y, max_y are 'b' range.
    power_or_sequence is the exponent for complex fractals, or the sequence string for Lyapunov.
    """
    pixels = np.zeros((height, width), dtype=np.float64)

    for x_idx in range(width):
        for y_idx in range(height):
            val1 = min_x + x_idx * (max_x - min_x) / width
            val2 = min_y + y_idx * (max_y - min_y) / height

            if fractal_type == 6: # Lyapunov
                pixels[y_idx, x_idx] = lyapunov_exponent(lyapunov_seq, val1, val2, maxiter, lyapunov_warmup)
            else: # Complex fractals
                c = val1 + val2 * 1j
                pixels[y_idx, x_idx] = fractal_smooth(c, maxiter, fractal_type, power_or_sequence, constant_c)

        if progress_callback and x_idx % max(1, width // 100) == 0:
            percent = int(100 * x_idx / width)
            progress_callback(percent)
    return pixels

def compute_blended_fractal(
    min_x, max_x, min_y, max_y, width, height,
    maxiter1, fractal_type1, power_or_sequence1, constant_c1,
    maxiter2, fractal_type2, power_or_sequence2, constant_c2,
    lyapunov_seq1="AB", lyapunov_seq2="AB", lyapunov_warmup=100,
    progress_callback=None
):
    """
    Compute two fractal arrays with different parameters for blending.
    Returns: pixels1, pixels2
    """
    pixels1 = np.zeros((height, width), dtype=np.float64)
    pixels2 = np.zeros((height, width), dtype=np.float64)

    for x_idx in range(width):
        for y_idx in range(height):
            val1 = min_x + x_idx * (max_x - min_x) / width
            val2 = min_y + y_idx * (max_y - min_y) / height

            if fractal_type1 == 6:
                pixels1[y_idx, x_idx] = lyapunov_exponent(lyapunov_seq1, val1, val2, maxiter1, lyapunov_warmup)
            else:
                c1 = val1 + val2 * 1j
                pixels1[y_idx, x_idx] = fractal_smooth(c1, maxiter1, fractal_type1, power_or_sequence1, constant_c1)

            if fractal_type2 == 6:
                pixels2[y_idx, x_idx] = lyapunov_exponent(lyapunov_seq2, val1, val2, maxiter2, lyapunov_warmup)
            else:
                c2 = val1 + val2 * 1j
                pixels2[y_idx, x_idx] = fractal_smooth(c2, maxiter2, fractal_type2, power_or_sequence2, constant_c2)

        if progress_callback and x_idx % max(1, width // 100) == 0:
            percent = int(100 * x_idx / width)
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