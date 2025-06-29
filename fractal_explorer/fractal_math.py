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

@njit(cache=True, fastmath=True)
def lyapunov_exponent(sequence, a, b, maxiter, warmup=100):
    """
    Calculates the Lyapunov exponent for a given sequence and parameters a, b.
    Sequence should be a string like "ABAB".
    'A' corresponds to parameter 'a', 'B' to 'b'.
    Returns -np.inf for invalid or diverged points.
    """
    x = 0.5  # Starting value for x, must be in (0,1)
    lyap_sum = 0.0
    seq_len = len(sequence)

    # Warmup iterations to let the system settle
    for i in range(warmup):
        r = a if sequence[i % seq_len] == 'A' else b
        x = r * x * (1.0 - x)
        if x <= 0 or x >= 1:
            return -np.inf

    # Main calculation loop
    for i in range(maxiter):
        r = a if sequence[(i + warmup) % seq_len] == 'A' else b
        x = r * x * (1.0 - x)
        if x <= 0 or x >= 1:
            return -np.inf
        derivative = r * (1.0 - 2.0 * x)
        if derivative == 0:
            return -np.inf
        lyap_sum += np.log(abs(derivative))

    if maxiter == 0:
        return 0.0
    return lyap_sum / maxiter

import mpmath

# --- Fractal Calculation Functions (Numba JIT compiled for standard precision) ---
@njit(cache=True, fastmath=True)
def fractal_smooth_numba(c_real, c_imag, maxiter, fractal_type, power_real, power_imag, constant_c_real=0.0, constant_c_imag=0.0):
    """
    Compute the smooth iteration count for a given point in the complex plane.
    Optimized for Numba. Inputs are real and imaginary parts.
    Returns a float (iteration count, possibly fractional).
    """
    bailout_radius_sq = 4.0  # Bailout radius squared (2*2)

    if fractal_type == 1:  # Julia
        z_real = c_real
        z_imag = c_imag
    else:  # Mandelbrot and others
        z_real = 0.0
        z_imag = 0.0

    # Determine actual power (real or complex)
    # Numba doesn't directly support complex exponentiation with complex base AND complex power easily in all cases
    # For simplicity, we'll assume power is either real (power_imag == 0) or use a simplified complex power for p=2
    is_complex_power = (power_imag != 0.0)

    # Pre-calculate abs(power) for smoothing if power is complex, otherwise just use power_real
    # This is a simplification for Numba; true complex power smoothing is more involved.
    log_abs_power = np.log(power_real) if not is_complex_power and power_real > 0 else np.log(2.0) # default to log(2)

    for n in range(maxiter):
        z_abs_sq = z_real * z_real + z_imag * z_imag
        if z_abs_sq > bailout_radius_sq:
            # Smoothing: n + 1 - log(log|Z_n| / log(bailout_radius)) / log|power|
            # With bailout_radius = 2, log(bailout_radius) = log(2)
            # log|Z_n| = 0.5 * log(z_abs_sq)
            # So term is: n + 1 - log(0.5 * log(z_abs_sq) / log(2)) / log_abs_power
            # Simplified: n + 1 - log(log2|Z_n|) / log_abs_power
            abs_z = np.sqrt(z_abs_sq)
            if abs_z < 1e-12: abs_z = 1e-12 # Avoid log(0)
            log_abs_z = np.log(abs_z)
            if log_abs_z < 1e-18: log_abs_z = 1e-18 # Avoid log of very small number leading to issues

            # Simplified smoothing for common case, adjust if power is not ~2
            # The classical formula is n + 1 - log(log|z|/log(2)) / log(d) where d is degree (power)
            # Our 'power' can be complex. For real power p, it's log(p).
            # If power is p=2 (real), log_abs_power = log(2)
            # smoothing = np.log(log_abs_z / np.log(2.0)) / log_abs_power
            # A common simplification for general powers when keeping bailout=2:
            smoothing = np.log(np.log(abs_z) / np.log(2.0)) / np.log(max(2.0, abs(power_real if not is_complex_power else np.sqrt(power_real**2 + power_imag**2)))))

            # Clamp smoothing value to avoid negative results if abs_z is very close to bailout
            # This can happen if log(log(abs_z)/log(2)) itself is negative
            if smoothing < 0: smoothing = 0.0

            return n + 1.0 - smoothing


        # Apply fractal-specific transformation
        # For z_next = z**power + c_eff (where c_eff is c or constant_c)
        # If power is real (p_real, 0): z_next = (x+iy)**p_real + c_eff
        # If power is complex (p_real, p_imag): z_next = (x+iy)**(p_real+ip_imag) + c_eff (hard in numba)

        # Current iteration uses z_real, z_imag
        # Target for next iteration: next_z_real, next_z_imag

        # Store z before modification for certain fractal types
        prev_z_real = z_real
        prev_z_imag = z_imag

        if fractal_type == 2:  # Burning Ship
            z_real = abs(prev_z_real)
            z_imag = abs(prev_z_imag)
        elif fractal_type == 3:  # Tricorn / Mandelbar (type 7)
            # Use z_prev for conjugation: conj(z_prev)**power + c
            # (x - iy)**power. If power is real p:
            # For power=2: (x-iy)^2 = x^2 - y^2 - 2ixy
            # The power operation below will handle z_real, z_imag
            z_imag = -prev_z_imag # Conjugate z
            # z_real remains prev_z_real
        elif fractal_type == 4:  # Celtic Mandelbrot
            z_real = abs(prev_z_real)
            # z_imag remains prev_z_imag
        elif fractal_type == 5:  # Buffalo
            # Simplified: z_transform = complex(0, 2 * prev_z_real * prev_z_imag)
            # Then z = complex(abs(z_transform.real), abs(z_transform.imag))
            # z_real = abs(0.0) = 0.0
            # z_imag = abs(2 * prev_z_real * prev_z_imag)
            # This seems to be the interpretation of the original code's first step
            # (z**2 - conj(z)**2)/2 = ( (x+iy)**2 - (x-iy)**2 ) / 2
            # = (x^2-y^2+2ixy - (x^2-y^2-2ixy) ) / 2
            # = (4ixy)/2 = 2ixy. So real part is 0, imag part is 2xy.
            _2xy = 2.0 * prev_z_real * prev_z_imag
            z_real = abs(0.0)
            z_imag = abs(_2xy)
        elif fractal_type == 8:  # Perpendicular Burning Ship
            z_real = abs(prev_z_imag)
            z_imag = abs(prev_z_real)
        elif fractal_type == 9:  # Perpendicular Buffalo
            # Similar to Buffalo, but perpendicular transform first
            # z_transform = complex(abs(prev_z_imag), abs(prev_z_real))
            # Then this z_transform is raised to power.
            # So, z_real = abs(prev_z_imag), z_imag = abs(prev_z_real)
            # This is also what the original code implies for z = complex(abs(z.imag), abs(z.real))
            # then z = z**power + c
            temp_real = abs(prev_z_imag)
            z_imag = abs(prev_z_real)
            z_real = temp_real


        # Perform z**power calculation
        # This part is tricky for complex powers in Numba without external libraries.
        # We assume real power for now if power_imag is 0,
        # or handle specific complex powers like power=2 if needed.
        # For z^p where p is real:
        if not is_complex_power: # Real power
            r = np.sqrt(z_real * z_real + z_imag * z_imag)
            if r == 0:
                new_z_real = 0.0
                new_z_imag = 0.0
            else:
                theta = np.arctan2(z_imag, z_real)
                r_pow_p = r ** power_real
                new_z_real = r_pow_p * np.cos(power_real * theta)
                new_z_imag = r_pow_p * np.sin(power_real * theta)
        else: # Complex power z^(pr + i*pi) = exp((pr + i*pi) * log(z))
              # log(z) = log|z| + i*arg(z)
              # (pr + i*pi)(log|z| + i*arg(z)) = pr*log|z| - pi*arg(z) + i*(pi*log|z| + pr*arg(z))
              # So, z^power = exp(A) * (cos(B) + i*sin(B)) where A = pr*log|z| - pi*arg(z), B = pi*log|z| + pr*arg(z)
            abs_val_z = np.sqrt(z_real * z_real + z_imag * z_imag)
            if abs_val_z == 0: # Avoid log(0) and division by zero
                new_z_real = 0.0
                new_z_imag = 0.0
            else:
                log_abs_z = np.log(abs_val_z)
                arg_z = np.arctan2(z_imag, z_real)

                term_A = power_real * log_abs_z - power_imag * arg_z
                term_B = power_imag * log_abs_z + power_real * arg_z

                exp_A = np.exp(term_A)
                new_z_real = exp_A * np.cos(term_B)
                new_z_imag = exp_A * np.sin(term_B)

        # Add c (Mandelbrot-like) or constant_c (Julia-like)
        if fractal_type == 1: # Julia
            z_real = new_z_real + constant_c_real
            z_imag = new_z_imag + constant_c_imag
        else: # Mandelbrot and others
            z_real = new_z_real + c_real
            z_imag = new_z_imag + c_imag

    return float(maxiter)


# --- Fractal Calculation Functions (Python + mpmath for high precision) ---
def fractal_smooth_mp(c_mp, maxiter, fractal_type, power_mp, constant_c_mp=None):
    """
    Compute smooth iteration count using mpmath for high precision.
    c_mp, power_mp, constant_c_mp are mpmath complex numbers.
    """
    bailout_radius_sq = mpmath.mpf('4.0') # Bailout radius 2, squared

    if fractal_type == 1: # Julia
        z_mp = c_mp
        if constant_c_mp is None: # Should be provided for Julia
            raise ValueError("constant_c_mp must be provided for Julia fractal type")
    else: # Mandelbrot and others
        z_mp = mpmath.mpc(0,0)
        if constant_c_mp is None: # Not used for Mandelbrot like, but function expects it
            constant_c_mp = mpmath.mpc(0,0) # Dummy

    # Determine log_abs_power for smoothing
    # For real power p, it's log(p). For complex power, log|power|.
    if mpmath.im(power_mp) == 0:
        abs_power_val = mpmath.fabs(mpmath.re(power_mp))
    else:
        abs_power_val = mpmath.fabs(power_mp) # mpmath.fabs on complex gives magnitude

    log_abs_power = mpmath.log(abs_power_val) if abs_power_val > 0 else mpmath.log(2) # Default to log(2)

    for n in range(maxiter):
        # z_abs_sq = z_mp.real**2 + z_mp.imag**2 # Less precise for mpf
        z_abs_sq = mpmath.norm(z_mp)**2 # mpmath.norm is |z|, so |z|^2
        if z_abs_sq > bailout_radius_sq:
            abs_z = mpmath.sqrt(z_abs_sq)
            if abs_z < mpmath.mpf('1e-50'): abs_z = mpmath.mpf('1e-50') # Avoid log(0)

            # Smoothing: n + 1 - log(log|Z_n| / log(bailout_radius)) / log|power|
            # Bailout radius is 2. log(bailout_radius) = log(2)
            # log|Z_n| = log(abs_z)
            # term = log(log(abs_z) / log(2)) / log_abs_power
            try:
                #smoothing_val = mpmath.log(mpmath.log(abs_z) / mpmath.log(2)) / log_abs_power
                # A common simplification for general powers when keeping bailout=2:
                smoothing_val = mpmath.log(mpmath.log(abs_z) / mpmath.log(mpmath.mpf('2.0'))) / mpmath.log(mpmath.fmax(mpmath.mpf('2.0'), abs_power_val))

            except ValueError: # e.g. log of negative if abs_z is too small
                smoothing_val = mpmath.mpf('0.0')

            if smoothing_val < 0: smoothing_val = mpmath.mpf('0.0') # Clamp
            return float(n + 1 - smoothing_val)


        # Apply fractal-specific transformation
        prev_z_mp = z_mp

        if fractal_type == 2:  # Burning Ship
            z_mp = mpmath.mpc(mpmath.fabs(prev_z_mp.real), mpmath.fabs(prev_z_mp.imag))
        elif fractal_type == 3 or fractal_type == 7:  # Tricorn / Mandelbar
            z_mp = mpmath.conj(prev_z_mp)
        elif fractal_type == 4:  # Celtic Mandelbrot
            z_mp = mpmath.mpc(mpmath.fabs(prev_z_mp.real), prev_z_mp.imag)
        elif fractal_type == 5:  # Buffalo
            # Simplified: z_transform = complex(0, 2 * prev_z_real * prev_z_imag)
            _2xy = 2 * prev_z_mp.real * prev_z_mp.imag
            z_mp = mpmath.mpc(mpmath.fabs(0), mpmath.fabs(_2xy))
        elif fractal_type == 8:  # Perpendicular Burning Ship
            z_mp = mpmath.mpc(mpmath.fabs(prev_z_mp.imag), mpmath.fabs(prev_z_mp.real))
        elif fractal_type == 9:  # Perpendicular Buffalo
            z_mp = mpmath.mpc(mpmath.fabs(prev_z_mp.imag), mpmath.fabs(prev_z_mp.real))
        # For type 0 (Mandelbrot) and 1 (Julia if not transformed above), z_mp is already correct for z**power

        # Perform z**power
        z_mp = mpmath.power(z_mp, power_mp)

        # Add c (Mandelbrot-like) or constant_c (Julia-like)
        if fractal_type == 1: # Julia
            z_mp += constant_c_mp
        else: # Mandelbrot and others
            z_mp += c_mp

    return float(maxiter)


def _get_precision_settings(min_x, max_x, min_y, max_y, use_mpmath_precision=False, mpmath_dps=30):
    """Determines if mpmath should be used and sets its precision."""
    if use_mpmath_precision:
        mpmath.mp.dps = mpmath_dps # Set desired decimal places
        # Check if coordinate range requires mpmath even if not explicitly enabled,
        # though this function is now primarily for when use_mpmath_precision is True.
        # threshold = 1e-15 # Standard double precision limit
        # if abs(max_x - min_x) < threshold or abs(max_y - min_y) < threshold:
        #     return True # mpmath is needed
        return True

    # Fallback or default: do not use mpmath, reset dps to default if it was changed
    # mpmath.mp.dps = 15 # Default mpmath precision
    return False


def _core_compute_fractal(
    min_x_orig, max_x_orig, min_y_orig, max_y_orig,
    width, height, maxiter,
    fractal_type, power_or_sequence, constant_c_orig,
    lyapunov_seq="AB", lyapunov_warmup=100,
    use_mpmath_flag=False, mpmath_dps_setting=30,
    progress_callback=None
):
    """
    Core fractal computation logic. Can use standard floats (Numba) or mpmath.
    """
    pixels = np.zeros((height, width), dtype=np.float64)

    is_mpmath_active = _get_precision_settings(min_x_orig, max_x_orig, min_y_orig, max_y_orig, use_mpmath_flag, mpmath_dps_setting)

    if is_mpmath_active:
        min_x = mpmath.mpf(str(min_x_orig))
        max_x = mpmath.mpf(str(max_x_orig))
        min_y = mpmath.mpf(str(min_y_orig))
        max_y = mpmath.mpf(str(max_y_orig))

        if isinstance(power_or_sequence, (complex, float, int)):
            power_mp = mpmath.mpc(power_or_sequence)
        else: # Should be Lyapunov sequence string, not used with fractal_smooth_mp
            power_mp = None

        constant_c_mp = mpmath.mpc(constant_c_orig) if constant_c_orig is not None else None
    else:
        # Prepare inputs for Numba version
        if isinstance(power_or_sequence, complex):
            power_real_nb = power_or_sequence.real
            power_imag_nb = power_or_sequence.imag
        elif isinstance(power_or_sequence, (float, int)):
            power_real_nb = float(power_or_sequence)
            power_imag_nb = 0.0
        else: # Lyapunov sequence, not for fractal_smooth_numba
            power_real_nb = 2.0 # Default for non-Lyapunov if type mismatch
            power_imag_nb = 0.0

        if constant_c_orig is not None:
            constant_c_real_nb = constant_c_orig.real
            constant_c_imag_nb = constant_c_orig.imag
        else:
            constant_c_real_nb = 0.0
            constant_c_imag_nb = 0.0

    for x_idx in range(width):
        for y_idx in range(height):
            if is_mpmath_active:
                # Linear interpolation with mpmath
                # Equivalent to: val1 = min_x + mpmath.mpf(x_idx) * (max_x - min_x) / mpmath.mpf(width)
                # And similar for val2
                # To avoid issues with width/height=0 (though UI should prevent)
                w_mpf = mpmath.mpf(width)
                h_mpf = mpmath.mpf(height)

                if w_mpf == 0: val1 = min_x
                else: val1 = min_x + (mpmath.mpf(x_idx) / w_mpf) * (max_x - min_x)

                if h_mpf == 0: val2 = min_y
                else: val2 = min_y + (mpmath.mpf(y_idx) / h_mpf) * (max_y - min_y)

            else: # Standard float precision
                val1 = min_x_orig + x_idx * (max_x_orig - min_x_orig) / width
                val2 = min_y_orig + y_idx * (max_y_orig - min_y_orig) / height

            if fractal_type == 6:  # Lyapunov
                # Lyapunov currently doesn't use mpmath path, uses original Numba version
                pixels[y_idx, x_idx] = lyapunov_exponent(
                    power_or_sequence, # Here power_or_sequence is lyapunov_seq
                    val1, val2, maxiter, lyapunov_warmup
                )
            else: # Complex fractals
                if is_mpmath_active:
                    c_mp_current = mpmath.mpc(val1, val2)
                    pixels[y_idx, x_idx] = fractal_smooth_mp(
                        c_mp_current, maxiter, fractal_type, power_mp, constant_c_mp
                    )
                else:
                    pixels[y_idx, x_idx] = fractal_smooth_numba(
                        val1, val2, maxiter, fractal_type,
                        power_real_nb, power_imag_nb,
                        constant_c_real_nb, constant_c_imag_nb
                    )

        if progress_callback and x_idx % max(1, width // 100) == 0:
            percent = int(100 * x_idx / width)
            progress_callback(percent)

    return pixels


def compute_fractal(
    min_x, max_x, min_y, max_y, width, height,
    maxiter, fractal_type, power_or_sequence, constant_c=0j,
    lyapunov_seq="AB", lyapunov_warmup=100,
    use_mpmath_precision=False, mpmath_dps=30, # New precision params
    progress_callback=None
):
    """
    Compute a single fractal array.
    For Lyapunov, power_or_sequence is the sequence string.
    For complex fractals, power_or_sequence is the exponent.
    """
    # If Lyapunov, power_or_sequence is actually the sequence string
    effective_power_or_seq = lyapunov_seq if fractal_type == 6 else power_or_sequence

    return _core_compute_fractal(
        min_x, max_x, min_y, max_y, width, height, maxiter,
        fractal_type, effective_power_or_seq, constant_c,
        lyapunov_seq, lyapunov_warmup, # These are passed through but lyapunov_seq is duplicated if fractal_type=6
        use_mpmath_precision, mpmath_dps,
        progress_callback
    )


def compute_blended_fractal(
    min_x, max_x, min_y, max_y, width, height,
    maxiter1, fractal_type1, power_or_sequence1, constant_c1,
    maxiter2, fractal_type2, power_or_sequence2, constant_c2,
    lyapunov_seq1="AB", lyapunov_seq2="AB", lyapunov_warmup=100,
    use_mpmath_precision=False, mpmath_dps=30, # New precision params
    progress_callback=None
):
    """
    Compute two fractal arrays with different parameters for blending.
    Returns: pixels1, pixels2 (both 2D numpy arrays)
    """

    def combined_progress_callback(stage_factor, stage_offset):
        if not progress_callback:
            return None
        def callback(percent):
            total_percent = int(stage_offset + percent * stage_factor)
            progress_callback(total_percent)
        return callback

    effective_power_or_seq1 = lyapunov_seq1 if fractal_type1 == 6 else power_or_sequence1
    pixels1 = _core_compute_fractal(
        min_x, max_x, min_y, max_y, width, height, maxiter1,
        fractal_type1, effective_power_or_seq1, constant_c1,
        lyapunov_seq1, lyapunov_warmup,
        use_mpmath_precision, mpmath_dps,
        combined_progress_callback(0.5, 0) # Scale progress to 0-50%
    )

    effective_power_or_seq2 = lyapunov_seq2 if fractal_type2 == 6 else power_or_sequence2
    pixels2 = _core_compute_fractal(
        min_x, max_x, min_y, max_y, width, height, maxiter2,
        fractal_type2, effective_power_or_seq2, constant_c2,
        lyapunov_seq2, lyapunov_warmup,
        use_mpmath_precision, mpmath_dps,
        combined_progress_callback(0.5, 50) # Scale progress to 50-100%
    )

    if progress_callback: # Ensure 100% is emitted at the very end
        progress_callback(100)

    return pixels1, pixels2


def blend_fractals_mask(pixels1, pixels2, blend_factor=0.5):
    """
    Weighted blend: output = (1-blend_factor)*pixels1 + blend_factor*pixels2
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