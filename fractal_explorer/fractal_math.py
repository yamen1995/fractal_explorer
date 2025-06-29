import numpy as np
from numba import njit
from typing import Tuple, List, Optional, Callable, Union

# --- Constants and Presets ---
JULIA_PRESETS: List[Tuple[str, Optional[Tuple[float, float]]]] = [
    ("Custom", None),
    ("-0.7 + 0.27015i (Classic)", (-0.7, 0.27015)),
    ("0.355 + 0.355i (Spiral)", (0.355, 0.355)),
    ("-0.4 + 0.6i (Nebula)", (-0.4, 0.6)),
    ("0.285 + 0.01i (Swirls)", (0.285, 0.01)),
    ("-0.70176 - 0.3842i (Snowflake)", (-0.70176, -0.3842)),
]

# --- Fractal Calculation Functions ---

@njit(cache=True, fastmath=True)
def lyapunov_exponent(sequence: str, a: float, b: float, maxiter: int, warmup: int = 100) -> float:
    """
    Calculates the Lyapunov exponent for a given sequence and parameters a, b.

    The Lyapunov exponent is a measure of the rate of separation of infinitesimally
    close trajectories in a dynamical system. Positive values typically indicate
    chaotic behavior, while negative values suggest stability.

    Args:
        sequence: A string composed of 'A' and 'B' (e.g., "ABAB").
                  'A' corresponds to parameter 'a', 'B' to 'b' in the logistic map iteration.
        a: Parameter 'a' for the logistic map x_next = r * x * (1 - x).
        b: Parameter 'b' for the logistic map x_next = r * x * (1 - x).
        maxiter: The number of iterations to compute the sum of logarithms of derivatives.
        warmup: The number of initial iterations to discard, allowing the system
                to settle into an attractor.

    Returns:
        The calculated Lyapunov exponent. Returns -np.inf if the trajectory
        diverges (x <= 0 or x >= 1) or if the derivative becomes zero,
        as the logarithm would be undefined. Returns 0.0 if maxiter is 0.
    """
    x: float = 0.5  # Starting value for x, must be in (0,1) for logistic map
    lyap_sum: float = 0.0
    seq_len: int = len(sequence)

    if seq_len == 0: # Avoid division by zero or empty sequence issues
        return -np.inf

    # Warmup iterations
    for i in range(warmup):
        r: float = a if sequence[i % seq_len] == 'A' else b
        x = r * x * (1.0 - x)
        if not (0 < x < 1):  # Check if x is outside the (0,1) interval
            return -np.inf

    # Main calculation loop
    for i in range(maxiter):
        r = a if sequence[(i + warmup) % seq_len] == 'A' else b
        x = r * x * (1.0 - x)
        if not (0 < x < 1): # Check if x is outside the (0,1) interval
            return -np.inf

        derivative: float = r * (1.0 - 2.0 * x)
        if abs(derivative) < 1e-12: # Avoid log(0)
            return -np.inf
        lyap_sum += np.log(abs(derivative))

    if maxiter == 0:
        return 0.0
    return lyap_sum / maxiter

@njit(cache=True, fastmath=True)
def fractal_smooth(
    c: complex,
    maxiter: int,
    fractal_type: int,
    power: Union[float, complex],
    constant_c: complex = 0j
) -> float:
    """
    Computes the "smooth" iteration count for a point in a fractal set.

    This function calculates how many iterations it takes for the magnitude of z
    to exceed a bailout value (typically 2). The "smooth" part refers to an
    adjustment that provides a fractional value, allowing for smoother color
    gradients in fractal visualizations.

    Args:
        c: The complex number representing the point in the complex plane.
           For Mandelbrot-like fractals, this is the 'c' in z = z^power + c.
           For Julia sets, this is the starting 'z' value.
        maxiter: The maximum number of iterations to perform.
        fractal_type: An integer identifying the type of fractal to compute.
            0: Mandelbrot
            1: Julia
            2: Burning Ship
            3: Tricorn
            4: Celtic Mandelbrot
            5: Buffalo
            7: Mandelbar (inverted Mandelbrot)
            8: Perpendicular Burning Ship
            9: Perpendicular Buffalo
        power: The exponent used in the fractal formula (e.g., 2 for the classic Mandelbrot).
               Can be a float or a complex number.
        constant_c: The constant 'c' used for Julia sets (z = z^power + constant_c).
                    Ignored for Mandelbrot-like fractals.

    Returns:
        A float representing the (potentially fractional) iteration count.
        If the point does not escape within `maxiter`, `maxiter` is returned.
    """
    if fractal_type == 1:  # Julia
        z: complex = c
    else:  # Mandelbrot, Burning Ship, Tricorn, etc.
        z: complex = 0j

    for n in range(maxiter):
        abs_z_sq: float = z.real * z.real + z.imag * z.imag # More efficient than abs(z)
        if abs_z_sq > 4: # Bailout condition (equivalent to abs(z) > 2)
            # Smooth iteration count formula
            # nu = n + 1 - log(log(|z|))/log(2)
            # Using log2(|z|) = log(|z|) / log(2)
            # And log(log2(|z|)) = log(log(|z|)/log(2))
            # Simplified: n + 1 - log(log2(|z|)) / log(degree)
            # For degree=2, log(degree) i.e. log(2) can be used.
            # Here, power can be non-integer or complex, so a more general form is used.
            # A common simplification is n + 1 - log(log(|z|)/log(bailout_radius)) / log(abs(power) or 2)
            # The current one is: n + 1 - log(log2(|z|))
            absz: float = np.sqrt(abs_z_sq)
            # Add small epsilon to prevent log of zero or very small numbers
            log_absz: float = np.log(absz + 1e-12)
            log2_val: float = log_absz / np.log(2.0) # log_2(|z|)

            # Further adjustment for smoothness, can be refined
            # This seems to be a common smooth coloring formula variant
            return float(n) + 1.0 - np.log(log2_val + 1e-12) / np.log(abs(power) if isinstance(power, (float, int)) and power != 0 else 2.0)


        # Fractal type specific calculations
        if fractal_type == 0:  # Mandelbrot
            z = z**power + c
        elif fractal_type == 1:  # Julia
            z = z**power + constant_c
        elif fractal_type == 2:  # Burning Ship (z = (|Re(z)| + i|Im(z)|)^power + c)
            z = complex(abs(z.real), abs(z.imag))**power + c
        elif fractal_type == 3:  # Tricorn (z = conj(z)^power + c)
            z = np.conj(z)**power + c
        elif fractal_type == 4:  # Celtic Mandelbrot (z = (|Re(z)| - Im(z)^2)^power + c - this is a variant)
                                 # Original seems to be z = (abs(real(z)) - real(imag(z)))^2 + c
                                 # The code implements z = (abs(real(z)) + i * imag(z))^power + c
            z = complex(abs(z.real), z.imag)**power + c
        elif fractal_type == 5:  # Buffalo (variant of Burning Ship)
                                 # Original definition can vary. The code implements:
                                 # temp_z = z^2 - conj(z)^2 / 2  (This is not what was implemented)
                                 # temp_z = (z*z - np.conj(z)*np.conj(z))/2  -- This is not it
                                 # The code had: z = (complex(z.real, z.imag)**2 - complex(z.real, -z.imag)**2) / 2
                                 # This simplifies to: z = ( (z.real + 1j*z.imag)**2 - (z.real - 1j*z.imag)**2 ) / 2
                                 # z = ( (z.r^2 - z.i^2 + 2*z.r*z.i*1j) - (z.r^2 - z.i^2 - 2*z.r*z.i*1j) ) / 2
                                 # z = ( 4*z.r*z.i*1j ) / 2 = 2*z.r*z.i*1j
                                 # Then: z = complex(abs(z.real), abs(z.imag))
                                 # And finally: z = z**power + c
                                 # Let's use the implemented logic directly for now.
            # Original logic in code:
            # z_sq = z * z
            # z_conj_sq = np.conj(z) * np.conj(z)
            # z_intermediate = (z_sq - z_conj_sq) / 2.0 # This is (z.real * z.imag * 2j)
            # z = complex(abs(z_intermediate.real), abs(z_intermediate.imag)) # abs(0) + abs(2*z.real*z.imag) * 1j
            # z = z**power + c
            # The one from the code:
            # z = (complex(z.real, z.imag)**2 - complex(z.real, -z.imag)**2) / 2
            # z = complex(abs(z.real), abs(z.imag))
            # z = z**power + c
            # Let's keep the existing logic as it was likely intended for a specific visual
            z_temp = z*z - (z.real - 1j*z.imag)**2 # (z.real + 1j*z.imag)**2 - (z.real - 1j*z.imag)**2
            z = z_temp / 2.0
            z = complex(abs(z.real), abs(z.imag))
            z = z**power + c
        elif fractal_type == 7:  # Mandelbar (z = conj(z)^power + c, but c is inverted on x-axis)
                                 # The code implies it's just like Tricorn.
                                 # For Mandelbar, often c.real is flipped.
                                 # Here, it is identical to Tricorn based on formula.
            z = np.conj(z)**power + c
        elif fractal_type == 8:  # Perpendicular Burning Ship (z = (|Im(z)| + i|Re(z)|)^power + c)
            z = complex(abs(z.imag), abs(z.real))**power + c
        elif fractal_type == 9:  # Perpendicular Buffalo
            # Similar to Buffalo, but with swapped real/imag for abs
            z_temp = z*z - (z.real - 1j*z.imag)**2
            z = z_temp / 2.0
            z = complex(abs(z.imag), abs(z.real)) # Swapped here
            z = z**power + c
        # Add more fractal types here as needed
        else: # Default or unknown fractal_type, fallback to Mandelbrot
            z = z**power + c


    return float(maxiter)

def _get_coordinate_precision_type(
    min_coord: float,
    max_coord: float
) -> type:
    """
    Determines the NumPy float type based on the range of coordinates.
    This is a simplified version, preferring float64 for general use.
    Using np.float128 can have compatibility and performance implications.

    Args:
        min_coord: The minimum coordinate value in a range.
        max_coord: The maximum coordinate value in a range.

    Returns:
        np.float64, as np.float128 usage is generally discouraged
        unless specific high-precision libraries are part of the stack.
    """
    # threshold = 1e-14 # A smaller threshold for considering higher precision
    # if abs(max_coord - min_coord) < threshold:
    #     try:
    #         return np.float128 # If available and truly needed
    #     except AttributeError:
    #         return np.float64 # Fallback if float128 is not available
    return np.float64

def compute_fractal(
    min_x: float, max_x: float, min_y: float, max_y: float,
    width: int, height: int,
    maxiter: int,
    fractal_type: int,
    power_or_sequence: Union[float, complex, str], # Exponent for complex, sequence for Lyapunov
    constant_c: complex = 0j, # For Julia sets
    lyapunov_seq: str = "AB", # Specific for Lyapunov fractal type
    lyapunov_warmup: int = 100, # Specific for Lyapunov
    progress_callback: Optional[Callable[[int], None]] = None
) -> np.ndarray:
    """
    Computes a 2D array representing a fractal image.

    This function generates the raw data (iteration counts or Lyapunov exponents)
    for a fractal by iterating over each pixel in the specified region of the
    complex plane or parameter space.

    Args:
        min_x: The minimum value for the x-axis (real part or parameter 'a').
        max_x: The maximum value for the x-axis.
        min_y: The minimum value for the y-axis (imaginary part or parameter 'b').
        max_y: The maximum value for the y-axis.
        width: The width of the output image in pixels.
        height: The height of the output image in pixels.
        maxiter: Maximum iterations for the fractal calculation.
        fractal_type: Integer code for the type of fractal.
        power_or_sequence: For complex fractals, this is the exponent (float or complex).
                           For Lyapunov fractals, this is the sequence string (e.g., "AB").
        constant_c: The complex constant for Julia sets.
        lyapunov_seq: The sequence string for Lyapunov fractals (overrides power_or_sequence if fractal_type is Lyapunov).
        lyapunov_warmup: Warmup iterations for Lyapunov calculations.
        progress_callback: An optional function called with progress percentage (0-100).

    Returns:
        A 2D NumPy array (height, width) of floats, containing iteration counts
        or Lyapunov exponents.

    Raises:
        ValueError: If `power_or_sequence` is not of the expected type for the
                    selected `fractal_type`.
    """
    # Determine precision for coordinate calculations - typically float64 is sufficient.
    # Higher precision types like np.float128 are rare and have overhead.
    coord_dtype_x = _get_coordinate_precision_type(min_x, max_x)
    coord_dtype_y = _get_coordinate_precision_type(min_y, max_y)

    pixels = np.zeros((height, width), dtype=np.float64) # Output is always float64

    # Create coordinate arrays with chosen precision
    # Note: Numba functions (fractal_smooth, lyapunov_exponent) will internally
    # use standard float/complex types unless explicitly typed for higher precision,
    # which is complex with Numba. Input `c` to fractal_smooth will be Python complex.

    # Generate real and imaginary parts for each pixel
    # Using dtype for linspace can be beneficial if higher precision is maintained throughout.
    # However, the Numba functions will likely operate on standard doubles.
    real_parts = np.linspace(min_x, max_x, width, dtype=coord_dtype_x)
    imag_parts = np.linspace(min_y, max_y, height, dtype=coord_dtype_y)

    for y_idx in range(height):
        for x_idx in range(width):
            if fractal_type == 6:  # Lyapunov fractal
                if not isinstance(power_or_sequence, str):
                    # Fallback or raise error if sequence not provided correctly via power_or_sequence
                    # For now, use dedicated lyapunov_seq, but UI should ensure correct passing.
                    # Consider raising ValueError if power_or_sequence is not string for Lyapunov
                    current_lyapunov_seq = lyapunov_seq
                else:
                    current_lyapunov_seq = power_or_sequence

                # Parameters for Lyapunov are directly from the grid
                param_a = real_parts[x_idx]
                param_b = imag_parts[y_idx]
                pixels[y_idx, x_idx] = lyapunov_exponent(
                    current_lyapunov_seq, param_a, param_b, maxiter, lyapunov_warmup
                )
            else:  # Complex plane fractals (Mandelbrot, Julia, etc.)
                if not isinstance(power_or_sequence, (float, complex, int)):
                    raise ValueError(
                        f"Exponent 'power_or_sequence' must be float or complex for fractal_type {fractal_type}, "
                        f"got {type(power_or_sequence)}"
                    )

                c_val = complex(real_parts[x_idx], imag_parts[y_idx])
                pixels[y_idx, x_idx] = fractal_smooth(
                    c_val, maxiter, fractal_type, power_or_sequence, constant_c
                )

        if progress_callback and (x_idx + 1) % max(1, width // 100) == 0 : # x_idx is 0-indexed
            percent = int(100 * (y_idx * width + x_idx + 1) / (width * height))
            progress_callback(percent)

    if progress_callback: # Ensure 100% is reported at the end
        progress_callback(100)

    return pixels

def compute_blended_fractal(
    min_x: float, max_x: float, min_y: float, max_y: float,
    width: int, height: int,
    maxiter1: int, fractal_type1: int, power_or_sequence1: Union[float, complex, str], constant_c1: complex,
    maxiter2: int, fractal_type2: int, power_or_sequence2: Union[float, complex, str], constant_c2: complex,
    lyapunov_seq1: str = "AB", lyapunov_seq2: str = "AB", # Specific for Lyapunov types
    lyapunov_warmup: int = 100,
    progress_callback: Optional[Callable[[int], None]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes two different fractal arrays over the same coordinate range.

    This is used for effects where two fractals are generated and then combined
    (e.g., by weighted average or masking).

    Args:
        min_x, max_x, min_y, max_y: Coordinate range.
        width, height: Dimensions of the output arrays in pixels.
        maxiter1, fractal_type1, power_or_sequence1, constant_c1: Parameters for the first fractal.
        maxiter2, fractal_type2, power_or_sequence2, constant_c2: Parameters for the second fractal.
        lyapunov_seq1, lyapunov_seq2: Sequences for Lyapunov fractals if chosen.
        lyapunov_warmup: Warmup iterations for Lyapunov calculations.
        progress_callback: Optional function for progress updates.

    Returns:
        A tuple containing two 2D NumPy arrays (pixels1, pixels2),
        each representing a fractal.

    Raises:
        ValueError: If `power_or_sequence` arguments are not of the expected type
                    for their respective `fractal_type`.
    """
    coord_dtype_x = _get_coordinate_precision_type(min_x, max_x)
    coord_dtype_y = _get_coordinate_precision_type(min_y, max_y)

    pixels1 = np.zeros((height, width), dtype=np.float64)
    pixels2 = np.zeros((height, width), dtype=np.float64)

    real_parts = np.linspace(min_x, max_x, width, dtype=coord_dtype_x)
    imag_parts = np.linspace(min_y, max_y, height, dtype=coord_dtype_y)

    total_pixels = width * height
    processed_pixels = 0

    for y_idx in range(height):
        for x_idx in range(width):
            # Fractal 1
            if fractal_type1 == 6: # Lyapunov
                if not isinstance(power_or_sequence1, str):
                    # Fallback or raise error
                    current_lyap_seq1 = lyapunov_seq1
                else:
                    current_lyap_seq1 = power_or_sequence1
                param_a = real_parts[x_idx]
                param_b = imag_parts[y_idx]
                pixels1[y_idx, x_idx] = lyapunov_exponent(
                    current_lyap_seq1, param_a, param_b, maxiter1, lyapunov_warmup
                )
            else: # Complex plane
                if not isinstance(power_or_sequence1, (float, complex, int)):
                    raise ValueError(f"power_or_sequence1 type error for fractal_type1 {fractal_type1}")
                c1_val = complex(real_parts[x_idx], imag_parts[y_idx])
                pixels1[y_idx, x_idx] = fractal_smooth(
                    c1_val, maxiter1, fractal_type1, power_or_sequence1, constant_c1
                )

            # Fractal 2
            if fractal_type2 == 6: # Lyapunov
                if not isinstance(power_or_sequence2, str):
                    current_lyap_seq2 = lyapunov_seq2
                else:
                    current_lyap_seq2 = power_or_sequence2
                param_a = real_parts[x_idx]
                param_b = imag_parts[y_idx]
                pixels2[y_idx, x_idx] = lyapunov_exponent(
                    current_lyap_seq2, param_a, param_b, maxiter2, lyapunov_warmup
                )
            else: # Complex plane
                if not isinstance(power_or_sequence2, (float, complex, int)):
                    raise ValueError(f"power_or_sequence2 type error for fractal_type2 {fractal_type2}")
                c2_val = complex(real_parts[x_idx], imag_parts[y_idx])
                pixels2[y_idx, x_idx] = fractal_smooth(
                    c2_val, maxiter2, fractal_type2, power_or_sequence2, constant_c2
                )

            processed_pixels +=1

        if progress_callback: # Report progress per row
            percent = int(100 * processed_pixels / total_pixels)
            progress_callback(percent)

    if progress_callback: # Ensure 100% is reported
        progress_callback(100)

    return pixels1, pixels2

def blend_fractals_mask(
    pixels1: np.ndarray,
    pixels2: np.ndarray,
    blend_factor: float = 0.5
) -> np.ndarray:
    """
    Blends two fractal arrays using a weighted average (linear interpolation).

    The formula used is: `output = (1 - blend_factor) * pixels1 + blend_factor * pixels2`.

    Args:
        pixels1: The first fractal array (e.g., iteration counts).
        pixels2: The second fractal array.
        blend_factor: The blending factor. 0.0 means only pixels1 is used,
                      1.0 means only pixels2 is used. 0.5 is an equal mix.

    Returns:
        A new NumPy array representing the blended fractal data.

    Raises:
        ValueError: if `pixels1` and `pixels2` do not have the same shape.
    """
    if pixels1.shape != pixels2.shape:
        raise ValueError("Input arrays pixels1 and pixels2 must have the same shape for blending.")

    blend_factor = np.clip(blend_factor, 0.0, 1.0) # Ensure factor is in [0,1]
    return (1.0 - blend_factor) * pixels1 + blend_factor * pixels2

def blend_fractals_alternating(
    pixels1: np.ndarray,
    pixels2: np.ndarray,
    mode: str = 'checker'
) -> np.ndarray:
    """
    Blends two fractal arrays by alternating between them based on a pattern.

    Args:
        pixels1: The first fractal array.
        pixels2: The second fractal array.
        mode: The pattern for alternation. Supported modes:
              'checker': Checkerboard pattern.
              'vertical': Alternating vertical stripes.
              'horizontal': Alternating horizontal stripes.
              If an unknown mode is provided, it defaults to using only `pixels1`.

    Returns:
        A new NumPy array with values taken from `pixels1` or `pixels2`
        according to the specified pattern.

    Raises:
        ValueError: if `pixels1` and `pixels2` do not have the same shape.
    """
    if pixels1.shape != pixels2.shape:
        raise ValueError("Input arrays pixels1 and pixels2 must have the same shape for alternating blend.")

    h, w = pixels1.shape
    mask: np.ndarray

    if mode == 'checker':
        # Create a checkerboard pattern: True for pixels1, False for pixels2
        iy, ix = np.indices((h, w))
        mask = (ix + iy) % 2 == 0
    elif mode == 'vertical':
        mask = np.zeros((h, w), dtype=bool)
        mask[:, ::2] = True # Every other column from pixels1
    elif mode == 'horizontal':
        mask = np.zeros((h, w), dtype=bool)
        mask[::2, :] = True # Every other row from pixels1
    else: # Default or unknown mode
        # This effectively returns pixels1 if an unknown mode is given.
        # Consider logging a warning or raising an error for unknown modes.
        mask = np.ones((h, w), dtype=bool)

    return np.where(mask, pixels1, pixels2)