import numpy as np
from typing import Dict, Optional, Literal

# Type alias for a colormap (list of RGB colors)
ColorLut = np.ndarray # Should be shape (N, 3) and dtype uint8

# --- Colormaps for Fractal Visualization ---
# Dictionary mapping colormap names to their LUTs (Look-Up Tables)
COLORMAPS: Dict[str, ColorLut] = {
    'plasma': np.array([
        [13, 8, 135], [62, 4, 153], [98, 2, 167], [132, 7, 173],
        [161, 19, 173], [187, 36, 169], [211, 55, 161], [230, 75, 150],
        [244, 97, 137], [253, 122, 122], [254, 148, 108], [251, 174, 96],
        [243, 201, 88], [231, 228, 84], [217, 255, 89]
    ], dtype=np.uint8),
    'viridis': np.array([
        [68, 1, 84], [71, 15, 100], [72, 29, 116], [69, 42, 129],
        [64, 56, 139], [58, 70, 147], [52, 84, 152], [46, 98, 155],
        [42, 111, 154], [38, 125, 152], [35, 138, 147], [34, 152, 142],
        [37, 165, 134], [47, 178, 124], [72, 191, 112], [103, 203, 99],
        [142, 214, 86], [189, 225, 73], [238, 236, 65]
    ], dtype=np.uint8),
    'inferno': np.array([
        [0, 0, 4], [20, 11, 53], [53, 9, 97], [88, 15, 109],
        [125, 21, 110], [158, 32, 100], [188, 47, 84], [216, 68, 62],
        [236, 93, 38], [245, 123, 18], [250, 157, 9], [249, 192, 25],
        [239, 226, 54], [254, 255, 141]
    ], dtype=np.uint8),
    'magma': np.array([
        [0, 0, 4], [21, 11, 59], [54, 9, 108], [90, 15, 128],
        [126, 21, 139], [162, 29, 139], [195, 42, 131], [222, 59, 117],
        [243, 80, 99], [254, 103, 81], [255, 130, 65], [255, 159, 52],
        [247, 188, 44], [252, 219, 45], [252, 252, 210]
    ], dtype=np.uint8),
    'jet': np.array([
        [0, 0, 127], [0, 0, 255], [0, 127, 255], [0, 255, 255],
        [127, 255, 127], [255, 255, 0], [255, 127, 0], [255, 0, 0],
        [127, 0, 0]
    ], dtype=np.uint8),
    'fire': np.array([
    [  0,   0,   0], [100,  10,   0], [180,  40,   0], [255,  90,   0],
    [255, 150,   0], [255, 210,  60], [255, 255, 255]
    ], dtype=np.uint8),
    'cool': np.array([
        [0, 255, 255], [0, 204, 204], [0, 153, 153], [0, 102, 102],
        [0, 51, 51], [0, 0, 0]
    ], dtype=np.uint8),
    'hot': np.array([
        [0, 0, 0], [255, 0, 0], [255, 51, 0], [255, 102, 0],
        [255, 153, 0], [255, 204, 0], [255, 255, 0], [255, 255, 51],
        [255, 255, 102], [255, 255, 153], [255, 255, 204], [255, 255, 255]
    ], dtype=np.uint8),
    'spring': np.array([
        [255, 0, 255], [204, 0, 204], [153, 0, 153], [102, 0, 102],
        [51, 0, 51], [0, 0, 0]
    ], dtype=np.uint8),
    'ocean': np.array([
    [  0,   0,  50], [  0,   0, 100], [  0,  50, 150], [  0, 100, 200],
    [  0, 150, 255], [ 50, 200, 255], [150, 255, 255]
    ], dtype=np.uint8),
    'sunset': np.array([
    [ 50,   0,  60], [120,   0, 120], [200,  30, 120], [255, 100,  60],
    [255, 170,  40], [255, 230,  80]
    ], dtype=np.uint8),
    'twilight': np.array([
        [  0,   0,  50], [ 30,   0, 100], [ 60,   0, 150], [100,   0, 200],
        [150,   0, 255], [200, 100, 255], [255, 200, 255]
    ], dtype=np.uint8),
    'forest': np.array([
    [ 10,  30,  10], [ 30,  60,  20], [ 50,  90,  30], [ 80, 120,  40],
    [120, 150,  50], [180, 200,  80], [220, 240, 150]
    ], dtype=np.uint8),
    'autumn': np.array([
        [255, 0, 0], [255, 51, 0], [255, 102, 0], [255, 153, 0],
        [255, 204, 0], [255, 255, 0], [204, 255, 0], [153, 255, 0],
        [102, 255, 0], [51, 255, 0], [0, 255, 0]
    ], dtype=np.uint8),
    'winter': np.array([
        [0, 0, 255], [0, 51, 204], [0, 102, 153], [0, 153, 102],
        [0, 204, 51], [0, 255, 0], [51, 255, 51], [102, 255, 102],
        [153, 255, 153], [204, 255, 204], [255, 255, 255]
    ], dtype=np.uint8),
    'cividis': np.array([ # Note: Cividis is perceptually uniform, good for data viz
        [0, 0, 128], [0, 0, 255], [0, 128, 255], [0, 255, 255],
        [128, 255, 128], [255, 255, 0], [255, 128, 0], [255, 0, 0],
        [128, 0, 0]
    ], dtype=np.uint8), # This is a simplified version, actual Cividis has more points
    'seismic': np.array([ # Diverging colormap
        [0, 0, 255], [0, 128, 255], [0, 255, 255], [128, 255, 128],
        [255, 255, 0], [255, 128, 0], [255, 0, 0], [128, 0, 0]
    ], dtype=np.uint8),
    'electric': np.array([
    [  0,   0,   0], [ 50,   0, 100], [150,   0, 255], [255,  50, 255],
    [255, 150, 100], [255, 255,   0], [255, 255, 255]
    ], dtype=np.uint8),
    'candy': np.array([
    [255, 182, 193], [255, 192, 203], [221, 160, 221], [176, 224, 230],
    [135, 206, 235], [144, 238, 144], [255, 250, 205]
    ], dtype=np.uint8)
}

# --- Function to Apply Colormap ---
def apply_colormap(
    arr: np.ndarray,
    colormap_name: str = 'plasma'
) -> np.ndarray:
    """
    Applies a named colormap to a 2D array of scalar values.

    The input array is first normalized to the range [0, 1]. These normalized
    values are then used as indices into the specified colormap's Look-Up Table (LUT).
    Handles NaN/infinity values by converting them to 0. If the input array has
    a constant value (max_val - min_val is very small), it maps all values to the
    first color in the LUT.

    Args:
        arr: A 2D NumPy array of float values (e.g., iteration counts from fractal calculation).
        colormap_name: The name of the colormap to apply (must be a key in COLORMAPS).
                       Defaults to 'plasma'.

    Returns:
        A 3D NumPy array (height, width, 3) of uint8 RGB values, representing the colored image.
        Returns an array of zeros of the same shape if the input array is empty.
    """
    if arr.size == 0:
        return np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

    # Replace NaN, -inf, +inf with a numeric value (e.g., 0)
    # This prevents errors in min/max calculation and normalization.
    # The choice of 0.0 for nan/inf is arbitrary but common.
    processed_arr = np.nan_to_num(arr, nan=0.0, neginf=0.0, posinf=0.0)

    min_val = processed_arr.min()
    max_val = processed_arr.max()

    # Normalize the array to [0, 1]
    # If min_val and max_val are the same (or very close), all normalized values will be 0.
    # This robustly handles constant arrays.
    if not np.isfinite(min_val) or not np.isfinite(max_val) or (max_val - min_val) < 1e-10:
        # If range is zero or non-finite, map all to the first color (index 0)
        normalized_arr = np.zeros_like(processed_arr, dtype=float)
    else:
        normalized_arr = (processed_arr - min_val) / (max_val - min_val)

    lut: ColorLut = COLORMAPS.get(colormap_name, COLORMAPS['plasma']) # Default to 'plasma' if name not found

    # Scale normalized values to indices of the LUT
    # Ensure indices are within the valid range [0, lut.shape[0] - 1]
    indices = np.clip((normalized_arr * (lut.shape[0] - 1)), 0, lut.shape[0] - 1).astype(np.uint8)

    return lut[indices]

BlendMode = Literal['linear', 'nonlinear', 'segment']

def blend_colormaps(
    arr: np.ndarray,
    colormap_name: str = 'plasma',
    colormap_2_name: str = 'viridis',
    blend_factor: float = 0.5,
    blend_mode: BlendMode = 'linear',
    nonlinear_power: float = 2.0,
    segment_point: float = 0.5
) -> np.ndarray:
    """
    Applies two different colormaps to an array and blends the results.

    The input array is first normalized. Then, two RGB color arrays are generated
    using `colormap_name` and `colormap_2_name`. These two color arrays are then
    blended based on the specified `blend_mode` and `blend_factor`.

    Args:
        arr: 2D NumPy array of scalar values.
        colormap_name: Name of the first colormap.
        colormap_2_name: Name of the second colormap.
        blend_factor: Factor for blending. Interpretation depends on `blend_mode`.
                      For 'linear' and 'nonlinear', it's typically in [0, 1].
        blend_mode: Method for blending:
            'linear': Linear interpolation: `(1-f)*c1 + f*c2`.
            'nonlinear': Nonlinear interpolation using `blend_factor` raised to `nonlinear_power`.
            'segment': Uses `colors1` if `normalized_value < segment_point`, else `colors2`.
        nonlinear_power: Exponent for 'nonlinear' blending.
        segment_point: Threshold for 'segment' blending, normalized to [0, 1].

    Returns:
        A 3D NumPy array (height, width, 3) of uint8 RGB values,
        representing the blended colored image.
        Returns an array of zeros if the input array is empty.
    """
    if arr.size == 0:
        return np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

    processed_arr = np.nan_to_num(arr, nan=0.0, neginf=0.0, posinf=0.0)
    min_val = processed_arr.min()
    max_val = processed_arr.max()

    if not np.isfinite(min_val) or not np.isfinite(max_val) or (max_val - min_val) < 1e-10:
        normalized_arr = np.zeros_like(processed_arr, dtype=float)
    else:
        normalized_arr = (processed_arr - min_val) / (max_val - min_val)

    # Apply the two colormaps
    # Note: apply_colormap itself handles normalization and LUT indexing.
    # Here, we are effectively re-doing normalization to get colors1 and colors2
    # This is slightly redundant but ensures the blend logic below has normalized_arr.
    # A more optimized path might pass normalized_arr to apply_colormap or have a variant.
    # However, current apply_colormap is self-contained.

    colors1 = apply_colormap(processed_arr, colormap_name) # arr is already processed_arr
    colors2 = apply_colormap(processed_arr, colormap_2_name)

    blended_colors: np.ndarray

    if blend_mode == 'linear':
        factor = np.clip(blend_factor, 0.0, 1.0)
        blended_colors = colors1 * (1.0 - factor) + colors2 * factor
    elif blend_mode == 'nonlinear':
        t = np.clip(blend_factor, 0.0, 1.0)
        # Ensure nonlinear_power is positive to avoid issues with t=0
        safe_power = max(1e-6, nonlinear_power) # Avoid power of 0 or negative if t can be 0
        t_nl = t ** safe_power
        blended_colors = colors1 * (1.0 - t_nl) + colors2 * t_nl
    elif blend_mode == 'segment':
        # Ensure segment_point is within [0,1] for normalized_arr comparison
        safe_segment_point = np.clip(segment_point, 0.0, 1.0)
        # Mask needs to be broadcastable to colors1/colors2 shape (H, W, 3)
        # normalized_arr is (H, W), so add new axis for broadcasting.
        mask = normalized_arr[..., np.newaxis] < safe_segment_point
        blended_colors = np.where(mask, colors1, colors2)
    else: # Default to linear blend if mode is unknown
        factor = np.clip(blend_factor, 0.0, 1.0)
        blended_colors = colors1 * (1.0 - factor) + colors2 * factor
        # Consider logging a warning for unrecognized blend_mode

    return np.clip(blended_colors, 0, 255).astype(np.uint8)

# Default colors for Lyapunov exponent visualization
DEFAULT_LYAPUNOV_COLORS: Dict[str, np.ndarray] = {
    'positive_chaos_start': np.array([255, 255, 0], dtype=np.uint8),  # Yellow
    'positive_chaos_end': np.array([255, 0, 0], dtype=np.uint8),      # Red
    'negative_order_start': np.array([173, 216, 230], dtype=np.uint8),# Light Blue (order, close to 0)
    'negative_order_end': np.array([0, 0, 139], dtype=np.uint8),      # Dark Blue (order, very negative)
    'divergent': np.array([0, 0, 0], dtype=np.uint8),                 # Black (for -inf or non-finite)
    'zero_boundary': np.array([200, 200, 200], dtype=np.uint8)        # Gray (for values very close to zero)
}

LyapunovColorKey = Literal[
    'positive_chaos_start', 'positive_chaos_end',
    'negative_order_start', 'negative_order_end',
    'divergent', 'zero_boundary'
]
CustomLyapunovColors = Dict[LyapunovColorKey, np.ndarray]


def apply_lyapunov_colormap(
    lyapunov_exponents: np.ndarray,
    custom_colors: Optional[CustomLyapunovColors] = None
) -> np.ndarray:
    """
    Applies a specialized colormap to Lyapunov exponent data.

    This colormap distinguishes between:
    - Divergent regions (non-finite exponents).
    - Chaotic regions (positive exponents), colored with a gradient.
    - Ordered regions (negative exponents), colored with a separate gradient.
    - Regions with exponents very close to zero (boundary).

    Args:
        lyapunov_exponents: 2D NumPy array of Lyapunov exponents.
        custom_colors: Optional dictionary to override default colors.
                       Keys should match `LyapunovColorKey`.

    Returns:
        A 3D NumPy array (height, width, 3) of uint8 RGB values.
        Returns an array of zeros if the input array is empty.
    """
    if lyapunov_exponents.size == 0:
        return np.zeros((lyapunov_exponents.shape[0], lyapunov_exponents.shape[1], 3), dtype=np.uint8)

    colors = DEFAULT_LYAPUNOV_COLORS.copy()
    if custom_colors:
        colors.update(custom_colors) # type: ignore

    height, width = lyapunov_exponents.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Epsilon for floating point comparisons (e.g., distinguishing from zero)
    epsilon = 1e-9

    # 1. Handle divergent points (NaN, +/- Inf)
    # These are typically set to -np.inf by the lyapunov_exponent function on divergence.
    is_divergent = ~np.isfinite(lyapunov_exponents)
    rgb_image[is_divergent] = colors['divergent']

    # Create a working copy for finite values, setting divergent points to 0 for now
    # This avoids issues with min/max on arrays containing non-finite values.
    finite_exp = np.where(is_divergent, 0.0, lyapunov_exponents)

    # 2. Positive exponents (chaotic region)
    is_positive = finite_exp > epsilon
    if np.any(is_positive):
        pos_exponents_values = finite_exp[is_positive]
        min_pos = pos_exponents_values.min()
        max_pos = pos_exponents_values.max()

        if (max_pos - min_pos) < epsilon: # All positive values are virtually the same
            norm_pos = 0.5 # Assign a mid-range color
        else:
            norm_pos = (pos_exponents_values - min_pos) / (max_pos - min_pos)

        # Interpolate color: (1-t)*start + t*end
        color_values = (colors['positive_chaos_start'].astype(np.float32) * (1.0 - norm_pos)[:, np.newaxis] +
                        colors['positive_chaos_end'].astype(np.float32) * norm_pos[:, np.newaxis])
        rgb_image[is_positive] = np.clip(color_values, 0, 255).astype(np.uint8)

    # 3. Negative exponents (ordered region)
    is_negative = finite_exp < -epsilon
    if np.any(is_negative):
        neg_exponents_values = finite_exp[is_negative] # These are negative
        # We want to map values closer to zero to 'negative_order_start'
        # and more negative values to 'negative_order_end'.
        # So, normalize based on absolute values, but reverse the gradient direction.
        abs_neg_exponents = np.abs(neg_exponents_values)
        min_neg_abs = abs_neg_exponents.min() # Smallest magnitude (closest to zero)
        max_neg_abs = abs_neg_exponents.max() # Largest magnitude (most negative)

        if (max_neg_abs - min_neg_abs) < epsilon: # All negative values are virtually the same
            norm_neg_abs = 0.5 # Assign a mid-range color
        else:
            # Normalize so 0 corresponds to min_neg_abs, 1 to max_neg_abs
            norm_neg_abs = (abs_neg_exponents - min_neg_abs) / (max_neg_abs - min_neg_abs)

        # Interpolate color: (1-t)*start + t*end
        # Here, t=0 (min_neg_abs) should map to negative_order_start (e.g. light blue)
        # t=1 (max_neg_abs) should map to negative_order_end (e.g. dark blue)
        color_values = (colors['negative_order_start'].astype(np.float32) * (1.0 - norm_neg_abs)[:, np.newaxis] +
                        colors['negative_order_end'].astype(np.float32) * norm_neg_abs[:, np.newaxis])
        rgb_image[is_negative] = np.clip(color_values, 0, 255).astype(np.uint8)

    # 4. Near-zero exponents (boundary between order and chaos)
    # These are finite, not positive ( > epsilon), and not negative ( < -epsilon)
    is_zero_boundary = ~is_divergent & ~is_positive & ~is_negative
    rgb_image[is_zero_boundary] = colors['zero_boundary']

    return rgb_image