import numpy as np

# --- Colormaps for Fractal Visualization ---
COLORMAPS = {
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
    'cividis': np.array([
        [0, 0, 128], [0, 0, 255], [0, 128, 255], [0, 255, 255],
        [128, 255, 128], [255, 255, 0], [255, 128, 0], [255, 0, 0],
        [128, 0, 0]
    ], dtype=np.uint8),
    'seismic': np.array([
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
def apply_colormap(arr, colormap_name='plasma'):
    arr = np.nan_to_num(arr, nan=0.0, neginf=0.0, posinf=0.0)
    min_val = arr.min()
    max_val = arr.max()
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val - min_val < 1e-10:
        normalized = np.zeros_like(arr)
    else:
        normalized = (arr - min_val) / (max_val - min_val)
    lut = COLORMAPS.get(colormap_name, COLORMAPS['plasma'])
    indices = (normalized * (lut.shape[0] - 1)).astype(np.uint8)
    return lut[indices]

def blend_colormaps(
    arr,
    colormap_name='plasma',
    colormap_2_name='viridis',
    blend_factor=0.5,
    blend_mode='linear',
    nonlinear_power=2.0,
    segment_point=0.5
):
    """
    Blend two colormaps using different modes.
    blend_mode: 'linear', 'nonlinear', 'segment'
    nonlinear_power: used if blend_mode == 'nonlinear'
    segment_point: used if blend_mode == 'segment', in [0,1]
    """
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val < 1e-10:
        normalized = np.zeros_like(arr)
    else:
        normalized = (arr - min_val) / (max_val - min_val)
    lut1 = COLORMAPS.get(colormap_name, COLORMAPS['plasma'])
    lut2 = COLORMAPS.get(colormap_2_name, COLORMAPS['viridis'])
    # Compute indices for each colormap separately
    indices1 = (normalized * (lut1.shape[0] - 1)).astype(np.uint8)
    indices2 = (normalized * (lut2.shape[0] - 1)).astype(np.uint8)
    colors1 = lut1[indices1]
    colors2 = lut2[indices2]

    if blend_mode == 'linear':
        blended = colors1 * (1 - blend_factor) + colors2 * blend_factor
    elif blend_mode == 'nonlinear':
        t = np.clip(blend_factor, 0, 1)
        t_nl = t ** nonlinear_power
        blended = colors1 * (1 - t_nl) + colors2 * t_nl
    elif blend_mode == 'segment':
        mask = normalized < segment_point
        blended = np.where(mask[..., None], colors1, colors2)
    else:
        blended = colors1 * (1 - blend_factor) + colors2 * blend_factor

    return blended.astype(np.uint8)


DEFAULT_LYAPUNOV_COLORS = {
    'positive_chaos_start': np.array([255, 255, 0], dtype=np.uint8),  # Yellow
    'positive_chaos_end': np.array([255, 0, 0], dtype=np.uint8),    # Red
    'negative_order_start': np.array([173, 216, 230], dtype=np.uint8), # Light Blue
    'negative_order_end': np.array([0, 0, 139], dtype=np.uint8),      # Dark Blue
    'divergent': np.array([0, 0, 0], dtype=np.uint8)               # Black
}

def apply_lyapunov_colormap(lyapunov_exponents, custom_colors=None):
    """
    Applies a specific colormap to Lyapunov exponents, distinguishing between
    chaotic (positive), ordered (negative), and divergent regions.

    The coloring uses gradients for positive and negative exponent ranges independently.
    Positive exponents are mapped from 'positive_chaos_start' to 'positive_chaos_end'.
    Negative exponents are mapped from 'negative_order_start' (for values closest to zero)
    to 'negative_order_end' (for values most negative).
    Non-finite values (like -np.inf from lyapunov_exponent function) are colored 'divergent'.

    Args:
        lyapunov_exponents (np.ndarray): Array of calculated Lyapunov exponents.
        custom_colors (dict, optional): A dictionary to override default colors.
            Expected keys: 'positive_chaos_start', 'positive_chaos_end',
                           'negative_order_start', 'negative_order_end', 'divergent'.

    Returns:
        np.ndarray: RGB image array (height, width, 3) of dtype np.uint8.
    """
    colors = DEFAULT_LYAPUNOV_COLORS.copy()
    if custom_colors:
        colors.update(custom_colors)

    height, width = lyapunov_exponents.shape
    # Initialize image to black. Points that don't fall into other categories will remain black.
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # --- Identify different regions based on exponent values ---
    is_divergent = ~np.isfinite(lyapunov_exponents) | (lyapunov_exponents == -np.inf)

    # Create a working copy for masking, ensuring divergent points don't interfere
    finite_exp_mask = np.copy(lyapunov_exponents)
    finite_exp_mask[is_divergent] = 0 # Neutral value for points already classified as divergent

    # Use a small epsilon for comparing floats to zero
    epsilon = 1e-9
    is_positive = finite_exp_mask > epsilon
    is_negative = finite_exp_mask < -epsilon
    # is_strictly_zero = np.abs(finite_exp_mask) <= epsilon (and not divergent)

    # --- Apply colors to regions ---

    # 1. Divergent points
    rgb_image[is_divergent] = colors['divergent']

    # 2. Positive exponents (chaotic)
    if np.any(is_positive):
        pos_exponents = lyapunov_exponents[is_positive] # Use original values for gradient
        min_pos = pos_exponents.min()
        max_pos = pos_exponents.max()

        if max_pos - min_pos < epsilon: # Avoid division by zero if all positives are effectively same
            norm_pos = np.zeros_like(pos_exponents) # Map all to 'positive_chaos_start'
        else:
            norm_pos = (pos_exponents - min_pos) / (max_pos - min_pos) # Normalize 0 to 1

        for i in range(3): # R, G, B
            rgb_image[is_positive, i] = (colors['positive_chaos_start'][i] * (1 - norm_pos) +
                                         colors['positive_chaos_end'][i] * norm_pos).astype(np.uint8)

    # 3. Negative exponents (ordered)
    if np.any(is_negative):
        neg_exponents = lyapunov_exponents[is_negative] # Use original values for gradient

        # Normalize based on absolute values:
        # Exponents closest to zero (e.g., -0.01, smallest absolute value) -> 'negative_order_start'
        # Exponents furthest from zero (e.g., -2.0, largest absolute value) -> 'negative_order_end'
        abs_neg_exponents = np.abs(neg_exponents)
        min_neg_abs = abs_neg_exponents.min()
        max_neg_abs = abs_neg_exponents.max()

        if max_neg_abs - min_neg_abs < epsilon: # If all negatives are effectively same magnitude
            norm_neg_abs = np.zeros_like(abs_neg_exponents) # Map all to 'negative_order_start'
        else:
            # norm = 0 for min_neg_abs (closest to zero), 1 for max_neg_abs (furthest from zero)
            norm_neg_abs = (abs_neg_exponents - min_neg_abs) / (max_neg_abs - min_neg_abs)

        for i in range(3): # R, G, B
            rgb_image[is_negative, i] = (colors['negative_order_start'][i] * (1 - norm_neg_abs) +
                                         colors['negative_order_end'][i] * norm_neg_abs).astype(np.uint8)

    # Points that are finite, not positive (<= epsilon), not negative (>= -epsilon), and not divergent
    # are effectively "zero" exponents. They will remain black as per initialization.
    # If a specific color for zero is desired, it can be added here:
    # is_zero_region = ~is_divergent & ~is_positive & ~is_negative
    # rgb_image[is_zero_region] = np.array([128, 128, 128], dtype=np.uint8) # e.g., Gray

    return rgb_image