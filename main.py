from skimage.feature import peak_local_max
import diplib as dip
import numpy as np

def getbox(input_image, ii, sz, x_positions, y_positions):
    """ Returns the specified subregion of input_image along with the left and top coordinate.
    Args:
        input_image: The original 2D image to crop from.
        ii: The index of the point to copy.
        sz: The size of the subregion to copy.
        x_positions: X coordinates of the center of the subregions.
        y_positions: Y coordinates of the center of the subregions.
    """
    sz_x, sz_y = input_image.shape

    # Calculate the index of the center of the subregion
    szl = int(sz / 2 + 0.5)

    # Get coordinates (adjusted for zero-indexing in Python)
    x = int(x_positions[ii] + 0.5)
    y = int(y_positions[ii] + 0.5)

    # Ensure coordinates are within bounds
    if x < 0 or y < 0 or x >= sz_x or y >= sz_y:
        raise ValueError(f"Point {ii} out of bounds position {x}, {y} dataset size {sz_x}, {sz_y}")

    # Calculate left, right, top, bottom coordinates for the box
    l = max(x - szl, 0)
    r = min(l + sz, sz_x)
    t = max(y - szl, 0)
    b = min(t + sz, sz_y)

    # Return the input_image in roi, the left coordinates, and the top coordinates
    return input_image[t:b, l:r], l, t

def make_subregions(inner_image_pos_idx, box_size, input_image):
    """ Creates subregions of size box_size around the points in inner_image_pos_idx.
    Args:
        inner_image_pos_idx: A 2D array of indices of the points to crop.
        box_size: The size of the subregions to crop.
        input_image: The original 2D image to crop from.
    Returns:
        scanning_roi_stack: A 3D array of the cropped subregions.
        leftcoord: A 1D array of the left coordinates of the subregions.
        topcoord: A 1D array of the top coordinates of the subregions.
    """
    x_positions, y_positions = inner_image_pos_idx[0], inner_image_pos_idx[1]
    if input_image.dtype != np.float32:
        raise ValueError("Data must be comprised of single floats")
    if len(x_positions) == 0 or len(y_positions) == 0:
        raise ValueError("Coordinate array(s) is/are empty.")
    if x_positions.shape != y_positions.shape:
        raise ValueError("Size of X and Y coordinates must match.")
    if box_size <= 0:
        raise ValueError("Box size must be a positive integer.")
    if input_image.ndim != 2:
        raise ValueError("Data should be a 2D array.")

    # Convert box_size to an integer
    box_sz = int(box_size)

    # Get the number of points
    n_rois = len(x_positions)
    
    # Initialize output arrays
    scanning_roi_stack = np.zeros((n_rois, box_sz, box_sz), dtype=float)
    leftcoord = np.zeros(n_rois, dtype=float)
    topcoord = np.zeros(n_rois, dtype=float)

    # Create subregions around each point
    for ii in range(n_rois):
        scanning_roi_stack[ii], leftcoord[ii], topcoord[ii] = getbox(input_image, ii, box_sz, x_positions, y_positions)

    return scanning_roi_stack, leftcoord, topcoord

def create_separable_filter(one_d_kernel, origin):
    """ Creates a separable filter from a 1D kernel.
    Args:
        one_d_kernel: The 1D kernel to use.
        origin: The origin of the kernel.
    Returns:
        adjusted_kernel: The adjusted kernel.
    """
    # Get the length of the 1D kernel
    length = len(one_d_kernel)

    # Create a full 2D kernel from the 1D kernel
    full_kernel = np.outer(one_d_kernel, one_d_kernel)

    # Calculate padding based on the desired origin
    pad_before = origin - 1
    pad_after = length - origin

    # Apply padding to create an adjusted kernel
    adjusted_kernel = np.pad(full_kernel, ((pad_before, pad_after), (pad_before, pad_after)), mode='constant')

    return adjusted_kernel

def get_tentative_peaks(image, min_distance=1,):
    # Define filters
    h2 = 1/16
    h1 = 1/4
    h0 = 3/8
    g0 = np.array([h2, h1, h0, h1, h2])
    g1 = np.array([h2, 0, h1, 0, h0, 0, h1, 0, h2])
    k0 = create_separable_filter(g0, 3)
    dip_image = dip.Image(image)
    # Filter image
    v0 = dip.Convolution(dip_image, k0, method="best")
    k1 = create_separable_filter(g1, 5)
    v1 = dip.Convolution(v0, k1, method="best")
    filtered_image = np.asarray(v0 - v1)
    filtered_image = filtered_image - np.min(filtered_image)
    tentative_peak_coordinates = peak_local_max(filtered_image, min_distance=min_distance)
    return tentative_peak_coordinates
