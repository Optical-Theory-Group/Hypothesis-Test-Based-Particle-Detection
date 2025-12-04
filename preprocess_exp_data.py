# Hypothesis-Test-Based-Particle-Detection
# -----------------------------------------
#
# This file is part of the project "Hypothesis-Test-Based-Particle-Detection".
# It implements preprocessing of experimental image files.
# Copyright (C) 2023-2025 [Kim, Neil H. and Foreman, Matthew R.]
# <matthew.foreman@ntu.edu.sg>
# Nanyang Technological University (NTU), Singapore.
#
# License: GNU General Public License v3.0
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this
# program. If not, see <https://www.gnu.org/licenses/>.


import os
import shutil
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from PIL import Image
import numpy as np
from math import ceil
import argparse
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import json
import cv2
from scipy import ndimage
from skimage import filters, morphology
from datetime import datetime


def rough_count_particles(tiff_path, ch=0.5, cw=0.5, min_component_size=20):
    """
    Count rough count of particle-like bright spots in a TIFF image using a simple
    thresholding and connected-component pipeline.
    
    The function performs the following steps:
    1. Load the TIFF image with Pillow and convert it to a NumPy array.
    2. If the image has 3 channels (RGB), convert to grayscale via per-pixel channel mean.
    3. Normalize grayscale intensities to [0, 1] based on 8-bit or 16-bit depth.
    4. Crop the central 50% region to avoid dark corners from a Gaussian beam profile.
    5. Denoise with a 5x5 Gaussian blur.
    6. Compute a global Otsu threshold and binarize.
    7. Remove small connected components (< 20 pixels) as noise.
    8. Label remaining connected components and return their count.

    Parameters
    ----------
    tiff_path : str | pathlib.Path
         Path to an 8-bit or 16-bit TIFF image file. RGB (3-channel) or single-channel
         images are supported.
    ch : float
         Fraction of image height to retain in the center crop (default 0.5).
    cw : float
         Fraction of image width to retain in the center crop (default 0.5).
    min_component_size : int
         Minimum size (in pixels) for connected components to be retained (default 20).
    Returns
    -------
    int
         The number of connected components (features) remaining after thresholding and
         size filtering, interpreted as a rough particle count.
    Raises
    ------
    ValueError
         If the image bit depth is not 8-bit (uint8) or 16-bit (uint16).
    FileNotFoundError
         If the provided path does not exist or cannot be opened.

    Notes
    -----
    - This is a "rough" heuristic count; it does not perform subpixel localization,
      deblending of overlapping particles, background modeling, or adaptive thresholding.
    - For densely packed or low SNR images, more advanced methods (e.g., LoG / DoG peak
      detection, wavelet filtering, background subtraction, or probabilistic modeling)
      may yield more accurate counts.

    """

    # Step 1: Open and convert to grayscale (average channels)
    with Image.open(tiff_path) as img:
        img = np.array(img)

    # Check if the TIFF file is RGB and convert to grayscale
    if img.ndim == 3 and img.shape[2] == 3:
        # Convert 16-bit RGB to grayscale
        grayscale = np.mean(img, axis=2)
    else:
        # Assume the image is already grayscale
        grayscale = img

    # Step 2: Normalize to [0, 1] based on bit depth
    if img.dtype == np.uint16:  # 16-bit image
        grayscale_norm = grayscale / 65535.0
    elif img.dtype == np.uint8:  # 8-bit image
        grayscale_norm = grayscale / 255.0
    else:
        raise ValueError("Unsupported image bit depth. Only 8-bit and 16-bit images are supported.")

    # crop to middle 50% of the image to avoid dark corners associated with Gaussian beam profile
    height, width = grayscale_norm.shape[:2]

    crop_height = int(height * ch)
    crop_width = int(width * cw)
    start_y = (height - crop_height) // 2
    start_x = (width - crop_width) // 2
    grayscale_norm = grayscale_norm[start_y:start_y + crop_height, start_x:start_x + crop_width]

    # Step 3: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(grayscale_norm, (5, 5), 0)

    # Step 4: Threshold (Otsu's method)
    thresh_val = filters.threshold_otsu(blurred)
    binary_mask = blurred > thresh_val

    # Step 5: Remove small objects (likely noise)
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_component_size)

    # Step 6: Label connected components
    _, num_features = ndimage.label(cleaned_mask)

    return num_features


def divide_images(input_file_path, output_dir, divide_dim, overlap=0, progress_callback=None,
                  crop_root_img_fraction=0.7):
    """
        Divide a (optionally centrally cropped) input image into overlapping sub-images (tiles)
        and save each tile as an individual TIFF file.
        The function:
          1. Loads the input image (converting RGB to grayscale).
          2. Optionally crops a centered region to mitigate dark corners (e.g., Gaussian beam profile).
          3. Computes a tiling grid with specified tile dimensions and pixel overlap.
          4. Iteratively extracts each tile and saves it to the output directory using a stable naming pattern.
          5. Optionally reports progress via a callback.

        Parameters
        ----------
        input_file_path : str or pathlib.Path
            Path to the source image file (e.g., TIFF, PNG, JPEG). Must be readable by Pillow.
        output_dir : str or pathlib.Path
            Directory where the generated tile images will be written. Created if it does not exist.
        divide_dim : tuple[int, int]
            (tile_width, tile_height) in pixels. Defines the size of each sub-image.
        overlap : int, default=0
            Number of pixels of overlap between adjacent tiles in both x and y directions.
            Must be less than the corresponding dimension in divide_dim to avoid infinite loops.
        progress_callback : callable, optional
            A function invoked after each tile is saved:
                progress_callback(processed_count: int, total_count: int)
            Useful for UI progress bars or logging.
        crop_root_img_fraction : float, default=0.7
            Fraction (0 < f <= 1) of the original image's width and height to retain, applied symmetrically
            about the center. Values outside (0, 1] trigger a warning and fallback to 0.7.
            Set to 1 to disable cropping.

        Behavior & Computation
        ----------------------
        Cropping:
            If crop_root_img_fraction < 1, a centered rectangle of size
            (floor(H * f), floor(W * f)) is extracted before tiling.
        Tiling Grid:
            Number of tiles horizontally = ceil((cropped_width  - overlap) / (tile_width  - overlap))
            Number of tiles vertically   = ceil((cropped_height - overlap) / (tile_height - overlap))
            Each tile's top-left corner advances by (tile_dim - overlap).
        File Naming:
            Tiles saved as: div_{row_index:02}_{col_index:02}.tiff
            Example: div_00_03.tiff for row 0, column 3.
        Grayscale Conversion:
            If the source image is RGB, it is converted to single-channel 'L' mode before processing.

        Returns
        -------
        None
            Writes tile images to disk; optionally emits progress via callback.

        Notes
        -----
        - Large overlap relative to tile size reduces stride and increases tile count.
        - Ensure (tile_width - overlap) > 0 and (tile_height - overlap) > 0.
        - For very large images / small strides, consider memory and filesystem limits.
        - Converting to grayscale may not be desirable if color information is needed; adapt as necessary.

    """

    # Make output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load image and convert to grayscale if needed
    with Image.open(input_file_path) as img:
        if img.mode == 'RGB':
            img = img.convert('L')
        img_array = np.array(img)
    img_height, img_width = img_array.shape

    # crop down to central image region according to the crop_root_img_fraction to avoid dark corners associated
    # with Gaussian beam profile
    if crop_root_img_fraction > 1 or crop_root_img_fraction < 0:
        print("Warning: Invalid crop fraction value. Using default of 0.7.")
        crop_root_img_fraction = 0.7

    if crop_root_img_fraction < 1:
        crop_pixel_height = int(img_height * crop_root_img_fraction)
        crop_pixel_width = int(img_width * crop_root_img_fraction)
        crop_top_bottom_pixels = (img_height - crop_pixel_height) // 2
        crop_left_right_pixels = (img_width - crop_pixel_width) // 2

        img_array = img_array[crop_top_bottom_pixels:crop_top_bottom_pixels + crop_pixel_height,
                              crop_left_right_pixels:crop_left_right_pixels + crop_pixel_width]
        img_height, img_width = img_array.shape

    # Save (optionally) cropped grayscale image
    if crop_root_img_fraction < 1:
        try:
            original_dir = os.path.dirname(input_file_path)
            cropped_dir = os.path.join(original_dir, 'cropped')
            os.makedirs(cropped_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(input_file_path))[0]
            cropped_path = os.path.join(cropped_dir, f"{base_name}_cropped.tiff")

            # Ensure grayscale (uint8/uint16 acceptable)
            if img_array.ndim == 3:
                img_array_to_save = np.mean(img_array, axis=2).astype(img_array.dtype)
            else:
                img_array_to_save = img_array
            Image.fromarray(img_array_to_save).save(cropped_path)
        except Exception as e:
            print(f"Warning: Failed to save cropped image ({e})")

    # Calculate number of sub-images in each dimension
    num_sub_images_x = ceil((img_width - overlap) / (divide_dim[0] - overlap))
    num_sub_images_y = ceil((img_height - overlap) / (divide_dim[1] - overlap))
    total_sub_images = num_sub_images_x * num_sub_images_y
    processed_sub_images = 0

    # Iterate and save sub-images
    for i in range(num_sub_images_y):
        for j in range(num_sub_images_x):
            # Calculate the coordinates of the sub-image
            start_x = j * (divide_dim[0] - overlap)
            start_y = i * (divide_dim[1] - overlap)
            end_x = start_x + divide_dim[0]
            end_y = start_y + divide_dim[1]
            sub_image = img_array[start_y:end_y, start_x:end_x]

            # Check if the extracted sub-image matches the specified dimensions
            if sub_image.shape[0] != divide_dim[1] or sub_image.shape[1] != divide_dim[0]:
                # Skip this sub-image if it's not the correct size (edge case)
                continue

            # Generate filename and extract the sub-image
            sub_image_filename = f'div_{i:02}_{j:02}.tiff'
            sub_image_path = os.path.join(output_dir, sub_image_filename)
            sub_image_pil = Image.fromarray(sub_image)

            # Try to save with full filename, fall back to shortened path if filename too long
            try:
                sub_image_pil.save(sub_image_path)
            except OSError as e:
                # If filename is too long, try a shortened output directory name
                if "too long" in str(e).lower() or "filename" in str(e).lower():
                    # This shouldn't happen with div_XX_XX.tiff filenames, but handle it anyway
                    print(f"Warning: Filename too long when saving {sub_image_filename}. \
                          This may indicate a path issue.")
                    raise
                else:
                    raise

            # Update progress
            processed_sub_images += 1
            if progress_callback:
                progress_callback(processed_sub_images, total_sub_images)


def gaussian_2d(x, y, x0, y0, sigma, amplitude, offset):
    """
    2D Gaussian function.

    Parameters
    ----------
        x, y: Coordinates where the Gaussian is evaluated.
        x0, y0: Center of the Gaussian.
        sigma: Standard deviation (width) of the Gaussian.
        amplitude: Peak amplitude of the Gaussian.
        offset: Baseline offset.

    Returns:
        float: The value of the 2D Gaussian function at the given coordinates.

    """
    return offset + amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma ** 2) + ((y - y0) ** 2) / (2 * sigma ** 2)))


def fit_gaussian_2d(image):
    """
    Fits a symmetric 2D Gaussian model to a single-channel image patch representing a particle.

    The function:
    1. Builds meshgrid coordinates for the image.
    2. Applies a Gaussian low-pass filter (sigma=3) to suppress high-frequency noise.
    3. Locates the peak in the filtered image to seed the optimizer.
    4. Performs heuristic signal/contrast checks to skip low-quality or empty patches.
    5. Uses scipy.optimize.curve_fit to fit a 2D Gaussian wrapper:
        G(x, y) = A * exp(-((x - x0)^2 + (y - y0)^2)/(2*sigma^2)) + offset.
    6. Validates fitted parameters (positive amplitude, reasonable sigma).

    Parameters
    ----------
    image : numpy.ndarray
        2D array (H x W) containing intensity values of the candidate particle region.
        Must be non-empty; assumed to be pre-cropped around a potential particle.

    Returns
    -------
    numpy.ndarray or None
        On success: array-like of length 5 -> (x0, y0, sigma, amplitude, offset)
            x0, y0 : float
                Subpixel coordinates of the Gaussian center in (x, y) order.
            sigma : float
                Shared (isotropic) standard deviation of the Gaussian.
            amplitude : float
                Peak height above the offset (not necessarily max(image)).
            offset : float
                Background intensity level.
        None if:
            - Contrast heuristics fail (no clear particle-like signal).
            - Optimization fails or yields non-physical parameters (sigma <= 0, sigma > 100, amplitude <= 0).

    Notes
    -----
    - Filtering improves robustness of the initial peak localization.
    - The contrast checks use both dynamic range and deviation from mean to reject flat regions.
    - The maxfev parameter (1000) guards against premature termination in tougher fits.
    - Assumes an isotropic Gaussian; anisotropy would require extending the model.

    """
    # Create meshgrid coordinates
    y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]]

    # Apply a low-pass filter to the image
    filtered_image = gaussian_filter(image, sigma=3)

    # Find the coordinates of the maximum value in the filtered image
    y0, x0 = np.unravel_index(np.argmax(filtered_image), filtered_image.shape)

    # Check if there's a meaningful signal in the image
    signal_max = np.max(filtered_image)
    signal_min = np.min(filtered_image)
    signal_mean = np.mean(filtered_image)
    signal_std = np.std(filtered_image)

    # Skip images with insufficient contrast (no clear particle)
    if signal_max - signal_min < 3 * signal_std or (signal_max - signal_mean) < 2 * signal_std:
        return None

    # Initial guess based on the filtered image
    initial_guess = (x0, y0, 1, np.max(image), np.min(image))

    try:
        params, _ = curve_fit(gaussian_2d_wrapper,
                              (y.ravel(), x.ravel()),
                              image.ravel(),
                              p0=initial_guess,
                              maxfev=1000)  # Increase max function evaluations

        # Validate results (ensure sigma is positive and reasonable)
        if params[2] <= 0 or params[2] > 100 or params[3] <= 0:
            return None

        return params
    except (RuntimeError, ValueError):
        # Catch fitting failures
        return None


def gaussian_2d_wrapper(coordinates, x0, y0, sigma, amplitude, offset):
    """
    Wrapper for the 2D Gaussian function to work with curve_fit.
    """

    y, x = coordinates
    return gaussian_2d(x, y, x0, y0, sigma, amplitude, offset).ravel()


###################################################################################
# ###### Some utility functions for timestamp-based file interval selection #######
###################################################################################
def get_time_differences(filenames):
    """
    Calculates the time differences between consecutive files
    based on HH-MM-SS timestamps in their filenames.

    Args:
        filenames: array of sorted filenames (strings) that contain timestamps
        Filename format assumed to be "HH-MM-SS_..." where HH is hours, MM is minutes, SS is seconds.
        Files not matching this leading pattern are skipped.

    Returns:
        list: A list of timedelta objects representing the time differences
              between consecutive files, sorted by their timestamps.
              Returns an empty list if there are fewer than two relevant files.
    """
    file_timestamps = []

    # Extract timestamps and associate with full paths
    for filename in filenames:
        if len(filename) >= 9 and filename[2] == '-' and filename[5] == '-' and filename[8] == '_':
            timestamp_str = filename[:8]  # e.g., "HH-MM-SS"
            try:
                # Assuming the files are from the same day, or we only care about time of day differences.
                dt_object = datetime.strptime(timestamp_str, "%H-%M-%S")
                file_timestamps.append((dt_object, filename)) # Store datetime object and original filename
            except ValueError:
                # Handle cases where the format doesn't match despite initial checks
                print(f"Warning: Could not parse timestamp from filename: {filename}")
                continue
        else:
            print(f"Skipping file (does not match expected format): {filename}")

    if not file_timestamps:
        print("No files with matching HH-MM-SS_ format found or parsed successfully.")
        return []

    difference_seconds = []
    # Calculate differences between consecutive files
    for i in range(1, len(file_timestamps)):
        current_time = file_timestamps[i][0]
        previous_time = file_timestamps[i-1][0]

        # Calculate the difference. NB We do not handle cases where the time wraps around midnight.
        difference = current_time - previous_time

        # Convert the timedelta object to total seconds
        difference_seconds.append(difference.total_seconds())

    return difference_seconds


def find_iqr_threshold(data, k=3):
    """
    Calculates the upper bound threshold for outliers using the IQR method.

    The IQR method defines potential outliers as data points that fall
    outside of the range [Q1 - k*IQR, Q3 + k*IQR]. This function specifically
    returns the upper bound (Q3 + k*IQR) as the threshold for 'long' intervals.

    Args:
        data (list or numpy.ndarray): A list or array of numerical data (e.g., time differences).
        k (float): The multiplier for the IQR. Common values are 1.5 (for mild outliers)
                   or 3.0 (for extreme outliers). Defaults to 1.5.

    Returns:
        float: The calculated upper bound threshold. Returns 0 if data is empty or too short.
    """
    # Guard for empty/short data; handle numpy arrays explicitly
    if data is None or len(data) < 2:
        print("Warning: Data is empty or has too few elements to calculate quartiles. Returning 0.")
        return 0.0

    # Convert to numpy array for robust percentile calculations
    data_np = np.array(data)

    # Calculate Q1 (25th and 75th percentile)
    Q1 = np.percentile(data_np, 25)
    Q3 = np.percentile(data_np, 75)

    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Calculate the upper bound threshold
    upper_bound_threshold = Q3 + k * IQR

    return upper_bound_threshold


def get_intervaled_tiffs_files(all_tiff_files, interval=0):
    """
    Select TIFF files at intervals based on timestamp gaps or fixed interval.
    This function filters a list of TIFF files to select representative images from
    distinct acquisition runs or sample regions. It supports two selection modes:
    automatic (timestamp-based) and manual (fixed interval).

    Parameters
    ----------
    all_tiff_files : list of str
        List of TIFF file paths to process. Can be empty or contain any number of files.
    interval : int, optional
        Downsampling of the TIFF files based on their timestamps can be performed. We look for significant
        time gaps between images to identify distinct image runs. The selection mode is determined by this parameter
        Selection mode parameter (default=0):
        - If 0: Use automatic timestamp-based selection with IQR method to detect
          large time gaps between images, selecting the first image after each gap.
        - If 1: Process all files (no downsampling).
        - If > 1: Use simple fixed-interval downsampling, taking every `interval`-th file.
    Filename format assumed to be "HH-MM-SS_....tiff" where HH is hours, MM is minutes, SS is seconds.
    Files not matching this leading pattern are skipped.

    Returns
    -------
    list of str
        Filtered list of TIFF file paths according to the selected interval strategy.
        - Empty list if input is empty
        - Single file if only one file provided
        - All files if fewer than 10 files and interval=0
        - First file only if timestamp extraction fails
        - Subset of files based on IQR threshold or fixed interval otherwise
    Notes
    -----
    - For automatic mode (interval=0), requires at least 10 files for IQR calculation
    - Timestamp extraction failures fall back to returning only the first file
    - Files are sorted before processing in automatic mode
    - Fixed interval mode always includes the first file
    - Robust to edge cases: empty input, single file, missing timestamps

    See Also
    --------
    get_time_differences : Extract time differences from file timestamps
    find_iqr_threshold : Calculate IQR-based threshold for gap detection

    """
    # we use the time stamps in the file name to calculate interval between each image and the select the first image occuring after a large inteveral. 
    # The large interval is found automatically using an interquartile range method. 
    # These images should then correspond to images from distinct image runs in different sample regions.
    # (enhanced: now robust to 0 or 1 file and filenames without timestamps.)

    # Normalize inputs
    if not all_tiff_files:
        return []

    if len(all_tiff_files) == 1:
        # With a single file, just return it regardless of interval strategy
        return [all_tiff_files[0]]

    if interval == 0:
        # Timestamp-based selection
        sorted_files = sorted(all_tiff_files)
        # If too few files, skip IQR/timestamp logic and use all files
        MIN_FILES_FOR_IQR = 10
        if len(sorted_files) < MIN_FILES_FOR_IQR:
            print(f"Few files detected ({len(sorted_files)} < {MIN_FILES_FOR_IQR}). Skipping timestamp/IQR and using all files.")
            return sorted_files
        time_diffs = get_time_differences(sorted_files)

        # If we couldn't compute diffs (e.g., no/invalid timestamps), fall back to first file only
        if not time_diffs:
            return [sorted_files[0]]

        # Ensure numpy array for safe elementwise comparison
        time_diffs_np = np.array(time_diffs, dtype=float)
        threshold = find_iqr_threshold(time_diffs_np)
        print(f"Threshold image interval is {threshold} seconds.")

        big_interval_inds = np.where(time_diffs_np >= threshold)[0] + 1  # this index corresponds to the next image after a big gap
        img_inds = np.insert(big_interval_inds, 0, 0)

        # Guard against any out-of-bounds or duplicates just in case
        img_inds = np.clip(img_inds, 0, len(sorted_files) - 1)
        img_inds = np.unique(img_inds)

        intervaled_tiff_files = [sorted_files[i] for i in img_inds]
    else:
        # Simple interval-based downsampling; always includes the first file
        intervaled_tiff_files = all_tiff_files[::max(1, int(interval))]

    return intervaled_tiff_files


# ############ MAIN LOOP #############
def main():
    """
    Main function to preprocess experimental TIFF images for particle detection analysis.

    This function performs the following steps:
    1. Parses command-line arguments or displays UI dialogs to collect user input
    2. Divides large TIFF images into smaller sub-images with optional overlap
    3. Performs Gaussian fitting on selected sub-images to estimate PSF (Point Spread Function) width
    4. Calculates mean sigma value from fitted Gaussian parameters
    5. Moves processed datasets to the './datasets' directory
    6. Generates configuration JSON files for each dataset with analysis parameters

    Command-line Arguments:
        -t, --terminal: Run without UI (command-line mode)
        -f, --folder: Path to folder containing TIFF images
        -s, --size: Size of square sub-images (sz x sz pixels)
        -o, --overlap: Overlap size in pixels (default: 0)
        -i, --interval: Process every Nth file; 0 uses IQR-based timestamp method (default: 0)
        -c, --crop: Crop fraction for raw images (default: 0.7)
        -m, --maxhindex: Maximum hypothesis index for analysis (default: 5)
        --save-plots: Save plots to file instead of displaying (for headless environments)
        --predefined-sigma: Optional predefined PSF sigma to use instead of estimating from images
        --config-subdir: Optional subfolder under ./configs to save config files

    Workflow:
        - In terminal mode: Validates folder path, lists TIFF files, and prompts for parameters
        - In UI mode: Opens dialogs for folder selection and parameter input
        - Automatically estimates optimal sub-image size based on particle density if requested
        - Divides images with progress tracking
        - Performs Gaussian fitting on user-selected sub-images containing particles
        - Filters outlier sigma values based on histogram mode (0.25*mode to 2*mode range)
        - Generates plots showing original vs fitted images and sigma histogram
        - Creates JSON config files with analysis parameters for each dataset

    Returns:
        None
    """
    # Parse command-line arguments    
    parser = argparse.ArgumentParser(description="Divide TIFF images into smaller sub-images and perform Gaussian fitting.")
    parser.add_argument('-t', '--terminal', action='store_true', help="Run the process without UI")
    parser.add_argument('-f', '--folder', type=str, help="Folder containing TIFF images to be divided")
    parser.add_argument('-s', '--size', type=int, help="Size of sub-image (sz x sz)")
    parser.add_argument('-o', '--overlap', type=int, default=0, help="Overlap size in pixels")
    parser.add_argument('-i', '--interval', type=int, default=0, help="Process every Nth file (e.g., 2 means every 2nd file). 0 implies an IQR method timestamps method is used.")
    parser.add_argument('-c', '--crop', type=float, default=0.7, help="Crop down size of raw image files (e.g., 0.7 means image height and width will be reduced to 0.7 of the original size)")
    parser.add_argument('-m', '--maxhindex', type=int, help="Set maximum hypothesis index in config file (ana_maximum_hypothesis_index). Default is 5")
    parser.add_argument('--save-plots', action='store_true',help="Save plots to file instead of displaying (useful for headless environments)")
    parser.add_argument('--predefined-sigma', type=float,help="Optional predefined PSF sigma to use instead of estimating from images")
    parser.add_argument('--config-subdir', type=str,help="Optional subfolder under ./configs where config files will be saved",)
    args = parser.parse_args()

    #  Terminal mode
    if args.terminal:
        # Print supplied arguments
        print("\n" + "=" * 70)
        print("TERMINAL MODE - Supplied Arguments:")
        print("=" * 70)
        print(f"  Folder:           {args.folder}")
        print(f"  Size:             {args.size}")
        print(f"  Overlap:          {args.overlap}")
        print(f"  Interval:         {args.interval}")
        print(f"  Crop:             {args.crop}")
        print(f"  Max H-Index:      {args.maxhindex}")
        print(f"  Save Plots:       {args.save_plots}")
        print(f"  Predefined Sigma: {args.predefined_sigma}")
        print(f"  Config Subdir:    {args.config_subdir}")
        print("=" * 70 + "\n")

        if not args.folder:
            print("Error: Folder must be provided.")
            return
        else:
            # Clean up the path - remove quotes and normalize path
            # Remove trailing slashes or backslashes
            input_folder = args.folder
            input_folder = input_folder.rstrip('\\/')
            input_folder = args.folder.strip('\'"')
            # Make sure path exists
            if not os.path.isdir(input_folder):
                print(f"Error: Folder '{input_folder}' does not exist or is not accessible.")
                return

        # Process file interval setting
        interval = args.interval
        if interval < 0:
            print("Warning: Invalid interval value. Using default of 1 (process all files).")
            interval = 1
        elif interval == 0:
            print("Processing files based on timestamp intervals.")
        elif interval >= 1:
            print(f"Processing every {interval}th file.")

        # Crop fraction for terminal mode (0..1, where 1 means no cropping)
        crop_img_fraction = args.crop

        # Optional config subfolder from CLI
        config_subdir = None
        if args.config_subdir:
            sub = args.config_subdir.strip().replace('\\', '/').strip('/')
            if sub and not os.path.isabs(sub) and '..' not in sub:
                config_subdir = sub
            else:
                print("Warning: Ignoring invalid --config-subdir value; using ./configs")

        # List TIFF files in the folder
        try:
            # Get all TIFF files in the folder
            all_tiff_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.tiff')])
            if not all_tiff_files:
                print(f"Warning: No TIFF files found in '{input_folder}'")
                all_tiff_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tiff', '.tif'))]
                if all_tiff_files:
                    print(f"Found {len(all_tiff_files)} files with .tif extension instead")
                else:
                    print("No TIFF or TIF files found in the folder. Exiting.")
                    return

            intervaled_tiff_files = get_intervaled_tiffs_files(all_tiff_files, interval)
            print(f"Intervaled (interval: {interval}) TIFF files to be divided into subimages:")
            for i, file in enumerate(intervaled_tiff_files):
                print(f"{i + 1}. {file}")
        except Exception as e:
            print(f"Error accessing folder or listing files: {e}")
            return

        if not args.size:
            print("Sub-image size not provided.")

            # Ask user if they want to automatically calculate sub-image size
            print("\nDo you want to automatically calculate sub-image size?")
            print("1. Yes")
            print("2. No, I will provide the size")
            choice = input("Enter your choice (1 or 2): ")

            if choice == '1':
                # List TIFF files in the folder
                print("\nIntervaled TIFF files to be divided into subimages:")
                for i, file in enumerate(intervaled_tiff_files):
                    print(f"{i + 1}. {file}")

                # Get user selection
                choice = int(input("\nSelect a file number to estimate sub-image size: ")) - 1
                if 0 <= choice < len(intervaled_tiff_files):
                    chosen_tiff_path = os.path.join(input_folder, intervaled_tiff_files[choice])
                    # Use user-specified crop fraction for both dimensions
                    particle_count = rough_count_particles(chosen_tiff_path, ch=crop_img_fraction, cw=crop_img_fraction)
                    # Suggest a size that would result in average of 1 particles per sub-image (P(n=5) ~ 0.003 in this case thus testing up to n=4 covers 99.7% of the cases)
                    with Image.open(chosen_tiff_path) as img:
                        width, height = img.size

                    # Account for cropping done by rough_count_particles using crop_img_fraction (square central crop)
                    cropped_area = (width * crop_img_fraction) * (height * crop_img_fraction)
                    area_per_particle = cropped_area / particle_count
                    sz = int(np.sqrt(area_per_particle * 1))
                    print(f"\n(Rougly) estimated {particle_count} particles in the entire image.")
                    print(f"Suggested sub-image: {sz}x{sz} pixels")
                    if not 20 < sz < 200:
                        print(
                            "Warning: The suggested size is outside the range of 20 to 200 pixels. "
                            "Clipping to this range."
                        )
                        print(
                            "If you want to use size outside this range, please run again with the --size "
                            "(or -s) argument."
                        )
                        if sz < 20:
                            sz = 20
                        elif sz > 200:
                            sz = 200
                else:
                    print("Invalid selection, using default size of 100")
                    return
            elif choice == '2':
                sz = int(input("Enter the desired sz of sub-image (sz x sz) in pixels: "))
            else:
                print("Invalid choice. Please enter 1 or 2.")
                return
        else:
            sz = args.size

        if not args.overlap:
            print("Overlap size not provided. Setting to default (0 pixels).")
            overlap = 0

    # UI mode
    else:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Image Division UI", "Welcome to Image Division UI.\n======================\nTo use terminal instead, use the --terminal flag with the -f (folder, required) argument (-s (size), -o (overlap), and -i (interval) are optional).")

        input_folder = filedialog.askdirectory(title="==== Select Folder Containing all TIFF images to be divided ====")
        if not input_folder:
            messagebox.showerror("Error", "No folder selected. Exiting.")
            return

        while not any(file.endswith('.tiff') for file in os.listdir(input_folder)):
            messagebox.showerror("Error", "No TIFF files found in the selected folder.")
            input_folder = filedialog.askdirectory(title="==== Select Folder Containing all TIFF images to be divided ====")

        if not input_folder:
            print("No folder selected. Exiting.")
            return

        while not os.path.exists(input_folder):
            retry = messagebox.askretrycancel("Error", "Selected folder does not exist.")
            if not retry:
                print("No valid folder selected. Exiting.")
                return

        # Ask user for crop fraction (0..1)
        crop_img_fraction = simpledialog.askfloat(
            "Crop fraction",
            "Enter crop fraction for the root images (0 to 1).\n1 disables cropping; default is 0.7",
            initialvalue=0.7,
            minvalue=0.0,
            maxvalue=1.0,
        )
        if crop_img_fraction is None:
            messagebox.showerror("Error", "No crop fraction provided. Exiting.")
            return

        # Ask for optional config subfolder under ./configs
        config_subdir = None
        sub = simpledialog.askstring(
            "Config subfolder (optional)",
            "Enter a subfolder name under ./configs to save config files (leave blank to use ./configs):",
            initialvalue=""
        )
        if sub:
            sub = sub.strip().replace('\\', '/').strip('/')
            if os.path.isabs(sub) or '..' in sub:
                messagebox.showerror("Invalid subfolder", "Please enter a simple subfolder name under ./configs.")
                return
            config_subdir = sub

        # Get file interval setting and file list
        interval = simpledialog.askinteger("Tiff file skip interval",
                                               """
Enter the interval for processing TIFF files (e.g., 2 means every 2nd file):

If 0: Use automatic timestamp-based selection with IQR method to detect
large time gaps between images, selecting the first image after each gap.
- If 1: Process all files (no downsampling).
- If > 1: Use simple fixed-interval downsampling, taking every `interval`-th file.
""",
                                               initialvalue=1, minvalue=0)
        all_tiff_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.tiff')])
        intervaled_tiff_files = get_intervaled_tiffs_files(all_tiff_files, interval)
        files_list = "\n".join(intervaled_tiff_files)

        # Ask user if they want to automatically calculate sub-image size
        auto_size = messagebox.askyesno("Sub-image Size", f"The following files will be processed:\n{files_list}\n================\n\nSuggest an appropriate sub-image size based on a rough estimate of particle density?")

        if auto_size:
            # List TIFF files in the folder
            # Create dialog to select a file
            dialog = tk.Toplevel()
            dialog.title("Select TIFF File")
            dialog.geometry("700x300")  # Set width=400, height=300
            listbox = tk.Listbox(dialog, width=700)
            for file in intervaled_tiff_files:
                listbox.insert(tk.END, file)
            listbox.pack(padx=10, pady=10)

            # Enable double-click selection
            def on_double_click(event):
                selected_file[0] = listbox.get(listbox.curselection())
                dialog.destroy()

            listbox.bind('<Double-Button-1>', on_double_click)

            selected_file = [None]

            def on_select():
                selected_file[0] = listbox.get(listbox.curselection())
                dialog.destroy()

            tk.Button(dialog, text="Select", command=on_select).pack(pady=5)
            dialog.wait_window()

            if selected_file[0]:
                tiff_path = os.path.join(input_folder, selected_file[0])
                # Use user-selected crop fraction for both dimensions (square central crop)
                particle_count = rough_count_particles(tiff_path, ch=crop_img_fraction, cw=crop_img_fraction)
                with Image.open(tiff_path) as img:
                    width, height = img.size
                # Account for cropping done by rough_count_particles using crop_img_fraction (square central crop)
                cropped_area = (width * crop_img_fraction) * (height * crop_img_fraction)
                area_per_particle = cropped_area / particle_count
                sz = int(np.sqrt(area_per_particle * 1))
                if sz < 20:
                    clip_yesno = messagebox.askyesno("Warning", f"Suggested {sz} pixels is below reasonable range (20-200). Set it to 20?")
                    if clip_yesno:
                        sz = 20
                    else:
                        sz = simpledialog.askinteger("Confirm Size", "Enter desired sub-image size:", initialvalue=sz, minvalue=10)
                elif sz > 200:
                    clip_yesno = messagebox.askyesno("Warning", f"Suggested {sz} pixels is above reasonable range (20-200). Set it to 200?")
                    if clip_yesno:
                        sz = 200
                    else:
                        sz = simpledialog.askinteger("Confirm Size", "Enter desired sub-image size:", initialvalue=sz, minvalue=10)
                else:
                    sz = simpledialog.askinteger("Confirm Size", f"Estimated {particle_count} particles in the entire image. \nSuggested sub-image size: {sz}x{sz}. If you want to use a different size, enter differently.", initialvalue=sz, minvalue=10)
                    messagebox.showinfo("Confirm Size", f"Estimated {particle_count} particles in the entire image. \nSuggested sub-image size: {sz}x{sz}")

            else:
                messagebox.showerror("Error", "No file selected.")
                sz = simpledialog.askinteger("Input", "Enter the desired sz of sub-image (sz x sz) in pixels:", initialvalue=100, minvalue=10)
        else:
            sz = simpledialog.askinteger("Input", "Enter the desired sz of sub-image (sz x sz) in pixels:", initialvalue=100, minvalue=1)
        if not sz:
            print("Invalid size. Exiting.")
            return

    ###################################################################################
    if not args.terminal:
        root = tk.Tk()
        root.title("Processing Images")
        progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        progress.pack(pady=20)
        progress_label = tk.Label(root, text="Starting...")
        progress_label.pack()
        current_file_label = tk.Label(root, text="")
        current_file_label.pack()

    # Function to update progress bar or terminal output
    def update_progress(processed, total):
        if not args.terminal:
            progress["value"] = (processed / total) * 100
            progress_label.config(text=f"Processed {processed} of {total} sub-images")
            root.update_idletasks()
        else:
            # Add terminal progress indicator
            percent = int((processed / total) * 100)
            bar_length = 30
            filled_length = int(bar_length * processed // total)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\r[{bar}] {percent}% - {processed}/{total} sub-images processed', end='', flush=True)
            if processed == total:
                print()  # Add newline after completion

    short_names = []
    if not args.terminal:
        overlap = simpledialog.askinteger("Input", "Enter desired overlap in pixels (default 0):", initialvalue=0, minvalue=0)
        if overlap is None:
            messagebox.showerror("Error", "No folder selected. Exiting.")
            return
        files_list = "\n".join([f"- {file}" for file in intervaled_tiff_files])
        messagebox.showinfo("Starting image division", f"Division of {len(intervaled_tiff_files)} tiff files into {sz}x{sz} subimages with {overlap} overlap will now begin (unit: pixel).\n\n{files_list}")
    else:
        overlap = args.overlap
        print(f"Division into {sz}x{sz} subimages with {overlap} overlap will now begin for intervaled tiff files (unit: pixel).")

    # Perform image division (use crop_img_fraction from terminal or UI selection)
    try:
        if not intervaled_tiff_files:
            raise FileNotFoundError("No TIFF files found in the selected folder.")

        print(f"Processing {len(intervaled_tiff_files)} of {len(all_tiff_files)} TIFF files (interval={interval}).")

        for file in intervaled_tiff_files:
            input_file_path = os.path.join(input_folder, file)

            # Always use the basename without extension for folder names to avoid clashing with the source TIFF file
            base_name, _ = os.path.splitext(file)
            output_dir = os.path.join(input_folder, base_name)

            # Try to create output directory with full base_name
            try:
                os.makedirs(output_dir, exist_ok=True)
                folder_name = base_name
            except OSError as e:
                # If filename is too long or other OS error, use shortened name
                if "too long" in str(e).lower() or "filename" in str(e).lower():
                    n1 = 40
                    n2 = 4
                    folder_name = base_name[:n1] + '___' + base_name[-n2:]
                    output_dir = os.path.join(input_folder, folder_name)
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"Warning: Filename too long. Using shortened name: {folder_name}")
                else:
                    raise

            short_names.append(folder_name)
            if not args.terminal:
                current_file_label.config(text=f"Dividing file: {file}")
                root.update_idletasks()
            divide_images(input_file_path, output_dir, (sz, sz), overlap, update_progress, crop_img_fraction)
            print(f"Processed {file}")
    except FileNotFoundError as e:
        if not args.terminal:
            messagebox.showerror("Error", f"File not found: {e.filename}")
        else:
            print(f"Error: File not found: {e.filename}")
        return

    if not args.terminal:
        root.destroy()
        messagebox.showinfo("Image Division UI", f"Image division completed successfully.\n\nShort names of processed files:\n{', '.join(short_names)}")
    else:
        print("Image division completed successfully.")
        print(f"Short names of processed files:\n{', '.join(short_names)}")

    # Select images for Gaussian fitting (skip if user provided predefined sigma)
    if args.terminal:
        if args.predefined_sigma is not None:
            # Skip all Gaussian fitting; use predefined sigma directly
            mean_sigma = float(args.predefined_sigma)
            print(f"\n[Predefined Sigma] Using provided sigma value: {mean_sigma:.3f}")
            print("[Predefined Sigma] Skipping Gaussian fitting and sub-image selection.\n")
        else:
            # Proceed with normal Gaussian fitting flow
            print("Please choose a folder from the following list to use for determining PSF width:")
            for i, short_name in enumerate(short_names):
                output_dir = os.path.join(input_folder, short_name)
                num_files = len(os.listdir(output_dir))
                print(f">> {i + 1}. {short_name} ({num_files} files)")
            folder_choice = int(input("Enter the number of the folder you want to use: ")) - 1
            selected_folder = os.path.join(input_folder, short_names[folder_choice])

        if args.predefined_sigma is not None:
            # Skip all Gaussian fitting; use predefined sigma directly
            mean_sigma = float(args.predefined_sigma)
            print(f"\n[Predefined Sigma] Using provided sigma value: {mean_sigma:.3f}")
            print("[Predefined Sigma] Skipping Gaussian fitting and sub-image selection.\n")
        else:
            # Proceed with normal Gaussian fitting flow
            print("Please choose a folder from the following list to use for determining PSF width:")
            for i, short_name in enumerate(short_names):
                output_dir = os.path.join(input_folder, short_name)
                num_files = len(os.listdir(output_dir))
                print(f">> {i + 1}. {short_name} ({num_files} files)")
            folder_choice = int(input("Enter the number of the folder you want to use: ")) - 1
            selected_folder = os.path.join(input_folder, short_names[folder_choice])

            selected_files = []
            print(f"Please choose at least 3 TIFF files from the folder {selected_folder} that clearly have particles in them:")
            tiff_files = [f for f in os.listdir(selected_folder) if f.endswith('.tiff')]

            # Display files with numbers (show only the first 20 if there are many)
            max_display = 20
            if len(tiff_files) > max_display:
                for i in range(max_display):
                    print(f">> {i + 1}. {tiff_files[i]}")
                print(f"... and {len(tiff_files) - max_display} more files")
            else:
                for i, tiff_file in enumerate(tiff_files):
                    print(f">> {i + 1}. {tiff_file}")

            # Ask user to select files by numbers
            # Default to the first 10% of files if the user presses Enter
            default_count = max(1, len(tiff_files) // 10)  # Ensure at least 1 file is selected
            default_selections = ', '.join(str(i + 1) for i in range(min(default_count, len(tiff_files))))

            prompt = f"Enter file numbers (e.g., '1, 4, 6, 11' or '1-5' or '1, 2, 4-7, 9') or press Enter for default (first {min(default_count, len(tiff_files))} files): "
            selections = input(prompt).strip()

            if not selections:
                selections = default_selections
                print(f"Using default selections: {selections}")

            # Parse the selections
            selected_indices = []
            for part in selections.split(','):
                part = part.strip()
                if '-' in part:
                    # Handle ranges (e.g., "1-5")
                    start, end = map(int, part.split('-'))
                    selected_indices.extend(range(start, end + 1))
                else:
                    # Handle individual numbers
                    try:
                        selected_indices.append(int(part))
                    except ValueError:
                        print(f"Warning: Ignoring invalid input '{part}'")

            # Convert to 0-based indices and get file paths
            selected_files = []
            for idx in selected_indices:
                if 1 <= idx <= len(tiff_files):
                    selected_files.append(os.path.join(selected_folder, tiff_files[idx - 1]))
                else:
                    print(f"Warning: Ignoring out-of-range index {idx}")

            if len(selected_files) < 3:
                print("Error: Please select at least 3 images.")
                return
    else:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Gaussian Fitting", "Please select at least 3 sub(divided)-images that clearly have particles. The images will be used to determine the PSF width. \nNote: only one particle (likely the brightest) per image will be analyzed.")
        # Get the last folder with divided images
        last_divided_folder = None
        for short_name in short_names:
            folder_path = os.path.join(input_folder, short_name)
            if os.path.isdir(folder_path):
                last_divided_folder = folder_path

        selected_files = filedialog.askopenfilenames(
            title="Select Images for Gaussian Fitting",
            filetypes=[("TIFF files", "*.tiff")],
            initialdir=last_divided_folder
        )
        if len(selected_files) < 3:
            messagebox.showerror("Error", "Exiting - 3+ images not selected.")
            return

    sigma_values = []
    images = []
    fitted_images = []
    skipped_files = []

    # If terminal mode and predefined sigma provided, bypass fitting and use directly
    if args.terminal and args.predefined_sigma is not None:
        mean_sigma = float(args.predefined_sigma)
        # Already announced earlier; no fitting performed
    else:
        for file in selected_files:
            image = np.array(Image.open(file))
            params = fit_gaussian_2d(image)

            if params is None:
                print(f"Skipping {os.path.basename(file)} - no clear particle detected")
                skipped_files.append(os.path.basename(file))
                continue

            sigma_values.append(params[2])
            fitted_image = gaussian_2d(*np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0])), *params).reshape(image.shape)
            images.append(image)
            fitted_images.append(fitted_image)
            print(f"Fitted parameters for {file}: (x, y) = ({params[0]:0.1f}, {params[1]:0.1f}), sigma = {params[2]:0.1f}, amplitude = {params[3]:0.1f}, background = {params[4]:0.1f}")

        # Calculate mode from histogram
        hist, bin_edges = np.histogram(sigma_values, bins=20)
        max_freq = np.max(hist)
        mode_bin_indices = np.where(hist == max_freq)[0]

        # Only proceed with mode filtering if there's a single clear mode
        if len(mode_bin_indices) == 1:
            mode = (bin_edges[mode_bin_indices[0]] + bin_edges[mode_bin_indices[0] + 1]) / 2

            # Create a filter mask for values within 0.25*mode to 2*mode
            filter_mask = [(0.25 * mode <= s <= 2 * mode) for s in sigma_values]

            # Apply filter to all three arrays simultaneously to keep them in sync
            filtered_sigma_values = [s for i, s in enumerate(sigma_values) if filter_mask[i]]
            filtered_images = [img for i, img in enumerate(images) if filter_mask[i]]
            filtered_fitted_images = [img for i, img in enumerate(fitted_images) if filter_mask[i]]

            # Count how many were filtered out
            filtered_out_count = len(sigma_values) - len(filtered_sigma_values)

            # Replace the original arrays with the filtered versions
            sigma_values = filtered_sigma_values
            images = filtered_images
            fitted_images = filtered_fitted_images

            print(f"\nMode sigma: {mode:.2f}")
            print(f"Filtered {filtered_out_count} outlier values")
        else:
            print("\nMultiple modes detected - skipping outlier filtering")

        if not sigma_values:
            if not args.terminal:
                messagebox.showerror("Error", "No valid particles detected in any of the selected images. Please select different images.")
            else:
                print("ERROR: No valid particles detected in any of the selected images. Please select different images.")
            return

        mean_sigma = np.mean(sigma_values)
        if not args.terminal:
            messagebox.showinfo("Gaussian Fitting", f"Gaussian fitting completed. Mean sigma: {mean_sigma:.2f}")
        else:
            print(f"Mean sigma: {mean_sigma:.2f}")

    if not (args.terminal and args.predefined_sigma is not None):
        # Plot the original and fitted images side-by-side for the selected images
        plt.figure(figsize=(5, 9))

        # Determine the common color range for the original and fitted images
        vmin = min(np.min(image) for image in images)
        vmax = max(np.max(image) for image in images)

        # Plot original and fitted images side-by-side for the selected images
        n_plot = min(5, len(images))  # Ensure at least 5 plots are shown
        for i in range(n_plot):
            plt.subplot(n_plot, 2, 2 * i + 1)
            plt.title(f'Original {i+1}', fontsize=8)
            plt.imshow(images[i], cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar().ax.tick_params(labelsize=6)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.subplot(n_plot, 2, 2 * i + 2)
            plt.title(f'Fitting {i+1} - Sigma: {sigma_values[i]:.2f}', fontsize=8)
            plt.imshow(fitted_images[i], cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar().ax.tick_params(labelsize=6)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)

        # Save or show the fitting plots
        if args.terminal and args.save_plots:
            fits_plot_path = os.path.join(input_folder, "gaussian_fits.png")
            plt.savefig(fits_plot_path)
            print(f"Gaussian fits plot saved to {fits_plot_path}")

        # Plot histogram of sigma values
        plt.figure(figsize=(4, 3))
        plt.hist(sigma_values, bins=20, edgecolor='black')
        plt.axvline(mean_sigma, color='r', linestyle='dashed', linewidth=1)
        plt.text(mean_sigma, plt.ylim()[1] * 0.9, f'Mean: {mean_sigma:.2f}', color='r', fontsize=8)
        plt.title('Histogram of Sigma Values', fontsize=10)
        plt.xlabel('Sigma', fontsize=8)
        plt.ylabel('Frequency', fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

        # Save or show the histogram plot
        if args.terminal and args.save_plots:
            hist_plot_path = os.path.join(input_folder, "sigma_histogram.png")
            plt.savefig(hist_plot_path)
            print(f"Sigma histogram plot saved to {hist_plot_path}")
        else:
            plt.show(block=False)

    # Move the folders containing the subdivided images to ./datasets for analysis
    if args.terminal or messagebox.askokcancel("Move Folders", f"The subdivided images folders will now be moved to {os.path.abspath('./datasets')} for analysis"):
        repo_root_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(repo_root_dir, 'datasets')
        os.makedirs(datasets_dir, exist_ok=True)

        for file in os.listdir(input_folder):
            if file.endswith('.tiff'):
                # Mirror the logic used during division to locate the output folders
                base_name, _ = os.path.splitext(file)

                # First, try to find folder with full base_name
                output_dir = os.path.join(input_folder, base_name)
                folder_name = base_name

                # If full name doesn't exist, try shortened name
                if not os.path.isdir(output_dir):
                    n1 = 40
                    n2 = 4
                    folder_name = base_name[:n1] + '___' + base_name[-n2:]
                    output_dir = os.path.join(input_folder, folder_name)

                if os.path.isdir(output_dir):
                    new_output_dir = os.path.join(datasets_dir, folder_name)
                    if os.path.exists(new_output_dir):
                        shutil.rmtree(new_output_dir)
                    try:
                        os.rename(output_dir, new_output_dir)
                    except OSError:
                        shutil.copytree(output_dir, new_output_dir)
                        shutil.rmtree(output_dir)
                    print(f"Moved {output_dir} to {new_output_dir}")
        print(f"All subdivided images have been moved to {os.path.abspath('./datasets')} for analysis.")
        if not args.terminal:
            moved_folders = [os.path.join(datasets_dir, short_name) for short_name in short_names]
            messagebox.showinfo("Move Folders", f"All subdivided images have been moved to {os.path.abspath('./datasets')} for analysis:\n\n" + "\n".join(moved_folders))

    # Create config JSON files for each dataset
    config_root = './configs'
    config_dir = os.path.join(config_root, config_subdir) if 'config_subdir' in locals() and config_subdir else config_root
    os.makedirs(config_dir, exist_ok=True)
    skipall_edits = False
    for short_name in short_names:
        folder_path = os.path.join(datasets_dir, short_name)
        if os.path.isdir(folder_path):
            config_data = {
                "image_folder_namebase": short_name,
                "code_version_date": "2025-11-19",
                "file_format": "tiff",
                "analyze_the_dataset?": True,
                "ana_random_seed": np.random.randint(0, 10000),
                "ana_predefined_psf_sigma": round(mean_sigma, 3),
                "ana_use_premature_hypothesis_choice?": False,
                "ana_maximum_hypothesis_index": args.maxhindex if args.maxhindex is not None else 5,
                "ana_delete_the_dataset_after_analysis?": False
            }

            # Allow terminal users to edit config before saving
            if args.terminal and not skipall_edits:
                print("\n==== Configuration for", short_name, "====")
                print(json.dumps(config_data, indent=4))
                edit_config = input("\nEdit this configuration before saving? Yes (y), No (n) or No for all remaining (na): ").strip().lower()
                if edit_config == 'y':
                    print("\nYou can edit the following parameters:")
                    for i, (key, value) in enumerate(config_data.items(), 1):
                        print(f"{i}. {key}: {value}")

                    while True:
                        param_num = input("\nEnter parameter number to edit (or 'done' to finish): ")
                        if param_num.lower() == 'done':
                            break
                        try:
                            param_num = int(param_num)
                            if 1 <= param_num <= len(config_data):
                                key = list(config_data.keys())[param_num - 1]
                                current_value = config_data[key]
                                print(f"Current value of '{key}': {current_value}")

                                # Handle different types of values
                                if isinstance(current_value, bool):
                                    new_value = input(f"Enter new value (true/false): ").strip().lower()
                                    config_data[key] = new_value == 'true'
                                elif isinstance(current_value, int):
                                    new_value = input(f"Enter new value (integer): ").strip()
                                    config_data[key] = int(new_value)
                                elif isinstance(current_value, float):
                                    new_value = input(f"Enter new value (float): ").strip()
                                    config_data[key] = float(new_value)
                                else:
                                    new_value = input(f"Enter new value (string): ").strip()
                                    config_data[key] = new_value

                                print(f"Updated '{key}' to: {config_data[key]}")
                            else:
                                print("Invalid parameter number!")
                        except ValueError:
                            print("Please enter a valid number or 'done'")

                    print("\nFinal configuration:")
                    print(json.dumps(config_data, indent=4))
                elif edit_config == 'na':
                    skipall_edits = True

            config_path = os.path.join(config_dir, f'{short_name}.json')
            with open(config_path, 'w') as config_file:
                json.dump(config_data, config_file, indent=4)

            if args.terminal:
                print(f"\nConfig file saved to: {os.path.abspath(config_path)}")
                print("=" * 50)
            else:
                config_path = os.path.join(os.path.abspath(config_dir), f"{short_name}.json")
                message = (
                    f"A new config file has been created - {config_path}:\n\n"
                    f"Config contents:\n{json.dumps(config_data, indent=4)}\n\n"
                    f"The calculated mean_sigma (mean psf width) value {mean_sigma:.3f} "
                    "is written as the analysis parameter in the config file.\n"
                    f"The image_folder_namebase of {short_name} is used.\n\n"
                    "Please open the config file to edit as necessary before running the analysis program."
                )
                messagebox.showinfo("Config Files", message)
        else:
            print(f"Folder {folder_path} does not exist.")
            print(f"Error: Folder {folder_path} does not exist. Config file will not be created.")
            if not args.terminal:
                messagebox.showerror("Error", f"Folder {folder_path} does not exist. Config file will not be created.")
                return

    if not args.terminal:
        if messagebox.askokcancel("Open Config Folder", "Do you want to open the folder containing the newly created Config Files?"):
            os.startfile(os.path.abspath(config_dir))

    if args.terminal:
        print("\n" + "=" * 70)
        print(f"NEXT STEPS:")
        print(f"1. Review and modify the config files in: {os.path.abspath(config_dir)}")
        print(f"2. Run the analysis program with these config files")
        print("=" * 70)

    print(f"\n-- Please open the config file {short_name}.json in {os.path.abspath(config_dir)} to edit as necessary before running the analysis program.")
    print("All done.")


if __name__ == "__main__":
    # if 'pydevd' in sys.modules or 'debugpy' in sys.modules:
    # #     # Run the main function without parallel processing ('-p' option value is False)
    #     sys.argv = ["preprocess_exp_data.py", "--folder", "./datasets/coreshell_np_data/130125_Au_nanoshell_Ag/130125_covidvirus_ctrl", "--terminal", "--size", "50", "--interval", "0", "--save-plot"] 
    #     # ['main.py', '-c', './configs/'] # -p for profiling. Default is False, and it will run on multiple processes.
    main()
