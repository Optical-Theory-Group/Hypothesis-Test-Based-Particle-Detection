import tifffile
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_generation import psfconvolution 
from process_algorithms import get_tentative_peaks, normal_gaussian_integrated_within_each_pixel
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

def gaussian_2d(x, y, x0, y0, sigma, amplitude):
    return amplitude / sigma / np.sqrt(2*np.pi) * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def generate_random_rgb_image_with_peaks(num_peaks=20):
    width, height = 100, 100  # Set desired size
    sigma = 2
    num_peaks=15

    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)

    # Initialize empty channels
    r_channel = np.zeros((height, width))
    g_channel = np.zeros((height, width))
    b_channel = np.zeros((height, width))

    for _ in range(num_peaks):
        x0 = np.random.randint(0, width)
        y0 = np.random.randint(0, height)
        r_intensity = np.random.randint(0, 800)
        g_intensity = np.random.randint(0, 800)
        b_intensity = np.random.randint(0, 800)

        r_channel += gaussian_2d(x, y, x0, y0, sigma, r_intensity)
        g_channel += gaussian_2d(x, y, x0, y0, sigma, g_intensity)
        b_channel += gaussian_2d(x, y, x0, y0, sigma, b_intensity)

    # Stack channels to create an RGB image
    random_image = np.stack([r_channel, g_channel, b_channel], axis=-1).astype(np.uint8)

    # Save as TIFF
    image = Image.fromarray(random_image, mode="RGB")
    image.save("random_rgb_with_peaks.tiff")

# generate_random_rgb_image_with_peaks()
# pass

def divide_large_tiff(input_file_path, sub_image_size, overlap, save_folder):
    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Open the TIFF image
    with Image.open(input_file_path) as img:
        # Check if the image is RGB
        if img.mode == 'RGB':
            # Convert to grayscale by averaging the channels
            img = img.convert('L')

        # Convert the image to a numpy array
        img_array = np.array(img)

    # Define the size of the sub-images and the overlap
    # sub_image_size = 100
    # overlap = 20

    # Get the dimensions of the input image
    img_height, img_width = img_array.shape

    # Calculate the number of sub-images in each dimension
    num_sub_images_x = math.ceil((img_width - overlap) / (sub_image_size - overlap))
    num_sub_images_y = math.ceil((img_height - overlap) / (sub_image_size - overlap))

    # Extract and save the sub-images
    for i in range(num_sub_images_y):
        for j in range(num_sub_images_x):
            start_x = j * (sub_image_size - overlap)
            start_y = i * (sub_image_size - overlap)
            end_x = start_x + sub_image_size
            end_y = start_y + sub_image_size

            sub_image = img_array[start_y:end_y, start_x:end_x]

            sub_image_filename = f'sub_image_{i}_{j}.tiff'
            sub_image_path = os.path.join(save_folder, sub_image_filename)

            # Save the sub-image
            sub_image_pil = Image.fromarray(sub_image)
            sub_image_pil.save(sub_image_path)
            pass

# Example usage
# input_file_path = '14-00-06_130824_covid_target_Au_Ag_ctrl_pdmschamber_df_image-1.tiff'
# sub_image_size = 100
# overlap = 20
# save_folder = 'div_images'
# divide_large_tiff(input_file_path, sub_image_size, overlap, save_folder)




# def fit_gaussian_2d(image, guess_xy):
#     x = np.arange(image.shape[1])
#     y = np.arange(image.shape[0])
#     x, y = np.meshgrid(x, y)
#     guess_x = 10
#     guess_y = 10
#     guess_sigma = 4
#     max_sigma = 7
#     alpha = (image.max() - image.min()) * 2 * np.pi * guess_sigma**2
#     guess_amp = alpha
#     initial_guess = (guess_x, guess_y, guess_sigma, guess_amp) 

#     def objective(params):
#         # Unnormalize the parameters
#         x0 = params[0] * image.shape[1]
#         y0 = params[1] * image.shape[0]
#         sigma = params[2] * max_sigma
#         amplitude = params[3] * alpha
#         return np.sum((gaussian_2d(x, y, x0, y0, sigma, amplitude) - image) ** 2)

#     def callback(params):
#         # Unnormalize the parameters
#         x0 = params[0] * image.shape[1]
#         y0 = params[1] * image.shape[0]
#         sigma = params[2] * max_sigma
#         amplitude = params[3] * alpha
#         print(f"Current parameters: x0={x0}, y0={y0}, sigma={sigma}, amplitude={amplitude}")
#         print(f"Current parameters: x0={params[0]}, y0={params[1]}, sigma={params[2]}, amplitude={params[3]}")

#     # Normalize the initial guess and bounds
#     norm_initial_guess = [initial_guess[0] / image.shape[1], initial_guess[1] / image.shape[0], initial_guess[2] / max_sigma, initial_guess[3] / alpha]
#     norm_bounds = [(0, 1), (0, 1), (0.01, 2), (0, 2)]

#     result = minimize(objective, norm_initial_guess, bounds=norm_bounds, callback=callback, options={'maxiter': 10000})
#     # result = minimize(objective, initial_guess, bounds=[(0, image.shape[1]), (0, image.shape[0]), (1, 10), (0, np.inf)], callback=callback, options={'maxiter': 10000})

#     # Scale the parameters back to their original range
#     x0, y0, sigma, amplitude = result.x
#     x0 *= image.shape[1]
#     y0 *= image.shape[0]
#     sigma *= max_sigma
#     amplitude *= alpha

#     return sigma  # Return the sigma value
    # return result.x[2]  # Return the sigma value


def gaussian_2d(x, y, x0, y0, sigma, amplitude):
    """2D Gaussian function"""
    return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def plot_fit_comparison(image, fitted_params, title="Fit Comparison"):
    """Plot the original image, fit, and residual side by side"""
    y, x = np.indices(image.shape)
    x0, y0, sigma, amplitude = fitted_params
    fitted = gaussian_2d(x, y, x0, y0, sigma, amplitude)
    residual = image - fitted
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original
    im1 = ax1.imshow(image)
    ax1.set_title('Original')
    plt.colorbar(im1, ax=ax1)
    ax1.plot(x0, y0, 'r+', markersize=10)
    
    # Plot fit
    im2 = ax2.imshow(fitted)
    ax2.set_title(f'Fit (σ={sigma:.2f})')
    plt.colorbar(im2, ax=ax2)
    ax2.plot(x0, y0, 'r+', markersize=10)
    
    # Plot residual
    im3 = ax3.imshow(residual)
    ax3.set_title('Residual')
    plt.colorbar(im3, ax=ax3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def fit_gaussian_2d(image, guess_xy=None, debug=True):
    """
    Fit a 2D Gaussian to an image
    
    Parameters:
    -----------
    image : ndarray
        2D image array
    guess_xy : tuple, optional
        (x, y) initial guess for center position
    debug : bool
        If True, print diagnostic information and show plots
        
    Returns:
    --------
    tuple : (x0, y0, sigma, amplitude)
        Fitted parameters
    """
    y, x = np.indices(image.shape)
    
    # Initial parameter estimation
    if guess_xy is None:
        # Estimate center as center of mass
        total = image.sum()
        if total == 0:
            y0, x0 = image.shape[0]//2, image.shape[1]//2
        else:
            y0 = (y * image).sum() / total
            x0 = (x * image).sum() / total
    else:
        x0, y0 = guess_xy
    
    # Estimate sigma as RMS distance from center, weighted by intensity
    dy = y - y0
    dx = x - x0
    r2 = dx**2 + dy**2
    weighted_r2 = r2 * image
    if weighted_r2.sum() > 0:
        sigma = np.sqrt(weighted_r2.sum() / (2 * image.sum()))
        sigma = np.clip(sigma, 1.0, 4.0)  # Wider initial sigma range
    else:
        sigma = 3.0
    
    # Estimate amplitude from peak value near the center
    y_idx = int(np.round(y0))
    x_idx = int(np.round(x0))
    window = 2
    roi = image[max(0, y_idx-window):min(image.shape[0], y_idx+window+1),
                max(0, x_idx-window):min(image.shape[1], x_idx+window+1)]
    amplitude = roi.max()
    
    initial_guess = (x0, y0, sigma, amplitude)
    
    if debug:
        print(f"Initial estimates:")
        print(f"Center: ({x0:.1f}, {y0:.1f})")
        print(f"Sigma: {sigma:.1f}")
        print(f"Amplitude: {amplitude:.1f}")
    
    def objective(params):
        """Modified objective function with better weighting"""
        x0, y0, sigma, amplitude = params
        model = gaussian_2d(x, y, x0, y0, sigma, amplitude)
        # Weight residuals by sqrt of intensity to handle noise better
        weights = 1.0 / np.sqrt(image + 1)
        return np.sum(((image - model) * weights)**2)
    
    # Set bounds relative to image size and initial guess
    bounds = [
        (max(0, x0-5), min(image.shape[1], x0+5)),  # x0: within ±5 pixels
        (max(0, y0-5), min(image.shape[0], y0+5)),  # y0: within ±5 pixels
        (2.0, 4.0),    # sigma: wider range
        (0.1*amplitude, 5.0*amplitude)  # amplitude: wider range
    ]
    
    iteration_count = 0
    best_objective = float('inf')
    best_params = None
    
    def callback(params):
        nonlocal iteration_count, best_objective, best_params
        current_obj = objective(params)
        if current_obj < best_objective:
            best_objective = current_obj
            best_params = params.copy()
        
        if debug and iteration_count % 5 == 0:
            print(f"Iteration {iteration_count}:")
            print(f"  x0={params[0]:.2f}, y0={params[1]:.2f}, "
                  f"sigma={params[2]:.2f}, amp={params[3]:.2f}")
            print(f"  Objective: {current_obj:.2f}")
        iteration_count += 1
    
    # Try multiple initial sigma values if first fit hits bounds
    sigma_tries = [sigma, sigma*0.7, sigma*1.3]
    best_result = None
    best_obj = float('inf')
    
    for try_sigma in sigma_tries:
        initial_guess = (x0, y0, try_sigma, amplitude)
        result = minimize(
            objective,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 100},
            callback=callback
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result
    
    result = best_result
    
    if debug:
        plot_fit_comparison(image, result.x, 
                          title=f"Final Fit (converged: {result.success})")
    
    return result.x

def process_tiff_images(input_directory):
    sigma_values = []
    total_particles = 0

    files = [f for f in os.listdir(input_directory) if f.endswith('.tiff') or f.endswith('.tif')]
    total_files = len(files)

    for idx, filename in enumerate(files):
        file_path = os.path.join(input_directory, filename)
        with Image.open(file_path) as img:
            if img.mode == 'RGB':
                img = img.convert('L')
            img_array = np.array(img)
            
            tentative_peaks = get_tentative_peaks(img_array, min_distance=1)
            plt.figure()
            plt.imshow(img_array, cmap='gray')
            for idx, peak in enumerate(tentative_peaks[:5]):
                plt.text(peak[1], peak[0], str(idx + 1), color='yellow', fontsize=12)
            plt.title(f'Tentative Peaks for {filename}')
            plt.show(block=False)
            guess_xy = tentative_peaks[0][::-1]
            # Cut the image around the guess_xy, centered, 20x20
            cut_size = 20
            half_size = cut_size // 2
            # guess_xy = (half_size, half_size)
            x_center, y_center = guess_xy
            x_start = max(0, x_center - half_size)
            x_end = min(img_array.shape[1], x_center + half_size)
            y_start = max(0, y_center - half_size)
            y_end = min(img_array.shape[0], y_center + half_size)
            cut_image = img_array[y_start:y_end, x_start:x_end]
            plt.close('all')
            plt.figure()
            plt.imshow(cut_image, cmap='gray')
            plt.show(block=False)
            pass
            
            # Check if the image contains significant signal
            if np.max(cut_image) > 0:
                sigma = fit_gaussian_2d(cut_image, (10,10))
                pass
                if sigma is not None :
                # if sigma is not None and sigma <= 100:
                    sigma_values.append(sigma)
                    total_particles += 1

        # Print progress
        progress = (idx + 1) / total_files * 100
        print(f'\rProcessing file: {filename} ({idx + 1}/{total_files}) - {progress:.2f}% complete - Total particles fitted: {total_particles}', end='')

    print()  # Move to the next line after the loop

    # Plot histogram of sigma values
    plt.hist(sigma_values, bins=30, alpha=0.7, color='blue')
    mean_sigma = np.mean(sigma_values)
    plt.axvline(mean_sigma, color='red', linestyle='dashed', linewidth=2)
    plt.text(mean_sigma, plt.ylim()[1] * 0.9, f'Mean: {mean_sigma:.2f}', color='red', ha='center')
    plt.xlabel('Sigma (width parameter)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Sigma Values')
    plt.show(block=False)
    pass

    return mean_sigma

# Example usage
input_directory = 'particle_images'
# input_directory = 'div_images'
# mean_sigma = process_tiff_images(input_directory)
# print(f'Mean sigma value: {mean_sigma}')


def generate_rgb_image_with_background_and_peaks(output_file="rgb_poisson_255.tiff"):
    width, height = 100, 100
    sigma = 2

    np.random.seed(42)

    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)

    # Create a non-zero background for each channel
    r_img = np.full((height, width), 10, dtype=np.float32) 
    g_img = np.full((height, width), 20, dtype=np.float32) 
    b_img = np.full((height, width), 30, dtype=np.float32)

    peak_info_p1 = {'x': 22.2, 'y': 25.5, 'psf_sigma': sigma, 'prefactor': np.array([222, 333, 255])*15}
    image_p1 = psfconvolution(peak_info_p1, image_width=width)
    r_img += image_p1[0]
    g_img += image_p1[1]
    b_img += image_p1[2]
    
    peak_info_p2 = {'x': 88.8, 'y': 77.7, 'psf_sigma': sigma, 'prefactor': np.array([345, 234, 432])*15}
    image_p2 = psfconvolution(peak_info_p2, image_width=width)
    r_img += image_p2[0]
    g_img += image_p2[1]
    b_img += image_p2[2]

    r_img = np.random.poisson(r_img)
    g_img = np.random.poisson(g_img)
    b_img = np.random.poisson(b_img)
    
    # Stack channels to create an RGB image
    rgb_image = np.stack([r_img, g_img, b_img], axis=-1)
    plt.imshow(rgb_image)
    plt.colorbar()
    plt.show(block=False)
    pass

    # # Normalize to fit within the range [0, 255]
    # rgb_image = np.clip(rgb_image, 0, 255).astype(np.float32)

    # Save the image as a TIFF file
    tifffile.imwrite(output_file, rgb_image)

    # Normalize the RGB image by dividing by 255
    normalized_rgb_image = rgb_image / 255.0

    # Save the normalized image as a new TIFF file
    normalized_output_file = os.path.splitext(output_file)[0] + '_normalized.tiff'
    tifffile.imwrite(normalized_output_file, normalized_rgb_image.astype(np.float32))

    print(f"Normalized image saved to: {normalized_output_file}")
    
    

    # Read the TIFF file using tifffile
    tiff_image = tifffile.imread(output_file)

    # Check the shape and data type of the image
    print(f"Image shape: {tiff_image.shape}, dtype: {tiff_image.dtype}")

    # Display the image
    plt.figure()
    plt.imshow(tiff_image)
    plt.colorbar()
    plt.title("TIFF Image")
    plt.show()
    pass

# Example usage
generate_rgb_image_with_background_and_peaks("rgb_poisson_255.tiff")
pass

def scale_rgb_tiff(input_file):
    """
    Reads an RGB TIFF file, scales each channel by 1000x, and saves the result to a new file.
    
    Parameters:
    -----------
    input_file : str
        Relative file path of the input TIFF file.
    """
    # Read the TIFF file
    input_path = os.path.abspath(input_file)
    output_path = os.path.splitext(input_path)[0] + '_1000x.tiff'

    # Load the image
    with tifffile.TiffFile(input_path) as tif:
        image = tif.asarray()

    # Ensure the image is RGB and values are between 0 and 1
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input TIFF must be an RGB image.")
    if not (0 <= image.min() <= image.max() <= 1):
        raise ValueError("RGB values must be between 0 and 1.")

    # Scale each channel by 1000x
    scaled_image = (image * 1000).astype(np.float32)

    # Save the scaled image to a new TIFF file
    tifffile.imwrite(output_path, scaled_image)

    print(f"Scaled image saved to: {output_path}")


# scale_rgb_tiff('./datasets/rgb/rgb_tiff.tiff')


