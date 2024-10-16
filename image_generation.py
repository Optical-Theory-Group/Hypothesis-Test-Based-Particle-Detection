import numpy as np
import cv2
from process_algorithms import integrate_gauss_1d

def psfconvolution(peak_info, image_width=512):
    """returns the pixel values to be added to the image based on psf convolution."""

    integral_x = integrate_gauss_1d(np.arange(image_width), peak_info['x'], peak_info['psf_sigma'])
    integral_y = integrate_gauss_1d(np.arange(image_width), peak_info['y'], peak_info['psf_sigma'])

    if isinstance(peak_info['prefactor'], (int, float)): # Case grayscale image
        output = np.outer(integral_y, integral_x) * peak_info['prefactor']
    else: # Case RGB image
        output = np.array([np.outer(integral_y, integral_x) * peak_info['prefactor'][i] for i in range(3)])

    return output

def lowfreq_background(image_width, x_freq, y_freq, amplitude=100, phase=0):
    """ Creates a low frequency background signal
    Args:
        image_width (int): width of the image
        x_freq (float): frequency of the cosine wave in x direction
        y_freq (float): frequency of the cosine wave in y direction
        amplitude (float): amplitude of the cosine wave
        angle (float): angle of the cosine wave in radians
    Returns:
        numpy.array(dtype=float): image_width x image_width array of the background signal
    """
    # Create the output image numpy array
    outputimg = np.zeros((image_width, image_width))
    center_x = center_y = image_width / 2
    # Fill the output image numpy array
    for x in range(image_width):
        for y in range(image_width):
            # Calculate the value of the cosine wave at this point
            value = amplitude * (1 + np.cos(2 * np.pi * (x_freq * (x - center_x) + y_freq * (y - center_y) + phase))) / 2
            # Add the value to the output image
            outputimg[y, x] = value

    return outputimg

def apply_vignette(image, strength=0.5, vignette_factor_min=0.5):

    # Assumes `image` is a grayscale image (2D numpy array)
    if len(image.shape) == 2:
        rows, cols = image.shape
    else: # If image is RGB
        raise ValueError("Input image t be a 2D numpy array. RGB images are not supported at this time.")

    # Create a vignette mask using Gaussian kernels
    kernel_x = cv2.getGaussianKernel(cols, cols/4)
    kernel_y = cv2.getGaussianKernel(rows, rows/4)
    kernel = kernel_y * kernel_x.T
    mask = kernel / np.linalg.norm(kernel)
    mask = cv2.GaussianBlur(mask, (0, 0), strength*cols/4)
    # Normalize mask to range from 0.5 to 1
    rescaled_mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask)) * (1 - vignette_factor_min) + vignette_factor_min

    # Apply the mask to the image
    vignetted_image = image * rescaled_mask

    return vignetted_image