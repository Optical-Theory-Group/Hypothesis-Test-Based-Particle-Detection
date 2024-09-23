import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import nmmn.plots
from process_algorithms import integrate_gauss_1d

parula = nmmn.plots.parulacmap()

def psfconvolution(peak_info, image_width=512):
    """returns the pixel values to be added to the image based on psf convolution."""

    integral_x = integrate_gauss_1d(np.arange(image_width), peak_info['x'], peak_info['psf_sd'])
    integral_y = integrate_gauss_1d(np.arange(image_width), peak_info['y'], peak_info['psf_sd'])

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
    rows, cols = image.shape

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

def neighboring_position(positions, distance=.5):
    # Randomly select a position
    base_x, base_y = random.choice(positions)
    angle = np.random.uniform(0, 2*np.pi)
    new_x = base_x + distance * np.cos(angle)
    new_y = base_y + distance * np.sin(angle)
    return new_x, new_y

def test_making_image(image_width=512, 
            background = 500,
            n_particles_or_clusters = 4,
            particle_psf_width = 2,
            particle_intensity_prefactor = 3000,
            particle_intensity_sd = 100,
            clustering = False, 
            cluster_chance = 0.5,
            cluster_size_mean = 2,
            cluster_internal_distance = .5,
            low_freq_background = False,
            low_freq_background_amp = 500,
            low_x_freq = 0.005,
            low_y_freq = 0.01,
            low_phase = 0,
            vignetting = False,
            vignette_width = 128,
            vignette_strength = 0.5,
            vignette_factor_min = 0.5,
            n_dust = 10,
            dust_psf_width_min = 10, 
            dust_psf_width_max = 30, 
            dust_intensity_prefactor = 1000,
            dust_intensity_sd = 200,):

    # Create a background image
    background_image = np.ones((image_width, image_width)) * background

    # Create particle peak info and store them in a list
    peaks_info = []
    for i in range(n_particles_or_clusters):
        x = np.random.randint(-.5, image_width - .5)
        y = np.random.randint(-.5, image_width - .5)
        prefactor = np.random.normal(particle_intensity_prefactor, particle_intensity_sd)
        psf_sd = particle_psf_width
        peaks_info.append({'x': x, 'y': y, 'prefactor': prefactor, 'psf_sd': psf_sd})
        if clustering:
            if np.random.random() < cluster_chance:
                n_cluster_members = int(np.random.poisson(cluster_size_mean))
                positions = [(x, y)]
                for j in range(n_cluster_members):
                    new_x, new_y = neighboring_position(positions, cluster_internal_distance)
                    new_prefactor = np.random.normal(particle_intensity_prefactor, particle_intensity_sd)
                    peaks_info.append({'x': new_x, 'y': new_y, 'prefactor': new_prefactor, 'psf_sd': psf_sd})
                    positions.append((new_x, new_y))

    # Create convolved image with particle locations and intensities
    particles_image = psfconvolution(peaks_info, image_width)

    if low_freq_background:
        # Create low-frequency background image
        lowfreq_background_image = lowfreq_background(image_width, x_freq=low_x_freq, y_freq=low_y_freq, amplitude=low_freq_background_amp, phase=low_phase)
    else:
        lowfreq_background_image = np.zeros((image_width, image_width))
    
    # Create dust peak info and store them in a list 
    peaks_info = []
    for i in range(n_dust):
        x = np.random.randint(-.5, image_width - .5)
        y = np.random.randint(-.5, image_width - .5)
        prefactor = np.random.normal(dust_intensity_prefactor, dust_intensity_sd)
        psf_sd = np.random.random() * (dust_psf_width_max - dust_psf_width_min) + dust_psf_width_min
        peaks_info.append({'x': x, 'y': y, 'prefactor': prefactor, 'psf_sd': psf_sd})

    # Create convolved image with dust locations and intensities
    dust_image = psfconvolution(peaks_info, image_width)

    # Add all the images together
    image_sum = background_image + particles_image + lowfreq_background_image + dust_image

    if vignetting:
        # Vignette effect is a multiplicative effect on the image
        output_image = apply_vignette(image_sum, strength=vignette_strength, vignette_factor_min=vignette_factor_min)
    else:
        output_image = image_sum

    # Apply Poisson noise to the image
    output_image = np.random.poisson(output_image)

    return output_image

# Test the function
# output = test_making_image(
#                     image_width=256, 
#                     background=500,
#                     n_particles_or_clusters=10,
#                     particle_psf_width=2,
#                     particle_intensity_prefactor=10000,
#                     particle_intensity_sd=500,
#                     clustering=True,
#                     cluster_chance=0.2,
#                     cluster_size_mean=2,
#                     cluster_internal_distance=3,
#                     n_dust=2,
#                     dust_psf_width_min=4,
#                     dust_psf_width_max=4,
#                     dust_intensity_prefactor=20000,
#                     dust_intensity_sd=3000,
#                     low_freq_background=False,
#                     low_freq_background_amp=0,
#                     low_y_freq=0.002,
#                     low_x_freq=0.003,
#                     vignetting=True,
#                     vignette_width=300,
#                     vignette_strength=.1,
#                     vignette_factor_min=.8
#                    )
# plt.imshow(output, cmap='gray')
# plt.show()
# pass