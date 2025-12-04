# Hypothesis-Test-Based-Particle-Detection
# -----------------------------------------
#
# This file is part of the project "Hypothesis-Test-Based-Particle-Detection".
# It implements the image generation functions.
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



import numpy as np
from process_algorithms import normal_gaussian_integrated_within_each_pixel
from tqdm import tqdm
import imageio
import json
import os


def psfconvolution(peak_info, image_width=512):
    """
    Returns the pixel values to be added to the image based on psf convolution.

    Args:
        peak_info (dict): Dictionary containing the following keys:
            x (float): x-coordinate of the peak
            y (float): y-coordinate of the peak
            psf_sigma (float): standard deviation of the PSF
            prefactor (float or list): prefactor of the peak
        image_width (int): width of the image

    Returns:
        numpy.array(dtype=float): image_width x image_width array of the pixel values to be added to the image
    """

    psf_factors_for_pixels_in_x = normal_gaussian_integrated_within_each_pixel(np.arange(image_width),
                                                                               peak_info['x'],
                                                                               peak_info['psf_sigma'])
    psf_factors_for_pixels_in_y = normal_gaussian_integrated_within_each_pixel(np.arange(image_width),
                                                                               peak_info['y'],
                                                                               peak_info['psf_sigma'])

    if isinstance(peak_info['prefactor'], (int, float)):  # Case grayscale image
        output = np.outer(psf_factors_for_pixels_in_y, psf_factors_for_pixels_in_x) * peak_info['prefactor']
    else:  # Case RGB image
        output = np.array([np.outer(psf_factors_for_pixels_in_y, psf_factors_for_pixels_in_x)
                           * peak_info['prefactor'][i] for i in range(3)])

    return output


def lowfreq_background(image_width, x_freq, y_freq, amplitude=100, phase=0):
    """
    Creates a low frequency background signal

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
            value = amplitude * (1 + np.cos(2 * np.pi * (x_freq * (x - center_x) +
                                                         y_freq * (y - center_y) + phase))) / 2
            # Add the value to the output image
            outputimg[y, x] = value

    return outputimg


def generate_test_images(image_folder_basename,
                         maximum_number_of_particles,
                         particle_intensity_mean,
                         particle_intensity_sd=0,
                         total_image_count=1,
                         psf_sigma=1,
                         image_side_length=20,
                         background_level=1,
                         generation_random_seed=42,
                         config_content=None,
                         minimum_number_of_particles=0,
                         file_extension='tiff'
                         ):
    """
    Generate test images (16-bit) with random number of particles between minimum_number_of_particles
    and maximum_number_of_particles.

    Parameters:
        image_folder_basename (str): The basename of the folder to store the images.
        maximum_number_of_particles (int): The maximum number of particles to be generated in the image.
        particle_intensity_mean (int or float): The mean intensity of the particles.
        particle_intensity_sd (int or float): The standard deviation of the particle intensity. Default is 0.
        total_image_count (int): The total number of images to be generated.
        psf_sigma (float): The sigma (width) of the point spread function in pixels.
        image_side_length (int): The side length of the image in pixels. This will be both the width and height of the image.
        background_level (int): The background intensity of the image.
        generation_random_seed (int): The random seed for this function.
        config_content (str): The content of the config file. Expected to be a string. Default is None.
        minimum_number_of_particles (int): The minimum number of particles in the image. Default is 0.
        file_extension (str): The format of the image file. Default is 'tiff'.

    Returns:
        str: The path of the folder containing the generated images
    """

    # Load the config file
    if config_content:
        config = json.loads(config_content)
        if 'file_format' in config:
            file_extension = config['file_format']

    # Check if the format is either 'tiff' or 'png'
    if file_extension not in ['tiff', 'png']:
        raise ValueError("File format must be either 'tiff' or 'png'.")

    # Set the random seed
    np.random.seed(generation_random_seed)

    # Take the total images count to be generated and calculate the number of images to be generated
    # for each particle count (particle_count=0, 1, 2, 3, ..., maximum_number_of_particles)
    number_of_types = maximum_number_of_particles - minimum_number_of_particles + 1
    number_of_images_per_type = int(np.ceil(total_image_count / number_of_types))

    # Print the number of images to be generated and folder to store the images.
    print(f'Generating images containing the following number of particles: {minimum_number_of_particles} - {maximum_number_of_particles}. It will produce {number_of_images_per_type} images per each case.')
    number_of_total_images = number_of_images_per_type * number_of_types
    print(f'Total {number_of_total_images} images will be generated.')

    # Address the potential issue of the total number of images being slightly more than the total number
    # of images requested to make the same number of image per for each particle count.
    user_input_total_image_count = total_image_count
    if number_of_total_images > user_input_total_image_count:
        print('-- This may be slightly more than the total number of images requested to make the same number of image per for each particle count.')

    # Create the folder to store the images
    image_folder_path = os.path.join("datasets", f"{image_folder_basename}")
    os.makedirs(image_folder_path, exist_ok=True)
    print(f'Image save destination: ./datasets/{image_folder_basename}')

    # Use the function input parameters - particle_intensity_mean, particle_intensity_sd, and
    # background_level - to determine the color mode of the image (gray or rgb).
    # Determine the color mode of the image (gray or rgb)
    if isinstance(particle_intensity_mean, (int, float)) and isinstance(particle_intensity_sd, (int, float)) and isinstance(background_level, (int, float)): # Case : gray scale
        color_mode = 'gray'  # gray scale
    elif len(particle_intensity_mean) == len(particle_intensity_sd) == len(background_level) == 3:  # Case : rgb
        color_mode = 'rgb'  # Red, Green, Blue
    else:
        raise ValueError("The color mode of the image is not recognized. Please check the following variables: particle_intensity_mean, particle_intensity_sd, and background_level.")

    # Print the color mode of the image
    with tqdm(total=number_of_total_images, desc="Generating Images", unit="image") as pbar:
        for number_of_particles_in_the_image in range(minimum_number_of_particles, maximum_number_of_particles+1):
            for image_index in range(number_of_images_per_type):
                # Initialize the image with the flat background intensity (no noise yet)
                if color_mode == 'gray':
                    image = np.ones((image_side_length, image_side_length), dtype=np.float32) * background_level
                else:  # Case : rgb
                    image = [np.ones((image_side_length, image_side_length), dtype=np.float32) * background_level[i] for i in range(3)]

                for _ in range(number_of_particles_in_the_image):

                    # Randomly draw the position of the particle, avoiding the edges of the image
                    x = np.random.rand() * (image_side_length - psf_sigma * 4) + psf_sigma * 2 - 0.5
                    y = np.random.rand() * (image_side_length - psf_sigma * 4) + psf_sigma * 2 - 0.5

                    # Randomly draw the relative intensity of the particle
                    if color_mode == 'gray':
                        particle_intensity = np.random.normal(particle_intensity_mean, particle_intensity_sd)
                        if particle_intensity < 0:
                            raise ValueError("Randomly drawn particle intensity is less than 0, which is not allowed.")

                        # Create peak info dictionary containing the x, y positions of the particle, the intensity
                        # of the particle, and the sigma (width) of the point spread function
                        peak_info = {'x': x, 'y': y, 'prefactor': particle_intensity, 'psf_sigma': psf_sigma}

                    else:  # Case : rgb
                        particle_intensities = np.array([np.random.normal(particle_intensity_mean,
                                                                          particle_intensity_sd) for i in range(3)])
                        if np.any(particle_intensities < 0):
                            raise ValueError("Randomly drawn particle intensity (at least one of r, g, or b) is less than 0, which is not allowed.")

                        # Create peak info dictionary
                        peak_info = {'x': x, 'y': y, 'prefactor': particle_intensities, 'psf_sigma': psf_sigma}

                    # Add the point spread function of the particle to the image
                    image += psfconvolution(peak_info, image_side_length)

                # Add Poisson noise
                image = np.random.poisson(image)  # Notice the the image type if float.
                img_filename = f"count{number_of_particles_in_the_image}-index{image_index}.{file_extension}"
                if file_extension == 'png' and np.any(image > 65535):
                    print(f"Warning: The pixel value(s) of {img_filename} exceeds 65535. Since png can store max 16-bits, such values will be clipped. This mimics saturation in the camera.")
                    image = np.clip(image, 0, 65535)

                # Adjust the shape of the image to match that of png or tiff
                if image.ndim == 3 and image.shape[0] == 3:  # This is when the image is in C x H x W format
                    image = np.transpose(image, (1, 2, 0))  # Change the shape to H x W x C format

                # Save the image
                img_filepath = os.path.join(image_folder_path, img_filename)
                if file_extension == 'png':
                    imageio.imwrite(img_filepath, image.astype(np.uint16))
                elif file_extension == 'tiff':
                    imageio.imwrite(img_filepath, image.astype(np.float32))

                # Update the progress bar
                pbar.update(1)

    # Save the content of the config file as "config_used.json"
    if config_content is not None:
        config_file_save_path = os.path.join(image_folder_path, 'config_used.json')
        with open(config_file_save_path, 'w') as f:
            json.dump(json.loads(config_content), f, indent=4)

    # Return the path of the folder containing the images
    return image_folder_path


def generate_separation_test_images(image_folder_basename='separation_test',
                                    sep_distance_ratio_to_psf_sigma=3,
                                    total_image_count=20,
                                    amp_to_background_level=5,
                                    psf_sigma=1,
                                    image_side_length=20,
                                    background_level=500,
                                    generation_random_seed=42,
                                    config_content=None,
                                    file_extension='tiff'):
    """ Generate test images (16-bit) with two particles separated by a distance of
        sep_distance_ratio_to_psf_sigma times the psf_sigma.
        *** RGB images are not supported in this function.

    Parameters:
        image_folder_basename (str): The name of the folder to store the images.
        sep_distance_ratio_to_psf_sigma (int or float): The ratio of the separation distance to the psf sigma.
        total_image_count (int): The total number of images to be generated.
        amp_to_background_level (int or float): The amplitude of the particles relative to the background.
        psf_sigma (float): The sigma (width) of the point spread function.
        image_side_length (int): The size of the image. Both width and height are the same.
        background_level (int): The background intensity of the image.
        generation_random_seed (int): The random seed for generating the images.
        config_content (str): The content of the config file.
        file_extension (str): The format of the image file. Default is 'tiff'.

    Returns:
        str: The path of the folder containing the images
    """
    # Load the config file
    if config_content:
        config = json.loads(config_content)
        if 'file_format' in config:
            file_extension = config['file_format']

    # Check if the format is either 'tiff' or 'png'
    if file_extension not in ['tiff', 'png']:
        raise ValueError("Format must be either 'tiff' or 'png'.")

    # Calculate the separation distance in pixels
    separation_distance = sep_distance_ratio_to_psf_sigma * psf_sigma

    # Check if the separation distance is greater than the size of the image
    if separation_distance >= image_side_length - 4 * psf_sigma - 2:
        # 4 * psf_sigma accounts for the required separation of the two particles from the edge
        # of the image and 2 accounts for the maximum random shift of center_x and center_y.
        raise ValueError(f"Separation {separation_distance} must be less than image_side_length - 4 * psf_sigma - 2 to be generally detectable.")

    # Set random seed
    np.random.seed(generation_random_seed)

    # Create the folder to store the images.
    image_folder_path = f"./datasets/{image_folder_basename}"
    os.makedirs(image_folder_path, exist_ok=True)

    # Set strings containing the psf and separation distance for file naming, replacing '.' with '_' to avoid confusing '.' as file extension separator.
    psf_str = f"{psf_sigma:.1f}".replace('.', '_')
    sep_str = f"{sep_distance_ratio_to_psf_sigma:.1f}".replace('.', '_')

    # Print the number of images to be generated and the folder to store the images.
    print(f'Generating {total_image_count} images with psf {psf_sigma} and separation {sep_str} times the psf in folder {image_folder_path}.')

    # Generate images
    with tqdm(total=total_image_count, desc="Generating Images", unit="image") as progress_bar:
        for image_index in range(total_image_count):
            # Print the image index every 20 images
            if image_index % 20 == 0:
                print(f"Generating image index {image_index}", end='\r')

            # Initialize the image with the background intensity
            image = np.ones((image_side_length, image_side_length), dtype=float) * background_level

            # Calculate the particle intensity
            particle_intensity = amp_to_background_level * background_level
            angle = np.random.uniform(0, 2*np.pi)

            # Set the middle position between particle 1 & 2 - Give random offset (-.5, .5) pixels in both x and y to randomize the center position relative to the pixel grid.
            center_x = image_side_length / 2 + np.random.uniform(-.5, .5)
            center_y = image_side_length / 2 + np.random.uniform(-.5, .5)

            # Set the x, y positions of particle 1
            x1 = center_x + separation_distance / 2 * np.cos(angle)
            y1 = center_y + separation_distance / 2 * np.sin(angle)

            # Check if the particle is out of bounds
            if (x1 <= -.5 + 2 * psf_sigma or x1 >= image_side_length - .5 - 2 * psf_sigma):
                raise ValueError(f"Particle 1 is out of bounds: x1={x1}, y1={y1}. The code logic does not allow this to happen. Check the code inside generate_separation_test_images().")

            # Add the point spread function of particle 1 to the image
            peak_info = {'x': x1, 'y': y1, 'prefactor': particle_intensity, 'psf_sigma': psf_sigma}
            image += psfconvolution(peak_info, image_side_length)

            # Set the x, y positions of particle 2
            x2 = center_x - separation_distance / 2 * np.cos(angle)
            y2 = center_y - separation_distance / 2 * np.sin(angle)

            # Check if the particle is out of bounds
            if (y1 <= -.5 + 2 * psf_sigma or y1 >= image_side_length - .5 - 2 * psf_sigma):
                raise ValueError(f"Particle 2 is out of bounds: x2={x2}, y2={y2}. The code logic does not allow this to happen. Check the code inside generate_separation_test_images().")

            # Add the point spread function of particle 2 to the image
            peak_info = {'x': x2, 'y': y2, 'prefactor': particle_intensity, 'psf_sigma': psf_sigma}
            image += psfconvolution(peak_info, image_side_length)

            # Add Poisson noise to the whole image
            image = np.random.poisson(image)  # This is the end of image processing.

            # Save the image
            img_filename = f"count2_psf{psf_str}_sep{sep_str}_index{image_index}.{file_extension}"
            if np.any(image) > 65535:
                print(f"Warning: The pixel value(s) of {img_filename} exceeds 65535. Such values will be clipped. This mimics saturation in the camera.")
                image = np.clip(image, 0, 65535)

            img_filepath = os.path.join(image_folder_path, img_filename)
            imageio.imwrite(img_filepath, image.astype(np.uint16))

            # Update the progress bar
            progress_bar.update(1)

    # Print the completion of image generation
    print(f"Image generation completed (total: {total_image_count}). Images saved to {image_folder_path}.")

    # Save the content of the config file
    if config_content is not None:
        config_file_save_path = os.path.join(image_folder_path, 'config_used.json')
        with open(config_file_save_path, 'w') as f:
            json.dump(json.loads(config_content), f, indent=4)

    return image_folder_path
