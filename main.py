from tqdm import tqdm
from collections import OrderedDict
import imageio
import ast
import matplotlib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pprint
import json
import argparse
from image_generation import psfconvolution
import csv
from PIL import Image as im
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from process_algorithms import generalized_maximum_likelihood_rule
from process_algorithms import merge_coincident_particles
import numpy as np
import glob
import shutil
from sklearn.metrics import confusion_matrix
import os
import concurrent.futures
import sys
from datetime import datetime, timedelta
from cProfile import Profile
from pstats import SortKey, Stats

def generate_test_images(image_folder_namebase, maximum_number_of_particles, particle_intensity_mean, particle_intensity_sd=0, n_total_image_count=1, psf_sigma=1, sz=20, bg=1, 
                         generation_random_seed=42, config_content=None, minimum_number_of_particles=0, file_format='tiff'):
    """ Generate test images (16-bit) with random number of particles between minimum_number_of_particles and maximum_number_of_particles.
    
    Parameters:
        image_folder_namebase (str): The name of the folder to store the images.
        maximum_number_of_particles (int): The maximum number of particles in the image.
        particle_intensity_mean (int or float): The mean intensity of the particles.
        particle_intensity_sd (int or float): The standard deviation of the particle intensity. Default is 0.
        n_total_image_count (int): The total number of images to be generated.
        psf_sigma (float): The sigma (width) of the point spread function.
        sz (int): The size of the image. Both width and height are the same.
        bg (int): The background intensity of the image.
        generation_random_seed (int): The random seed for generating the images.
        config_content (str): The content of the config file.
        minimum_number_of_particles (int): The minimum number of particles in the image. Default is 0.
        file_format (str): The format of the image file. Default is 'tiff'.
        
    Returns:
        str: The path of the folder containing the images
    """

    # Load the config file 
    if config_content:
        config = json.loads(config_content)
        if 'file_format' in config:
            file_format = config['file_format']

    # Check if the format is either 'tiff' or 'png'
    if file_format not in ['tiff', 'png']:
        raise ValueError("File format must be either 'tiff' or 'png'.")

    # Set the random seed
    np.random.seed(generation_random_seed)
    # Set the minimum relative intensity of a particle
    number_of_counts = maximum_number_of_particles - minimum_number_of_particles + 1
    number_of_images_per_count = int(np.ceil(n_total_image_count / number_of_counts))
    
    # Print the number of images to be generated and folder to store the images. 
    print(f'Generating images containing {minimum_number_of_particles} to {maximum_number_of_particles} particles. It will produce {number_of_images_per_count} images per count.')
    num_total_images = number_of_images_per_count * number_of_counts
    print(f'Image generation complete (total: {num_total_images}).')
    if number_of_images_per_count * number_of_counts > n_total_image_count:
        print('This may be slightly more than the total number of images requested to make the same number of image per for each particle count.')
    print(f'Image save destination: ./datasets/{image_folder_namebase}')

    # Create the folder to store the images
    image_folder_path = os.path.join("datasets", f"{image_folder_namebase}") 
    os.makedirs(image_folder_path, exist_ok=True)

    # Determine the color mode of the image (gray or rgb)
    color_mode = ''
    if isinstance(particle_intensity_mean, (int, float)): # Case : gray scale
        color_mode = 'gray'
    elif len(particle_intensity_mean) == len(particle_intensity_sd) == len(bg) == 3: # Case : rgb
        color_mode = 'rgb' 
    else:
        raise ValueError("The color mode of the image is not recognized. Please check the following variables: particle_intensity_mean, particle_intensity_sd, and bg.")

    with tqdm(total=num_total_images, desc="Generating Images", unit="image") as pbar:
        for n_particles in range(minimum_number_of_particles, maximum_number_of_particles+1):
            for img_idx in range(number_of_images_per_count):
                # Initialize the image and the chosen mean intensity(s)
                if color_mode == 'gray':
                    image = np.ones((sz, sz), dtype=float) * bg
                else:   
                    image = [np.ones((sz, sz), dtype=float) * bg[i] for i in range(3)]

                for _ in range(n_particles):
                    # Randomly draw the position of the particle, avoiding the edges of the image
                    x = np.random.rand() * (sz - psf_sigma * 4) + psf_sigma * 2 - 0.5
                    y = np.random.rand() * (sz - psf_sigma * 4) + psf_sigma * 2 - 0.5

                    # Randomly draw the relative intensity of the particle (mean: 1, std: amp_sd)
                    if color_mode == 'gray':
                        particle_intensity = np.random.normal(particle_intensity_mean, particle_intensity_sd)
                        if particle_intensity < 0:
                            raise ValueError("Randomly drawn particle intensity is less than 0, which is not allowed.")

                        # Create peak info dictionary
                        peak_info = {'x': x, 'y': y, 'prefactor': particle_intensity, 'psf_sigma': psf_sigma}

                    else: # Case : rgb
                        particle_intensities = np.array([np.random.normal(particle_intensity_mean, particle_intensity_sd) for i in range(3)])
                        if np.any(particle_intensities < 0):
                            raise ValueError("Randomly drawn particle intensity (at least one of r, g, or b) is less than 0, which is not allowed.")

                        # Create peak info dictionary
                        peak_info = {'x': x, 'y': y, 'prefactor': particle_intensities, 'psf_sigma': psf_sigma}

                    # Add the point spread function of the particle to the image
                    image += psfconvolution(peak_info, sz)

                # Add Poisson noise
                image = np.random.poisson(image).astype(np.uint16) # This is the end of image processing.

                # Adjust the shape of the image to match that of png or tiff
                if image.ndim == 3 and image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))

                # Save the image
                img_filename = f"count{n_particles}-index{img_idx}.{file_format}"
                img_filepath = os.path.join(image_folder_path, img_filename)
                imageio.imwrite(img_filepath, image)

                # Update the progress bar
                pbar.update(1)
    
    # Save the content of the config file
    if config_content is not None:
        config_file_save_path = os.path.join(image_folder_path, 'config_used.json')
        with open(config_file_save_path, 'w') as f:
            json.dump(json.loads(config_content), f, indent=4)

    # Return the path of the folder containing the images
    return image_folder_path

def generate_separation_test_images(image_folder_namebase='separation_test', sep_distance_ratio_to_psf_sigma=3, n_total_image_count=20, amp_to_bg=5, psf_sigma=1, 
                                    sz=20, bg=500, generation_random_seed=42, config_content=None, file_format='tiff'):
    """ Generate test images (16-bit) with two particles separated by a distance of sep_distance_ratio_to_psf_sigma times the psf sigma.
        RGB images are not supported in this function.
    
    Parameters:
        image_folder_namebase (str): The name of the folder to store the images.
        sep_distance_ratio_to_psf_sigma (int or float): The ratio of the separation distance to the psf sigma.
        n_total_image_count (int): The total number of images to be generated.
        amp_to_bg (int or float): The amplitude of the particles relative to the background.
        psf_sigma (float): The sigma (width) of the point spread function.
        sz (int): The size of the image. Both width and height are the same.
        bg (int): The background intensity of the image.
        generation_random_seed (int): The random seed for generating the images.
        config_content (str): The content of the config file.
        file_format (str): The format of the image file. Default is 'tiff'.
        
    Returns:
        str: The path of the folder containing the images
    """                            
    # Load the config file
    if config_content:
        config = json.loads(config_content)
        if 'file_format' in config:
            file_format = config['file_format']

    # Check if the format is either 'tiff' or 'png'
    if file_format not in ['tiff', 'png']:
        raise ValueError("Format must be either 'tiff' or 'png'.")

    # Calculate the separation distance in pixels
    separation_distance = sep_distance_ratio_to_psf_sigma * psf_sigma

    # Check if the separation distance is greater than the size of the image
    if separation_distance >= sz - 4 * psf_sigma - 2: 
        # 4 * psf_sigma accounts for the required separation of the two particles from the edge of
        # the image and 2 accounts for the maximum random shift of center_x and center_y.

        raise ValueError(f"Separation {separation_distance} must be less than sz - 4 * psf_sigma - 2 to be generally detectable.")

    # Set random seed
    np.random.seed(generation_random_seed)

    # Create the folder to store the images. 
    image_folder_path = f"./datasets/{image_folder_namebase}"
    os.makedirs(image_folder_path, exist_ok=True)
    
    # Set strings containing the psf and separation distance for file naming, replacing '.' with '_' to avoid confusing '.' as file extension separator.
    psf_str = f"{psf_sigma:.1f}".replace('.', '_')
    sep_str = f"{sep_distance_ratio_to_psf_sigma:.1f}".replace('.', '_')

    # Print the number of images to be generated and the folder to store the images.
    print(f'Generating {n_total_image_count} images with psf {psf_sigma} and separation {sep_str} times the psf in folder {image_folder_path}.')

    # Generate images

    with tqdm(total=n_total_image_count, desc="Generating Images", unit="image") as pbar:
        for img_idx in range(n_total_image_count):

            # Print the image index every 20 images
            if img_idx % 20 == 0:
                print(f"Generating image index {img_idx}", end='\r')

            # Initialize the image with the background intensity
            image = np.ones((sz, sz), dtype=float) * bg

            # Calculate the particle intensity
            particle_intensity = amp_to_bg * bg
            angle = np.random.uniform(0, 2*np.pi)

            # Set the middle position between particle 1 & 2 - Give random offset (-.5, .5) pixels in both x and y to randomize the center position relative to the pixel grid.
            center_x = sz / 2 + np.random.uniform(-.5, .5)
            center_y = sz / 2 + np.random.uniform(-.5, .5)

            # Set the x, y positions of particle 1
            x1 = center_x + separation_distance / 2 * np.cos(angle)
            y1 = center_y + separation_distance / 2 * np.sin(angle)

            # Check if the particle is out of bounds
            if (x1 <= -.5 + 2 * psf_sigma or x1 >= sz - .5 - 2 * psf_sigma): 
                raise ValueError(f"Particle 1 is out of bounds: x1={x1}, y1={y1}. The code logic does not allow this to happen. Check the code inside generate_separation_test_images().")

            # Add the point spread function of particle 1 to the image
            peak_info = {'x': x1, 'y': y1, 'prefactor': particle_intensity, 'psf_sigma': psf_sigma}
            image += psfconvolution(peak_info, sz)

            # Set the x, y positions of particle 2
            x2 = center_x - separation_distance / 2 * np.cos(angle)
            y2 = center_y - separation_distance / 2 * np.sin(angle)

            # Check if the particle is out of bounds
            if (y1 <= -.5 + 2 * psf_sigma or y1 >= sz - .5 - 2 * psf_sigma):
                raise ValueError(f"Particle 2 is out of bounds: x2={x2}, y2={y2}. The code logic does not allow this to happen. Check the code inside generate_separation_test_images().")
            
            # Add the point spread function of particle 2 to the image
            peak_info = {'x': x2, 'y': y2, 'prefactor': particle_intensity, 'psf_sigma': psf_sigma}
            image += psfconvolution(peak_info, sz)

            # Add Poisson noise to the whole image
            image = np.random.poisson(image).astype(np.uint16) # This is the end of image processing.

            # Save the image
            img_filename = f"count2_psf{psf_str}_sep{sep_str}_index{img_idx}.{file_format}"
            img_filepath = os.path.join(image_folder_path, img_filename)
            imageio.imwrite(img_filepath, image)

            # Update the progress bar
            pbar.update(1)

    # Print the completion of image generation
    print(f"Image generation completed (total: {n_total_image_count}). Images saved to {image_folder_path}.")

    # Save the content of the config file
    if config_content is not None:
        config_file_save_path = os.path.join(image_folder_path, 'config_used.json')
        with open(config_file_save_path, 'w') as f:
            json.dump(json.loads(config_content), f, indent=4)

    return image_folder_path


def report_progress(progresscount, totalrealisations, starttime=None, statusmsg=''):
    """ Calls updated_progress() to print updated progress bar and returns the updated progress count.

    Parameters:
        progresscount (int): Current counter recording progress.
        totalrealisations (int): Total number of realisations to be calculation
        starttime (datetime): Time at which simulation was started
    Return:
        progresscount (int): Input value incremented by 1.
    """
    # update progress trackers and inform user if they have so requested
    progresscount += 1
    runtime = datetime.now() - starttime  
    runtime -= timedelta(microseconds=runtime.microseconds)
    runtimesecs = runtime.total_seconds() if runtime.total_seconds() > 0 else .1  #
    remaintime = (runtime / progresscount) * (totalrealisations - progresscount)
    remaintime = remaintime - timedelta(microseconds=remaintime.microseconds)

    strmsg = '{}/{}' \
            ' in : {} ({:.4f}/s  Remaining time estimate: {}). {}'.format(progresscount, totalrealisations,
                                                    runtime, progresscount / runtimesecs, remaintime, statusmsg)

    update_progress(progresscount / totalrealisations, strmsg)

    return progresscount

def update_progress(progress, status='', barlength=20):
    """ Prints a progress bar to console

    Parameters:
        progress (float): Variable ranging from 0 to 1 indicating fractional progress.
        status (TYPE, optional): Status text to suffix progress bar. The default is ''.
        barlength (str, optional): Controls width of progress bar in console. The default is 20.

    Returns nothing.
    """
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = 'error: progress var must be float\r\n'
    if progress < 0:
        progress = 0
        status = 'Halt...\r\n'
    if progress >= 1:
        progress = 1
        status += ' Done.\r\n'
    block = int(round(barlength * progress))
    text = '\rPercent: [{0}] {1:.2f}% {2}'.format('#' * block + '-' * (barlength - block), progress * 100, status)
    try:
        _, erase_length = os.get_terminal_size(0)  # get the width of the terminal
    except OSError:
        erase_length = len(text) + 10
    clear_line = '\r' + ' ' * erase_length + '\r'
    sys.stdout.write(clear_line + text)
    sys.stdout.flush()

def analyze_whole_folder(image_folder_namebase, code_version_date, use_exit_condi=False, last_h_index=7, psf_sigma=1.39, analysis_rand_seed=0, config_content=None, parallel=False, display_xi_graph=False, timeout_per_image=120):
    '''Analyzes all the images in the dataset folder.

    Parameters:
        image_folder_namebase (str): The name of the folder containing the images.
        code_version_date (str): The version of the code.
        use_exit_condi (bool): Whether to use the exit condition.
        last_h_index (int): The last index of the hypothesis.
        psf_sigma (float): The sigma of the point spread function.
        analysis_rand_seed (int): The random seed for the analysis.
        config_content (str): The content of the config file.
        parallel (bool): Whether to analyze the images in parallel.
        display_xi_graph (bool): Whether to display the xi graph.
        timeout_per_image (int): The maximum time allowed for processing each image. (sec)

    Returns:
        analyses_folder (str): The path of the folder containing the analyses outputs.
    '''

    # Set random seed
    np.random.seed(analysis_rand_seed)

    # Get the list of image files in the folder
    images_folder = os.path.join('./datasets', image_folder_namebase)
    original_images_folder = images_folder

    # Print the folder being analyzed
    print(f"Looking into the folder {images_folder}")

    # Check if the folder exists
    if not os.path.exists(images_folder):
        # If the folder does not exist, print a message and find another folder that starts with image_folder_namebase
        print(f"Folder {images_folder} does not exist.")
        base_dir = './datasets'
        matching_folders = [f for f in os.listdir(base_dir) if f.startswith(image_folder_namebase)]

        # Still check whether there are folders starting with image_folder_namebase. If then, use the first one
        if matching_folders:
            print(f"Found folders starting with '{image_folder_namebase}': {matching_folders}")
            images_folder = os.path.join(base_dir, matching_folders[0])

            # Print a message to inform the user that the program is working with the folder that starts with image_folder_namebase
            print(f"Note: The program is working with {images_folder} which is close to the user input {original_images_folder}. **")
        else:
            raise ValueError(f"No folder starting with '{image_folder_namebase}' found in '{base_dir}'.")

    # Read all png and tiff files
    image_files = glob.glob(os.path.join(images_folder, '*.png')) + glob.glob(os.path.join(images_folder, '*.tiff'))

    # If there are no images in the folder, raise an error
    if len(image_files) == 0:
        raise ValueError("There are no images in this folder.")

    # Print the number of images loaded
    print(f"Images loaded (total of {len(image_files)}):")

    # Create a folder to store the analysis outputs
    analyses_folder = os.path.join('./analyses', image_folder_namebase + '_code_ver' + code_version_date)
    os.makedirs(analyses_folder, exist_ok=True)

    # Save the content of the config file
    if config_content is not None:
        config_file_save_path = os.path.join(analyses_folder, f'{image_folder_namebase}_code_ver{code_version_date}_config_used.json')
        with open(config_file_save_path, 'w') as f:
            json.dump(json.loads(config_content), f, indent=4)

    # Prepare the label (=actual count) prediction (=estimated count) log file
    label_prediction_log_file_path = os.path.join(analyses_folder, f'{image_folder_namebase}_code_ver{code_version_date}_label_prediction_log.csv')

    # Create the "analyses" folder if it doesn't exist
    if not os.path.exists('./analyses'):
        os.makedirs('./analyses')

    # Mark the start time and print a message indicating the beginning of the image analysis
    starttime = datetime.now()
    print('Beginning image analysis...')

    # TESTING - MAKE SURE TO REMOVE after 10/31/2024 ---- #
    TESTING = False
    image_rand_seeds = list(range(60000))
    np.random.shuffle(image_rand_seeds)
    filename = image_files[0]
    if TESTING:
        progress = 0
        for analysis_rand_seed_per_image in image_rand_seeds:
            analysis_result = analyze_image(filename, psf_sigma, last_h_index, analysis_rand_seed_per_image, analyses_folder, display_xi_graph=display_xi_graph, use_exit_condi=use_exit_condi)
            progress += 1
            if progress % 500 == 0:
                print(f"Progress: {progress}/{len(image_files)}")
    pass


    # --------------------------------------------------- #
    

    # Create a list of random seeds for each image
    image_rand_seeds = list(range(len(image_files)))
    np.random.shuffle(image_rand_seeds)

    with open(label_prediction_log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Input Image File', 'Actual Particle Count', 'Estimated Particle Count', "Determined Particle Intensities"])

    # Check if the analysis is to be done in parallel (or sequentially).
    if parallel:
        print("Analyzing images in parallel...")

        # Analyze the images in parallel using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Create a list of futures for each image
            futures = [executor.submit(analyze_image, filename, psf_sigma, last_h_index, analysis_rand_seed_per_image, analyses_folder, use_exit_condi=use_exit_condi )
                        for analysis_rand_seed_per_image, filename in zip(image_rand_seeds, image_files)]
            # Initialize the progress counter
            progress = 0
            # Write the results to the main log file
                # Iterate over the futures that are completed. 
                # for cfresult in concurrent.futures.as_completed(futures):
            for future, future_filename in zip(futures, image_files):
                try:
                    cfresult = future.result(timeout=timeout_per_image)

                    # analysis_result = cfresult.result(timeout=timeout_per_image)
                    analysis_result = cfresult

                    # Extract the results from the analysis result
                    actual_num_particles = analysis_result['actual_num_particles']
                    estimated_num_particles = analysis_result['estimated_num_particles']
                    input_image_file = analysis_result['image_filename']
                    determined_particle_intensities = analysis_result['determined_particle_intensities']

                    # Write the results to the label_prediction log file
                    with open(label_prediction_log_file_path, 'a', newline='') as f:    
                        writer = csv.writer(f)
                        writer.writerow([input_image_file, actual_num_particles, estimated_num_particles, determined_particle_intensities])

                    # Set status message on whether the analysis overestimated, underestimated, or correctly estimated the number of particles
                    if actual_num_particles == estimated_num_particles:
                        statusmsg = f'\"{input_image_file}\" - Actual Count {actual_num_particles} == Estimated {estimated_num_particles}'
                    elif actual_num_particles > estimated_num_particles:
                        statusmsg = f'\"{input_image_file}\" - Actual Count: {actual_num_particles} > Estimated {estimated_num_particles}'
                    else:
                        statusmsg = f'\"{input_image_file}\" - Actual Count {actual_num_particles} < Estimated {estimated_num_particles}'

                except concurrent.futures.TimeoutError:
                    print(f"Task exceeded the maximum allowed time of {timeout_per_image} seconds and was cancelled. File: {future_filename} ")
                    # cfresult.cancel()
                    statusmsg = f'Task cancelled due to timeout. File: {future_filename} '
                except Exception as e:
                    print(f"Error in cfresult.result(): {e} File: {future_filename} ")
                    statusmsg = f'Error: {e}'

                # Report the progress
                report_progress(progress, len(image_files), starttime, statusmsg)

                # Increment the progress counter
                progress += 1

    else: # If the analysis is to be done sequentially
        print("Analyzing images sequentially...")

        # Initialize the progress counter
        progress = 0

        # Iterate over the images
        for analysis_rand_seed_per_image, filename in zip(image_rand_seeds, image_files):
            # Analyze the image
            try:
                analysis_result = analyze_image(filename, psf_sigma, last_h_index, analysis_rand_seed_per_image, analyses_folder, display_xi_graph=display_xi_graph, use_exit_condi=use_exit_condi)

                # Extract the results from the analysis result
                actual_num_particles = analysis_result['actual_num_particles']
                estimated_num_particles = analysis_result['estimated_num_particles']
                input_image_file = analysis_result['image_filename']
                determined_particle_intensities = analysis_result['determined_particle_intensities']

                # Write the results to the label_prediction log file
                with open(label_prediction_log_file_path, 'a', newline='') as f: 
                    writer = csv.writer(f)
                    writer.writerow([input_image_file, actual_num_particles, estimated_num_particles, determined_particle_intensities])

                # Set status message on whether the analysis overestimated, underestimated, or correctly estimated the number of particles
                if actual_num_particles == estimated_num_particles:
                    statusmsg = f'\"{input_image_file}\" - Actual Count {actual_num_particles} == Estimated {estimated_num_particles}\n'
                elif actual_num_particles > estimated_num_particles:
                    statusmsg = f'\"{input_image_file}\" - Actual Count: {actual_num_particles} > Estimated {estimated_num_particles}\n'
                else:
                    statusmsg = f'\"{input_image_file}\" - Actual Count {actual_num_particles} < Estimated {estimated_num_particles}\n'

            except:
                print(f"Task exceeded the maximum allowed time of {timeout_per_image} seconds and was cancelled. File: {filename} ")
                statusmsg = f'Error: {e} File: {filename} '

            # Report the progress
            report_progress(progress, len(image_files), starttime, statusmsg)
            # Increment the progress counter
            progress += 1
            
    return analyses_folder  # Return the path of the folder containing the analyses outputs

def analyze_image(image_filename, psf_sigma, last_h_index, analysis_rand_seed_per_image, analyses_folder, display_fit_results=False, display_xi_graph=False, use_exit_condi=False, tile_width=40, tile_stride=30, tiling_width_threshold=160):
    """ Analyze an image using the generalized maximum likelihood rule.
    
    Parameters:
        image_filename (str): The name of the image file.
        psf_sigma (float): The sigma of the point spread function.
        last_h_index (int): The last index of the hypothesis.
        analysis_rand_seed_per_image (int): The random seed for the analysis of the image.
        analyses_folder (str): The path of the folder containing the analyses outputs.
        display_fit_results (bool): Whether to display the fit results. Default is False.
        display_xi_graph (bool): Whether to display the xi graph. Default is False.
        use_exit_condi (bool): Whether to use the exit condition. Default is False.
        tile_width (int): The width of the tile. Default is 40. Tiling only occurs if the image is larger than the tile size.
        tile_stride (int): The stride of the tile. Default is 30. This is the distance between adjacent tiles in pixels. It is less than the tile width to ensure overlap between adjacent tiles.
        
    Returns:
        image_analysis_results (dict) or tile_combined_results (dict): The results of the image analysis or, if the image is too big to analyze at once, the combined results of the tiles. 
        (Currently, the latter is not implemented and the function returns None.)
    """
    # Print the name of the image file
    image = np.array(im.open(image_filename))

    # Extract the number of particles from image_filename
    basename = os.path.basename(image_filename)
    count_part = basename.split('-')[0]
    foldername = os.path.basename(os.path.dirname(image_filename))

    # If it is a separation test, set the number of particles to 2
    if count_part.startswith("separation") or os.path.basename(foldername).startswith("separation"):
        num_particles = 2
    else:
        num_particles = count_part.split('count')[1] # Get the part right after 'count'
        num_particles = num_particles.split('_')[0] # Get the part before '_'

    # Convert the number of particles (str) to an integer
    actual_num_particles = int(num_particles)

    # Get the size of the image (width and height are both sz)
    sz = image.shape[0]

    # If the image is smaller than the tile size, then process the whole image (no need to divide into tiles)
    if sz < tiling_width_threshold: 

        # Call the generalized_maximum_likelihood_rule (GMLR) function to analyze the image
        estimated_num_particles, fit_results, test_metrics = generalized_maximum_likelihood_rule(roi_image=image, psf_sigma=psf_sigma, 
                                                                last_h_index=last_h_index, random_seed=analysis_rand_seed_per_image, display_fit_results=display_fit_results, 
                                                                display_xi_graph=display_xi_graph, use_exit_condi=use_exit_condi) 


        # Extract xi, lli, and penalty from test_metrics
        xi = test_metrics['xi']
        lli = test_metrics['lli']
        penalty = test_metrics['penalty']
        fisher_info = test_metrics['fisher_info']
        fit_parameters = [result['theta'] for result in fit_results]

        # Create a list of tuples containing hypothesis_index, xi, lli, and penalty
        file_h_info = [f"{image_filename} (h{h_index})" for h_index in range(len(xi))]
        true_counts = [actual_num_particles for _ in range(len(xi))]
        h_numbers = [h_index for h_index in range(len(xi))]
        selected_bools = [1 if estimated_num_particles == h_index else 0 for h_index in range(len(xi))]

        # Extract the determined particle intensities (to see the particle intensity distribution)
        determined_particle_intensities = []
        if estimated_num_particles > 0:
            for i in range(1, estimated_num_particles + 1):
                determined_particle_intensities.append(fit_parameters[estimated_num_particles][i][0])
        determined_particle_intensities

        # Combine variables into list named metric_data 
        metric_data = list(zip(file_h_info, true_counts, h_numbers, selected_bools, xi, lli, penalty, fisher_info, fit_parameters))

        # Save the results to a CSV file ending with '_analysis_log.csv'
        image_analysis_log_filename = f"{analyses_folder}/image_log/{os.path.splitext(os.path.basename(image_filename))[0]}_analysis_log.csv"

        os.makedirs(os.path.dirname(image_analysis_log_filename), exist_ok=True)

        with open(image_analysis_log_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow(['image_filename (h number)', 'selected?', 'xi', 'lli', 'penalty', 'fisher_info', 'fit_parameters'])
            writer.writerow(['image_filename (h number)', 'true_count', 'h number', 'selected?', 'xi', 'lli', 'penalty', 'fisher_info', 'fit_parameters'])
            writer.writerows(metric_data)

        image_analysis_results = {
                                'actual_num_particles': actual_num_particles,
                                'estimated_num_particles': estimated_num_particles,
                                'image_filename': image_filename,
                                'determined_particle_intensities': determined_particle_intensities,
                                'metric_data': metric_data
                                }

        # Return the results of the image analysis 
        return image_analysis_results

    else:
        # Divide the image into tiles, following the tiling_stride.
        tile_sz = (tile_width, tile_width)
        x_low_end_list = [0]
        y_low_end_list = [0]
        while x_low_end_list[-1] + tile_width < sz:
            x_low_end_list.append(x_low_end_list[-1] + tile_stride)
        while y_low_end_list[-1] + tile_width < sz:
            y_low_end_list.append(y_low_end_list[-1] + tile_stride)
        n_x, n_y = 1, 1
        while n_x * tile_stride + tile_width < sz:
            n_x += 1
        while n_y * tile_stride + tile_width < sz:
            n_y += 1
        print(f"{image_filename} is divided into {(n_x + 1) * (n_y + 1)} tiles.")

        # Create a dictionary to store tile information
        tile_dicts_array = np.zeros((n_y + 1, n_x + 1), dtype=object)
        y_low_end_list = [tile_stride * (n) for n in range(n_y + 1)]
        x_low_end_list = [tile_stride * (n) for n in range(n_x + 1)]
        img_height, img_width = image.shape
        for y_index, y_low_end in enumerate(y_low_end_list):
            y_high_end = min(y_low_end + tile_sz[0], img_height)
            for x_index, x_low_end in enumerate(x_low_end_list):
                x_high_end = min(x_low_end + tile_sz[1], img_width)
                tile_dicts_array[x_index][y_index] = {'x_low_end': x_low_end, 'y_low_end': y_low_end, 'image_slice': image[y_low_end:y_high_end, x_low_end:x_high_end], 'particle_locations': []}

        for tile_dict in tile_dicts_array.flatten():
            # Call generalized_maximum_likelihood_rule for each tile
            est_num_particle_tile, fit_results, test_metrics = generalized_maximum_likelihood_rule(tile_dict['image_slice'], psf_sigma, last_h_index, analysis_rand_seed_per_image, display_xi_graph=display_xi_graph, use_exit_condi=True)

            # Use the estimated number of particles to see the fit under the corresponding hypothesis
            chosen_fit = fit_results[est_num_particle_tile]

            # Extract the particle locations from the chosen fit and store them in the tile_dict
            particle_locations = []
            for particle_index in range(1, est_num_particle_tile + 1):
                loc = chosen_fit['theta'][particle_index][1:3]
                particle_locations.append(loc)
            tile_dict['particle_locations'] = particle_locations

            # Set the log file name for the current tile
            tilename = f"tile_x{tile_dict['x_low_end']}-{min(tile_dict['x_low_end'] + tile_width, img_width)}_y{tile_dict['y_low_end']}-{min(tile_dict['y_low_end'] + tile_width, img_height)}"
            print(f"Processing tile {tilename} in image {image_filename}", end='\r')

            image_filename_base = os.path.basename(image_filename).split('.')[0]
            tile_analysis_log_filename = f"{analyses_folder}/image_log/{image_filename_base}_{tilename}_analysis_log.csv"

            # Extract xi, lli, and penalty from test_metrics and fit_parameters from fit_results for this tile
            xi = test_metrics['xi']
            lli = test_metrics['lli']
            penalty = test_metrics['penalty']
            fisher_info = test_metrics['fisher_info']
            fit_parameters = [result['theta'] for result in fit_results]

            # Create a list of tuples containing hypothesis_index, xi, lli, and penalty for this tile
            file_tile_h_info = [f"{image_filename} {tilename} (h{h_index})" for h_index in range(len(xi))]

            # Extract the determined particle intensities (to see the particle intensity distribution)
            h_numbers = [h_index for h_index in range(len(xi))]
            selected_bools = [1 if est_num_particle_tile == h_index else 0 for h_index in range(len(xi))]
            determined_particle_intensities = []
            if est_num_particle_tile > 0:
                for i in range(1, est_num_particle_tile + 1):
                    determined_particle_intensities.append(fit_parameters[est_num_particle_tile][i][0])
            determined_particle_intensities

            # Combine variables into list named file_tile_metric_data
            file_tile_metric_data = list(zip(file_tile_h_info, h_numbers, selected_bools, xi, lli, penalty, fisher_info, fit_parameters))

            # Write the data to the tile analysis log (CSV) file
            os.makedirs(os.path.dirname(tile_analysis_log_filename), exist_ok=True)
            with open(tile_analysis_log_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['image_filename (h number)', 'h number', 'selected?', 'xi', 'lli', 'penalty', 'fisher_info', 'fit_parameters'])
                writer.writerows(file_tile_metric_data)
                pass

        merge_coincident_particles(image, tile_dicts_array, psf_sigma)

        # Todo: Assign the tile_combined_result as below

        # tile_combined_result = {
        #                         'actual_num_particles': actual_num_particles,
        #                         'estimated_num_particles': estimated_num_particles,
        #                         'image_filename': image_filename,
        #                         'determined_particle_intensities': determined_particle_intensities,
        #                         'metric_data': None
        #                         }

        ## 'metric_data' being None will be used to indicate that the analysis was done on tiles.

        tile_combined_result = None

        return tile_combined_result

def generate_intensity_histogram(label_pred_log_file_path, image_folder_namebase, code_version_date, display=False, savefig=True):
    """ Generate the intensity histogram of the determined particle intensities.
    
    Parameters:
        label_pred_log_file_path (str): The path of the label prediction log file.
        image_folder_namebase (str): The name of the folder containing the images.
        code_version_date (str): The version of the code.
        display (bool): Whether to display the histogram. Default is False.
        savefig (bool): Whether to save the histogram as a PNG file. Default is True.
        
    Returns:
        None
    """

    # Read the CSV file
    df = pd.read_csv(label_pred_log_file_path)
    try:
        intensities = df["Determined Particle Intensities"]
    except KeyError:
        raise KeyError("The column name 'Determined Particle Intensities' is not found in the CSV file.")

    # Convert the string of intensities to a list of floats
    all_intensities = []
    for entry in intensities:
        all_intensities.extend(ast.literal_eval(entry))

    # Generate intensity histogram
    _, ax = plt.subplots()
    ax.hist(all_intensities, bins=20)
    ax.set_xlabel('Particle Intensity')
    ax.set_ylabel('Frequency')
    ax.set_title('Intensity Histogram')
    if display:
        plt.show(block=False)
    if savefig:
        png_file_path = os.path.dirname(label_pred_log_file_path)
        png_file_name = f'/{image_folder_namebase}_code_ver{code_version_date}_particle_intensities_hist.png'
        png_file_path += png_file_name
        plt.savefig(png_file_path, dpi=300)

def generate_confusion_matrix(label_pred_log_file_path, image_folder_namebase, code_version_date, display=False, savefig=True):
    """ Generate the confusion matrix and calculate the metrics.
    
    Parameters:
        label_pred_log_file_path (str): The path of the label prediction log file.
        image_folder_namebase (str): The name of the folder containing the images.
        code_version_date (str): The version of the code.
        display (bool): Whether to display the confusion matrix. Default is False.
        savefig (bool): Whether to save the confusion matrix as a PNG file. Default is True.
        
    Returns:
        None
    """
    # Read the CSV file
    df = pd.read_csv(label_pred_log_file_path)
    if df.empty:
        raise ValueError("The CSV file is empty. No data to process.")
    # Extract the actual and estimated particle numbers
    try:
        actual = df['Actual Particle Count']
    except KeyError:
        try:
            actual = df['Actual Particle Number'] 
        except KeyError:
            raise KeyError("The column name 'Actual Particle Count' or 'Actual Particle Number' is not found in the CSV file.")
    try:
        estimated = df['Estimated Particle Count']
    except KeyError:
        try: 
            estimated = df['Estimated Particle Number']
        except KeyError:
            raise KeyError("The column name 'Estimated Particle Count' or 'Estimated Particle Number' is not found in the CSV file.")
    
    # Generate the confusion matrix
    matrix = confusion_matrix(actual, estimated)
    
    # Save the confusion matrix as a CSV file ending with '_confusion_mat.csv'
    matrix_df = pd.DataFrame(matrix)
    csv_file_path = os.path.dirname(label_pred_log_file_path)
    csv_file_name = f'/{image_folder_namebase}_code_ver{code_version_date}_confusion_mat.csv'
    csv_file_path += csv_file_name
    matrix_df.to_csv(csv_file_path, index=False)

    # Calculate the metrics
    row_sums = matrix.sum(axis=1)
    actual_counts = np.array([0, 1, 2, 3, 4, 5])
    estimated_counts = np.arange(matrix.shape[1])
    correct_counts_per_row = np.diag(matrix)
    false_counts = matrix.sum() - correct_counts_per_row.sum()
    accuracy = correct_counts_per_row.sum() / (correct_counts_per_row.sum() + false_counts)
    overestimation_rate = np.triu(matrix, k=1).sum() / matrix.sum()
    underestimation_rate = np.tril(matrix, k=-1).sum() / matrix.sum()
    miss_by_one_rate = (np.diag(matrix, k=1).sum() + np.diag(matrix, k=-1).sum() + correct_counts_per_row.sum()) / matrix.sum()

    # Generate repeated actual and estimated counts
    actual_repeats = []
    estimated_repeats = []

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            actual_repeats.extend([actual_counts[i]] * int(matrix[i, j]))
            estimated_repeats.extend([estimated_counts[j]] * int(matrix[i, j]))

    # Convert to numpy arrays
    actual_repeats = np.array(actual_repeats)
    estimated_repeats = np.array(estimated_repeats)

    # Mean Absolute Error (MAE) and RMSE
    mae = mean_absolute_error(actual_repeats, estimated_repeats)
    rmse = np.sqrt(mean_squared_error(actual_repeats, estimated_repeats))

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Overestimation Rate: {overestimation_rate:.3f}")
    print(f"Underestimation Rate: {underestimation_rate:.3f}")
    print(f"Miss-by-One Rate: {miss_by_one_rate:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"Root Mean Squared Error: {rmse:.3f}")

    # Prepare metrics for saving
    scores = {
        'Accuracy': accuracy,
        'Overestimation Rate': overestimation_rate,
        'Underestimation Rate': underestimation_rate,
        'Miss-by-One Rate': miss_by_one_rate,
        'Mean Absolute Error': mae,
        'Root Mean Squared Error': rmse
    }

    # Save the metrics as a CSV file ending with '_scores.csv'
    metrics_df = pd.DataFrame(scores, index=[0])
    csv_file_path = os.path.dirname(label_pred_log_file_path)
    csv_file_name = f'/{image_folder_namebase}_code_ver{code_version_date}_scores.csv'
    csv_file_path += csv_file_name
    metrics_df.to_csv(csv_file_path, index=False)

    # Normalize the confusion matrix
    normalized_matrix = np.zeros(matrix.shape)
    
    row_sums = matrix.sum(axis=1)
    if display or savefig:
        _, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [4,1]})  # Increase the size of the figure.

        # Debugging output
        print("Original matrix:\n", matrix)
        print("Row sums before normalization:", row_sums)

        # Adjusted normalization with debugging
        for row in range(matrix.shape[0]):
            normalized_matrix[row] = matrix[row] / row_sums[row] if row_sums[row] != 0 else np.zeros(matrix.shape[1])

        # Debugging output to check if normalization is as expected
        print("Normalized matrix:\n", normalized_matrix)

        folder_name = os.path.basename(os.path.dirname(label_pred_log_file_path))
        ax = axs[0]
        sns.heatmap(normalized_matrix, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, vmin=0, vmax=1)  # Plot the heatmap on the new axes.
        ax.set_title(f'{folder_name}')
        ax.set_xlabel('Estimated Particle Count')
        ax.set_ylabel('Actual Particle Count')
        ytick_labels = [f"{i} (count: {row_sums[i]})" for i in range(len(row_sums))]
        ax.set_yticklabels(ytick_labels, rotation=0)
        
        # Draw lines between rows
        for i in range(matrix.shape[0]+1):
            ax.axhline(i, color='black', linewidth=1)
        
        # Text messages
        text_message = f"Accuracy: {accuracy:.3f}\n"+ \
            f"Overestimation Rate: {overestimation_rate:.3f}\n"+ \
            f"Underestimation Rate: {underestimation_rate:.3f}\n"+ \
            f"Miss-by-One Rate: {miss_by_one_rate:.3f}\n"+ \
            f"Mean Absolute Error: {mae:.3f}\n"+ \
            f"Root Mean Squared Error: {rmse:.3f}"
        ax = axs[1]
        ax.axis('off')
        # Add text messages to the figure
        ax.text(0.01, 0.5, text_message, ha='left', va='center')  # Adjust x, y values as needed
        plt.tight_layout()

        if display:
            plt.show(block=False)

        if savefig:
            png_file_path = os.path.dirname(label_pred_log_file_path)
            png_file_name = f'/{image_folder_namebase}_code_ver{code_version_date}_confusion_mat.png'
            png_file_path += png_file_name
            plt.savefig(png_file_path, dpi=300)

def combine_log_files(analyses_folder, image_folder_namebase, code_version_date, delete_individual_files=False):
    ''' Combines the log files in the image_log folder into one file called fitting_results.csv.
    
    Parameters:
        analyses_folder (str): The path of the folder containing the analyses outputs.
        image_folder_namebase (str): The name of the folder containing the images.
        code_version_date (str): The version of the code.
        delete_individual_files (bool): Whether to delete the individual log files. Default is False.
    
    Returns:
        None
    '''

    # Create the fitting_results.csv file
    whole_metrics_log_filename = os.path.join(analyses_folder, f'{image_folder_namebase}_code_ver{code_version_date}_metrics_log_per_image_hypothesis.csv')
    print(f"{whole_metrics_log_filename=}")
    
    os.makedirs(os.path.dirname(whole_metrics_log_filename), exist_ok=True)

    # Get all the *_fittings.csv files in the image_log folder
    individual_image_log_files = glob.glob(os.path.join(analyses_folder, 'image_log', '*_analysis_log.csv'))

    # Open the fitting_results.csv file in write mode
    with open(whole_metrics_log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_filename (h number)', 'true_count', 'h number', 'selected?', 'xi', 'lli', 'penalty', 'fisher_info', 'fit_parameters'])

        # Iterate over the fittings_files
        for log_file in individual_image_log_files:
            # Get the image file name without the extension
            # image_file_name = os.path.splitext(os.path.basename(log_file))[0].split('_analysis_log')[0] + ".tiff"
            # Read the fitting file
            with open(log_file, 'r') as f_ind:
                reader = csv.reader(f_ind)
                # Skip the first row (header)
                next(reader)
                rows = list(reader)
                writer.writerows(rows)

    # Delete the image_log directory and all its contents
    if delete_individual_files:
        dir_path = os.path.join(analyses_folder, 'image_log')
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print('Deleting individual image log files.')

def make_metrics_histograms(file_path = "./analyses/PSF 1_0_2024-06-13/PSF 1_0_2024-06-13_metrics_log_per_image_hypothesis.csv", metric_of_interest='penalty'):
    """ Make histograms of the metric of interest for each true count and h number in the given CSV file.
    
    Parameters:
        file_path (str): The path of the CSV file containing the metrics log.
        metric_of_interest (str): The metric of interest. Default is 'penalty'.
        
    Returns:
        None
    """

    # Fix legacy formats: 
    # - Step 1: Open the CSV file for reading
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # - Step 2: Check the first line
    first_line = lines[0].strip()
    expected_header = 'image_filename (h number),xi,lli,penalty,fisher_info,fit_parameters'
    new_header = 'image_filename (h number),accepted?,xi,lli,penalty,fisher_info,fit_parameters'
    # - Step 3: Prepare the modified version if needed
    if first_line == expected_header:
        lines[0] = new_header + '\n'
    # - Step 4 & 5: Open the file for writing and write the modified content
    with open(file_path, 'w') as file:
        file.writelines(lines)

    # Step 1: Read the CSV file
    df = pd.read_csv(file_path)

    # Add true_count and h number columns if they are not present in the dataframe.
    # Extract the true count and h number from the image_filename (h number) column if they are not present in the dataframe.
    overwrite_needed = False
    if 'true_count' not in df.columns:
        df['true_count'] = df['image_filename (h number)'].apply(lambda x: int(x.split('/')[-1].split('\\')[-1].split('count')[-1].split('-')[0]))
        overwrite_needed = True
    if 'h number' not in df.columns:
        df['h number'] = df['image_filename (h number)'].apply(lambda x: int(x.split('/')[-1].split('\\')[-1].split('h')[-1].split(')')[0]))
        overwrite_needed = True

    if overwrite_needed:
        # Rename the original file as back up
        backup_file_path = file_path.rsplit('.', 1)[0] + '_backup_' + datetime.now().strftime("%Y-%m-%d") + '.' + file_path.rsplit('.', 1)[1]
        os.rename(file_path, backup_file_path)
        
        # save the df to csv in the order: image_filename (h number), true_count, est_count, h number, selected, xi, lli, penalty, fisher_info, fit_parameters
        df = df[['image_filename (h number)', 'true_count', 'h number', 'selected?', 'xi', 'lli', 'penalty', 'fisher_info', 'fit_parameters']]
        df.to_csv(file_path, index=False)

    # Identify all unique true_count values
    unique_true_counts = df['true_count'].unique()
    
    for true_count in unique_true_counts:
        if true_count == 2:
            pass
        # Filter DataFrame for the current true_count
        df_filtered = df[df['true_count'] == true_count]
        
        # Group by 'h number'
        grouped = df_filtered.groupby('h number')
        
        # Determine the number of unique h_numbers for the current true_count
        unique_h_numbers = sorted(df_filtered['h number'].unique())
        num_h_numbers = len(unique_h_numbers)
        
        # Determine the max y-value for consistent y-range across subplots
        min_metric, max_metric = 0, 0
        edges = np.linspace(0, 1, 41)
        # for _, group_data in grouped:
        valid_data = grouped[metric_of_interest]
        # if not valid_data.empty:
        min_metric = valid_data.min().min()
        max_metric = valid_data.max().max()
        edges = np.linspace(min_metric, max_metric, 41)
        max_y_value = 0
        for h_number, group_data in grouped:
            numerical_data = group_data[metric_of_interest].astype(float)
            counts, _ = np.histogram(numerical_data, bins=edges)
            max_y_value = max(max_y_value, max(counts))
        max_y_value *= 1.1
        
        # Create a figure with subplots
        fig, axs = plt.subplots(num_h_numbers, 1, figsize=(5, 1.6 * num_h_numbers))
        
        # Ensure axs is iterable
        if num_h_numbers == 1:
            axs = [axs]
        
        # Get a colormap
        cmap = matplotlib.colormaps['turbo_r']
        color_map = {h_number: cmap(i / len(unique_h_numbers)) for i, h_number in enumerate(unique_h_numbers)}
        
        # Plot histograms for each h number
        for i, (h_number, group_data) in enumerate(grouped):
            ax = axs[i] if num_h_numbers > 1 else axs[0]
            ax.hist(group_data[metric_of_interest], bins=edges, color=color_map[h_number])
            ax.set_xlim(min_metric, max_metric)
            ax.set_ylim(0, max_y_value)  # Set consistent y-range
            ax.set_ylabel('Count')
            ax.legend([f'H{h_number}'])
        
        # Adjust layout and save the figure
        deepest_folder_name = os.path.basename(os.path.dirname(file_path))
        plt.suptitle(f'{metric_of_interest} values for true count {true_count} in {deepest_folder_name}')
        plt.tight_layout()
        fig_path = os.path.join(os.path.dirname(file_path), f'{metric_of_interest} hist per h for true count {true_count}.png')
        plt.savefig(fig_path)
        plt.close(fig)  # Close the figure to free memory
        print(f'saved: {metric_of_interest} hist per h for true count {true_count}.png')

def process(config_files_dir, parallel=False):
    ''' Process the config files in the config_files_dir directory.
    
    Parameters:
        config_files_dir (str): The path of the directory containing the config files.
        parallel (bool): Whether to run the processing in parallel. Default is False.
        
    Returns:
        analyses_folder (str): The path of the folder containing the analyses outputs.
    '''
    # Load the config files
    try:
        config_files = os.listdir(config_files_dir)
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error accessing directory {config_files_dir}: {e}")
        return None

    # Filter out non-JSON files
    config_files = [file for file in config_files if file.endswith('.json')]
    
    # Print the config files
    print(f"Config files loaded (total of {len(config_files)}):")
    for config_file in config_files:
        print("> " + config_file)

    # Process the config files
    for i, config_file in enumerate(config_files):
        print(f"Processing {config_file} ({i+1}/{len(config_files)})")
        try:
            with open(os.path.join(config_files_dir, config_file), 'r') as f:
                config = json.load(f, object_pairs_hook=OrderedDict)
                
                # Pretty print the config file
                pprint.pprint(config)

                # Set the required fields for each type of processing
                required_fields_common = ['image_folder_namebase', 'code_version_date']

                required_fields_for_separation_test = ['separation_test_image_generation?', 
                                                       'sep_image_count', 
                                                       'sep_intensity_prefactor_to_bg_level', 
                                                       'sep_psf_sigma', 
                                                       'sep_distance_ratio_to_psf_sigma', 
                                                       'sep_img_width', 
                                                       'sep_bg_level', 
                                                       'sep_random_seed']

                required_fields_for_generation = ['generate_regular_dataset?', 
                                                  'gen_random_seed', 
                                                  'gen_total_image_count', 
                                                  'gen_psf_sigma', 
                                                  'gen_img_width', 
                                                  'gen_minimum_particle_count', 
                                                  'gen_maximum_particle_count', 
                                                  'gen_bg_level', 
                                                  'gen_particle_intensity_mean', 
                                                  'gen_particle_intensity_sd']

                required_fields_for_analysis = ['analyze_the_dataset?', 
                                                'ana_random_seed', 
                                                'ana_predefined_psf_sigma', 
                                                'ana_use_premature_hypothesis_choice?', 
                                                'ana_maximum_hypothesis_index']

                # Check if the required_fields_common are strings
                for field in required_fields_common:
                    if field in config and not isinstance(config[field], str):
                        # The config file cannot be used to run the code. Print an error message and exit
                        print(f"Error: '{field}' should be a string.")
                        exit()
                    else:
                        # Replace '.' with '_' in the field value to avoid issues with file paths
                        if '.' in config[field]:
                            before_change = config[field]
                            config[field] = config[field].replace('.', '_')
                            print(f"Modified field '{field}' value to replace '.' with '_' - before: {before_change}, after: {config[field]}")
                
                # Check if all fields ending with '?' are boolean
                for field in config:
                    if field.endswith('?') and not isinstance(config[field], bool):
                        # The config file cannot be used to run the code. Print an error message and exit
                        print(f"Error: '{field}' should be a boolean.")
                        exit()
                        
                # Assign the required fields based on the type of processing
                required_fields = required_fields_common

                if config['separation_test_image_generation?']:
                    required_fields += required_fields_for_separation_test
                elif config['generate_regular_dataset?']:
                    required_fields += required_fields_for_generation
                elif config['analyze_the_dataset?']:
                    required_fields += required_fields_for_analysis

                # Check if all required fields are present in the config file
                for field in required_fields:
                    if field not in config:
                        # If config['separation_test_image_generation?'] is True, then all fields starting with 'sep' are required.
                        if config['separation_test_image_generation?'] and field.startswith('sep'):
                            print(f"Error: '{field}' should be set for separation test image generation.")
                            exit()
                        # If config['generate_regular_dataset?'] is True, then all fields starting with 'gen' are required.
                        if config['generate_regular_dataset?'] and field.startswith('gen'):
                            print(f"Error: '{field}' should be set for image generation.")
                            exit()
                        # If config['analyze_the_dataset?'] is True, then all fields starting with 'analysis' are required.
                        if config['analyze_the_dataset?'] and field.startswith('ana'):
                            print(f"Error: '{field}' should be set for image analysis.")
                            exit()
                
        # If the config file is not found or invalid, print an error message and continue to the next config file
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error: {config_file} file not found or invalid. Skipping to the next config file.")
            continue

        # Generate separation test images
        if config['separation_test_image_generation?']:
            generate_separation_test_images(image_folder_namebase=config['image_folder_namebase'], 
                                            # code_ver=config['code_version_date'],
                                            sep_distance_ratio_to_psf_sigma = config['sep_distance_ratio_to_psf_sigma'],
                                            n_total_image_count=config['sep_image_count'],
                                            amp_to_bg=config['sep_intensity_prefactor_to_bg_level'], 
                                            psf_sigma=config['sep_psf_sigma'], 
                                            sz=config['sep_img_width'], 
                                            bg=config['sep_bg_level'], 
                                            generation_random_seed=config['sep_random_seed'], 
                                            config_content=json.dumps(config)
                                            )

        # Generate regular dataset
        elif config['generate_regular_dataset?']:
            generate_test_images(image_folder_namebase=config['image_folder_namebase'], 
                                # code_ver=config['code_version_date'],
                                n_total_image_count=config['gen_total_image_count'],
                                minimum_number_of_particles=config['gen_minimum_particle_count'], 
                                maximum_number_of_particles=config['gen_maximum_particle_count'], 
                                particle_intensity_mean=config['gen_particle_intensity_mean'], 
                                particle_intensity_sd=config['gen_particle_intensity_sd'], 
                                
                                psf_sigma=config['gen_psf_sigma'], sz=config['gen_img_width'], 
                                bg=config['gen_bg_level'], 
                                generation_random_seed=config['gen_random_seed'], 
                                config_content=json.dumps(config))

        # Analyze dataset
        if config['analyze_the_dataset?']:
            # parallel = False # Debug purpose
            timeout = config.get('ana_timeout_per_image', 600)
            analyses_folder_path = analyze_whole_folder(image_folder_namebase=config['image_folder_namebase'], 
                                                code_version_date=config['code_version_date'], 
                                                use_exit_condi=config['ana_use_premature_hypothesis_choice?'], 
                                                last_h_index=config['ana_maximum_hypothesis_index'], 
                                                analysis_rand_seed=config['ana_random_seed'], 
                                                psf_sigma=config['ana_predefined_psf_sigma'], 
                                                config_content=json.dumps(config), 
                                                parallel=parallel, 
                                                timeout_per_image=timeout)
            # Get the dataset name and code version date
            image_folder_namebase = config['image_folder_namebase']
            code_version_date = config['code_version_date']
            plt.close('all')

            # Combine analysis log files into one.
            combine_log_files(analyses_folder_path, image_folder_namebase, code_version_date, delete_individual_files=True)
            
            # Generate confusion matrix
            label_prediction_log_file_path = os.path.join(analyses_folder_path, f'{image_folder_namebase}_code_ver{code_version_date}_label_prediction_log.csv')
            try:
                generate_confusion_matrix(label_prediction_log_file_path, image_folder_namebase, code_version_date, display=False, savefig=True)
            except Exception as e:
                print(f"Error generating confusion matrix: {e}")
            # generate_intensity_histogram(label_prediction_log_file_path, image_folder_namebase, code_version_date, display=False, savefig=True)

            # Delete the dataset after analysis
            if config['ana_delete_the_dataset_after_analysis?']:
                dir_path =os.path.join("datasets", f"{config['image_folder_namebase']}")
                shutil.rmtree(dir_path)
                print('Deleting image data.')

            
def main():
    """ Main function to run the analysis pipeline. """
    
    # Start the batch job timer
    batchjobstarttime = datetime.now()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process config files.')
    parser.add_argument('--config-file-folder', '-c', type=str, help='Folder containing config files to run.')
    parser.add_argument('--profile', '-p', type=bool, default=False, help='Boolean to decide whether to profile or not.')
    args = parser.parse_args()

    # Check if config-file-folder is provided
    if (args.config_file_folder is None):
        print("Please provide the folder name for config files using --config-file-folder or -c option.")
        exit()
    config_files_dir = args.config_file_folder
    
    if args.profile is True:
        with Profile() as profile:
            process(config_files_dir=config_files_dir, parallel=False)
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.TIME)
                .dump_stats('profile_results.prof')
            )
            # os.system('snakeviz profile_results.prof &')
    else:
        process(config_files_dir, parallel=True)

    # End the batch job timer
    batchjobendtime = datetime.now()
    # Print the time taken for the batch job
    print(f'\nBatch job completed in {batchjobendtime - batchjobstarttime}')


# Run the main function if the script is executed from the command line
if __name__ == '__main__':

    # Run the main function with parallel processing ('-p' option value is True)
    # sys.argv = ['main.py', '-c', './example_config_folder/', '-p', 'True'] # -p for profiling. If True, it will run on a single process.

    # Run the main function without parallel processing ('-p' option value is False)
    sys.argv = ['main.py', '-c', './configs/'] # -p for profiling. Default is False, and it will run on multiple processes.
    # sys.argv = ['main.py', '-c', './configs/'] # -p for profiling. Default is False, and it will run on multiple processes.


    main()