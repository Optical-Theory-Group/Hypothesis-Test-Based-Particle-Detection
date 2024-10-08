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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from process_algorithms import generalized_likelihood_ratio_test, generalized_maximum_likelihood_rule
from process_algorithms import make_subregions, create_separable_filter, get_tentative_peaks
import math
import numpy as np
import diplib as dip
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
    """ Generate test images with random number of particles between minimum_number_of_particles and maximum_number_of_particles.
    
    Parameters:
        image_folder_namebase (str): The name of the folder to store the images.
        maximum_number_of_particles (int): The maximum number of particles in the image.
        amp_to_bg_min (int): Minimum amplitude to background ratio.
        amp_to_bg_max (int): Maximum amplitude to background ratio.
        amp_sd (float): Standard deviation of the amplitude.

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
    print(f'Total number of images generated from this config is {number_of_images_per_count * number_of_counts}. Note that this number may be slightly higher than the total number of images requested.')
    print(f'Image save destination: ./image_dataset/{image_folder_namebase}.')

    # Create the folder to store the images
    image_folder_path = os.path.join("image_dataset", f"{image_folder_namebase}") 
    os.makedirs(image_folder_path, exist_ok=True)

    # Determine the color mode of the image (gray or rgb)
    color_mode = ''
    if len(particle_intensity_mean) == len(particle_intensity_sd) == len(bg) == 3: # Case : rgb
        color_mode = 'rgb' 
    elif isinstance(particle_intensity_mean, (int, float)): # Case : gray scale
        color_mode = 'gray'
    else:
        raise ValueError("The color mode of the image is not recognized. Please check the following variables: particle_intensity_mean, particle_intensity_sd, and bg.")

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

                # Convolve the psf with the particle position and add it to the image
                image += psfconvolution(peak_info, sz)

            # Add Poisson noise
            image = np.random.poisson(image).astype(np.uint16) # This is the end of image processing.
            img_filename = f"count{n_particles}-index{img_idx}.{file_format}"
            if image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))

            # Save the image
            imageio.imwrite(img_filename, image)
    
    # Save the content of the config file
    if config_content is not None:
        config_file_save_path = os.path.join(image_folder_path, 'config_used.json')
        with open(config_file_save_path, 'w') as f:
            json.dump(json.loads(config_content), f, indent=4)

    # Return the path of the folder containing the images
    return image_folder_path

def generate_separation_test_images(image_folder_namebase='separation_test', sep_distance_ratio_to_psf_sigma=3, n_total_image_count=20, amp_to_bg=5, psf_sigma=1, 
                                    sz=20, bg=500, generation_random_seed=42, config_content=None, file_format='tiff'):
    if config_content:
        config = json.loads(config_content)
        if 'file_format' in config:
            file_format = config['file_format']

    if file_format not in ['tiff', 'png']:
        raise ValueError("Format must be either 'tiff' or 'png'.")


    separation_distance = sep_distance_ratio_to_psf_sigma * psf_sigma
    if separation_distance > sz:
        raise ValueError(f"Separation {separation_distance} is greater than the size of the image {sz}.")
    # Set the random seed
    np.random.seed(generation_random_seed)
    # Create the folder to store the images
    psf_str = f"{psf_sigma:.1f}".replace('.', '_')
    sep_str = f"{sep_distance_ratio_to_psf_sigma:.1f}".replace('.', '_')
    image_folder_path = f"./image_dataset/{image_folder_namebase}"
    os.makedirs(image_folder_path, exist_ok=True)
    print(f'Generating {n_total_image_count} images with psf {psf_sigma} and separation {sep_str} times the psf in folder {image_folder_path}.')

    # Generate the images
    sz_original = sz
    for img_idx in range(n_total_image_count):
        if img_idx % 20 == 0:
            print(f"Generating image index {img_idx}", end='\r')
        particle_intensity = bg * amp_to_bg
        angle = np.random.uniform(0, 2*np.pi)
        # Center pos of the image - give random offset of the size of the pixel
        center_x = sz / 2 + np.random.uniform(-.5, .5)
        center_y = sz / 2 + np.random.uniform(-.5, .5)
        # print(f"{center_x=}, {center_y=}")
        # Particle 1
        x1 = center_x + separation_distance / 2 * np.cos(angle)
        y1 = center_y + separation_distance / 2 * np.sin(angle)

        retry_count = 0
        while retry_count < 1000 and (x1 < -.5 * 2 * psf_sigma or x1 > sz - .5 - 2 * psf_sigma or y1 < -.5 * 2 * psf_sigma or y1 > sz - .5 - 2 * psf_sigma):
            angle = np.random.uniform(0, 2*np.pi)
            x1 = center_x + separation_distance / 2 * np.cos(angle)
            y1 = center_y + separation_distance / 2 * np.sin(angle)
            retry_count += 1
        
        if retry_count == 1000:
            print(f"Warning: Particles could not be fitted inside the image. The separation and the psf are probably too large. {img_idx} will be skipped.")
            continue

        peak_info = {'x': x1, 'y': y1, 'prefactor': particle_intensity, 'psf_sigma': psf_sigma}
        image = np.ones((sz, sz), dtype=float) * bg
        image += psfconvolution(peak_info, sz)
        # Particle 2
        x2 = center_x - separation_distance / 2 * np.cos(angle)
        y2 = center_y - separation_distance / 2 * np.sin(angle)
        peak_info = {'x': x2, 'y': y2, 'prefactor': particle_intensity, 'psf_sigma': psf_sigma}
        image += psfconvolution(peak_info, sz)
        # Add Poisson noise
        image = np.random.poisson(image, size=(image.shape)) # This is the resulting (given) image.
        img_filename = f"count2_psf{psf_str}_sep{sep_str}_index{img_idx}.{file_format}"
        pil_image = im.fromarray(image.astype(np.uint16))
        pil_image.save(os.path.join(image_folder_path, img_filename))

    # Save the content of the config file
    if config_content is not None:
        config_file_save_path = os.path.join(image_folder_path, 'config_used.json')
        with open(config_file_save_path, 'w') as f:
            json.dump(json.loads(config_content), f, indent=4)

    return image_folder_path

def report_progress(progresscount, totalrealisations, starttime=None, statusmsg=''):
    """
        Calls updated_progress() to print updated progress bar and returns the updated progress count.

        Input:
            progresscount: int
                Current counter recording progress.
            totalrealisations: int
                Total number of realisations to be calculation
            starttime: datetime
                Time at which simulation was started

        Return:
            progresscount: int
                Input value incremented by 1.
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
    """
        Prints a progress bar to console

        Parameters
        ----------
        progress : float
            Variable ranging from 0 to 1 indicating fractional progress.
        status : TYPE, optional
            Status text to suffix progress bar. The default is ''.
        barlength : str, optional
            Controls width of progress bar in console. The default is 20.

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
    # sys.stdout.write(text)
    # sys.stdout.flush()

def analyze_whole_folder(image_folder_namebase, code_version_date, use_exit_condi=False, last_h_index=7, psf_sigma=1.39, analysis_rand_seed=0, config_content=None, parallel=False, display_xi_graph=False, timeout=120):
    '''Analyzes all the images in the dataset folder.'''
    # Set random seed
    np.random.seed(analysis_rand_seed)

    # Get a list of image files in the folder
    images_folder = os.path.join('./image_dataset', image_folder_namebase)
    original_images_folder = images_folder
    
    print(f"Looking into the folder {images_folder}")

    # Check if the folder exists
    if not os.path.exists(images_folder):
        print(f"Folder {images_folder} does not exist.")
        # Find another folder that starts with image_folder_namebase
        base_dir = './image_dataset'
        matching_folders = [f for f in os.listdir(base_dir) if f.startswith(image_folder_namebase)]
        if matching_folders:
            print(f"Found folders starting with '{image_folder_namebase}': {matching_folders}")
            images_folder = os.path.join(base_dir, matching_folders[0])
        else:
            raise ValueError(f"No folder starting with '{image_folder_namebase}' found in '{base_dir}'.")

    print(f"Note: The program is working with {images_folder} which is close to the user input {original_images_folder}. **")
    # Read all png and tiff files
    image_files = glob.glob(os.path.join(images_folder, '*.png')) + glob.glob(os.path.join(images_folder, '*.tiff'))
    if len(image_files) == 0:
        raise ValueError("There are no images in this folder.")

    print(f"Images loaded (total of {len(image_files)}):")

    # Create a folder to store the logs
    log_folder = os.path.join('./runs', image_folder_namebase + '_code_ver' + code_version_date)
    os.makedirs(log_folder, exist_ok=True)

    # Save the content of the config file
    if config_content is not None:
        config_file_save_path = os.path.join(log_folder, f'{image_folder_namebase}_code_ver{code_version_date}_config_used.json')
        with open(config_file_save_path, 'w') as f:
            json.dump(json.loads(config_content), f, indent=4)

    # Prepare the label (actual count) prediction (estimated count) log file
    label_prediction_log_file_path = os.path.join(log_folder, f'{image_folder_namebase}_code_ver{code_version_date}_label_prediction_log.csv')
    with open(label_prediction_log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Input Image File', 'Actual Particle Count', 'Estimated Particle Count', "Determined Particle Intensities"])

    # Create the "runs" folder if it doesn't exist
    runs_folder = './runs'
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    # For each image file, 
    starttime = datetime.now()
    print('Beginning image analysis...')

    # Create a list of random seeds for each image
    image_rand_seeds = list(range(len(image_files)))
    np.random.shuffle(image_rand_seeds)

    # Analyze the images in parallel or sequentially
    if parallel:
        # Analyze the images in parallel using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Create a list of futures for each image
            futures = [executor.submit(analyze_image, filename, psf_sigma, last_h_index, analysis_rand_seed_per_image, log_folder, use_exit_condi=use_exit_condi )
                        for analysis_rand_seed_per_image, filename in zip(image_rand_seeds, image_files)]
            progress = 0
            # Write the results to the main log file
            with open(label_prediction_log_file_path, 'a', newline='') as f:  #####
                # Iterate over the futures that are completed. 
                for cfresult in concurrent.futures.as_completed(futures):
                    # If an exception is raised, print the exception and continue to the next image
                    if cfresult._exception is not None:
                        if isinstance(cfresult._exception, Warning):
                            print("Encountered a Warning:", cfresult._exception)
                        else:
                            print("Encountered an Exception:", cfresult._exception)
                            print("Proceeding without addressing the exception.")
                    # Get the result of the future and write the results to the main log file??k
                    try:
                        analysis_result = cfresult.result(timeout=timeout)
                        actual_num_particles = analysis_result['actual_num_particles']
                        estimated_num_particles = analysis_result['estimated_num_particles']
                        input_image_file = analysis_result['image_filename']
                        determined_particle_intensities = analysis_result['determined_particle_intensities']

                        writer = csv.writer(f)
                        writer.writerow([input_image_file, actual_num_particles, estimated_num_particles, determined_particle_intensities])

                        if actual_num_particles == estimated_num_particles:
                            statusmsg = f'\"{input_image_file}\" - Actual Count {actual_num_particles} == Estimated {estimated_num_particles}'
                        elif actual_num_particles > estimated_num_particles:
                            statusmsg = f'\"{input_image_file}\" - Actual Count: {actual_num_particles} > Estimated {estimated_num_particles}'
                        else:
                            statusmsg = f'\"{input_image_file}\" - Actual Count {actual_num_particles} < Estimated {estimated_num_particles}'
                    except concurrent.futures.TimeoutError:
                        print(f"Task exceeded the maximum allowed time of {timeout} seconds and was cancelled.")
                        cfresult.cancel()  # Attempt to cancel the future
                        statusmsg = 'Task cancelled due to timeout.'
                    except Exception as e:
                        print(f"Error in cfresult.result(): {e}")
                        statusmsg = f'Error: {e}'

                    # statusmsg += f' Test results saved to {filename}'
                    report_progress(progress, len(image_files), starttime, statusmsg)
                    progress += 1
    else:
        progress = 0
        for analysis_rand_seed_per_image, filename in zip(image_rand_seeds, image_files):
            analysis_result = analyze_image(filename, psf_sigma, last_h_index, analysis_rand_seed_per_image, log_folder, display_xi_graph=display_xi_graph, use_exit_condi=use_exit_condi)
            actual_num_particles = analysis_result['actual_num_particles']
            estimated_num_particles = analysis_result['estimated_num_particles']
            input_image_file = analysis_result['image_filename']
            determined_particle_intensities = analysis_result['determined_particle_intensities']

            with open(label_prediction_log_file_path, 'a', newline='') as f: 
                writer = csv.writer(f)
                writer.writerow([input_image_file, actual_num_particles, estimated_num_particles, determined_particle_intensities])

                statusmsg = f'{image_folder_namebase} '
                if actual_num_particles == estimated_num_particles:
                    statusmsg += f'\"{input_image_file}\" - Actual Count {actual_num_particles} == Estimated {estimated_num_particles}\n'
                elif actual_num_particles > estimated_num_particles:
                    statusmsg += f'\"{input_image_file}\" - Actual Count: {actual_num_particles} > Estimated {estimated_num_particles}\n'
                else:
                    statusmsg += f'\"{input_image_file}\" - Actual Count {actual_num_particles} < Estimated {estimated_num_particles}\n'

            report_progress(progress, len(image_files), starttime, statusmsg)
            progress += 1
            
    return log_folder #, determined_particle_intensities

# def process_tile(image_array, psf_sigma, last_h_index, random_seed, use_exit_condi, display_fit_results=False, display_xi_graph=False):
#     # Find tentative peaks
#     tentative_peaks = get_tentative_peaks(image_array, min_distance=1)
#     rough_peaks_xy = [peak[::-1] for peak in tentative_peaks]

#     # Run GMRL
#     estimated_num_particles, fit_results, test_metrics = generalized_maximum_likelihood_rule(roi_image=image_array, rough_peaks_xy=rough_peaks_xy, \
#                                                         psf_sigma=psf_sigma, last_h_index=last_h_index, random_seed=random_seed, display_fit_results=display_fit_results, display_xi_graph=display_xi_graph, use_exit_condi=use_exit_condi) 

#     return estimated_num_particles, fit_results, test_metrics

def merge_conincident_particles(image, tile_dicts_array, psf):

    plt.close('all')
    _, axs = plt.subplots(2, 1, figsize=(5,10))
    markers = ['1', '2', '|',  '_', '+', 'x',] * 100
    palette = sns.color_palette('Paired', len(tile_dicts_array.flatten()))
    plt.sca(axs[0])
    plt.imshow(image, cmap='gray')     
    len_all_locations = sum([len(tile_dict['particle_locations']) for tile_dict in tile_dicts_array.flatten()])
    plt.title(f'Tiled - Sum of all particle detections: {len_all_locations}')
    ax = plt.gca()
    i = 0
    for tile_dict in tile_dicts_array.flatten():
        locations = tile_dict['particle_locations']
        rectangle = plt.Rectangle((tile_dict['x_low_end'], tile_dict['y_low_end']), tile_dict['image_slice'].shape[1], tile_dict['image_slice'].shape[0], edgecolor=palette[i], facecolor='none', linewidth=1, )
        ax.add_patch(rectangle)
        for loc in locations:
            plt.scatter(loc[0] + tile_dict['x_low_end'], loc[1] + tile_dict['y_low_end'], marker=markers[i], s=300, color=palette[i], linewidths=2)
        # plt.scatter(locations[0] + tile_dict['x_low_end'], locations[1] + tile_dict['y_low_end'], marker=markers[i], s=300, color=palette[i], linewidths=3)
        i += 1
        
    overlap = 0
    # abs_locations_of_unique_particles = []
    new_tile_dicts_array = tile_dicts_array.copy()
    for ref_tile_x in range(tile_dicts_array.shape[0]):
        for ref_tile_y in range(tile_dicts_array.shape[1]):
            ref_tile = tile_dicts_array[ref_tile_x][ref_tile_y]

            # Check the right tile first 
            if ref_tile_x < tile_dicts_array.shape[0] - 1: # If the tile is not the rightmost tile
                compare_tile = tile_dicts_array[ref_tile_x+1][ref_tile_y]  # Get the tile to the right
                # Initialize the replacement information
                replacement_info = {'ref_indices_to_del': [], 'com_indices_to_del': [], 'average_location_in_ref_frame': []}
                # For each particle location in the reference tile
                for ref_loc_index, relative_ref_loc in enumerate(ref_tile['particle_locations']):
                    # For each particle location in the compare tile
                    for com_loc_index, relative_compare_loc in enumerate(compare_tile['particle_locations']):
                        # Calculate the absolute locations of the particles
                        absolute_ref_loc = relative_ref_loc + np.array([ref_tile['x_low_end'], ref_tile['y_low_end']])
                        absolute_compare_loc = relative_compare_loc + np.array([compare_tile['x_low_end'], compare_tile['y_low_end']])
                        # If the distance between the two locations is less than psf, then consider them as the same particle.
                        if (absolute_ref_loc[0] - absolute_compare_loc[0])**2 + (absolute_ref_loc[1] - absolute_compare_loc[1])**2 < psf**2:
                            overlap += 1
                            # Add to the replacement information that the reference particle index should be deleted
                            replacement_info['ref_indices_to_del'].append(ref_loc_index)

                # Create a new reference tile containing information of ref_tile except particle locations
                new_ref_tile = ref_tile.copy()
                new_ref_tile['particle_locations'] = []
                # For each particle location in the reference tile
                for i in range(len(ref_tile['particle_locations'])):
                    # If the particle location index is not in the list of indices to delete, then add it to the new reference tile
                    if i not in replacement_info['ref_indices_to_del']:
                        new_ref_tile['particle_locations'].append(ref_tile['particle_locations'][i])
                ref_tile = new_ref_tile
                new_tile_dicts_array[ref_tile_x][ref_tile_y] = ref_tile

            # Check the bottom tile
            if ref_tile_y < tile_dicts_array.shape[1] - 1:
                compare_tile = tile_dicts_array[ref_tile_x][ref_tile_y+1] 
                replacement_info = {'ref_indices_to_del': [], 'com_indices_to_del': [], 'average_location_in_ref_frame': []}
                for ref_loc_index, relative_ref_loc in enumerate(ref_tile['particle_locations']):
                    for com_loc_index, relative_compare_loc in enumerate(compare_tile['particle_locations']):
                        # If the distance between the two locations is less than psf, then consider them as the same particle.
                        absolute_ref_loc = relative_ref_loc + np.array([ref_tile['x_low_end'], ref_tile['y_low_end']])
                        absolute_compare_loc = relative_compare_loc + np.array([compare_tile['x_low_end'], compare_tile['y_low_end']])
                        if (absolute_ref_loc[0] - absolute_compare_loc[0])**2 + (absolute_ref_loc[1] - absolute_compare_loc[1])**2 < psf**2:
                            overlap += 1
                            # Delete the information from tile_dict_j and add the average to tile_dict_i
                            replacement_info['ref_indices_to_del'].append(ref_loc_index)
                            replacement_info['com_indices_to_del'].append(com_loc_index)

                # Create a new reference tile containing information of ref_tile except particle locations
                new_ref_tile = ref_tile.copy()
                new_ref_tile['particle_locations'] = []
                # For each particle location in the reference tile
                for i in range(len(ref_tile['particle_locations'])):
                    # If the particle location index is not in the list of indices to delete, then add it to the new reference tile
                    if i not in replacement_info['ref_indices_to_del']:
                        new_ref_tile['particle_locations'].append(ref_tile['particle_locations'][i])
                ref_tile = new_ref_tile
                new_tile_dicts_array[ref_tile_x][ref_tile_y] = ref_tile
                        # ref_tile['particle_locations'].pop(i)
                    # else: # if it is in the list of indices to delete, delete that particle location from compare_tile
                    # else:
                    #     abs_locations_of_unique_particles.append(ref_tile['particle_locations'][i] + np.array([ref_tile['x_low_end'], ref_tile['y_low_end']]))
                        
                # for i in range(len(compare_tile['particle_locations'])):
                #     if i not in replacement_info['com_indices_to_del']:
                #         new_com_tile['particle_locations'].append(compare_tile['particle_locations'][i])
                # for i in range(len(replacement_info['average_location_in_ref_frame'])):
                #     new_ref_tile['particle_locations'].append(replacement_info['average_location_in_ref_frame'][i])
                # new_tile_dicts_array[ref_tile_x][ref_tile_y] = ref_tile
                # compare_tile = new_com_tile
                pass


    print(f"{overlap=}")
    for x in range(new_tile_dicts_array.shape[0]):
        for y in range(new_tile_dicts_array.shape[1]):
            print(f"Tile ({x}, {y}): {(new_tile_dicts_array[x][y]['particle_locations'])}")

    merged_locations = []
    for tile in new_tile_dicts_array.flatten():
        for loc in tile['particle_locations']:
            absolute_loc = loc + np.array([tile['x_low_end'], tile['y_low_end']])
            merged_locations.append(absolute_loc)

    plt.sca(axs[1])
    ax = plt.gca()
    plt.title(f'Same locations merged (count:{len(merged_locations)})')
    plt.imshow(image, cmap='gray')     
    for loc in merged_locations:
        plt.scatter(loc[0], loc[1], marker=markers[i], s=200, color='red', linewidths=1)
    plt.show()
    pass


    return merged_locations



def analyze_image(image_filename, psf_sigma, last_h_index, analysis_rand_seed_per_image, log_folder, display_fit_results=False, display_xi_graph=False, use_exit_condi=False, tile_width=40, tile_stride=30):
    # Print the name of the image file
    image = np.array(im.open(image_filename))

    # Extract the number of particles from image_filename
    basename = os.path.basename(image_filename)
    count_part = basename.split('-')[0]
    foldername = os.path.basename(os.path.dirname(image_filename))
    if count_part.startswith("separation") or os.path.basename(foldername).startswith("separation"):
        num_particles = 2
    else:
        num_particles = count_part.split('count')[1]
        num_particles = num_particles.split('_')[0]
    actual_num_particles = int(num_particles)

    sz = image.shape[0]
    if sz < tile_width + tile_stride: # If the image is smaller than the tile size, then process the whole image (no need to divide into tiles)

        # Run GMRL
        estimated_num_particles, fit_results, test_metrics = generalized_maximum_likelihood_rule(roi_image=image, \
                                                            psf_sigma=psf_sigma, last_h_index=last_h_index, random_seed=analysis_rand_seed_per_image, display_fit_results=display_fit_results, display_xi_graph=display_xi_graph, use_exit_condi=use_exit_condi) 

        image_analysis_log_filename = f"{log_folder}/image_log/{os.path.splitext(os.path.basename(image_filename))[0]}_analysis_log.csv"

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
        determined_particle_intensities = []
        if estimated_num_particles > 0:
            for i in range(1, estimated_num_particles + 1):
                determined_particle_intensities.append(fit_parameters[estimated_num_particles][i][0])
        determined_particle_intensities

        metric_data = list(zip(file_h_info, true_counts, h_numbers, selected_bools, xi, lli, penalty, fisher_info, fit_parameters))

        # Write the data to the CSV files
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
        
        return image_analysis_results

    # estimated_num_particles, fit_results, test_metrics = generalized_maximum_likelihood_rule(roi_image=image_array, rough_peaks_xy=rough_peaks_xy, \
    #                                                     psf_sigma=psf_sigma, last_h_index=last_h_index, random_seed=random_seed, display_fit_results=display_fit_results, display_xi_graph=display_xi_graph, use_exit_condi=use_exit_condi) 
        # estimated_num_particles, fit_results, test_metrics = process_tile(image, psf_sigma, last_h_index, analysis_rand_seed_per_image, use_exit_condi, display_fit_results=display_fit_results, display_xi_graph=display_xi_graph)
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

        tile_dicts_array = np.zeros((n_y + 1, n_x + 1), dtype=object)
        y_low_end_list = [tile_stride * (n) for n in range(n_y + 1)]
        x_low_end_list = [tile_stride * (n) for n in range(n_x + 1)]
        img_height, img_width = image.shape
        tile_count = 0
        for y_index, y_low_end in enumerate(y_low_end_list):
            y_high_end = min(y_low_end + tile_sz[0], img_height)
            for x_index, x_low_end in enumerate(x_low_end_list):
                x_high_end = min(x_low_end + tile_sz[1], img_width)
                tile_dicts_array[x_index][y_index] = {'x_low_end': x_low_end, 'y_low_end': y_low_end, 'image_slice': image[y_low_end:y_high_end, x_low_end:x_high_end], 'particle_locations': []}
                print(f"Loading tile {tile_count} from image {image_filename}", end='\r')
                tile_count += 1

        # (x,y) or all detected particles
        # particle_locations = []
        for tile_dict in tile_dicts_array.flatten():
            # est_num_particle_tile, fit_results, test_metrics = generalized_maximum_likelihood_rule(tile_dict['image_slice'], psf_sigma, last_h_index, analysis_rand_seed_per_image, display_fit_results=display_fit_results, display_xi_graph=display_xi_graph, use_exit_condi=use_exit_condi)
            est_num_particle_tile, fit_results, test_metrics = generalized_maximum_likelihood_rule(tile_dict['image_slice'], psf_sigma, last_h_index, analysis_rand_seed_per_image, display_xi_graph=display_xi_graph, use_exit_condi=True)
            # Choose the fit_result with its index matching est_num_particle_tile
            chosen_fit = fit_results[est_num_particle_tile]
            particle_locations = []
            for particle_index in range(1, est_num_particle_tile + 1):
                loc = chosen_fit['theta'][particle_index][1:3]
                particle_locations.append(loc)
            tile_dict['particle_locations'] = particle_locations

            # Set the log file name for the current tile
            tilename = f"tile_x{tile_dict['x_low_end']}-{min(tile_dict['x_low_end'] + tile_width, img_width)}_y{tile_dict['y_low_end']}-{min(tile_dict['y_low_end'] + tile_width, img_height)}"
            print(f"Processing tile {tilename} in image {image_filename}", end='\r')

            image_filename_base = os.path.basename(image_filename).split('.')[0]
            tile_analysis_log_filename = f"{log_folder}/image_log/{image_filename_base}_{tilename}_analysis_log.csv"

            # Extract xi, lli, and penalty from test_metrics and fit_parameters from fit_results - for this tile/generate_se
            xi = test_metrics['xi']
            lli = test_metrics['lli']
            penalty = test_metrics['penalty']
            fisher_info = test_metrics['fisher_info']
            fit_parameters = [result['theta'] for result in fit_results]

            # Create a list of tuples containing hypothesis_index, xi, lli, and penalty - for this tile
            file_tile_h_info = [f"{image_filename} {tilename} (h{h_index})" for h_index in range(len(xi))]
            # true_counts = [actual_num_particles for _ in range(len(xi))]
            h_numbers = [h_index for h_index in range(len(xi))]
            selected_bools = [1 if est_num_particle_tile == h_index else 0 for h_index in range(len(xi))]
            determined_particle_intensities = []
            if est_num_particle_tile > 0:
                for i in range(1, est_num_particle_tile + 1):
                    determined_particle_intensities.append(fit_parameters[est_num_particle_tile][i][0])
            determined_particle_intensities
            file_tile_metric_data = list(zip(file_tile_h_info, h_numbers, selected_bools, xi, lli, penalty, fisher_info, fit_parameters))

            # Write the data to the CSV files
            os.makedirs(os.path.dirname(tile_analysis_log_filename), exist_ok=True)
            with open(tile_analysis_log_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['image_filename (h number)', 'h number', 'selected?', 'xi', 'lli', 'penalty', 'fisher_info', 'fit_parameters'])
                writer.writerows(file_tile_metric_data)
                pass

            # if tile_dict['y_low_end'] == 510:
            #     pass
            #     break

        deduplicate_locations = merge_conincident_particles(image, tile_dicts_array, psf_sigma)

        pass

        
        # # Find tentative peaks
        # tentative_peaks = get_tentative_peaks(image, min_distance=1)
        # rough_peaks_xy = [peak[::-1] for peak in tentative_peaks]

        # # Run GMRL
        # estimated_num_particles, fit_results, test_metrics = generalized_maximum_likelihood_rule(roi_image=image, rough_peaks_xy=rough_peaks_xy, \
        #                                                     psf_sigma=psf_sigma, last_h_index=last_h_index, random_seed=analysis_rand_seed_per_image, display_fit_results=display_fit_results, display_xi_graph=display_xi_graph, use_exit_condi=use_exit_condi) 



        # image_analysis_results = {
        #                         'actual_num_particles': actual_num_particles,
        #                         'estimated_num_particles': estimated_num_particles,
        #                         'image_filename': image_filename,
        #                         'determined_particle_intensities': determined_particle_intensities,
        #                         }
    
    # return image_analysis_results
    return None

def generate_intensity_histogram(label_pred_log_file_path, image_folder_namebase, code_version_date, display=False, savefig=True):
    # Read the CSV file
    df = pd.read_csv(label_pred_log_file_path)
    try:
        intensities = df[ "Determined Particle Intensities"]
    except KeyError:
        intensities = df[ "Determined Particle Intensities"]

    all_intensities = []
    for entry in intensities:
        all_intensities.extend(ast.literal_eval(entry))

    # Generate intensity histogram
    fig, ax = plt.subplots()
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
    # Read the CSV file
    df = pd.read_csv(label_pred_log_file_path)
    # Extract the actual and estimated particle numbers
    try:
        actual = df['Actual Particle Count']
    except KeyError:
        actual = df['Actual Particle Number'] 
    try:
        estimated = df['Estimated Particle Count']
    except KeyError:
        estimated = df['Estimated Particle Number']
    
    # Generate the confusion matrix
    matrix = confusion_matrix(actual, estimated)
    # if matrix[-1].sum == 0:
    #     matrix = matrix[:-1, :]
    
    # Save the confusion matrix as a CSV file
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

    metrics_df = pd.DataFrame(scores, index=[0])
    csv_file_path = os.path.dirname(label_pred_log_file_path)
    csv_file_name = f'/{image_folder_namebase}_code_ver{code_version_date}_scores.csv'
    csv_file_path += csv_file_name
    metrics_df.to_csv(csv_file_path, index=False)

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

        # normalized_matrix = np.divide(matrix, row_sums[:, None] + epsilon, out=np.zeros_like(matrix, dtype=np.float64), where=row_sums!=0)
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
        pass
        if display:
            plt.show(block=False)

        if savefig:
            png_file_path = os.path.dirname(label_pred_log_file_path)
            png_file_name = f'/{image_folder_namebase}_code_ver{code_version_date}_confusion_mat.png'
            png_file_path += png_file_name
            plt.savefig(png_file_path, dpi=300)

def combine_log_files(log_folder, image_folder_namebase, code_version_date, delete_individual_files=False):
    '''Combines the log files in the image_log folder into one file called fitting_results.csv.'''
    # Create the fitting_results.csv file
    whole_metrics_log_filename = os.path.join(log_folder, f'{image_folder_namebase}_code_ver{code_version_date}_metrics_log_per_image_hypothesis.csv')
    print(f"{whole_metrics_log_filename=}")
    
    os.makedirs(os.path.dirname(whole_metrics_log_filename), exist_ok=True)

    # Get all the *_fittings.csv files in the image_log folder
    individual_image_log_files = glob.glob(os.path.join(log_folder, 'image_log', '*_analysis_log.csv'))

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
        dir_path = os.path.join(log_folder, 'image_log')
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print('Deleting individual image log files.')

def make_metrics_histograms(file_path = "./runs/PSF 1_0_2024-06-13/PSF 1_0_2024-06-13_metrics_log_per_image_hypothesis.csv", metric_of_interest='penalty'):
    # metric_of_interest = 'lli'
    # metric_of_interest = 'xi'
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

def main():
    # Start the batch job timer
    batchjobstarttime = datetime.now()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process config files.')
    parser.add_argument('--config-file-folder', '-c', type=str, help='Folder containing config files to run.')
    parser.add_argument('--profile', '-p', type=bool, default=False, help='Boolean to decide whether to profile or not.')
    args = parser.parse_args()

    # Override config file folder argument for testing purposes
    # print('Forcing the config file folder to ./config_files/300524 for testing purposes - remove the line below this code in main() to restore correct behaviour.')
    # args.config_file_folder = './config_files/300524'

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

    batchjobendtime = datetime.now()
    print(f'\nBatch job completed in {batchjobendtime - batchjobstarttime}')

def process(config_files_dir, parallel=False, timeout=120):
    '''Process the config files in the config_files_dir directory.'''
    config_files = os.listdir(config_files_dir)

    print(f"Config files loaded (total of {len(config_files)}):")
    for config_file in config_files:
        print("> " + config_file)

    for i, config_file in enumerate(config_files):
        print(f"Processing {config_file} ({i+1}/{len(config_files)})")
        try:
            with open(os.path.join(config_files_dir, config_file), 'r') as f:
                config = json.load(f)
                
                # Pretty print the config file
                pprint.pprint(config)

                required_fields_common = ['image_folder_namebase', 'code_version_date']

                required_fields_for_separation_test = ['separation_test_image_generation?', 
                                                       'sep_image_count', 
                                                       'sep_intensity_prefactor_to_bg_level', 
                                                       'sep_psf_sigma', 
                                                       'sep_distance_ratio_to_psf_sigma', 
                                                       'sep_img_width', 
                                                       'sep_bg_level', 
                                                       'sep_random_seed']

                required_fields_for_generation = ['genereate_regular_dataset?', 
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
                        print(f"Error: '{field}' should be a string.")
                        exit()
                    else:
                        # Replace '.' with '_' in the field value
                        if '.' in config[field]:
                            before_change = config[field]
                            config[field] = config[field].replace('.', '_')
                            print(f"Modified field '{field}' value to replace '.' with '_' - before: {before_change}, after: {config[field]}")
                
                # Check if all fields ending with '?' are boolean
                for field in config:
                    if field.endswith('?') and not isinstance(config[field], bool):
                        print(f"Error: '{field}' should be a boolean.")
                        exit()
                        
                # Check if the required fields are present in the config file
                required_fields = required_fields_common

                if config['separation_test_image_generation?']:
                    required_fields += required_fields_for_separation_test
                elif config['genereate_regular_dataset?']:
                    required_fields += required_fields_for_generation
                elif config['analyze_the_dataset?']:
                    required_fields += required_fields_for_analysis

                for field in required_fields:
                    if field not in config:
                        # If config['separation_test_image_generation?'] is True, then all fields starting with 'sep' are required.
                        if config['separation_test_image_generation?'] and field.startswith('sep'):
                            print(f"Error: '{field}' should be set for separation test image generation.")
                            exit()
                        # If config['genereate_regular_dataset?'] is True, then all fields starting with 'gen' are required.
                        if config['genereate_regular_dataset?'] and field.startswith('gen'):
                            print(f"Error: '{field}' should be set for image generation.")
                            exit()
                        # If config['analyze_the_dataset?'] is True, then all fields starting with 'analysis' are required.
                        if config['analyze_the_dataset?'] and field.startswith('ana'):
                            print(f"Error: '{field}' should be set for image analysis.")
                            exit()
                
        # If the config file is not found or invalid, print an error message and continue to the next config file
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error: {config_file} file not found or invalid")
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

        elif config['genereate_regular_dataset?']:
# def generate_test_images(image_folder_namebase, maximum_number_of_particles, particle_intensity_mean, particle_intensity_sd=0, n_total_image_count=1, psf_sigma=1, sz=20, bg=1, 
#                          generation_random_seed=42, config_content=None, minimum_number_of_particles=0, format='tiff'):
            generate_test_images(image_folder_namebase=config['image_folder_namebase'], 
                                # code_ver=config['code_version_date'],
                                n_total_image_count=config['gen_total_image_count'],
                                minimum_number_of_particles=config['gen_minimum_particle_count'], 
                                maximum_number_of_particles=config['gen_maximum_particle_count'], 
                                particle_intensity_mean=config['particle_intensity_mean'], 
                                particle_intensity_sd=config['particle_intensity_sd'], 
                                
                                psf_sigma=config['gen_psf_sigma'], sz=config['gen_img_width'], 
                                bg=config['gen_bg_level'], 
                                generation_random_seed=config['gen_random_seed'], 
                                config_content=json.dumps(config))
        # Analyze the dataset
        if config['analyze_the_dataset?']:
            # parallel = False # Debug purpose
            log_folder_path = analyze_whole_folder(image_folder_namebase=config['image_folder_namebase'], 
                                                code_version_date=config['code_version_date'], 
                                                use_exit_condi=config['ana_use_premature_hypothesis_choice?'], 
                                                last_h_index=config['ana_maximum_hypothesis_index'], 
                                                analysis_rand_seed=config['ana_random_seed'], 
                                                psf_sigma=config['ana_predefined_psf_sigma'], 
                                                config_content=json.dumps(config), 
                                                parallel=parallel, 
                                                timeout=timeout)
            # Get the dataset name and code version date
            image_folder_namebase = config['image_folder_namebase']
            code_version_date = config['code_version_date']
            plt.close('all')

            # Combine analysis log files into one.
            combine_log_files(log_folder_path, image_folder_namebase, code_version_date, delete_individual_files=True)
            
            # Generate confusion matrix
            label_prediction_log_file_path = os.path.join(log_folder_path, f'{image_folder_namebase}_code_ver{code_version_date}_label_prediction_log.csv')
            generate_confusion_matrix(label_prediction_log_file_path, image_folder_namebase, code_version_date, display=False, savefig=True)
            # generate_intensity_histogram(label_prediction_log_file_path, image_folder_namebase, code_version_date, display=False, savefig=True)

            # Delete the dataset after analysis
            if config['ana_delete_the_dataset_after_analysis?']:
                dir_path =os.path.join("image_dataset", f"{config['image_folder_namebase']}_code_ver{config['code_version_date']}")
                shutil.rmtree(dir_path)
                print('Deleting image data.')

if __name__ == '__main__':

    # sys.argv = ['main.py', '-c', './config_sep_test_scale_intensity/'] 
    # main()
    # pass 
    # img_folder_path = generate_test_images(image_folder_namebase='test', code_ver='2024-07-31', n_total_image_count=1, minimum_number_of_particles=100, maximum_number_of_particles=100, 
    #                                        amp_to_bg_min=5, generation_random_seed=42, amp_to_bg_max=5, amp_sd=0, psf_sigma=1.0, sz=256, bg=100)
    # # print("image generated")
    # image_files = glob.glob(os.path.join(img_folder_path, '*.png')) + glob.glob(os.path.join(img_folder_path, '*.tiff'))
    # image_file = image_files[0]

    # Get a list of image files in the folder
    # images_folder = os.path.join('./image_dataset', 'test_code_ver2024-07-31')
    # image_files = glob.glob(os.path.join(images_folder, '*.png')) + glob.glob(os.path.join(images_folder, '*.tiff'))
    # if len(image_files) == 0:
    #     raise ValueError("There are no images in this folder.")
    # image_file = image_files[0]

    # analyze_image(image_file, psf_sigma=1.0, last_h_index=5, analysis_rand_seed_per_image=1, log_folder='runs/test_2024-07-24', tile_width=40, tile_stride=30)
    # combine_log_files('runs/test_2024-07-24', 'test', '2024-07-24', delete_individual_files=False)
    # pass
    
    # sys.argv = ['main.py', '-c', './config_test/'] 
    # sys.argv = ['main.py', '-c', './config_files/'] 
    # sys.argv = ['main.py', '-c', './config_scale1_test/'] 
    # sys.argv = ['main.py', '-c', './config_3/'] 
    # sys.argv = ['main.py', '-c', './config_/'] 
    # sys.argv = ['main.py', '-c', './config_/', '-p', 'True']
    sys.argv = ['main.py', '-c', './config_files/']
    # sys.argv = ['main.py', '-c', './config_files/', '-p', 'True']
    # print(f"Manually setting argv as {sys.argv}. Delete this line and above to restore normal behaviour. (inside main.py, if __name__ == '__main__': )")
    main()
    # filepath = "./runs/weighted FIM size 20 factors are theta_code_ver2024-07-09/weighted FIM size 20 factors are theta_code_ver2024-07-09_metrics_log_per_image_hypothesis.csv"
    # make_metrics_histograms(file_path=filepath, metric_of_interest='penalty')
    # make_metrics_histograms(file_path=filepath, metric_of_interest='lli')
    # make_metrics_histograms(file_path=filepath, metric_of_interest='xi')
    # items
    # pass

