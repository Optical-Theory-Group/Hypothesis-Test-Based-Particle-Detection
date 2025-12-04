# Hypothesis-Test-Based-Particle-Detection
# -----------------------------------------
#
# This file is part of the project "Hypothesis-Test-Based-Particle-Detection".
# It implements the main hypothesis test based algorithm.
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


from process_algorithms import generalized_maximum_likelihood_rule
from process_algorithms import merge_coincident_particles
from image_generation import generate_test_images, generate_separation_test_images
from collections import OrderedDict
import ast
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pprint
import json
import argparse
import csv
from PIL import Image as im
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
import shutil
from sklearn.metrics import confusion_matrix
import concurrent.futures
import sys
from datetime import datetime, timedelta
from cProfile import Profile
from pstats import SortKey, Stats
import tifffile


#######################################################################
# ########### Define some utility functions ###########################
#######################################################################
def report_progress(progresscount, totalrealisations, starttime=None, statusmsg=''):
    """
    Calls update_progress() to print updated progress bar and returns the updated progress count.

    Parameters:
        progresscount (int): Current counter recording progress.
        totalrealisations (int): Total number of realisations to be calculation
        starttime (datetime): Time at which the analysis had started
    Return:
        progresscount (int): Input value incremented by 1.
    """
    # update progress trackers
    progresscount += 1
    runtime = datetime.now() - starttime
    runtime -= timedelta(microseconds=runtime.microseconds)
    remaintime = (runtime / progresscount) * (totalrealisations - progresscount)
    remaintime = remaintime - timedelta(microseconds=remaintime.microseconds)

    # Store the progress message
    strmsg = '{}/{} in : {} (Remaining: {}). {}'.format(progresscount, totalrealisations,
                                                        runtime, remaintime, statusmsg)

    # Call the update_progress function
    update_progress(progresscount / totalrealisations, strmsg)

    return progresscount


def update_progress(progress, status='', barlength=10):
    """
    Prints a progress bar to console

    Parameters:
        progress (float): Variable ranging from 0 to 1 indicating fractional progress.
        status (str, optional): Status text to suffix progress bar. The default is ''.
        barlength (int, optional): Controls width of progress bar in console. The default is 10.

    Return:
        None
    """
    # Ensure progress is a float
    if isinstance(progress, int):
        progress = float(progress)

    # Check that progress is a float
    if not isinstance(progress, float):
        progress = 0
        status = 'error: progress var must be float\r\n'

    # Clamp progress between 0 and 1
    if progress < 0:
        progress = 0
        status = 'Halt...\r\n'
    if progress >= 1:
        progress = 1
        status += ' \nReached 100 percent.\r\n'

    # Build the progress bar string
    block = int(round(barlength * progress))
    text = '\rPercent: [{0}] {1:.2f}% {2}'.format('#' * block + '-' * (barlength - block), progress * 100, status)
    try:
        _, erase_length = os.get_terminal_size(0)  # get the width of the terminal to know how many characters to erase
    except OSError:
        erase_length = len(text) + 10  # If the terminal size cannot be obtained, use the length of the text as the erase length
    clear_line = '\r' + ' ' * erase_length + '\r'
    sys.stdout.write(clear_line + text)
    sys.stdout.flush()


#######################################################################
# ########### Define some (batch) processing functions ####################
#######################################################################
def analyze_whole_folder(image_folder_basename,
                         code_version_date,
                         timeout_per_image,
                         use_exit_condition=False,
                         last_h_index=7,
                         psf_sigma=1.39,
                         analysis_rand_seed=0,
                         config_content=None,
                         parallel=False,
                         display_xi_graph=False
                         ):
    """
    Analyzes all the images in the dataset folder.

    Parameters:
        image_folder_basename (str): The name of the folder containing the images.
        code_version_date (str): The version of the code.
        use_exit_condition (bool): Whether to use the exit condition.
        last_h_index (int): The last index of the hypothesis.
        psf_sigma (float): The sigma of the point spread function.
        analysis_rand_seed (int): The random seed for the analysis.
        config_content (str): The content of the config file.
        parallel (bool): Whether to analyze the images in parallel.
        display_xi_graph (bool): Whether to display the xi graph.
        timeout_per_image (int): The maximum time allowed for processing each image. (sec)

    Returns:
        analyses_folder (str): The path of the folder containing the analyses outputs.
    """
    # Set random seed
    np.random.seed(analysis_rand_seed)

    # Get the list of image files in the folder
    images_folder = os.path.join('./datasets', image_folder_basename)
    original_images_folder = images_folder

    # Print the folder being analyzed
    print(f"Looking into the folder {images_folder} to perform analysis")

    # If the folder does not exist, print a message and find another folder that starts with 
    # image_folder_basename
    if not os.path.exists(images_folder):
        print(f"Folder {images_folder} does not exist.")
        base_dir = './datasets'
        alternative_folders = [f for f in os.listdir(base_dir) if f.startswith(image_folder_basename)]

        # Still check whether there are folders starting with image_folder_basename. If then, use the first one
        if alternative_folders:
            print(f"Found folders starting with '{image_folder_basename}': {alternative_folders}")
            images_folder = os.path.join(base_dir, alternative_folders[0])

            # Print a message to inform the user that the program is working with an alternative folder,
            # close to the user input
            print(f"Note: The program is working with {images_folder} which is close to the user input {original_images_folder}. **")
        else:
            raise ValueError(f"No folder starting with '{image_folder_basename}' found in '{base_dir}'.")

    # Read all png and tiff files
    all_image_files = glob.glob(os.path.join(images_folder, '*.png')) + glob.glob(os.path.join(images_folder, '*.tiff'))
    print(f"Total number of image files in the folder: {len(all_image_files)}")
    print(f"Number of png files: {len(glob.glob(os.path.join(images_folder, '*.png')))}")
    print(f"Number of tiff files: {len(glob.glob(os.path.join(images_folder, '*.tiff')))}")

    # Check the first image whether it is grayscale or RGB, and print the information to the user.
    if all_image_files[0].lower().endswith(('.tif', '.tiff')):
        first_image = tifffile.imread(all_image_files[0])
    else:
        first_image = np.array(im.open(all_image_files[0]))
    if first_image.ndim == 2:  # Grayscale image
        print(f"== The first image ({all_image_files[0]}) is grayscale, indicating all images in this dataset are grayscale.")
    elif first_image.ndim == 3 and first_image.shape[2] == 3:  # RGB image
        print(f"== The first image ({all_image_files[0]}) is rgb, indicating all images in this dataset are rgb.")
    else:
        print(f"== The first image ({all_image_files[0]}) has an unexpected format. All images in this dataset might likely be in unexpected format, neither grayscale nor rgb.")

    # If there are no images in the folder, raise an error
    if len(all_image_files) == 0:
        raise ValueError("There are no images in this folder.")

    # Print the number of images loaded
    print(f"Images loaded (total of {len(all_image_files)}):")

    # Filter files that start with "count"
    image_files_with_count_label = [f for f in all_image_files if os.path.basename(f).startswith('count')]
    print(f"Number of image files starting with 'count': {len(image_files_with_count_label)}")
    print('- These files will have "Actual Particle Count" in the label_prediction log file and "true count" in the metrics log file.')
    print(f"Number of image files not starting with 'count': {len(all_image_files) - len(image_files_with_count_label)}")
    print('- These files will have "Actual Particle Count" nor "true count" written in the log files.')

    # Create a folder to store the analysis outputs
    # Use shortened name if the full path would be too long
    full_folder_name = image_folder_basename + '_code_ver' + code_version_date

    # Check if the path would be too long and shorten preemptively
    max_safe_length = 60  # Conservative limit to ensure file operations work
    if len(full_folder_name) > max_safe_length:
        # Use shortened name - drop code_ver and keep more of original name
        short_folder_name = image_folder_basename[:max_safe_length]
        print(f"Warning: Folder name too long. Using shortened name: {short_folder_name}")
    else:
        short_folder_name = full_folder_name

    analyses_folder = os.path.join('./analyses', short_folder_name)
    os.makedirs(analyses_folder, exist_ok=True)

    # Save the content of the config file
    if config_content is not None:
        config_file_save_path = os.path.join(analyses_folder, f'{short_folder_name}_config_used.json')
        try:
            with open(config_file_save_path, 'w') as f:
                json.dump(json.loads(config_content), f, indent=4)
        except OSError as e:
            if "too long" in str(e).lower() or "filename" in str(e).lower():
                # Further shorten the config filename
                short_config_name = f'{short_folder_name[:30]}_config.json'
                config_file_save_path = os.path.join(analyses_folder, short_config_name)
                with open(config_file_save_path, 'w') as f:
                    json.dump(json.loads(config_content), f, indent=4)
                print(f"Warning: Config filename too long. Using: {short_config_name}")
            else:
                raise

    # Prepare the label (=actual count) prediction (=estimated count) log file
    label_prediction_log_file_path = os.path.join(analyses_folder, f'{short_folder_name}_label_prediction_log.csv')

    # Create the "analyses" folder if it doesn't exist
    os.makedirs('./analyses', exist_ok=True)

    # Mark the start time and print a message indicating the beginning of the image analysis
    starttime = datetime.now()
    print('Beginning image analysis...')

    # Create a list of random seeds for each image
    image_rand_seeds = list(range(len(all_image_files)))
    np.random.shuffle(image_rand_seeds)

    print("Creating the label_prediction log file...")
    # Create the folder to store the label_prediction log file
    os.makedirs(os.path.dirname(label_prediction_log_file_path), exist_ok=True)
    try:
        with open(label_prediction_log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Input Image File', 'Actual Particle Count', 'Estimated Particle Count', "Determined Particle Intensities"])
    except Exception as e:
        print("Error in creating the label_prediction log. Could be that the folder/file name is too long. Error: ", e)

    # Check if the analysis is to be done in parallel (or sequentially).
    if parallel:
        print("Analyzing images in parallel...")

        # Analyze the images in parallel using ProcessPoolExecutor
        executor = concurrent.futures.ProcessPoolExecutor()
        try:
        # with concurrent.futures.ProcessPoolExecutor() as executor:
            # Create a list of futures for each image
            futures = [executor.submit(analyze_image, filename, psf_sigma, last_h_index,
                                       analysis_rand_seed_per_image, analyses_folder,
                                       use_exit_condition=use_exit_condition)
                       for analysis_rand_seed_per_image, filename in zip(image_rand_seeds, all_image_files)]
            print(f"Number of futures submitted: {len(futures)}")

            # Initialize the progress counter
            progress = 0

            first_future_flag = True
            timed_out = 0
            for future, future_filename in zip(futures, all_image_files):
                if first_future_flag:
                    print(f"First future processing: {future_filename}")
                    first_future_flag = False

                try:
                    analysis_result = future.result(timeout=timeout_per_image)
                    actual_num_particles = analysis_result['actual_num_particles']
                    estimated_num_particles = analysis_result['estimated_num_particles']
                    input_image_file = analysis_result['image_filename']
                    input_basename = os.path.basename(input_image_file)
                    determined_particle_intensities = analysis_result['determined_particle_intensities']
                    with open(label_prediction_log_file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            input_image_file,
                            actual_num_particles,
                            estimated_num_particles,
                            determined_particle_intensities
                        ])
                    sign = '+' if estimated_num_particles - actual_num_particles >= 0 else ''
                    if estimated_num_particles == -1:
                        if actual_num_particles < 0:
                            statusmsg = (
                                f'"{input_basename}" Error in analysis. '
                                'Estimated count = -1 (true count unknown)'
                            )
                        else:
                            statusmsg = (
                                f'"{input_basename}" Error in analysis. '
                                f'Estimated count = -1 (true count: {actual_num_particles})'
                            )
                    else:
                        if actual_num_particles < 0:
                            statusmsg = (
                                f'"{input_basename}" count readout: '
                                f'{estimated_num_particles} (true count unknown)'
                            )
                        else:
                            diff = estimated_num_particles - actual_num_particles
                            statusmsg = (
                                f'"{input_basename}" {actual_num_particles} -> '
                                f'{estimated_num_particles} ({sign}{diff})'
                            )
                except concurrent.futures.TimeoutError:
                    if future.cancel():
                        timed_out += 1
                    statusmsg = (
                        f'TIMEOUT after {timeout_per_image}s (file: {future_filename}). '
                        'Task canceled.'
                    )
                except Exception as e:
                    statusmsg = f'error: {e} (file: {future_filename})'

                report_progress(progress, len(futures), starttime, statusmsg)
                progress += 1

            if timed_out:
                print(f"Total timed-out tasks canceled: {timed_out}")

            # ensure a newline after the progress bar in parallel mode too
            print()

        finally:
            # Cancel any futures that are not done and don’t wait on shutdown
            canceled = 0
            for f in futures:
                if not f.done():
                    if f.cancel():
                        canceled += 1
            executor.shutdown(wait=False, cancel_futures=True)
            if canceled:
                print(f"Canceled {canceled} unfinished tasks.")

    else:  # If the analysis is to be done sequentially
        print("Analyzing images in serial...")
        # Initialize the progress counter
        progress = 0

        # Iterate over the images
        for analysis_rand_seed_per_image, filename in zip(image_rand_seeds, all_image_files):
            try:
                analysis_result = analyze_image(filename, psf_sigma, last_h_index,
                                                analysis_rand_seed_per_image, analyses_folder,
                                                display_xi_graph=display_xi_graph,
                                                use_exit_condition=use_exit_condition)

                # Extract the results from the analysis result
                actual_num_particles = analysis_result['actual_num_particles']
                estimated_num_particles = analysis_result['estimated_num_particles']
                input_image_file = analysis_result['image_filename']
                input_basename = os.path.basename(input_image_file)
                determined_particle_intensities = analysis_result['determined_particle_intensities']

                # Write the results to the label_prediction log file
                with open(label_prediction_log_file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([input_image_file, actual_num_particles,
                                     estimated_num_particles, determined_particle_intensities])

                # Build status message about estimation accuracy
                if actual_num_particles < 0:
                    statusmsg = (
                        f'"{input_basename}" count readout: '
                        f'{estimated_num_particles} (true count unknown)'
                    )
                else:
                    diff = estimated_num_particles - actual_num_particles
                    sign = '+' if diff >= 0 else ''
                    statusmsg = (
                        f'"{input_basename}" {actual_num_particles} -> '
                        f'{estimated_num_particles} ({sign}{diff})'
                    )

            except Exception as e:
                statusmsg = f'Error: {e} File: {filename} '

            total_count = len(all_image_files)
            report_progress(progress, total_count, starttime, statusmsg)

            # Increment the progress counter
            progress += 1
            print()  # Print a newline after the progress bar

    print('Returning')
    # Return the path of the folder containing the analyses outputs and the shortened folder name
    return analyses_folder, short_folder_name


def analyze_image(image_filename,
                  psf_sigma,
                  last_h_index,
                  analysis_rand_seed_per_image,
                  analyses_folder,
                  display_fit_results=False,
                  display_xi_graph=False,
                  use_exit_condition=False,
                  tile_width=40,
                  tile_jump_distance=30,
                  tiling_width_threshold=160
                  ):
    """ Analyze an image using the generalized maximum likelihood rule.

    Parameters:
        image_filename (str): The name of the image file.
        psf_sigma (float): The sigma of the point spread function.
        last_h_index (int): The last index of the hypothesis.
        analysis_rand_seed_per_image (int): The random seed for the analysis of the image.
        analyses_folder (str): The path of the folder containing the analyses outputs.
        display_fit_results (bool): Whether to display the fit results. Default is False.
        display_xi_graph (bool): Whether to display the xi graph. Default is False.
        use_exit_condition (bool): Whether to use the exit condition. Default is False.
        tile_width (int): The width of the tile. Default is 40. Tiling only occurs if the image is
                          larger than the tile size.
    tile_jump_distance (int): Default is 30. This is the distance between adjacent tiles in pixels.
                          It is less than the tile width to ensure overlap between adjacent tiles.

    Returns:
        image_analysis_results (dict) or tile_combined_results (dict): The results of the image analysis
                    or, if the image is too big to analyze at once, the combined results of the tiles.
        (Currently, the latter is not implemented and the function returns None.)
    """
    # Print the name of the image file
    if image_filename.lower().endswith(('.tif', '.tiff')):
        # entire_image = tifffile.imread(image_filename).astype(np.float32)
        entire_image = tifffile.imread(image_filename)
    else:
        # entire_image = np.array(im.open(image_filename), dtype=np.float32)
        entire_image = np.array(im.open(image_filename))

    # Check if the image is grayscale or RGB
    if entire_image.ndim == 2:  # Grayscale image
        color_mode = 'gray'
    elif entire_image.ndim == 3 and entire_image.shape[2] == 3:  # RGB image
        color_mode = 'rgb'
    else:
        raise ValueError(
            "Unexpected dimension for file: "
            f"{image_filename}. Expected 2 (grayscale) or 3 (RGB). "
            f"Instead got {entire_image.ndim} dimensions."
        )

    # Extract the number of particles from image_filename
    basename = os.path.basename(image_filename)

    actual_num_particles = -1
    # If the image is a separation test, set the number of particles to 2
    if basename.startswith("separation"):
        actual_num_particles = 2
    elif basename.startswith('count'):
        actual_num_particles = int(basename.split('-')[0].split('count')[1].split('_')[0])
    else:
        actual_num_particles = -1

    # foldername = os.path.basename(os.path.dirname(image_filename))

    def analyze_region_of_interest(
        tiling,
        roi_image,
        psf_sigma,
        last_h_index,
        analysis_rand_seed_per_image,
        display_fit_results,
        display_xi_graph,
        use_exit_condition,
        roi_name=None,
        color_mode=None,
    ):
        """ Analyze the region of interest (ROI) of the image. """
        # Call the generalized_maximum_likelihood_rule (GMLR) function to analyze the image
        estimated_num_particles, fit_results, test_metrics = generalized_maximum_likelihood_rule(
            roi_image=roi_image,
            psf_sigma=psf_sigma,
            last_h_index=last_h_index,
            random_seed=analysis_rand_seed_per_image,
            display_fit_results=display_fit_results,
            display_xi_graph=display_xi_graph,
            use_exit_condition=use_exit_condition,
            roi_name=roi_name,
            color_mode=color_mode,
        )

        # Extract xi, lli, and penalty from test_metrics
        xi = test_metrics['xi']
        xi_aic = test_metrics['xi_aic']
        xi_bic = test_metrics['xi_bic']
        lli = test_metrics['lli']
        penalty = test_metrics['penalty']
        penalty_aic = test_metrics['penalty_aic']  # Akaike Information Criterion (AIC)
        penalty_bic = test_metrics['penalty_bic']  # Bayesian Information Criterion (BIC)

        # Extract the Fisher Information Matrix and the fit parameters (theta) from the test_metrics
        fisher_info = test_metrics['fisher_info']  # Fisher Information Matrix
        fit_parameters = [result['theta'] for result in fit_results]
        chosen_fit = fit_parameters[estimated_num_particles]  # TODO: validate (2025.03.10, Neil)

        # Create lists for hypothesis indexing and selection flags
        roi_name_h_index = [f"{roi_name} (h{h_index})" for h_index in range(len(xi))]
        true_counts = [actual_num_particles for _ in range(len(xi))]
        h_numbers = [h_index for h_index in range(len(xi))]
        selected_bools = [1 if estimated_num_particles == h_index else 0 for h_index in range(len(xi))]

        # Extract the determined particle intensities (to see the particle intensity distribution)
        particle_intensities = []
        if estimated_num_particles > 0:
            for i in range(1, estimated_num_particles + 1):
                particle_intensities.append(
                    fit_parameters[estimated_num_particles][i][0]
                )
            
        # Create a list of tuples containing the results of the individual hypothesis tests
        individual_hypothesis_test_results = list(zip(roi_name_h_index, true_counts, h_numbers, selected_bools, xi, lli, penalty, fisher_info, fit_parameters, xi_aic, xi_bic, penalty_aic, penalty_bic))

        # Save the results to a CSV file ending with '_analysis_log.csv'
        analysis_log_filename = f"{analyses_folder}/image_log/{roi_name}_analysis_log.csv"
        os.makedirs(os.path.dirname(analysis_log_filename), exist_ok=True)
        with open(analysis_log_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['roi_name (h number)', 'true_count', 'h number', 'selected?', 'xi', 'lli', 'penalty', 'fisher_info', 'fit_parameters', 'xi_aic', 'xi_bic', 'penalty_aic', 'penalty_bic'])
            writer.writerows(individual_hypothesis_test_results)

        chosen_fit = fit_parameters[estimated_num_particles]
        return estimated_num_particles, chosen_fit, particle_intensities


    # Get the size of the image (width and height are both image_side_length)
    image_side_length = entire_image.shape[0]

    # Check if the image is too big to analyze at once. If it is, use tiling.
    if image_side_length < tiling_width_threshold:
        tiling = False
    else:
        tiling = True

    if tiling:
        # Divide the image into tiles, following the tile jump distance.
        tile_image_side_length = tile_width
        x_low_end_list = [0]
        y_low_end_list = [0]
        while x_low_end_list[-1] + tile_width < image_side_length:
            x_low_end_list.append(x_low_end_list[-1] + tile_jump_distance)
        while y_low_end_list[-1] + tile_width < image_side_length:
            y_low_end_list.append(y_low_end_list[-1] + tile_jump_distance)
        n_x, n_y = 1, 1
        while n_x * tile_jump_distance + tile_width < image_side_length:
            n_x += 1
        while n_y * tile_jump_distance + tile_width < image_side_length:
            n_y += 1
        print(f"{basename} is divided into {(n_x + 1) * (n_y + 1)} tiles.")

        # Create a dictionary to store tile information
        tile_dicts_array = np.zeros((n_y + 1, n_x + 1), dtype=object)
        y_low_end_list = [tile_jump_distance * (n) for n in range(n_y + 1)]
        x_low_end_list = [tile_jump_distance * (n) for n in range(n_x + 1)]
        img_height, img_width = entire_image.shape

        # Populate tile_dict with x_low_end, y_low_end, and image_slice (pixel values) - but the particle locations and intensities are not yet determined
        for y_index, y_low_end in enumerate(y_low_end_list):
            y_high_end = min(y_low_end + tile_image_side_length, img_height)
            for x_index, x_low_end in enumerate(x_low_end_list):
                x_high_end = min(x_low_end + tile_image_side_length, img_width)
                tile_dicts_array[x_index][y_index] = {'x_low_end': x_low_end, 'y_low_end': y_low_end, 'image_slice': entire_image[y_low_end:y_high_end, x_low_end:x_high_end], 'particle_locations': [], 'particle_intensities': []}

        for tile_dict in tile_dicts_array.flatten():
            # Set the tile file name for the current tile
            tilename = f"tile_x{tile_dict['x_low_end']}-{min(tile_dict['x_low_end'] + tile_width, img_width)}_y{tile_dict['y_low_end']}-{min(tile_dict['y_low_end'] + tile_width, img_height)}"
            print(f"Processing tile {tilename} in image {basename}", end='\r')
            roi_name = f"{basename} {tilename})"
            estimated_num_particles_for_roi, chosen_fit, _ = analyze_region_of_interest(tiling, tile_dict['image_slice'], psf_sigma, last_h_index, analysis_rand_seed_per_image, display_fit_results, display_xi_graph, use_exit_condition, roi_name=basename, color_mode=color_mode)
            particle_locations = []
            for particle_index in range(1, estimated_num_particles_for_roi + 1):
                loc = chosen_fit[particle_index][1:3]
                particle_locations.append(loc)
            tile_dict['particle_locations'] = particle_locations
        
        # Combine the results of the tiles
        resulting_locations, determined_intensities = merge_coincident_particles(entire_image, tile_dicts_array, psf_sigma)
        estimated_num_particles = len(resulting_locations)
    else:
        estimated_num_particles, chosen_fit, determined_intensities = analyze_region_of_interest(tiling, entire_image, psf_sigma, last_h_index, analysis_rand_seed_per_image, display_fit_results, display_xi_graph, use_exit_condition, roi_name=basename, color_mode=color_mode)

    analyze_image_result = {
        'actual_num_particles': actual_num_particles,
        'estimated_num_particles': estimated_num_particles,
        'image_filename': image_filename,
        'determined_particle_intensities': determined_intensities,
        }

    return analyze_image_result


def process(config_files_dir,
            parallel=False,
            move_finished_config_file=True):
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
                required_fields_common = ['image_folder_namebase',
                                          'code_version_date'
                                          ]

                required_fields_for_separation_test = ['separation_test_image_generation?',
                                                       'sep_image_count',
                                                       'sep_intensity_prefactor_to_bg_level',
                                                       'sep_psf_sigma',
                                                       'sep_distance_ratio_to_psf_sigma',
                                                       'sep_img_width',
                                                       'sep_bg_level',
                                                       'sep_random_seed'
                                                       ]

                required_fields_for_generation = ['generate_regular_dataset?',
                                                  'gen_random_seed',
                                                  'gen_total_image_count',
                                                  'gen_psf_sigma',
                                                  'gen_img_width',
                                                  'gen_minimum_particle_count',
                                                  'gen_maximum_particle_count',
                                                  'gen_bg_level',
                                                  'gen_particle_intensity_mean',
                                                  'gen_particle_intensity_sd'
                                                  ]

                required_fields_for_analysis = ['analyze_the_dataset?',
                                                'ana_random_seed',
                                                'ana_predefined_psf_sigma',
                                                'ana_use_premature_hypothesis_choice?',
                                                'ana_maximum_hypothesis_index'
                                                ]

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

                if 'separation_test_image_generation?' in config and config['separation_test_image_generation?']:
                    required_fields += required_fields_for_separation_test
                elif 'generate_regular_dataset?' in config and config['generate_regular_dataset?']:
                    required_fields += required_fields_for_generation
                elif 'analyze_the_dataset?' in config and config['analyze_the_dataset?']:
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
        if 'separation_test_image_generation?' in config and config['separation_test_image_generation?']:
            generate_separation_test_images(image_folder_basename=config['image_folder_namebase'],
                                            sep_distance_ratio_to_psf_sigma=config['sep_distance_ratio_to_psf_sigma'],
                                            total_image_count=config['sep_image_count'],
                                            amp_to_background_level=config['sep_intensity_prefactor_to_bg_level'],
                                            psf_sigma=config['sep_psf_sigma'],
                                            image_side_length=config['sep_img_width'],
                                            background_level=config['sep_bg_level'],
                                            generation_random_seed=config['sep_random_seed'],
                                            config_content=json.dumps(config)
                                            )

        # Generate regular dataset
        elif 'generate_regular_dataset?' in config and config['generate_regular_dataset?']:
            generate_test_images(image_folder_basename=config['image_folder_namebase'],
                                 total_image_count=config['gen_total_image_count'],
                                 minimum_number_of_particles=config['gen_minimum_particle_count'],
                                 maximum_number_of_particles=config['gen_maximum_particle_count'],
                                 particle_intensity_mean=config['gen_particle_intensity_mean'],
                                 particle_intensity_sd=config['gen_particle_intensity_sd'],
                                 psf_sigma=config['gen_psf_sigma'], image_side_length=config['gen_img_width'],
                                 background_level=config['gen_bg_level'],
                                 generation_random_seed=config['gen_random_seed'],
                                 config_content=json.dumps(config)
                                 )

        # Analyze dataset
        if 'analyze_the_dataset?' in config and config['analyze_the_dataset?']:
            timeout = config.get('ana_timeout_per_image', 3600)  # time out = 1 hour per image
            analyses_folder_path, short_folder_name = analyze_whole_folder(
                image_folder_basename=config['image_folder_namebase'],
                code_version_date=config['code_version_date'],
                use_exit_condition=config['ana_use_premature_hypothesis_choice?'],
                last_h_index=config['ana_maximum_hypothesis_index'],
                analysis_rand_seed=config['ana_random_seed'],
                psf_sigma=config['ana_predefined_psf_sigma'],
                config_content=json.dumps(config),
                parallel=parallel,
                timeout_per_image=timeout
                )

            print('Done analyzing the dataset.')

            # Get the dataset name and code version date
            image_folder_basename = config['image_folder_namebase']
            code_version_date = config['code_version_date']

            print('Combining logs and generating analysis figures...')
            # Combine analysis log files into one.
            combine_log_files(analyses_folder_path,
                              image_folder_basename,
                              code_version_date,
                              delete_individual_files=True
                              )
            print('Combined analysis log files.')

            # Generate confusion matrix
            label_prediction_log_file_path = os.path.join(analyses_folder_path, 
                                                          f'{short_folder_name}_label_prediction_log.csv')
            try:
                generate_confusion_matrix(label_prediction_log_file_path,
                                          image_folder_basename,
                                          code_version_date,
                                          display=False,
                                          savefig=True
                                          )
            except Exception as e:
                print(f"Error generating confusion matrix: {e}")

            # Count occurence
            print('Counting occurrences...')
            count_occurence(label_prediction_log_file_path)

            print('Done counting occurrences.')
            # Delete the dataset after analysis
            if config['ana_delete_the_dataset_after_analysis?']:
                print('Deleting dataset')

                dir_path = os.path.join("datasets", f"{config['image_folder_namebase']}")
                shutil.rmtree(dir_path)
                print('Deleting image data.')
                print('-------------------------------------')

        if move_finished_config_file:
            print('Moving config')

            # Move the processed config file to the "finished configs" subfolder
            finished_configs_dir = os.path.join(config_files_dir, "finished_configs")
            os.makedirs(finished_configs_dir, exist_ok=True)
            shutil.move(os.path.join(config_files_dir, config_file), os.path.join(finished_configs_dir, config_file))


#######################################################################
# ########### Define some analysis functions ##########################
#######################################################################
def generate_intensity_histogram(label_pred_log_file_path,
                                 image_folder_basename,
                                 code_version_date='',
                                 display=False,
                                 savefig=True
                                 ):
    """
    Generate and optionally display/save a histogram of particle intensities from detection results.
    This function reads particle intensity data from a CSV file containing detection results,
    processes the intensity values, and creates a histogram visualization.
    Args:
        label_pred_log_file_path (str): Path to the CSV file containing particle detection results.
            The CSV must contain a column named "Determined Particle Intensities" with intensity
            values stored as string representations of lists.
        image_folder_basename (str): Base name of the image folder, used for naming the output
            histogram file.
        code_version_date (str, optional): Version date string to include in the output filename.
            Defaults to empty string.
        display (bool, optional): If True, displays the histogram plot in a window.
            Defaults to False.
        savefig (bool, optional): If True, saves the histogram as a PNG file in the same directory
            as the input CSV file. Defaults to True.
    Returns:
        None

    Notes:
        - The intensity values in the CSV are expected to be stored as string representations
            of Python lists and are parsed using ast.literal_eval().
        - When savefig=True, the output file is saved with the naming pattern:
            '{image_folder_basename}_code_ver{code_version_date}_particle_intensities_hist.png'
        - The histogram uses 20 bins by default.
        - The saved figure has a resolution of 300 DPI.
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
    ax.set_ylabel('Occurences')
    ax.set_title('Intensity Histogram')
    if display:
        plt.show(block=False)
    if savefig:
        png_file_path = os.path.dirname(label_pred_log_file_path)
        png_file_name = f'/{image_folder_basename}_code_ver{code_version_date}_particle_intensities_hist.png'
        png_file_path += png_file_name
        plt.savefig(png_file_path, dpi=300)


def generate_confusion_matrix(label_pred_log_file_path,
                              image_folder_basename,
                              code_version_date,
                              display=False,
                              savefig=True
                              ):
    """ Generate the confusion matrix and calculate the metrics.

    Parameters:
        label_pred_log_file_path (str): The path of the label prediction log file.
        image_folder_basename (str): The name of the folder containing the images.
        code_version_date (str): The version of the code.
        display (bool): Whether to display the confusion matrix. Default is False.
        savefig (bool): Whether to save the confusion matrix as a PNG file. Default is True.

    Returns:
        None
    """
    # Read the CSV file
    df = pd.read_csv(label_pred_log_file_path)

    # Check if the CSV file is empty or only contains headers
    if df.empty or len(df) == 1:
        print("The CSV file is empty or only contains headers. No data to process.")
        return

    # Check if the CSV file contains the columns 'Actual Particle Count' and 'Estimated Particle Count'
    if 'Actual Particle Count' not in df.columns and 'Actual Particle Number' not in df.columns:
        print("The column name 'Actual Particle Count' or 'Actual Particle Number' is not found in the CSV file.")
        return

    # Check if any of the actual counts are -1 (unknown)
    # If there are unknown counts, skip generating the confusion matrix
    if (df['Actual Particle Count'] == -1).any():
        print("Some images have unknown actual particle counts (-1). Skipping confusion matrix generation.")
        return

    actual = df['Actual Particle Count']
    estimated = df['Estimated Particle Count']

    # Generate the confusion matrix
    matrix = confusion_matrix(actual, estimated)

    # Save the confusion matrix as a CSV file ending with '_confusion_mat.csv'
    matrix_df = pd.DataFrame(matrix)
    csv_file_path = os.path.dirname(label_pred_log_file_path)
    csv_file_name = f'/{image_folder_basename}_code_ver{code_version_date}_confusion_mat.csv'
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

    print(f"Flat weight Accuracy: {accuracy:.3f}")
    print(f"Within-One Accuarcy: {miss_by_one_rate:.3f}")
    print(f"Overestimation Rate: {overestimation_rate:.3f}")
    print(f"Underestimation Rate: {underestimation_rate:.3f}")
    print(f"Mean Absolute Error (flat weight): {mae:.3f}")
    print(f"Root Mean Squared Error (flat weight): {rmse:.3f}")

    # Prepare metrics for saving
    scores = {
        'Flat weight Accuracy': accuracy,
        'Within-One Accuracy': miss_by_one_rate,
        'Overestimation Rate': overestimation_rate,
        'Underestimation Rate': underestimation_rate,
        'Mean Absolute Error (flat weight)': mae,
        'Root Mean Squared Error (flat weight)': rmse
    }

    # Save the metrics as a CSV file ending with '_scores.csv'
    metrics_df = pd.DataFrame(scores, index=[0])
    csv_file_path = os.path.dirname(label_pred_log_file_path)
    csv_file_name = f'/{image_folder_basename}_code_ver{code_version_date}_scores.csv'
    csv_file_path += csv_file_name
    metrics_df.to_csv(csv_file_path, index=False)

    # Normalize the confusion matrix
    normalized_matrix = np.zeros(matrix.shape)

    # Plot the heatmap of the confusion matrix
    row_sums = matrix.sum(axis=1)
    if display or savefig:
        _, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [4,1]})

        # Debugging output
        print("Original matrix:\n", matrix)
        print("Row sums before normalization:", row_sums)

        # Adjusted normalization with debugging
        for row in range(matrix.shape[0]):
            normalized_matrix[row] = matrix[row] / row_sums[row] if row_sums[row] != 0 else np.zeros(matrix.shape[1])

        # Debugging output to check if normalization is as expected
        print("Normalized matrix:\n", np.array2string(normalized_matrix, formatter={'float_kind':lambda x: f"{x:.3f}"}))

        folder_name = os.path.basename(os.path.dirname(label_pred_log_file_path))
        ax = axs[0]
        # Plot the heatmap on the new axes.
        sns.heatmap(normalized_matrix, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax, vmin=0, vmax=1)
        ax.set_title(f'{folder_name}')
        ax.set_xlabel('Estimated Particle Count')
        ax.set_ylabel('Actual Particle Count')
        ytick_labels = [f"{i} (count: {row_sums[i]})" for i in range(len(row_sums))]
        ax.set_yticklabels(ytick_labels, rotation=0)

        # Draw lines between rows
        for i in range(matrix.shape[0]+1):
            ax.axhline(i, color='black', linewidth=1)

        # Text messages
        text_message = f"Accuracy: {accuracy:.3f}\n" + \
            f"Overestimation Rate: {overestimation_rate:.3f}\n" + \
            f"Underestimation Rate: {underestimation_rate:.3f}\n" + \
            f"Miss-by-One Rate: {miss_by_one_rate:.3f}\n" + \
            f"Mean Absolute Error: {mae:.3f}\n" + \
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
            png_file_name = f'/{image_folder_basename}_code_ver{code_version_date}_confusion_mat.png'
            png_file_path += png_file_name
            plt.savefig(png_file_path, dpi=300)


def combine_log_files(analyses_folder, image_folder_basename, code_version_date, delete_individual_files=False):
    ''' Combines the log files in the image_log folder into one file called fitting_results.csv.

    Parameters:
        analyses_folder (str): The path of the folder containing the analyses outputs.
        image_folder_basename (str): The name of the folder containing the images.
        code_version_date (str): The version of the code.
        delete_individual_files (bool): Whether to delete the individual log files. Default is False.

    Returns:
        None
    '''
    # First ensure the analyses_folder exists
    os.makedirs(analyses_folder, exist_ok=True)

    # Create the fitting_results.csv file (with a shorter name if necessary)
    short_basename = image_folder_basename[:20] if len(image_folder_basename) > 20 else image_folder_basename
    whole_metrics_log_filename = os.path.normpath(os.path.join(
        analyses_folder,
        f'{short_basename}_metrics_log.csv'  # Shortened filename
        ))
    print(f"Creating log with all metrics: {whole_metrics_log_filename}")
    os.makedirs(os.path.dirname(whole_metrics_log_filename), exist_ok=True)

    # Get all the *_fittings.csv files in the image_log folder
    individual_image_log_files = glob.glob(os.path.join(analyses_folder, 'image_log', '*_analysis_log.csv'))

    # Open the fitting_results.csv file in write mode
    with open(whole_metrics_log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_filename (h number)',
                         'true_count',
                         'h number',
                         'selected?',
                         'xi',
                         'lli',
                         'penalty',
                         'fisher_info',
                         'fit_parameters',
                         'xi_aic',
                         'xi_bic',
                         'penalty_aic',
                         'penalty_bic'
                         ])

        # Iterate over the fittings_files
        for log_file in individual_image_log_files:
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


def count_occurence(label_pred_log_file_path):
    """Count occurrences of each estimated particle count and save to CSV."""
    try:
        # Read the CSV file
        df = pd.read_csv(label_pred_log_file_path)

        # Count occurrences of each estimated count
        count_series = df['Estimated Particle Count'].value_counts().sort_index()

        # Convert to DataFrame with columns 'Estimated Count' and 'Occurrence'
        count_df = pd.DataFrame({
            'Estimated Count': count_series.index,
            'Occurrence': count_series.values
        })

        # Save to CSV
        output_path = label_pred_log_file_path.split('label_prediction_log')[0] + 'occurence_count.csv'
        count_df.to_csv(output_path, index=False)
        print(count_df)
        print(f"Occurrence counts saved to: {output_path}")

    except Exception as e:
        print(f"Error in counting occurrences: {e}")


def main(move_finished_config_file=True):
    """
    Main function to run the particle detection analysis pipeline. 
    This function serves as the entry point for the particle counting algorithm. It handles
    command-line arguments, manages configuration files, and executes the analysis process
    with optional profiling capabilities.

    Args:
        move_finished_config_file (bool, optional): Flag to determine whether to move
            processed configuration files after completion. Defaults to True.

    Command-line Arguments:
        --config-file-folder, -c (str): Path to the folder containing configuration files
            to be processed. This argument is required.
        --profile, -p (bool): Flag to enable profiling of the analysis pipeline. When True,
            generates a profile report saved as 'profile_results.prof'. Defaults to False.
        --parallel, -x (bool): Flag to enable parallel processing. When True, the analysis
            runs in parallel mode. Defaults to False.

    Returns:
        None

    Side Effects:
        - Processes configuration files from the specified directory
        - May move configuration files to a different location based on move_finished_config_file
        - Generates profiling statistics if profiling is enabled
        - Prints execution time and progress information to stdout

    Notes:
        - When profiling is enabled, the process runs in sequential mode (parallel=False)
        - Execution time is displayed upon completion
    """
    # Start the batch job timer
    batchjobstarttime = datetime.now()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process config files.')
    parser.add_argument('--config-file-folder', '-c', type=str, help='Folder containing config files to run.')
    parser.add_argument('--parallel', '-x', type=bool, default=False, help='Boolean to decide whether to run in parallel or not.')
    parser.add_argument('--profile', '-p', type=bool, default=False, help='Boolean to decide whether to profile or not.')
    args = parser.parse_args()

    print("Arguments provided by the user:")
    print(f"Config file folder: {args.config_file_folder}")
    print(f"Profile: {args.profile}")

    # Check if config-file-folder is provided
    if (args.config_file_folder is None):
        print("Please provide the folder name for config files using --config-file-folder or -c option.")
        exit()
    config_files_dir = args.config_file_folder

    parallel_flag = args.parallel

    if args.profile is True:
        with Profile() as profile:
            process(config_files_dir=config_files_dir,
                    parallel=parallel_flag,
                    move_finished_config_file=move_finished_config_file)
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.TIME)
                .dump_stats('profile_results.prof')
            )
            # os.system('snakeviz profile_results.prof &')
    else:
        process(config_files_dir,
                parallel=parallel_flag,
                move_finished_config_file=move_finished_config_file
                )

    # Print the time taken for the batch job
    # End the batch job timer
    batchjobendtime = datetime.now()

    # Calculate the time taken for the batch job
    time_taken = batchjobendtime - batchjobstarttime

    # Print the time taken for the batch job in seconds
    print(f'Batch job completed in {time_taken.total_seconds():.2f} seconds')


# Run the main function if the script is executed from the command line
if __name__ == '__main__':
    if 'pydevd' in sys.modules or 'debugpy' in sys.modules:

        # Run the main function without parallel processing ('-p' option value is False)
        sys.argv = ['main.py', '-c', './configs/']  # -p for profiling. Default is False, and it will run on multiple processes.

        # Run the main function with profiling (Useful for debugging)
        # sys.argv = ['main.py', '-c', './configs/', '-p', 'True'] # -p for profiling. When True, it will run on **seriallly (instead of parallel)** to profile the code.

    # Call the main function
    main()
