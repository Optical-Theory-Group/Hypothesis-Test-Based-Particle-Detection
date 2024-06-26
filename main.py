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

def main_glrt_tester(input_image, psf_sd=1.39, significance=0.05, consideration_limit_level=2, fittype=0, ):
    """ Performs image processing on the input image, including the preprocessing, detection, and fitting steps.
    Args:
        input_image: The image to process.
        psf_sd: The sigma value of the PSF.
        significance: The maximum p-value to consider a detection significant.
        consideration_limit_level: The compression reduction factor.
        fittype: The type of fit to use (0: Bg and Intensity, 1: Bg, Intensity, X, Y).
    Returns:
        (nothing)
    """
    # Stripping the edges, because the edges are cannot be processed using our strategy.
    required_box_size = int(np.ceil(3 * (2 * psf_sd + 1)))
    # required_box_size = 3 * (2 * psf_sd + 1)
    xbegin = math.ceil(required_box_size / 2) 
    ybegin = xbegin 
    # Note that in line with python indexing, the end is not included in the range
    xend = math.floor(input_image.shape[1] - required_box_size / 2)
    yend = math.floor(input_image.shape[0] - required_box_size / 2)
    # Get the inner indices
    inner_xidxs = np.arange(xbegin, xend) 
    inner_yidxs = np.arange(ybegin, yend) 

    # Crop the image
    inner_image = input_image[np.ix_(inner_xidxs, inner_yidxs)]    

    # Define the filter
    h2 = 1/16
    h1 = 1/4
    h0 = 3/8
    g = {}
    g[0] = [h2, h1, h0, h1, h2]
    g[1] = [h2, 0, h1, 0, h0, 0, h1, 0, h2]

    # ROI location mask initialization. At first, all the inner_image is considered as ROI.
    consideration_mask = np.ones(inner_image.shape)

    # If consideration_limit_level is above 0, 
    # then we consider only the pixels that are above the mean + std of the difference between V1 and V2  
    if consideration_limit_level:
        # Compute V1 and V2
        k0 = np.array(g[0])
        adjusted_kernel_0 = create_separable_filter(k0, 3)
        dip_image = dip.Image(inner_image)
        v0 = dip.Convolution(dip_image, adjusted_kernel_0, method="best")
        k1 = np.array(g[1])
        adjusted_kernel_1 = create_separable_filter(k1, 5)
        v1 = dip.Convolution(v0, adjusted_kernel_1, method="best")

        # Compute the difference between V1 and V2
        w = v0 -v1

        # Compute the mean and std of the difference between V1 and V2
        consideration_mask = w > np.mean(dip.Image(w), axis=(0,1)) + consideration_limit_level * np.std(dip.Image(w), axis=(0,1))

        viz_consideration_mask = True
        # Visualize the consideration mask
        if viz_consideration_mask:
           _,ax=plt.subplots()
           ax.imshow(consideration_mask), ax.set_title('consideration_mask'), plt.show(block=False)

    # Get the indices of the pixels that are considered as ROI
    inner_image_pos_idx = np.array(consideration_mask).nonzero()
    idxs = np.vstack((inner_image_pos_idx[0] + xbegin, inner_image_pos_idx[1] + ybegin))
 
    # Create subregions
    roi_stack, leftcoords, topcoords = make_subregions(idxs, int(np.ceil(3 * (2 * psf_sd + 1))), input_image.astype('float32'))
   
    # Create a list of tuples of the subregions and their coordinates 
    roi_infos = [(roi_stack[i], leftcoords[i], topcoords[i]) for i in range(len(roi_stack))]

    # Initialize the pfa and significance_mask arrays
    pfa_array = np.ones(inner_image.shape)
    significance_mask = np.zeros(inner_image.shape, dtype=bool)
    # Perform the generalized likelihood ratio test on each subregion
    for i in range(len(roi_infos)):
        if i % 10 == 0:
            print(f'Processing subregion {i / len(roi_infos) * 100:.3f}%')
        _, _, _, pfa = generalized_likelihood_ratio_test(roi_image=roi_infos[i][0], psf_sd=psf_sd, iterations=10, fittype=fittype)
        pfa_array[inner_image_pos_idx[0][i], inner_image_pos_idx[1][i]] = pfa

    # Create a custom colormap for the pfa image, with red-to-yellow for the values below 'significance', and green-to-blue for the values above.
    n_colors = 1024
    # The colormap is divided into three parts: from 0 to significance, from significance to 1, and from 1 to the maximum value in pfa_array
    q = significance
    color_list = [plt.cm.autumn(i / (n_colors * q - 1)) for i in range(int(n_colors * q))]
    color_list += [plt.cm.winter(i / (n_colors * (1 - q) - 1)) for i in range(int(n_colors * q), n_colors)]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', color_list)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    # Plotting pfa image 
    plt.figure(), plt.title('pfa')
    plt.imshow(pfa_array, cmap=custom_cmap, norm=norm)
    plt.colorbar(), plt.show(block=False)

    _, _, pfa_adj = fdr_bh(pfa_array, significance, method='dep')
    pfa_adj = pfa_adj.flatten()
    pfa_adj_array = np.ones(inner_image.shape)    
    for i, p_adj in enumerate(pfa_adj):
        pfa_adj_array[inner_image_pos_idx[0][i], inner_image_pos_idx[1][i]] = p_adj

    # Create a custom colormap for the pfa image, with red-to-yellow for the values below 'significance-val * max-val', and tinted_green-to-blue for the values above.
    max_pfa_adj = np.max(pfa_adj_array)
    # The colormap is divided into two parts: from 0 to significance * max_pfa_adj and from significance * max_pfa_adj to 1 
    q = significance
    n_colors = int(1024 * max_pfa_adj)
    color_list = [plt.cm.autumn(i / (n_colors * q - 1)) for i in range(0, int(n_colors * q))]
    color_list += [plt.cm.winter(i / (n_colors * (1-q) - 1)) for i in range(int(n_colors * q), n_colors)] 
    color_list += [plt.cm.cool(i / (n_colors * (max_pfa_adj - 1) - 1)) for i in range(int(n_colors), int(n_colors * max_pfa_adj))]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', color_list)
    norm = mcolors.Normalize(vmin=0, vmax=max_pfa_adj)
    # Plotting pfa_adj array 
    plt.figure(), plt.title('pfa_adj')
    plt.imshow(pfa_adj_array, cmap=custom_cmap, norm=norm)
    plt.colorbar(), plt.show(block=False)

    significance_mask[inner_image_pos_idx] = (pfa_adj <= significance).flatten()
    
    show_significance_mask = True
    if show_significance_mask:
        plt.figure(), plt.imshow(significance_mask), plt.title('significance_mask'), plt.show(block=False)

    # Examination of the significance mask might be implemented here. 
 
def make_and_process_image_glrt(x=3.35, y=6.69, sz=12, intensity=10, bg=4, psf=1.39, show_fig=False, verbose=False):

    peaks_info = [{'x': x, 'y': y, 'prefactor': intensity, 'psf_sd': psf}]
    image = psfconvolution(peaks_info, sz)
    # Adding background
    image += np.ones(image.shape)*bg
    image = np.random.poisson(image, size=(image.shape))
    if show_fig:    
        plt.imshow(image)
        plt.colorbar()

    fittype = 0
    params,crlb,statistics,p_value = generalized_likelihood_ratio_test(roi_image=image, psf_sd=1.39, iterations=10, fittype=fittype)

    if verbose:
        for i, p in enumerate(params):
            if i == len(params) - 1:
                print('-----')
            if i != 1: 
                print(f"{p:.2f}")
            else:
                print(f'{sz-p:.2f}')
        print('-----') 

    if fittype == 1 and show_fig:
        plt.plot(x,y,'ro', markersize=15, label='true center')
        plt.plot(params[0],params[0],'*', markersize=15, label='guessed center', markerfacecolor='aqua')

    if show_fig:
        plt.legend()
    crlbs = []
    for i, v in enumerate(crlb):
        crlbs += np.sqrt(v)
        if verbose:
            print(f'{np.sqrt(v):.2f}\n----\n{p_value=}')

    if show_fig:
        titlestr = f"Ground truth: x={x:.2f}, y={y:.2f}, intensity={intensity}, bg={bg}\n"\
            f"Fitted: {params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f}, {params[3]:.2f}\n"\
            f"p-value: {p_value:.15f}"
        plt.title(titlestr), plt.tight_layout(), plt.show(block=False)
        
    t_g = statistics[0]

    return t_g, p_value

def generate_test_images(image_folder_namebase,  code_version, maximum_number_of_particles, amp_to_bg_min, amp_to_bg_max, amp_sd, n_total_image_count, psf_sd, sz, bg, generation_random_seed, config_content='', minimum_number_of_particles=0):
    # Set the random seed
    np.random.seed(generation_random_seed)
    # Set the minimum relative intensity of a particle
    relative_intensity_min = 0.1
    number_of_counts = maximum_number_of_particles - minimum_number_of_particles + 1
    number_of_images_per_count = int(np.ceil(n_total_image_count / number_of_counts))
    
    # Print the number of images to be generated and folder to store the images. 
    print(f'Generating images containing {minimum_number_of_particles} to {maximum_number_of_particles} particles. It will produce {number_of_images_per_count} images per count.')
    print(f'Total number of images generated from this config is {number_of_images_per_count * number_of_counts}. Note that this number may be slightly higher than the total number of images requested.')
    print(f'Saving dataset to: ./image_dataset/{image_folder_namebase}_code_ver{code_version}.')

    # Create the folder to store the images
    image_folder_path = os.path.join("image_dataset", f"{image_folder_namebase}_code_ver{code_version}")
    os.makedirs(image_folder_path, exist_ok=True)

    for n_particles in range(minimum_number_of_particles, maximum_number_of_particles+1):
        for img_idx in range(number_of_images_per_count):
            image = np.ones((sz, sz), dtype=float) * bg
            chosen_mean_intensity = (np.random.rand() * (amp_to_bg_max - amp_to_bg_min) + amp_to_bg_min) * bg
            for _ in range(n_particles):
                # [ToDo] Refine the following ranges.
                x = np.random.rand() * (sz - psf_sd * 4) + psf_sd * 2 - 0.5
                y = np.random.rand() * (sz - psf_sd * 4) + psf_sd * 2 - 0.5
                # x = np.random.rand() * (sz - 1) #+ psf_sd * 2 - 0.5
                # y = np.random.rand() * (sz - 1) #+ psf_sd * 2 - 0.5
                # x = np.random.rand() * (sz - psf_sd * 6 - 1) + psf_sd * 3 - .5
                # y = np.random.rand() * (sz - psf_sd * 6 - 1) + psf_sd * 3 - .5
                relative_intensity = np.random.normal(1, amp_sd)
                if relative_intensity < relative_intensity_min:
                    relative_intensity = relative_intensity_min
                    print('Warning: Randomly drawn particle intensity is less than 0.1 * "expected intensity". Forcing it to be 0.1 * "expected intensity"')
                amplitude = relative_intensity * chosen_mean_intensity
                peak_info = [{'x': x, 'y': y, 'prefactor': amplitude, 'psf_sd': psf_sd}]
                image += psfconvolution(peak_info, sz)
            # Add Poisson noise
            image = np.random.poisson(image, size=(image.shape)) # This is the resulting (given) image.
            img_filename = f"count{n_particles}-index{img_idx}.tiff"
            pil_image = im.fromarray(image.astype(np.uint16))
            pil_image.save(os.path.join(image_folder_path, img_filename))
    
    # Save the content of the config file
    config_file_save_path = os.path.join(image_folder_path, 'config_used.json')
    with open(config_file_save_path, 'w') as f:
        json.dump(json.loads(config_content), f, indent=4)

def generate_separation_test_images(subfolder_name='separation_test', separation=3, n_images_per_separation=20, amp_to_bg=5, psf_sd=1, sz=20, bg=500, generation_random_seed=42, ):
    if separation > sz:
        raise ValueError(f"Separation {separation} is greater than the size of the image {sz}.")
    # Set the random seed
    np.random.seed(generation_random_seed)
    # Create the folder to store the images
    image_folder_path = os.path.join("image_dataset", f"{subfolder_name}{separation}")
    os.makedirs(image_folder_path, exist_ok=True)
    print(f'Generating {n_images_per_separation} images with separation {separation} in folder {image_folder_path}.')
    # Generate the images
    for img_idx in range(n_images_per_separation):
        image = np.ones((sz, sz), dtype=float) * bg
        particle_intensity = bg * amp_to_bg
        angle = np.random.uniform(0, 2*np.pi)
        # Particle 1
        while True:
            x1 = sz / 2 + separation / 2 * np.cos(angle)
            y1 = sz / 2 + separation / 2 * np.sin(angle)
            if 2 * psf_sd + 1 <= x1 <= sz - 2 * psf_sd - 1 and 2 * psf_sd + 1 <= y1 <= sz - 2 * psf_sd - 1:
                break
        peak_info = [{'x': x1, 'y': y1, 'prefactor': particle_intensity, 'psf_sd': psf_sd}]
        image += psfconvolution(peak_info, sz)
        # Particle 2
        x2 = sz / 2 - separation / 2 * np.cos(angle)
        y2 = sz / 2 - separation / 2 * np.sin(angle)
        peak_info = [{'x': x2, 'y': y2, 'prefactor': particle_intensity, 'psf_sd': psf_sd}]
        image += psfconvolution(peak_info, sz)
        # Add Poisson noise
        image = np.random.poisson(image, size=(image.shape)) # This is the resulting (given) image.
        img_filename = f"separation{separation}-index{img_idx}.tiff"
        pil_image = im.fromarray(image.astype(np.uint16))
        pil_image.save(os.path.join(image_folder_path, img_filename))

def report_progress(progresscount, totalrealisations, starttime=None, statusmsg=''):
    """
        Displays progress bar for current progress.

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

def analyze_whole_folder(image_folder_namebase, code_version_date, use_exit_condi=True, last_h_index=7, psf_sd=1.39, analysis_rand_seed=0, config_content='', parallel=True, display_fit_results=False, display_xi_graph=False):
    '''Analyzes all the images in the dataset folder.'''
    # Set random seed
    np.random.seed(analysis_rand_seed)

    # Get a list of image files in the folder
    images_folder = os.path.join('./image_dataset', image_folder_namebase + '_code_ver' + code_version_date)
    image_files = glob.glob(os.path.join(images_folder, '*.png')) + glob.glob(os.path.join(images_folder, '*.tiff'))

    print(f"Images loaded (total of {len(image_files)}):")

    # Create a folder to store the logs
    log_folder = os.path.join('./runs', image_folder_namebase + '_code_ver' + code_version_date)
    os.makedirs(log_folder, exist_ok=True)

    # Save the content of the config file
    if config_content:
        config_file_save_path = os.path.join(log_folder, f'{image_folder_namebase}_code_ver{code_version_date}_config_used.json')
        with open(config_file_save_path, 'w') as f:
            json.dump(json.loads(config_content), f, indent=4)

    # Prepare the label (actual count) prediction (estimated count) log file
    label_prediction_log_file_path = os.path.join(log_folder, f'{image_folder_namebase}_code_ver{code_version_date}_label_prediction_log.csv')
    with open(label_prediction_log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Input Image File', 'Actual Particle Count', 'Estimated Particle Count'])

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
            futures = [executor.submit(analyze_image, filename, psf_sd, last_h_index, analysis_rand_seed_per_image, use_exit_condi, log_folder, )
                        for analysis_rand_seed_per_image, filename in zip(image_rand_seeds, image_files)]
            progress = 0
            # Write the results to the main log file
            with open(label_prediction_log_file_path, 'a', newline='') as f: 
                # Iterate over the futures that are completed. 
                for cfresult in concurrent.futures.as_completed(futures):
                    # If an exception is raised, print the exception and continue to the next image
                    if cfresult._exception is not None:
                        if isinstance(cfresult._exception, Warning):
                            print("Encountered a Warning:", cfresult._exception)
                        else:
                            print("Encountered an Exception:", cfresult._exception)
                            print("Proceeding without addressing the exception.")
                    # Get the result of the future and write the results to the main log file
                    try:
                        analysis_result = cfresult.result()
                        actual_num_particles = analysis_result['actual_num_particles']
                        estimated_num_particles = analysis_result['estimated_num_particles']
                        input_image_file = analysis_result['image_filename']

                        writer = csv.writer(f)
                        writer.writerow([input_image_file + ".tiff", actual_num_particles, estimated_num_particles])

                        if actual_num_particles == estimated_num_particles:
                            statusmsg = f'\"{input_image_file}\" - Actual Count {actual_num_particles} == Estimated {estimated_num_particles}'
                        elif actual_num_particles > estimated_num_particles:
                            statusmsg = f'\"{input_image_file}\" - Actual Count: {actual_num_particles} > Estimated {estimated_num_particles}'
                        else:
                            statusmsg = f'\"{input_image_file}\" - Actual Count {actual_num_particles} < Estimated {estimated_num_particles}'
                    except Exception as e:
                        print(f"Error in cfresult.result(): {e}")
                        statusmsg = f'Error: {e}'

                    # statusmsg += f' Test results saved to {filename}'
                    report_progress(progress, len(image_files), starttime, statusmsg)
                    progress += 1
    else:
        progress = 0
        for analysis_rand_seed_per_image, filename in zip(image_rand_seeds, image_files):
            analysis_result = analyze_image(filename, psf_sd, last_h_index, analysis_rand_seed_per_image, use_exit_condi, log_folder, display_fit_results=display_fit_results, display_xi_graph=display_xi_graph)
            actual_num_particles = analysis_result['actual_num_particles']
            estimated_num_particles = analysis_result['estimated_num_particles']
            input_image_file = analysis_result['image_filename']

            with open(label_prediction_log_file_path, 'a', newline='') as f: 
                writer = csv.writer(f)
                writer.writerow([input_image_file + ".tiff", actual_num_particles, estimated_num_particles])

                statusmsg = f'{image_folder_namebase} '
                if actual_num_particles == estimated_num_particles:
                    statusmsg += f'\"{input_image_file}\" - Actual Count {actual_num_particles} == Estimated {estimated_num_particles}\n'
                elif actual_num_particles > estimated_num_particles:
                    statusmsg += f'\"{input_image_file}\" - Actual Count: {actual_num_particles} > Estimated {estimated_num_particles}\n'
                else:
                    statusmsg += f'\"{input_image_file}\" - Actual Count {actual_num_particles} < Estimated {estimated_num_particles}\n'

            report_progress(progress, len(image_files), starttime, statusmsg)
            progress += 1
            
    return log_folder

def analyze_image(image_filename, psf_sd, last_h_index, analysis_rand_seed_per_image, use_exit_condi, log_folder, display_fit_results=False, display_xi_graph=False):
    # Print the name of the image file
    image = np.array(im.open(image_filename))

    # Extract the number of particles from image_filename
    basename = os.path.basename(image_filename)
    count_part = basename.split('-')[0]
    if count_part.startswith("separation"):
        num_particles = 2
    else:
        num_particles = count_part.split('count')[1]
    actual_num_particles = int(num_particles)

    # Find tentative peaks
    tentative_peaks = get_tentative_peaks(image, min_distance=1)
    rough_peaks_xy = [peak[::-1] for peak in tentative_peaks]

    # Run GMRL
    estimated_num_particles, fit_results, test_metrics = generalized_maximum_likelihood_rule(roi_image=image, rough_peaks_xy=rough_peaks_xy, \
                                                        psf_sd=psf_sd, last_h_index=last_h_index, random_seed=analysis_rand_seed_per_image, display_fit_results=display_fit_results, display_xi_graph=display_xi_graph, use_exit_condi=use_exit_condi) 

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
                              }
    
    return image_analysis_results

def generate_confusion_matrix(label_pred_log_file_path, image_folder_namebase, code_version_date, display=False, savefig=True):
    # Read the CSV file
    df = pd.read_csv(label_pred_log_file_path)
    if os.path.isfile(label_pred_log_file_path):
        # Read the CSV file
        df = pd.read_csv(label_pred_log_file_path)
        # Rest of the code...
    else:
        print("File not found.")
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
        _, axs = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [4,1]})  # Increase the size of the figure.

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
        sns.heatmap(normalized_matrix, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax, vmin=0, vmax=1)  # Plot the heatmap on the new axes.
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
            process(parallel=True, config_files_dir=config_files_dir)
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.TIME)
                .dump_stats('profile_results.prof')
            )
            # os.system('snakeviz profile_results.prof &')
    else:
        process(config_files_dir)

    batchjobendtime = datetime.now()
    print(f'\nBatch job completed in {batchjobendtime - batchjobstarttime}')

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

def process(config_files_dir, parallel=True):
    '''Process the config files in the config_files_dir directory.'''
    config_files = os.listdir(config_files_dir)

    print(f"Config files loaded (total of {len(config_files)}):")
    for config_file in config_files:
        print("> " + config_file)

    for i, config_file in enumerate(config_files):
        print(f"Processing {config_file} ({i+1}/{len(config_files)})")
        print(config_file)
        try:
            with open(os.path.join(config_files_dir, config_file), 'r') as f:
                config = json.load(f)
                
                # Pretty print the config file
                pprint.pprint(config)

                # Check if the required fields are present in the config file
                required_fields = ['image_folder_namebase', 'code_version_date',\
                                    'generate_the_dataset', 'gen_random_seed', 'gen_total_image_count', 'gen_psf_sd', 'gen_img_width', 
                                    "gen_minimum_particle_count",
                                    "gen_maximum_particle_count",
                                    'gen_bg_level', 'gen_intensity_prefactor_to_bg_level_ratio_min', 'gen_intensity_prefactor_to_bg_level_ratio_max', 'gen_intensity_prefactor_coefficient_of_variation', 
                                    'analysis_random_seed', 'analysis_predefined_psf_sd', 'analysis_use_premature_hypothesis_choice', 'analysis_maximum_hypothesis_index',]

                # Modify field values with '.'
                for field in required_fields:
                    if field in config and isinstance(config[field], str):
                        if '.' in config[field]:
                            before_change = config[field]
                            config[field] = config[field].replace('.', '_')
                            print(f"Modified field '{field}' value to replace '.' with '_' - before: {before_change}, after: {config[field]}")
                
                # Check if the required fields are present in the config file
                for field in required_fields:
                    if field not in config:
                        # If config['generate_the_dataset'] is True, then all fields starting with 'gen' are required.
                        if config['generate_the_dataset'] and field.startswith('gen'):
                            print(f"Error: '{field}' should be set for image generation.")
                            exit()
                        # If config['analyze_the_dataset'] is True, then all fields starting with 'analysis' are required.
                        if config['analyze_the_dataset'] and field.startswith('analysis'):
                            print(f"Error: '{field}' should be set for image analysis.")
                            exit()
        # If the config file is not found or invalid, print an error message and continue to the next config file
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error: {config_file} file not found or invalid")
            continue

        # Generate the dataset 
        if config['generate_the_dataset']:
            generate_test_images(image_folder_namebase=config['image_folder_namebase'], 
                                code_version=config['code_version_date'],
                                n_total_image_count=config['gen_total_image_count'],
                                minimum_number_of_particles=config['gen_minimum_particle_count'], 
                                maximum_number_of_particles=config['gen_maximum_particle_count'], 
                                amp_to_bg_min=config['gen_intensity_prefactor_to_bg_level_ratio_min'], 
                                amp_to_bg_max=config['gen_intensity_prefactor_to_bg_level_ratio_max'], 
                                amp_sd=config['gen_intensity_prefactor_coefficient_of_variation'], 
                                
                                psf_sd=config['gen_psf_sd'], sz=config['gen_img_width'], 
                                bg=config['gen_bg_level'], 
                                generation_random_seed=config['gen_random_seed'], 
                                config_content=json.dumps(config))
        # Analyze the dataset
        if config['analyze_the_dataset']:
            log_folder_path = analyze_whole_folder(image_folder_namebase=config['image_folder_namebase'], 
                                                code_version_date=config['code_version_date'], 
                                                use_exit_condi=config['analysis_use_premature_hypothesis_choice'], 
                                                last_h_index=config['analysis_maximum_hypothesis_index'], 
                                                analysis_rand_seed=config['analysis_random_seed'], psf_sd=config['analysis_predefined_psf_sd'], 
                                                config_content=json.dumps(config), parallel=True)
            # Get the dataset name and code version date
            image_folder_namebase = config['image_folder_namebase']
            code_version_date = config['code_version_date']

            # Combine analysis log files into one.
            combine_log_files(log_folder_path, image_folder_namebase, code_version_date, delete_individual_files=True)
            
            # Generate confusion matrix
            label_prediction_log_file_path = os.path.join(log_folder_path, f'{image_folder_namebase}_code_ver{code_version_date}_label_prediction_log.csv')
            generate_confusion_matrix(label_prediction_log_file_path, image_folder_namebase, code_version_date, display=True, savefig=True)

            # Delete the dataset after analysis
            if config['analysis_delete_the_dataset_after_analysis']:
                dir_path =os.path.join("image_dataset", f"{config['image_folder_namebase']}_code_ver{config['code_version_date']}")
                shutil.rmtree(dir_path)
                print('Deleting image data.')

def quick_analysis():
    config_files_dir = './config_files/300524'
    config_files = os.listdir(config_files_dir)
    config_file = config_files[0]
    with open(os.path.join(config_files_dir, config_file), 'r') as f:
        config = json.load(f)
        # Pretty print the config file
        pprint.pprint(config)
        analysis_rand_seed_per_image = 7258
        filename = "./image_dataset/PSF 1.1/count1-index88.tiff"
        # log_folder = analyze_whole_folder(image_folder_name=config['image_folder_name'], code_version_date=config['code_version_date'], use_exit_condi=config['analysis_use_premature_hypothesis_choice'], last_h_index=config['analysis_maximum_hypothesis_index'], \
        #                     analysis_rand_seed=config['analysis_random_seed'], psf_sd=config['analysis_predefined_psf_sd'], config_content=json.dumps(config), parallel=parallel)
        psf_sd = config['analysis_predefined_psf_sd']
        last_h_index = config['analysis_maximum_hypothesis_index']
        use_exit_condi = config['analysis_use_premature_hypothesis_choice']
        image_folder_namebase = config['image_folder_namebase']
        code_version_date = config['code_version_date']
        log_folder = os.path.join('./runs', image_folder_namebase + '_code_ver' + code_version_date)
        analysis_result = analyze_image(filename, psf_sd, last_h_index, analysis_rand_seed_per_image, use_exit_condi, log_folder, display_fit_results=True, display_xi_graph=True)
    pass

def make_metrics_histograms(file_path = "./runs/PSF 1_0_2024-06-13/PSF 1_0_2024-06-13_metrics_log_per_image_hypothesis.csv", ):
    metric_of_interest = 'penalty'
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

if __name__ == '__main__':
    # make_metrics_histograms()
    sys.argv = ['main.py', '-c', './config_files/hi_accu_new']
    print(f"Manually setting argv as {sys.argv}. Delete this line and above to restore normal behaviour. (inside main.py, if __name__ == '__main__': )")
    main()
    # items = [1]
    # for item in items:
    #     if item == 1 or item == 1.1:
    #         numstr = f'{item:.1f}'.replace('.', '_')
    #         # label_prediction_log_filepath = f'./runs/PSF {numstr}_2024-06-13/label_prediction_log.csv'
    #         label_prediction_log_filepath = f"C:/github_repos/Generalized-Likelihood-Ratio-Particle-Counting/runs/PSF {numstr}_2024-06-13/label_prediction_log.csv"
    #         # save_path = os.path.dirname(label_prediction_log_filepath)
    #     else:
    #         numstr = f'{item:.1f}'
    #         # label_prediction_log_filepath = f'./runs/PSF_{numstr}_2024-05-29/actual_vs_counted.csv'
    #         label_prediction_log_filepath = f"C:/github_repos/Generalized-Likelihood-Ratio-Particle-Counting/runs/PSF_{numstr}_2024-05-29/actual_vs_counted.csv"
    #         # save_path = os.path.dirname(label_prediction_log_filepath)
    #     generate_confusion_matrix(label_prediction_log_filepath, 'PSF_1.0', '2024-06-13', display=False, savefig=True)
    pass