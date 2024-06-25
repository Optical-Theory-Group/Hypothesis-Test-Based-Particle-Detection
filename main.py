import pprint
import json
import argparse
from image_generation import psfconvolution
import csv
from PIL import Image as im
import os
import pandas as pd
import seaborn as sns
from process_algorithms import generalized_likelihood_ratio_test, fdr_bh, generalized_maximum_likelihood_rule
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

def test_glrt4_with_2_particles_image():
    # intensity = 2500
    psf_sd = 1.39
    sz = 20# Size of the width and height of the input image to be generated
    scaling = 3000  # As in the point spread function := scaling * normalized 2D gaussian
    bg = 500
    image = np.zeros((sz, sz))
    x = 5
    y = 5
    peaks_info = [{'x': x, 'y': y, 'prefactor': scaling, 'psf_sd': psf_sd}]
    image += psfconvolution(peaks_info, sz)
    show_fig = True
    image += np.ones(image.shape)*bg
    np.random.seed(42)
    image = np.random.poisson(image, size=(image.shape))
    if show_fig:    
        plt.imshow(image)
        plt.colorbar()

    fittype = 1
    psf_sd = 1.39
    h0_params,h1_params,crlbs1,p_value = generalized_likelihood_ratio_test(roi_image=image, psf_sd=1.39, iterations=10, fittype=fittype)
        
    for p in h1_params:
        print(f'{p=}')
    print(f'{crlbs1=}')
    print(f'{p_value=}')
    
    pass

def generate_test_images(dataset_name, mean_area_per_particle=0, amp_to_bg_min=2, amp_to_bg_max=50, amp_sd=0.1, n_images_per_count=10, psf_sd=1.39, sz=20, bg=500, random_seed=42, config_content=''):
    np.random.seed(random_seed)

    relative_intensity_min = 0.1
    
    n_particle_min = 0
    n_particle_max = sz * sz // mean_area_per_particle
    print(f'Generating {n_images_per_count} images per count {n_particle_min} to {n_particle_max} in folder image_dataset/{dataset_name}.\n\
        Total number of images generated from this config is {n_images_per_count * (n_particle_max - n_particle_min + 1)}.')

    # Create the folder to store the images
    image_folder_path = os.path.join("image_dataset", dataset_name)
    os.makedirs(image_folder_path, exist_ok=True)

    for n_particles in range(n_particle_min, n_particle_max+1):
        for img_idx in range(n_images_per_count):
            image = np.ones((sz, sz), dtype=float) * bg
            chosen_mean_intensity = (np.random.rand() * (amp_to_bg_max - amp_to_bg_min) + amp_to_bg_min) * bg
            for _ in range(n_particles):
                x = np.random.rand() * (sz - psf_sd * 4) + psf_sd * 2
                y = np.random.rand() * (sz - psf_sd * 4) + psf_sd * 2
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
            pil_image.save(os.path.join("image_dataset", dataset_name, img_filename))
    
    # Save the content of the config file
    config_file_save_path = os.path.join(image_folder_path, 'config_used.json')
    with open(config_file_save_path, 'w') as f:
        json.dump(json.loads(config_content), f, indent=4)

# def separation_test():
#     pass

# def psf_sd_test():
#     pass

# def imagewidth_test():
#     pass

def visualize_ctable(fname):
    df = pd.read_csv(fname)
    plt.figure()
    sns.heatmap(df, annot=True, fmt='g', cmap='Reds', cbar=False, annot_kws={"size": 15})
    plt.xlabel("Estimated number of particles", size=15)
    plt.ylabel("Actual number of particles", size=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show(block=False)
    plt.tight_layout()


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
            '   in   : {} ({:.4f}/s  Remaining run time estimate: {}). {}'.format(progresscount, totalrealisations,
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
        sys.stdout.write(text)
        sys.stdout.flush()

def analyze_whole_folder(dataset_name, analysis_name, use_exit_condi=True, last_h_index=7, psf_sd=1.39, rand_seed=0, config_content='', parallel=True):
    # Get a list of image files in the folder
    images_folder = os.path.join('./image_dataset', dataset_name)
    image_files = glob.glob(os.path.join(images_folder, '*.png')) + glob.glob(os.path.join(images_folder, '*.tiff'))

    print(f"Images loaded (total of {len(image_files)}):")

    # Create a folder to store the logs
    log_folder = os.path.join('./runs', dataset_name + '_' + analysis_name)
    os.makedirs(log_folder, exist_ok=True)

    # Save the content of the config file
    config_file_save_path = os.path.join(log_folder, 'config_used.json')
    with open(config_file_save_path, 'w') as f:
        json.dump(json.loads(config_content), f, indent=4)

    # Create a folder to store the logs for each image
    main_log_file_path = os.path.join(log_folder, 'actual_vs_counted.csv')
    with open(main_log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Input Image File', 'Actual Particle Number', 'Estimated Particle Number'])

    # Create the "runs" folder if it doesn't exist
    runs_folder = './runs'
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    # For each image file, 
    starttime = datetime.now()
    print('Beginning image analysis...')
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(analyze_image, filename, psf_sd, last_h_index, rand_seed, use_exit_condi, log_folder, seed)
                        for seed, filename in enumerate(image_files)]
            
            progress = 0
            with open(main_log_file_path, 'a', newline='') as f: 
                for cfresult in concurrent.futures.as_completed(futures):
                    if cfresult._exception is not None:
                        raise RuntimeError(cfresult._exception)
                    analysis_result = cfresult.result()
                    
                    # scores_csv_filename = analysis_result['csv_files1']
                    # fits_csv_filename = analysis_result['csv_files2']
                    actual_num_particles = analysis_result['actual_num_particles']
                    estimated_num_particles = analysis_result['estimated_num_particles']
                    input_image_file = analysis_result['input_image_file']
                    # filename = analysis_result['filename']

                    writer = csv.writer(f)
                    writer.writerow([input_image_file + ".tiff", actual_num_particles, estimated_num_particles])

                    statusmsg = f'{dataset_name} '
                    if actual_num_particles == estimated_num_particles:
                        statusmsg += f'\"{input_image_file}.tiff\" - Actual Number {actual_num_particles} == Estimated {estimated_num_particles}'
                    elif actual_num_particles > estimated_num_particles:
                        statusmsg += f'\"{input_image_file}.tiff\" - Actual Number: {actual_num_particles} > Estimated {estimated_num_particles}'
                    else:
                        statusmsg += f'\"{input_image_file}.tiff\" - Actual Number {actual_num_particles} < Estimated {estimated_num_particles}'

                    # statusmsg += f' Test results saved to {filename}'
                    report_progress(progress, len(image_files), starttime, statusmsg)
                    progress += 1
    else:
        progress = 0
        for seed, filename in enumerate(image_files):
            analysis_result = analyze_image(filename, psf_sd, last_h_index, rand_seed, use_exit_condi, log_folder, seed)
            actual_num_particles = analysis_result['actual_num_particles']
            estimated_num_particles = analysis_result['estimated_num_particles']
            input_image_file = analysis_result['input_image_file']

            with open(main_log_file_path, 'a', newline='') as f: 
                writer = csv.writer(f)
                writer.writerow([input_image_file + ".tiff", actual_num_particles, estimated_num_particles])

                statusmsg = f'{dataset_name} '
                if actual_num_particles == estimated_num_particles:
                    statusmsg += f'\"{input_image_file}.tiff\" - Actual Number {actual_num_particles} == Estimated {estimated_num_particles}\n'
                elif actual_num_particles > estimated_num_particles:
                    statusmsg += f'\"{input_image_file}.tiff\" - Actual Number: {actual_num_particles} > Estimated {estimated_num_particles}\n'
                else:
                    statusmsg += f'\"{input_image_file}.tiff\" - Actual Number {actual_num_particles} < Estimated {estimated_num_particles}\n'

            report_progress(progress, len(image_files), starttime, statusmsg)
            progress += 1
            
    return log_folder

def analyze_image(image_filename, psf_sd, last_h_index, rand_seed, use_exit_condi, log_folder, seed):
    # In case random numbers are used in analyse we seed here such that different parallel processes do not use the same random numbers
    np.random.seed(seed) 

    # Print the name of the image file
    image = np.array(im.open(image_filename))

    # Extract the number of particles from image_filename
    basename = os.path.basename(image_filename)
    count_part = basename.split('-')[0]
    num_particles = count_part.split('count')[1]
    actual_num_particles = int(num_particles)

    # Find tentative peaks
    tentative_peaks = get_tentative_peaks(image, min_distance=1)
    rough_peaks_xy = [peak[::-1] for peak in tentative_peaks]

    # Run GMRL
    estimated_num_particles, fit_results, test_metrics = generalized_maximum_likelihood_rule(roi_image=image, rough_peaks_xy=rough_peaks_xy, \
                                                        psf_sd=psf_sd, last_h_index=last_h_index, random_seed=rand_seed, use_exit_condi=use_exit_condi) 

    # Get the input image file name
    input_image_file = os.path.splitext(os.path.basename(image_filename))[0]
    scores_csv_filename = f"{log_folder}/image_log/{os.path.splitext(os.path.basename(image_filename))[0]}_scores.csv"
    fits_csv_filename = f"{log_folder}/image_log/{os.path.splitext(os.path.basename(image_filename))[0]}_fittings.csv"

    # Extract xi, lli, and penalty from test_metrics
    xi = test_metrics['xi']
    lli = test_metrics['lli']
    penalty = test_metrics['penalty']
    fisher_info = test_metrics['fisher_info']
    # Create a list of tuples containing hypothesis_index, xi, lli, and penalty
    metric_data = list(zip(range(len(xi)), xi, lli, penalty, fisher_info))
    # metric_data = list(zip(range(len(xi)), xi, lli, penalty, ))

    # Create a list of tuples containing fit_results_for_max_xi
    fitted_theta = fit_results[estimated_num_particles]['theta']
    fitting_data = [[fitted_theta]]  # Convert fitted_theta to a list

    # Write the data to the CSV files
    os.makedirs(os.path.dirname(scores_csv_filename), exist_ok=True)
    os.makedirs(os.path.dirname(fits_csv_filename), exist_ok=True)

    with open(scores_csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['hypothesis_index', 'xi', 'lli', 'penalty, fisher_info'])
        writer.writerows(metric_data)

    with open(fits_csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['theta'])
        writer.writerows(fitting_data)

    image_analysis_results = {'scores_csv_file': scores_csv_filename,
                              'csv_files2': fits_csv_filename,
                              'actual_num_particles': actual_num_particles,
                              'estimated_num_particles': estimated_num_particles,
                              'input_image_file': input_image_file,
                              'image_filename': image_filename,
                              }
    
    
    return image_analysis_results

def generate_confusion_matrix(csv_file, save_path, display=False, ):

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Extract the actual and estimated particle numbers
    actual = df['Actual Particle Number']
    estimated = df['Estimated Particle Number']
    
    # Load the JSON file
    # json_file = os.path.join(os.path.dirname(csv_file), 'config_used.json')
    
    # Get the analysis_max_h_number from the JSON file
    # analysis_max_h_number = config_data.get('analysis_max_h_number', 0)
    
    # Set the matrix size based on analysis_max_h_number
    
    # Generate the confusion matrix
    matrix = confusion_matrix(actual, estimated)
    if matrix[-1].sum == 0:
        matrix = matrix[:-1, :]
    normalized_matrix = np.zeros(matrix.shape)
    
    if display:
        row_sums = matrix.sum(axis=1)
        row_sums = np.reshape(row_sums, (len(row_sums),-1))
        _, ax = plt.subplots(figsize=(8, 5))  # Increase the size of the figure.
        normalized_matrix = matrix / row_sums
        folder_name = os.path.basename(os.path.dirname(csv_file))
        sns.heatmap(normalized_matrix, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax)  # Plot the heatmap on the new axes.
        ax.set_title(f'{folder_name}')
        ax.set_xlabel('Estimated Particle Number')
        ax.set_ylabel('Actual Particle Number')
        ytick_labels = [f"{i} (count: {row_sums[i][0]})" for i in range(len(row_sums))]
        ax.set_yticklabels(ytick_labels, rotation=0)
        
        # Draw lines between rows
        for i in range(matrix.shape[0]):
            ax.axhline(i, color='black', linewidth=1)
        
        plt.tight_layout()
        plt.show()
    
    # Save the confusion matrix as a CSV file
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_csv(save_path, index=False)


def plot_confusion_matrices_from_all_folders_inside_run_folder():

    # Get the list of folders in the specified directory
    folder_path = "./runs/"
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    # Iterate over the folders
    for folder in folders:
        # Get the path to the actual_vs_counted.csv file
        csv_file_path = os.path.join(folder_path, folder, "actual_vs_counted.csv")

        # Generate and display the confusion matrix
        generate_confusion_matrix(csv_file_path, display=True, save_path=None)


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

def main():
    batchjobstarttime = datetime.now()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process config files.')
    parser.add_argument('--config-file-folder', '-c', type=str, help='Folder containing config files to run.')
    parser.add_argument('--profile', '-p', type=bool, help='Boolean to decide whether to profile or not.')
    args = parser.parse_args()

    print('Overriding arguments for testing purposes - remove lines in main() to restore correct behaviour')
    args.config_file_folder = './config_files/jupiter_run_140524'
    args.profile = False

    # Check if config-file-folder is provided
    if (args.config_file_folder is None):
        print("Please provide the folder name for config files using --config-file-folder or -c option.")
        exit()
    config_files_dir = args.config_file_folder
    
    if args.profile is True:
        # config_files_dir = './config_files/profile_test'
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
    print(f'\n\nBatch job completed in {batchjobendtime - batchjobstarttime}')


def process(config_files_dir, parallel=True):
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
                required_fields = ['dataset_name', \
                                    'generate_dataset', 'gen_randseed', 'gen_n_img_per_count', 'gen_psf_sd', 'gen_img_width', 'gen_min_mean_area_per_particle', 'gen_bg_level', \
                                    'gen_particle_int_mean_to_bg_level_min', 'gen_particle_int_mean_to_bg_level_max', 'gen_particle_int_sd_to_mean_int', 'generated_img_folder_removal_after_counting',\
                                    'analysis_name', 'analysis_randseed', 'analysis_psf_sd', 'analysis_use_exit_condition', 'analysis_max_h_number',]
                for field in required_fields:
                    if field not in config:
                        if config['generate_dataset'] and field.startswith('gen'):
                            print(f"Error: '{field}' is not a valid field when 'generate_dataset' is True")
                            exit()
                        if config['analyze_dataset'] and field.startswith('analysis'):
                            print(f"Error: '{field}' is not a valid field when 'analyze_dataset' is True")
                            exit()

        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error: {config_file} file not found or invalid")
            continue

        if config['generate_dataset']:
            generate_test_images(dataset_name=config['dataset_name'], mean_area_per_particle=config['gen_min_mean_area_per_particle'], amp_to_bg_min=config['gen_particle_int_mean_to_bg_level_min'], amp_to_bg_max=config['gen_particle_int_mean_to_bg_level_max'], amp_sd=config['gen_particle_int_sd_to_mean_int'], \
                                n_images_per_count=config['gen_n_img_per_count'], psf_sd=config['gen_psf_sd'], sz=config['gen_img_width'], bg=config['gen_bg_level'], random_seed=config['gen_randseed'], config_content=json.dumps(config))

        if config['analyze_dataset']:
            log_folder = analyze_whole_folder(dataset_name=config['dataset_name'], analysis_name=config['analysis_name'], use_exit_condi=config['analysis_use_exit_condition'], last_h_index=config['analysis_max_h_number'], \
                                rand_seed=config['analysis_randseed'], psf_sd=config['analysis_psf_sd'], config_content=json.dumps(config), parallel=parallel)

            combine_log_files(log_folder, 'common_stock_for_comparison', '2024-06-23', delete_individual_files=True)

            if config['generated_img_folder_removal_after_counting']:
                shutil.rmtree(config['dataset_name'])
        
        # Generate confusion matrix
        main_log_file_path = os.path.join(log_folder, 'actual_vs_counted.csv')
        generate_confusion_matrix(main_log_file_path, os.path.join(log_folder, 'confusion_matrix.csv'), display=True)


if __name__ == '__main__':
    main()
    # plot_confusion_matrices_from_all_folders_inside_run_folder()
    