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

def generate_test_images(dataset_name, amp_to_bg=20, amp_sd=0.1, n_images=10, psf_sd=1.39, sz=20, bg=500, random_seed=42):
    np.random.seed(random_seed)
    n_particle_min = 0
    n_particle_max = sz // 5
    print(f'Generating {n_images} images in folder image_dataset/{dataset_name}')

    # Create the folder to store the images
    os.makedirs(os.path.join("image_dataset", dataset_name), exist_ok=True)
    for img_idx in range(n_images):
        image = np.ones((sz, sz), dtype=float) * bg
        num_particles = np.random.randint(n_particle_min, n_particle_max+1)
        # num_particles = 1
        for _ in range(num_particles):
            x = np.random.rand() * (sz - 3.1) + 2.1
            y = np.random.rand() * (sz - 3.1) + 2.1
            amplitude = np.random.normal(1, amp_sd) * amp_to_bg * bg
            if amplitude <= 0: 
                print('Warning: Amplitude is less than or equal to 0')
                peaks_info = []
            else:
                peaks_info = [{'x': x, 'y': y, 'prefactor': amplitude, 'psf_sd': psf_sd}]
            image += psfconvolution(peaks_info, sz)
        # Add Poisson noise
        image = np.random.poisson(image, size=(image.shape)) # This is the resulting (given) image.
        img_filename = f"img{img_idx}_{num_particles}particles.tiff"
        # plt.imshow(image, cmap='gray')
        pil_image = im.fromarray(image.astype(np.uint16))
        pil_image.save(os.path.join("image_dataset", dataset_name, img_filename))
        # plt.close()

    
# # snr_test()
# pass

# def n_particles_test():
#     pass

# def separation_test():
#     pass

# def intensity_spread_test():
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

def analyze_whole_folder(dataset_name, analysis_name, use_exit_condi=True, last_h_index=7, psf_sd=1.39, rand_seed=0, config_content=''):
    # Get a list of image files in the folder
    images_folder = os.path.join('./image_dataset', dataset_name)
    image_files = glob.glob(os.path.join(images_folder, '*.png')) + glob.glob(os.path.join(images_folder, '*.tiff'))

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

    # For each image file, 
    for filename in image_files:
        # Load image
        image = np.array(im.open(filename))
        image = np.array(image)

        # Extract the number of particles from filename
        basename = os.path.basename(filename)
        parts = basename.split('_')
        num_particles_part = parts[1]
        # Split this part on 'particles' to get the number
        actual_num_particles = int(num_particles_part.split('particles')[0])

        # Find tentative peaks
        tentative_peaks = get_tentative_peaks(image, min_distance=1)
        rough_peaks_xy = [peak[::-1] for peak in tentative_peaks]

        # Run GMRL
        estimated_num_particles, fit_results, test_metrics = generalized_maximum_likelihood_rule(roi_image=image, rough_peaks_xy=rough_peaks_xy, \
                                                            psf_sd=psf_sd, last_h_index=last_h_index, random_seed=rand_seed, use_exit_condi=use_exit_condi) 

        # Create the "runs" folder if it doesn't exist
        runs_folder = './runs'
        if not os.path.exists(runs_folder):
            os.makedirs(runs_folder)

        # Get the input image file name
        input_image_file = os.path.splitext(os.path.basename(filename))[0]
        csv_file1 = f"{log_folder}/image_log/{os.path.splitext(os.path.basename(filename))[0]}_scores.csv"
        csv_file2 = f"{log_folder}/image_log/{os.path.splitext(os.path.basename(filename))[0]}_fittings.csv"

        # Extract xi, lli, and penalty from test_metrics
        xi = test_metrics['xi']
        lli = test_metrics['lli']
        penalty = test_metrics['penalty']
        # Create a list of tuples containing hypothesis_index, xi, lli, and penalty
        data1 = list(zip(range(len(xi)), xi, lli, penalty))

        # Create a list of tuples containing fit_results_for_max_xi
        fitted_theta  = fit_results[estimated_num_particles]['theta']
        data2 = [[fitted_theta]]  # Convert fitted_theta to a list

        # Write the data to the CSV files
        os.makedirs(os.path.dirname(csv_file1), exist_ok=True)
        os.makedirs(os.path.dirname(csv_file2), exist_ok=True)

        with open(csv_file1, 'w', newline='') as file1:
            writer1 = csv.writer(file1)
            writer1.writerow(['hypothesis_index', 'xi', 'lli', 'penalty'])
            writer1.writerows(data1)

        with open(csv_file2, 'w', newline='') as file2:
            writer2 = csv.writer(file2)
            writer2.writerow(['theta'])
            writer2.writerows(data2)
        print(f"Data saved to \n\t{csv_file1} and \n\t{csv_file2}")

        # Print the test results
        print(f'======   Test result: num_particles={actual_num_particles}, estimated_num_particles={estimated_num_particles}   ======')
        if actual_num_particles == estimated_num_particles:
            print('Test passed: Accurate particle count')
        elif actual_num_particles > estimated_num_particles:
            print('Test failed: Particle count - underestimation')
        else:
            print('Test failed: Particle count - overestimation')

        with open(main_log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([input_image_file + ".tiff", actual_num_particles, estimated_num_particles])
        print(f'Test results saved to to \t\n{filename=} ')
        
    return log_folder

def generate_confusion_matrix(csv_file, save_path, display=False, ):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Extract the actual and estimated particle numbers
    actual = df['Actual Particle Number']
    estimated = df['Estimated Particle Number']

    # Generate the confusion matrix
    matrix = confusion_matrix(actual, estimated)

    # Display the confusion matrix
    if display:
        sns.heatmap(matrix, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Estimated Particle Number')
        plt.ylabel('Actual Particle Number')
        plt.show()

    # Save the confusion matrix as a CSV file
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_csv(save_path, index=False)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process config files.')
    parser.add_argument('--config-file-folder', type=str, default='config_files', help='Folder containing config files')
    args = parser.parse_args()

    config_files_dir = args.config_file_folder
    config_files = os.listdir(config_files_dir)

    print(f"Config files loaded (total of {len(config_files)}):")
    for config_file in config_files:
        print(config_file)

    for i, config_file in enumerate(config_files):
        print(f"Processing {config_file}, {i+1}/{len(config_files)}")
        print(config_file)
        try:
            with open(os.path.join(config_files_dir, config_file), 'r') as f:
                config = json.load(f)
                # Check if the required fields are present in the config file
                required_fields = ['dataset_name', \
                                    'generate_dataset', 'gen_randseed', 'gen_n_img', 'gen_psf_sd', 'gen_img_width', 'gen_bg_level', \
                                    'gen_particle_int_to_bg_level', 'gen_particle_int_sd_to_mean_intt', 'generated_img_folder_removal_after_counting',\
                                    'analysis_name', 'analysis_randseed', 'analysis_psf_sd', 'use_exit_condition', 'max_hypothesis_index',]
                for field in required_fields:
                    if field not in config:
                        print(f"Error: '{field}' is missing in the config file")

        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error: {config_file} file not found or invalid")
            continue

        if config['generate_dataset']:
            generate_test_images(dataset_name=config['dataset_name'], amp_to_bg=config['gen_particle_int_to_bg_level'], amp_sd=config['gen_particle_int_sd_to_mean_int'], \
                                n_images=config['gen_n_img'], psf_sd=config['gen_psf_sd'], sz=config['gen_img_width'], bg=config['gen_bg_level'], random_seed=config['gen_randseed'])

        if config['analyze_dataset']:
            log_folder = analyze_whole_folder(dataset_name=config['dataset_name'], analysis_name=config['analysis_name'], use_exit_condi=config['use_exit_condition'], last_h_index=config['max_hypothesis_index'], \
                                rand_seed=config['analysis_randseed'], psf_sd=config['analysis_psf_sd'], config_content=json.dumps(config))

        if config['generated_img_folder_removal_after_counting']:
            shutil.rmtree(config['dataset_name'])
        
        # Generate confusino matrix
        main_log_file_path = os.path.join(log_folder, 'actual_vs_counted.csv')
        generate_confusion_matrix(main_log_file_path, os.path.join(log_folder, 'confusion_matrix.csv'), display=True)

if __name__ == '__main__':
    main()