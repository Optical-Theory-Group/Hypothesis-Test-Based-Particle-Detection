from image_generation import psfconvolution
from PIL import Image as im
import os
import datetime
from matplotlib.cm import ScalarMappable
import pandas as pd
import seaborn as sns
from process_algorithms import generalized_likelihood_ratio_test, fdr_bh, generalized_maximum_likelihood_rule
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from process_algorithms import generalized_likelihood_ratio_test, generalized_maximum_likelihood_rule
import math
import numpy as np
from main import make_subregions, create_separable_filter, get_tentative_peaks
import diplib as dip

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
 
def make_and_process_image(x=3.35, y=6.69, sz=12, intensity=10, bg=4, psf=1.39, show_fig=False, verbose=False):

    image = psfconvolution(particle_x=x, particle_y=y, multiplying_constant=intensity, psf_sd=psf, imgwidth=sz)
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

# make_and_process_image()

def vary_particle_width():
    bg = 0
    intensity = 1000 
    psfs = np.array([.25, .5, 1, 1.5, 2, 4])*1.39

    df = pd.DataFrame(columns = ['bg', 'p'])
    p_per_psf = []
    for _, psf in enumerate(psfs):
        for _ in range(30):
            x = np.random.rand() + 5.5 
            y = np.random.rand() + 5.5
            p = make_and_process_image(x=x, y=y, intensity=intensity, bg=bg, psf=psf, sz=20, show_fig=False)
            p_per_psf.append(p)
            df = pd.concat([df, pd.DataFrame.from_records([{'psf':psf, 'p':p}])], ignore_index=True)
    plt.figure()
    plt.title("p-values for different psfs")
    sns.swarmplot(data=df, x='psf', y='p', zorder=.5)
    plt.show(block=False)
    pass

def test_glrt4_with_2_particles_image():
    # intensity = 2500
    psf_sd = 1.39
    sz = 20# Size of the width and height of the input image to be generated
    scaling = 3000  # As in the point spread function := scaling * normalized 2D gaussian
    bg = 500
    image = np.zeros((sz, sz))
    x = 5
    y = 5
    image += psfconvolution(particle_x=x, particle_y=y, multiplying_constant=scaling, psf_sd=psf_sd, imgwidth=sz)
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

def test_gmlr():
    np.random.seed(42)
    psf_sd = 1.39
    sz = 30# Size of the width and height of the input image to be generated
    amplitude = 1000 #* np.random.rand() # As in the point spread function := amplitude * normalized 2D gaussian
    # amplitude = 1000 #* np.random.rand() # As in the point spread function := amplitude * normalized 2D gaussian
    bg = 500
    image = np.zeros((sz, sz))

    show_generated_input_image = True
    weak_peak_test = True
    if show_generated_input_image:    
        _, ax = plt.subplots(1,2, figsize=(10,5))
        ax[0].set_xlim(0-.5, sz-.5)
        ax[0].set_ylim(sz-.5, 0-.5) 
        ax[1].set_xlim(0-.5, sz-.5)
        ax[1].set_ylim(sz-.5, 0-.5) 
    if weak_peak_test:
        for j in range(3):
            y = 9 * (j + 0.2)
            for i in range(3):
                x = 8 * (i + 0.9)
                image += psfconvolution(particle_x=x, particle_y=y, multiplying_constant=amplitude, psf_sd=psf_sd, imgwidth=sz)
                amplitude -= 100
                # ax[0].plot(x, y, 'ro', markersize=15)
                if show_generated_input_image:    
                    ax[0].imshow(image)
                    ax[0].text(x-.5, y+.5, 'x', fontsize=9, color='red') 
                    ax[0].text(x-.5, y+.5, f'  {amplitude:.1e}', fontsize=9, color='red') 
                    ax[1].text(x-.5, y+.5, 'o', fontsize=20, color='gray')
        plt.show(block=False)
    else:
        num_particles = np.random.randint(0, 8)
        for _ in range(num_particles):
            x = np.random.rand()*(sz-3) + 1.1
            y = np.random.rand()*(sz-3) + 1.1
            amplitude = max(np.random.normal(1,.3), 0.1) * amplitude
            image += psfconvolution(particle_x=x, particle_y=y, multiplying_constant=amplitude, psf_sd=psf_sd, imgwidth=sz) 
    # Adding background
    image += np.ones(image.shape)*bg
    image = np.random.poisson(image, size=(image.shape))
    tentative_peak_coordinates = get_tentative_peaks(image, min_distance=1)

    show_generated_input_image = True
    if show_generated_input_image:    
        im = ax[0].imshow(image)
        plt.colorbar(im, ax=ax[0])  # Add colorbar to ax[0]
        cmap = plt.get_cmap('plasma')
        ax[1].set_xlim(0-.5, sz-.5)
        ax[1].set_ylim(sz-.5, 0-.5) 
        ax[1].set_aspect('equal')
        for i, coord in enumerate(tentative_peak_coordinates):
            x, y = coord
            color = cmap(i / len(tentative_peak_coordinates))
            ax[1].text(y-.5, x+.5, f'x', fontsize=20, color=color) 
            plt.pause(0.1)
        # Add horizontal colorbar
        norm = plt.Normalize(vmin=0, vmax=len(tentative_peak_coordinates)-1)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        plt.show(block=False)
        cbar = plt.colorbar(sm, ax=ax[1])
        cbar.ax.invert_yaxis()
        cbar.set_ticks(range(len(tentative_peak_coordinates)))
        cbar.set_ticklabels(range(1, len(tentative_peak_coordinates)+1))
        cbar.set_label('Peak Index')
        plt.tight_layout()
        plt.pause(0.1)

    generalized_maximum_likelihood_rule(roi_image=image, tentative_peak_coordinates=tentative_peak_coordinates, psf_sd=1.39)

def make_images(n_img=100, psf_sd=1.39, sz=50, bg=500, brightness=9000):
    for img_idx in range(n_img):
        image = np.ones((sz, sz), dtype=float) * bg
        num_particles = np.random.randint(0, 6)
        for _ in range(num_particles):
            x = np.random.rand()*(sz-3.1) + 2.1
            y = np.random.rand()*(sz-3.1) + 2.1
            amplitude = np.random.normal(1,.2) * brightness
            if 1/np.pi/psf_sd**2 * amplitude < 0.1 * bg:
                amplitude = 0.1 * bg * np.pi * psf_sd**2
            image += psfconvolution(particle_x=x, particle_y=y, multiplying_constant=amplitude, psf_sd=psf_sd, imgwidth=sz) 
            # Create the directory if it doesn't exist
        image = np.random.poisson(image, size=(image.shape))
        os.makedirs('./test_images', exist_ok=True)
        fname = f'{img_idx}_{num_particles}particles.png'
        plt.imsave(arr=image, fname=f'./test_images/{fname}')

def generate_test_images(amp_to_bgs=[20, 3], amp_sds=[0, .3], n_per_condition=10, psf_sd=1.39, sz=20, bg=500, ):
    print(f'Generating test images with {n_per_condition} images per condition')
    n_particle_min = 0
    n_particle_max = sz // 5
    idstring = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for amp_to_bg in amp_to_bgs:
        for amp_sd in amp_sds:
            folder_name = f"amp_to_bg_{amp_to_bg}_amp_sd_{amp_sd}-{idstring}"
            os.makedirs(folder_name, exist_ok=True)
            for img_idx in range(n_per_condition):
                image = np.ones((sz, sz), dtype=float) * bg
                num_particles = np.random.randint(n_particle_min, n_particle_max+1)
                # num_particles = 1
                for _ in range(num_particles):
                    x = np.random.rand() * (sz - 3.1) + 2.1
                    y = np.random.rand() * (sz - 3.1) + 2.1
                    amplitude = np.random.normal(1, amp_sd) * amp_to_bg * bg
                    if amplitude <= 0: 
                        print('Warning: Amplitude is less than or equal to 0')
                    image += psfconvolution(particle_x=x, particle_y=y, multiplying_constant=amplitude, psf_sd=psf_sd, imgwidth=sz)
                # Add Poisson noise
                image = np.random.poisson(image, size=(image.shape)) # This is the resulting (given) image.
                img_filename = f"img{img_idx}_{num_particles}particles.tiff"
                plt.imshow(image, cmap='gray')
                plt.savefig(os.path.join(folder_name, img_filename), format='tiff')
                plt.close()

def test_separation():
    pass

def test_model(specific_test_data_foldername, psf_sd=1.39):
    # List up all png files.
    tiff_files = []
    for file in os.listdir(specific_test_data_foldername):
        if file.endswith(".tiff"):
            tiff_files.append(os.path.join(specific_test_data_foldername, file))

    # Define load_tiff_file function
    def load_tiff_file(file_path):
        
        image = np.array(im.open(file_path))
        image = np.array(image)
        return image

    confusion_table = np.zeros((21,21))
    for file in tiff_files:
        image = load_tiff_file(file)
        # Extract the true number of particles from the filename
        basename = os.path.basename(file)
        # Extract the true number of particles from the filename
        parts = basename.split('_')
        num_particles_part = parts[1]
        # Split this part on 'particles' to get the number
        actual_num_particles = int(num_particles_part.split('particles')[0])

        # Find tentative peaks
        tentative_peaks = get_tentative_peaks(image, min_distance=1)
        rough_peaks_xy = [peak[::-1] for peak in tentative_peaks]
        estimated_num_particles = generalized_maximum_likelihood_rule(roi_image=image, rough_peaks_xy=rough_peaks_xy, psf_sd=psf_sd,) 
        print(f'============ Test result: {basename=} num_particles={actual_num_particles}, estimated_num_particles={estimated_num_particles}')                                                            
        confusion_table[actual_num_particles, estimated_num_particles] += 1
        
    # Save the confusion table
    result_foldername = specific_test_data_foldername + "-test_result"
    os.makedirs(result_foldername, exist_ok=True)
    df = pd.DataFrame(confusion_table)
    csv_path = os.path.join(result_foldername, f'row-actual_col-est.csv')
    df.to_csv(csv_path, index=False)

    
test_model('amp_to_bg_20_amp_sd_0-20240410_172153')
pass


def snr_test(amp_sd=0, psf_sd=1.39, sz=40, bg=500, n_particle_min=0, n_particle_max=4, n_per_condition=100):
    # Build confusion table
    for amp_to_bg in [20, 2]:
        confusion_table = np.zeros((n_particle_max+1, n_particle_max+1))
        idstring = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for i in range(n_per_condition):
            # Image Generation
            image = np.ones((sz, sz), dtype=float) * bg
            num_particles = np.random.randint(n_particle_min, n_particle_max+1)
            # num_particles = 1
            for _ in range(num_particles):
                x = np.random.rand() * (sz - 3.1) + 2.1
                y = np.random.rand() * (sz - 3.1) + 2.1
                amplitude = np.random.normal(1, amp_sd) * amp_to_bg * bg
                if amplitude <= 0: 
                    print('Warning: Amplitude is less than or equal to 0')
                image += psfconvolution(particle_x=x, particle_y=y, multiplying_constant=amplitude, psf_sd=psf_sd, imgwidth=sz)
            # Add Poisson noise
            image = np.random.poisson(image, size=(image.shape)) # This is the resulting (given) image.
            # plt.imshow(image)
            # plt.show(block=False)
            tentative_peaks = get_tentative_peaks(image, min_distance=1)
            rough_peaks_xy = [peak[::-1] for peak in tentative_peaks]
            estimated_num_particles = generalized_maximum_likelihood_rule(roi_image=image, rough_peaks_xy=rough_peaks_xy, psf_sd=psf_sd,) 
                                                                        #   display_xi_graph=True,
                                                                        #   display_fit_results=True)
                                    
            print(f'============ Test result: {i=} num_particles={num_particles}, estimated_num_particles={estimated_num_particles}')                                                            

            confusion_table[num_particles, estimated_num_particles] += 1

            # Save the image for later access to the data used for this test.
            # Create the folder path
            folder_name = f'snr_test/test_images/{idstring}'
            os.makedirs(folder_name, exist_ok=True)
            image_path = os.path.join(folder_name, f'img{i}_{num_particles}particles.png')
            # image_path = os.path.join(folder_name, f'amp-to-bg{amp_to_bg}_bg{bg}_{i}.png')
            plt.imsave(image_path, image, cmap='gray')
        # Save the confusion table
        folder_name = f'snr_test/test_results/'
        os.makedirs(folder_name, exist_ok=True)
        df = pd.DataFrame(confusion_table)
        csv_path = os.path.join(folder_name, f'row-actual_col-est_{idstring}.csv')
        df.to_csv(csv_path, index=False)
        pass
    
# snr_test()
pass

def n_particles_test():
    pass

def separation_test():
    pass

def intensity_spread_test():
    pass

def psf_sd_test():
    pass

def imagewidth_test():
    pass

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
    
# visualize_ctable("./snr_test/test_results/row-actual_col-est_20240410_132353.csv")
# test_model(specific_test_data_foldername='test_images')
test_model('amp_to_bg_3_amp_sd_0-20240410_170325')