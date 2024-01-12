import matplotlib.colors as mcolors
import math
import numpy as np
from PIL import Image as im
from scipy.ndimage import maximum_filter
# import skimage.measure as measure
# from scipy.ndimage import gaussian_filter, binary_closing, distance_transform_edt, label
# from skimage.segmentation import watershed
import diplib as dip
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from image_generation import psfconvolution
from process_algorithms import generalized_likelihood_ratio_test, fdr_bh
from mpl_toolkits.mplot3d import Axes3D

def getbox(input_image, ii, sz, x_positions, y_positions):
    """ Copies the specified subregion in input_image to dataout.
    Args:
        input_image: The original 2D image to crop from.
        ii: The index of the point to copy.
        sz: The size of the subregion to copy.
        x_positions: X coordinates of the center of the subregions.
        y_positions: Y coordinates of the center of the subregions.
    """
    sz_x, sz_y = input_image.shape

    # Calculate the index of the center of the subregion
    szl = int(sz / 2 + 0.5)

    # Get coordinates (adjusted for zero-indexing in Python)
    x = int(x_positions[ii] + 0.5)
    y = int(y_positions[ii] + 0.5)

    # Ensure coordinates are within bounds
    if x < 0 or y < 0 or x >= sz_x or y >= sz_y:
        raise ValueError(f"Point {ii} out of bounds position {x}, {y} dataset size {sz_x}, {sz_y}")

    # Calculate left, right, top, bottom coordinates for the box
    l = max(x - szl, 0)
    r = min(l + sz, sz_x)
    t = max(y - szl, 0)
    b = min(t + sz, sz_y)

    # Return the input_image in roi, the left coordinates, and the top coordinates
    return input_image[t:b, l:r], l, t

def make_subregions(inner_image_pos_idx, box_size, input_image):
    """ Creates subregions of size box_size around the points in inner_image_pos_idx.
    Args:
        inner_image_pos_idx: A 2D array of indices of the points to crop.
        box_size: The size of the subregions to crop.
        input_image: The original 2D image to crop from.
    Returns:
        scanning_roi_stack: A 3D array of the cropped subregions.
        leftcoord: A 1D array of the left coordinates of the subregions.
        topcoord: A 1D array of the top coordinates of the subregions.
    """
    x_positions, y_positions = inner_image_pos_idx[0], inner_image_pos_idx[1]
    if input_image.dtype != np.float32:
        raise ValueError("Data must be comprised of single floats")
    if len(x_positions) == 0 or len(y_positions) == 0:
        raise ValueError("Coordinate array(s) is/are empty.")
    if x_positions.shape != y_positions.shape:
        raise ValueError("Size of X and Y coordinates must match.")
    if box_size <= 0:
        raise ValueError("Box size must be a positive integer.")
    if input_image.ndim != 2:
        raise ValueError("Data should be a 2D array.")

    # Convert box_size to an integer
    box_sz = int(box_size)

    # Get the number of points
    n_rois = len(x_positions)
    
    # Initialize output arrays
    scanning_roi_stack = np.zeros((n_rois, box_sz, box_sz), dtype=float)
    leftcoord = np.zeros(n_rois, dtype=float)
    topcoord = np.zeros(n_rois, dtype=float)

    for ii in range(n_rois):
        scanning_roi_stack[ii], leftcoord[ii], topcoord[ii] = getbox(input_image, ii, box_sz, x_positions, y_positions)

    return scanning_roi_stack, leftcoord, topcoord
    
def create_separable_filter(one_d_kernel, origin):
    """ Creates a separable filter from a 1D kernel.
    Args:
        one_d_kernel: The 1D kernel to use.
        origin: The origin of the kernel.
    Returns:
        adjusted_kernel: The adjusted kernel.
    """
    # Get the length of the 1D kernel
    length = len(one_d_kernel)

    # Create a full 2D kernel from the 1D kernel
    full_kernel = np.outer(one_d_kernel, one_d_kernel)

    # Calculate padding based on the desired origin
    pad_before = origin - 1
    pad_after = length - origin

    # Apply padding to create an adjusted kernel
    adjusted_kernel = np.pad(full_kernel, ((pad_before, pad_after), (pad_before, pad_after)), mode='constant')

    return adjusted_kernel


def main_image_analysis_controller(input_image, psf_sd=1.39, significance=0.05, consideration_limit_level=2, fittype=0, split=True):
    """ Performs image processing on the input image, including the preprocessing, detection, and fitting steps.
    Args:
        input_image: The image to process.
        psf_sd: The sigma value of the PSF.
        min_pixels: The minimum number of pixels in a cluster.
        significance: The significance level.
        consideration_limit_level: The compression reduction factor.
        fittype: The type of fit to use.
        split: Boolean on whether to split clusters or not.
    Returns:
        cluster_properties: A dictionary of the detection parameters.
            - coords: A 2D array of the coordinates of the points.
            - circularity: A 1D array of the circularity of the points.
            - clusterSize: A 1D array of the size of the clusters.
            - pH1: A 1D array of the pH1 values.
    """

    # Stripping the edges, because the edges are cannot be processed using our strategy.
    required_box_size = int(np.ceil(3 * (2 * psf_sd + 1)))
    # required_box_size = 3 * (2 * psf_sd + 1)
    xbegin = math.ceil(required_box_size / 2) 
    ybegin = xbegin 
    # Note that in line with python indexing, the end is not included in the range
    xend = math.floor(input_image.shape[1] - required_box_size / 2)
    yend = math.floor(input_image.shape[0] - required_box_size / 2)
    
    inner_xidxs = np.arange(xbegin, xend) 
    inner_yidxs = np.arange(ybegin, yend) 

    # Define the filter
    h2 = 1/16
    h1 = 1/4
    h0 = 3/8
    g = {}
    g[0] = [h2, h1, h0, h1, h2]
    g[1] = [h2, 0, h1, 0, h0, 0, h1, 0, h2]
    
    # Crop the image
    inner_image = input_image[np.ix_(inner_xidxs, inner_yidxs)]

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
        v1 = dip.Convolution(dip_image, adjusted_kernel_1, method="best")

        # Compute the difference between V1 and V2
        w = v0 -v1
        check_w = True
        # Visualize the difference between V1 and V2
        if check_w:
            _,ax=plt.subplots(1,3)
            imw = ax[2].imshow(w), ax[2].set_title('w')
            vmin, vmax = imw.get_clim()
            ax[0].imshow(v0, vmin=vmin, vmax=vmax), ax[0].set_title('v0')
            ax[1].imshow(v1, vmin=vmin, vmax=vmax), ax[1].set_title('v1')
            plt.colorbar(imw, ax=ax.ravel().tolist(), orientation='vertical')

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

    # Plotting pfa array as an image with log scale colormap
    plt.figure()
    pfa_array_log = np.zeros((3,3))
    min_z_idx = np.unravel_index(np.argmin(pfa_array), pfa_array.shape)
    min_row, min_col = min_z_idx[0], min_z_idx[1]
    for i in range(3):
        for j in range(3):
                pfa_array_log[i, j] = pfa_array[min_row+i-1,min_col+j-1]
    plt.imshow(pfa_array_log, cmap='viridis', norm=mcolors.LogNorm())
    plt.colorbar()
    plt.title('PFA Image with Log Scale Colormap')
    plt.show(block=False)

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

    # Plotting pfa_adj array as an image with log scale colormap
    plt.figure()
    pfa_array_log = np.zeros((3,3))
    min_z_idx = np.unravel_index(np.argmin(pfa_adj_array), pfa_adj_array.shape)
    min_row, min_col = min_z_idx[0], min_z_idx[1]
    for i in range(3):
        for j in range(3):
                pfa_array_log[i, j] = pfa_adj_array[min_row+i-1,min_col+j-1]
    plt.imshow(pfa_array_log, cmap='viridis', norm=mcolors.LogNorm())
    plt.colorbar()
    plt.title('PFA adj Image with Log Scale Colormap')
    plt.show(block=False)
    
    significance_mask[inner_image_pos_idx] = (pfa_adj <= significance).flatten()
    
    show_significance_mask = True
    if show_significance_mask:
        plt.figure(), plt.imshow(significance_mask), plt.title('significance_mask'), plt.show(block=False)

    # Add an extra dimension to the image (as the third dimension, whose size is 1)
    image = np.expand_dims(input_image, axis=2)

    # coords = []
    # circularity = []
    # clusterSize = []
    # pH1 = []

    # min_pixels = (psf_sd * 1.5) ** 2
    min_pixels = 0 
    if min_pixels:
        pass
        # ll = np.zeros((significance_mask.shape[1], significance_mask.shape[0], significance_mask.shape[2]), dtype=np.int32)
        # ll = np.zeros((significance_mask.shape[1], significance_mask.shape[0]), dtype=np.int32)
        # h = binary_closing(significance_mask)
            
        # if split and np.sum(significance_mask) > 0 and np.sum(h) > 0:
        #     D = -distance_transform_edt(h)
        #     D[~h] = np.inf
        #     # ll[:, :, i - 1], _ = label(~watershed(D, 1) * significance_mask[:, :, i])

        # else:
        #     # ll[:, :, i - 1], _ = label(significance_mask[:, :, i])
        # # msrResults = measure.regionprops(label(ll[:, :, i - 1]))
        # for region in msrResults:
        #     if region.area > min_pixels:
        #         coords.append([region.centroid[1], region.centroid[0], i - 1])
        #         circularity.append(region.equivalent_diameter / region.perimeter)
        #         clusterSize.append(region.area)
        #         pH1.append(min(max(2 * region.area / ((2 * psf_sd) ** 2 * np.pi), 0), 1))
    
        # cluster_properties = {
        #     'significance_mask': significance_mask,
        #     'll': ll,
        #     'circularity': circularity,
        #     'clusterSize': clusterSize,
        #     'pH1': pH1
        # }
    else:
        # Define the footprint for the local maximum detection
        neighborhood_size = ((int(2 * psf_sd + 1), int(2 * psf_sd + 1)))
        footprint = np.ones(neighborhood_size)

        # Apply the maximum filter
        filtered_image = maximum_filter(inner_image, footprint=footprint)

        # Identify local maxima
        local_maxima = (inner_image == filtered_image)

        # Apply the significance_mask mask if needed
        local_maxima_masked = local_maxima & significance_mask

        im_max = local_maxima_masked
        # im_max = (max_filtered_image >= threshold) & significance_mask
        # im_max = (inner_image >= 0.999 * np.max(inner_image, axis=(0, 1), keepdims=True)) & significance_mask
        coords = np.argwhere(im_max)
        show_immax = True
        if show_immax:
            _,ax = plt.subplots()
            ax.imshow(im_max), ax.set_title('im_max')
            plt.show(block=False)

       # Convert to 0s and 1s
        # modified_significance_mask = significance_mask.astype(int) 
        # for coord in coords:
            # modified_significance_mask[coord[0], coord[1]] = -1  # Set the value at each coordinate in coords to 0
        # Define the colors for each pixel value
        # cmap = mcolors.ListedColormap(['darkred', 'yellow', 'darkblue'])
        # bounds = [-2, -0.5, 0.5, 2]  # Define boundaries for the colors
        # norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Plotting
        # plt.imshow(modified_significance_mask, cmap=cmap, norm=norm)
        # plt.figure()
        # plt.imshow(modified_significance_mask, cmap='viridis_r')
        # plt.show(block=False)
        
        # if image.ndim == 2:
            # coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
    
        # pH1 = 1 - pfa[im_max]
        # cluster_properties = {}
    
    # cluster_properties['pfa_adj'] = pfa
    # cluster_properties['pH1'] = np.array(pH1)

    # ax.imshow(pfa)
    # title = ''
    # for i, coord in enumerate(coords):
    #     title += f'{coord}: pH1={pH1[i]:.3f}, '
    # ax.set_title(title)
    # plt.show(block=False)
    
    # for i, coord in enumerate(coords):
    #     plt.text(coord[1], coord[0], f'pH1={pH1[i]:.3f}', color='red')
    # inner_image = image[inner_xidxs, inner_yidxs, :]
    # pass

def save_an_image(fname='zzzz.png'):
    sz = 20
    # image = makeimg(imgwidth=sz, pparticlepixel=8/6400, particle_multiplying_constant=10000, psf_sd=1.39, bgfreq=0.01, bgamp=0,
    #                 clusterp=0, avgclustersz=0, particledistancecluster=0, pdustpixel=0, dustpsfr=0, dust_multiplying_constant=0, vignettesd=0, ) 
    image = np.zeros((sz,sz))
    # for _ in range(3):
        # pass
        # image += psfconvolution(particle_x=np.random.rand()*(sz-1), particle_y=np.random.rand()*(sz-1), multiplying_constant=1000, psf_sd=1.39, imgwidth=sz)
        
    image = psfconvolution(particle_x=sz/2-0.5, particle_y=sz/2-0.5, multiplying_constant=1000, psf_sd=1.39, imgwidth=sz)
    image += 500 * np.ones((sz, sz))

    image = np.random.poisson(image, size=(image.shape))
    plt.imshow(image), plt.show(block=False)
    data = im.fromarray(image)
    data.save(fname)
    # plt.imsave(arr=image, fname=fname)

fname = 'presentation1.png'

# save_an_image(fname)

def main_test(readfname='image1.png'):
    image = im.open(readfname)
    image = np.array(image)
    plt.imshow(image)
    plt.show(block=False)
    # sz = 34
    # image = psfconvolution(particle_x=14, particle_y=14, multiplying_constant=1000, psf_sd=1.39, imgwidth=sz)
    main_image_analysis_controller(image, consideration_limit_level=0, fittype=0)

# fname = '3particles_nonoise.tif'
# main_test(fname)
def test_as_single_roi(readfname):
    image = im.open(readfname)
    image = np.array(image)
    generalized_likelihood_ratio_test(image, 1.39, 8, 0)
 
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

def test1():
    sz = 18
    intensity = 1500
    image = psfconvolution(sz/2 -0.5, sz/2 -0.5, intensity, 1.39, sz)
    image += 500 * np.ones((sz, sz))
    image = np.random.poisson(image, size=(image.shape))
    plt.imshow(image, cmap='gray')
    plt.yticks([])
    plt.xticks([])
    plt.show(block=False)
    main_image_analysis_controller(image, consideration_limit_level=0, fittype=0)
    pass

def test_glrt4_with_2_particles_image():
    intensity = 2500
    psf_sd = 1.39
    sz = 20
    bg = 500
    show_fig = True
    x = 8.35
    y = 7.69
    image = psfconvolution(particle_x=x, particle_y=y, multiplying_constant=intensity, psf_sd=psf_sd, imgwidth=sz)
    x = 13.35
    y = 11.69
    image += psfconvolution(particle_x=x, particle_y=y, multiplying_constant=intensity, psf_sd=psf_sd, imgwidth=sz)
    # Adding background
    image += np.ones(image.shape)*bg
    image = np.random.poisson(image, size=(image.shape))
    if show_fig:    
        plt.imshow(image)
        plt.colorbar()

    fittype = 1
    psf_sd = 1.39
    params0,params1,crlbs1,p_value = generalized_likelihood_ratio_test(roi_image=image, psf_sd=1.39, iterations=10, fittype=fittype)
        
    for p in params1:
        print(f'{p=}')
    print(f'{crlbs1=}')
    print(f'{p_value=}')
    
    pass

test_glrt4_with_2_particles_image()