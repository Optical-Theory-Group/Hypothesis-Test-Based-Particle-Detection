import seaborn as sns
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import erf, gammaln
from scipy.stats import norm
from skimage.feature import peak_local_max
import diplib as dip

# Define penalty factor constants for the out-of-bounds particles and negative intensity particles.
POSITION_PENALTY_FACTOR = 1e5
INTENSITY_PENALTY_FACTOR = 1e5

# Print numpy arrays with 3 decimal points
np.set_printoptions(precision=4, formatter={'float': '{:0.3f}'.format}, linewidth=np.inf)

def normalize(th, hypothesis_index, roi_min, roi_max, psf_sigma, szx, szy):
    """ Normalize the theta values for the optimization process.
    Args:
        th (list): The list of particle parameters, un-normalized and fully structured.
        hypothesis_index (int): The index of the hypothesis being tested. (it matches the number of particles each hypothesis assumes)
        roi_min (float): The minimum value of the region of interest.
        roi_max (float): The maximum value of the region of interest.
        psf_sigma (float): The standard deviation of the point spread function.
        szx (int): The size of the x-axis.
        szy (int): The size of the y-axis.
    Returns:
        ndarray: The normalized and flattened theta values.
    """

    # If the image we are dealing with is a grayscale image, process the theta values as follows:
    if isinstance(roi_max, (int, np.integer, float)) and isinstance(roi_min, (int, np.integer, float)): # If the image is a grayscale image, then roi_max and roi_min should be a single value.

        # First, normalize theta into n_th (: normalized theta) 
        n_th = np.zeros((hypothesis_index + 1, 3)) # 3 is for intensity, x, and y
        n_th[0][0] = th[0][0] / roi_max
        n_th[0][1] = np.nan
        n_th[0][2] = np.nan

        for particle_index in range(1, hypothesis_index + 1):
            n_th[particle_index][0] = th[particle_index][0] / (roi_max - roi_min) / 2 / np.pi / psf_sigma**2
            n_th[particle_index][1] = th[particle_index][1] / szx
            n_th[particle_index][2] = th[particle_index][2] / szy
            
        # Manipulate theta to use with scipy.optimize.minimize
        nf_th = n_th.flatten() # nf_th: normalized and flattened theta
        nft_th = nf_th[~np.isnan(nf_th)] # ntf_th: normalized, flattened, and trimmed theta

        return nft_th # Return the normalized, flattened, and trimmed theta values

    # If the image we are dealing with is an RGB image, process the theta values as follows:
    elif len(roi_max) == 3 and len(roi_min) == 3: # If the image is an RGB image, then roi_max and roi_min should be a list of 3 values.
        # Format of nf_th: [Bg_r, Bg_g, Bg_b, i_r1, i_g1, i_b1, x1, y1, ...] - There is nothing to trim when dealing with RGB images due to the way the theta values are structured.
        nf_th = [] # nf = normalized and flattened

        # Append the background values (rgb)
        nf_th.append([th['bg'][ch] / roi_max[ch] for ch in range(3)])

        # Append the particle intensity (rgb) and position values 
        for particle_index in range(hypothesis_index):
            nf_th.append([th['particles'][particle_index]['I'][ch] / (roi_max[ch] - roi_min[ch]) / 2 / np.pi / psf_sigma**2 for ch in range(3)])
            nf_th.append([th['particles'][particle_index]['x'] / szx, th['particles'][particle_index]['y'] /szy])
        
        # Flatten the list of lists
        nf_th = np.array([item for sublist in nf_th for item in sublist])
        
        return nf_th # Return the normalized, flattened, and trimmed theta values

    else: # If the image is neither grayscale nor RGB, raise an error
        print("Error: roi_max and roi_min should be either of both length 1 or 3. Check required.")

def denormalize(norm_flat_trimmed_theta, hypothesis_index, roi_min, roi_max, psf_sigma, szx, szy):
    """ Denormalize the normalized theta values.
    
    Args:
        norm_flat_trimmed_theta (ndarray): The normalized and flattened theta values.
        hypothesis_index (int): The index of the hypothesis being tested.
        roi_min (float): The minimum value of the region of interest.
        roi_max (float): The maximum value of the region of interest.
        psf_sigma (float): The standard deviation of the point spread function.
        szx (int): The size of the x-axis.
        szy (int): The size of the y-axis.
        
    Returns:
        ndarray: The denormalized theta values.
    """
    # Denormalize the normalized theta values as follows when dealing with a gray_scale image
    if isinstance(roi_max, (int, np.integer, float)) and isinstance(roi_min, (int, float, np.integer)):
        nan_padded_nft_theta = np.insert(norm_flat_trimmed_theta, [1, 1], np.nan)
        structured_norm_theta_gray = np.reshape(nan_padded_nft_theta, (-1, 3))
        theta = np.zeros((hypothesis_index + 1, 3))
        theta[0][0] = structured_norm_theta_gray[0][0] * roi_max
        theta[0][1] = theta[0][2] = np.nan
        for particle_index in range(1, hypothesis_index + 1):
            theta[particle_index][0] = structured_norm_theta_gray[particle_index][0] * (roi_max - roi_min) * 2 * np.pi * psf_sigma**2
            theta[particle_index][1] = structured_norm_theta_gray[particle_index][1] * szx
            theta[particle_index][2] = structured_norm_theta_gray[particle_index][2] * szy
        return theta
        
    # Denormalize the normalized theta values as follows when dealing with an RGB image
    elif len(roi_max) == 3 and len(roi_min) == 3:
        # Format of norm_flat_trimmed_theta: [Bg_r, Bg_g, Bg_b, i_r1, i_g1, i_b1, x1, y1, ...]
        theta = {'bg': [0, 0, 0], 'particle': [{} for _ in range(hypothesis_index)]}

        # First, denormalize the background values (rgb) and save it to the structured_theta dictionary
        bg_rgb = norm_flat_trimmed_theta[:3]

        # Denormalize the background values (rgb)
        theta['bg'] = [norm_bg_ch * roi_max[ch] for ch, norm_bg_ch in enumerate(bg_rgb)]

        # Next, denormalize the particle intensity (rgb) and position values and save it to the structured_theta dictionary
        for particle_index in range(1, hypothesis_index + 1):
            # See this to look at the desired structure of theta: theta['particle'][particle_index]['x']
            # 3, 8, 13... are the indices of the first element of each particle's intensity values
            i_rgb_start_idx = 3 + 5 * (particle_index - 1)
            norm_i_rgb = norm_flat_trimmed_theta[i_rgb_start_idx:i_rgb_start_idx + 3]
            # 6, 11, 16... are the indices of the first element of each particle's position values
            i_pos_start_idx = 6 + 5 * (particle_index - 1)
            norm_position = norm_flat_trimmed_theta[i_pos_start_idx:i_pos_start_idx + 2]
            # Denormalize the particle intensity (rgb) and position values 
            theta['particle'][particle_index - 1] = {'I': [norm_i_rgb[ch] * (roi_max[ch] - roi_min[ch]) * 2 * np.pi * psf_sigma**2 for ch in range(3)], 
                                                    'x': norm_position[0] * szx, 
                                                    'y': norm_position[1] * szy}
        return theta

    else: # If the image is neither grayscale nor RGB, raise an error
        print("Error: roi_max and roi_min should be either of both length 1 or 3. Check required.")

def position_penalty_function(t, width):
    """ Returns the value of the cup (penalty) function at t.
    Args:
        t (float): The input value.
        width (float): The width of the boundary where there is no penalty (i.e., the image's x/y width depending on the context).
    Returns:
        float: The value of the cup (penalty) function at t.
    """
    return -POSITION_PENALTY_FACTOR * (t + .5)**3 if t < -.5 else (POSITION_PENALTY_FACTOR * (t - width + .5)**3 if t >= width - .5 else 0)
    
def ddt_position_penalty_function(t, width):
    """ Returns the derivative of the cup (penalty) function at t.
    Args:
        t (float): The input value.
        width (float): The width of the boundary where there is no penalty (i.e., the image's x/y width depending on the context).
    Returns:
        float: The derivative of the cup (penalty) function at t.
    """
    return -3 * POSITION_PENALTY_FACTOR * (t + .5)**2 if t < -.5 else (3 * POSITION_PENALTY_FACTOR * (t - width + .5)**2 if t >= width - .5 else 0)
    
def d2dt2_position_penalty_function(t, width):
    """ Returns the second derivative of the cup (penalty) function at t.
    Args:
        t (float): The input value.
        width (float): The width of the boundary where there is no penalty (i.e., the image's x/y width depending on the context).
    Returns:
        float: The second derivative of the cup (penalty) function at t.
    """
    return -6 * POSITION_PENALTY_FACTOR * (t + .5) if t < -.5 else (6 * POSITION_PENALTY_FACTOR * (t - width + .5) if t >= width - .5 else 0)

def intensity_penalty_function(t):
    """
    Args:
        t (int, float, np.ndarray, list): The intensity value(s) of the pixel(s). 
                                      Can be a single value (for grayscale images) 
                                      or a list/array of values (for RGB images).
    Returns:
        float: The computed intensity penalty. For RGB images, the penalties for 
            each channel are summed.
    """
    scale = INTENSITY_PENALTY_FACTOR
    if isinstance(t, (np.ndarray, list)):  # Case: RGB image
        return sum(-scale * (ti**3) if ti < 0 else 0 for ti in t)
    else:  # Case: grayscale image
        return -scale * (t**3) if t < 0 else 0

def ddt_intensity_penalty_function(t):
    """
        Args:
            t (int, float, np.ndarray, list): The intensity value(s) of the pixel(s). 
                                          Can be a single value (for grayscale images) 
                                          or a list/array of values (for RGB images).
        Returns:
            float: The computed derivative of the intensity penalty. For RGB images, 
                the derivatives for each channel are summed.
        """
    scale = INTENSITY_PENALTY_FACTOR
    if isinstance(t, (np.ndarray, list)):  # Case: RGB image
        return sum(-3 * scale * ti**2 if ti < 0 else 0 for ti in t)
    else:  # Case: grayscale image
        return -3 * scale * t**2 if t < 0 else 0


def d2dt2_intensity_penalty_function(t):
    """
    Args:
        t (int, float, np.ndarray, list): The intensity value(s) of the pixel(s). 
                                      Can be a single value (for grayscale images) 
                                      or a list/array of values (for RGB images).
    Returns:
        float: The computed second derivative of the intensity penalty. For RGB images, 
            the second derivatives for each channel are summed.
    """
    scale = INTENSITY_PENALTY_FACTOR
    if isinstance(t, (np.ndarray, list)):  # Case: RGB image
        return sum(-6 * scale * ti if ti < 0 else 0 for ti in t)
    else:  # Case: grayscale image
        return -6 * scale * t if t < 0 else 0


def out_of_bounds_particle_penalty(theta, szx, szy):
    """ Returns a penalty for particles that are out of bounds.
    Args:
        theta (list): The list of particle parameters.
        szx (int): The size of the x-axis.
        szy (int): The size of the y-axis.
    Returns:
        float: The summed penalties for particles that are out of bounds.
    """
    penalty = 0
    if isinstance(theta, (np.ndarray, list)): # Case: grayscale image
        for i in range(1, len(theta)):
            intensity_term = intensity_penalty_function(theta[i][0])
            x_term = position_penalty_function(theta[i][1], szx)
            y_term = position_penalty_function(theta[i][2], szy)
            penalty += intensity_term + x_term + y_term
    elif isinstance(theta, dict): # Case: rgb image
        for i in range(len(theta['particle'])):
            intensity_term = intensity_penalty_function(theta['particle'][i]['I'])
            x_term = position_penalty_function(theta['particle'][i]['x'], szx)
            y_term = position_penalty_function(theta['particle'][i]['y'], szy)
            penalty += intensity_term + x_term + y_term
    else:
        print("Error inside out_of_bounds_particle_penalty(). theta should be either a list/ndarray or a dictionary. Check required.")

    return penalty

def jac_oob_penalty(theta, szx, szy, roi_max, roi_min, psf_sigma):
    """ Returns the derivative of the out of bounds penalty."""
    if isinstance(theta, (np.ndarray, list)): # Case: grayscale image
        ddt_oob = np.zeros((len(theta), 3))
        # Treat the background term (ddt_oob[0][0] is zero as there is no penalty for the background)
        ddt_oob[0][1] = ddt_oob[0][2] = np.nan
        # Treat the particle terms
        for i in range(1, len(theta)):
            ddt_oob[i][0] = ddt_intensity_penalty_function(theta[i][0]) * (roi_max - roi_min) * 2 * np.pi * psf_sigma**2 # (roi_max - roi_min) * 2 * np.pi * psf_sigma**2 is the normalization factor for particle intensity
            ddt_oob[i][1] = ddt_position_penalty_function(theta[i][1], szx) * szx # szx is the normalization factor for the x-coordinate
            ddt_oob[i][2] = ddt_position_penalty_function(theta[i][2], szy) * szx # szy is the normalization factor for the y-coordinate
    else: # Case: rgb image
        # REF - Format of nf_th: [Bg_r, Bg_g, Bg_b, i_r1, i_g1, i_b1, x1, y1, ...]
        # REF - ddt_nll_rgb = np.zeros((hypothesis_index + 1, 5))  # row: particle index (starts from 1 while 0 is for background), col: I_r, I_g, I_b, x, y
        ddt_oob = np.zeros((len(theta['particle'])+1, 5))
        # Treat the background term (ddt_oob[0][i]'s are zeros as there is no penalty for the background)
        ddt_oob[0][3] = ddt_oob[0][4] = np.nan
        # Treat the particle terms
        for i in range(len(theta['particle'])):
            ddt_oob[i][0] = ddt_intensity_penalty_function(theta['particle'][i]['I'][0]) * (roi_max[0] - roi_min[0]) * 2 * np.pi * psf_sigma**2
            ddt_oob[i][1] = ddt_intensity_penalty_function(theta['particle'][i]['x']) * szx
            ddt_oob[i][2] = ddt_intensity_penalty_function(theta['particle'][i]['y']) * szy

    return ddt_oob 

def hess_oob_penalty(theta, szx, szy, roi_max, roi_min, psf_sigma):
    """ Returns the Hessian of the out of bounds penalty."""
    if isinstance(theta, (np.ndarray, list)): # Case: grayscale image
        d2dt2_oob_2d = np.zeros((len(theta)* 3 - 2, len(theta)* 3 - 2))
        # Treat the background term (d2dt2_oob_2d[0][0] is zero as there is no penalty for the background)
        d2dt2_oob_2d[0, :] = d2dt2_oob_2d[:, 0] = 0 # No penalty for the background
        # Treat the particle terms
        for pidx in range(1, len(theta)):
            # i0, i0
            d2dt2_oob_2d[(pidx - 1) * 3 + 1][(pidx - 1) * 3 + 1] = d2dt2_intensity_penalty_function(theta[pidx][0]) * ((roi_max - roi_min) * 2 * np.pi * psf_sigma**2)**2 
            # i0, i1, # i0, i2 # i1, i1 # i1, i2 are all zeros already.
            # i2, i2
            d2dt2_oob_2d[(pidx - 1) * 3 + 3][(pidx - 1) * 3 + 3] = d2dt2_position_penalty_function(theta[pidx][2], szy) * szx**2 
    else: # Case: rgb image
        # REF - Format of nf_th: [Bg_r, Bg_g, Bg_b, i_r1, i_g1, i_b1, x1, y1, ...]
        # REF - d2dt2_nll_2d = np.zeros((hypothesis_index * 5 + 3, hypothesis_index * 5 + 3))
        d2dt2_oob_2d = np.zeros((len(theta['particle']) * 5 + 3, len(theta['particle']) * 5 + 3))
        # Treat the background term - they are already zeros
        # Treat the particle terms
        for pidx in range(len(theta['particle'])):
            d2dt2_oob_2d[(pidx-1)*5 + 3][(pidx-1)*5 + 3] = d2dt2_intensity_penalty_function(theta['particle'][pidx]['I'][0]) * ((roi_max[0] - roi_min[0]) * 2 * np.pi * psf_sigma**2)**2
            d2dt2_oob_2d[(pidx-1)*5 + 4][(pidx-1)*5 + 4] = d2dt2_intensity_penalty_function(theta['particle'][pidx]['I'][1]) * ((roi_max[1] - roi_min[1]) * 2 * np.pi * psf_sigma**2)**2
            d2dt2_oob_2d[(pidx-1)*5 + 5][(pidx-1)*5 + 5] = d2dt2_intensity_penalty_function(theta['particle'][pidx]['I'][2]) * ((roi_max[2] - roi_min[2]) * 2 * np.pi * psf_sigma**2)**2
            d2dt2_oob_2d[(pidx-1)*5 + 6][(pidx-1)*5 + 6] = d2dt2_position_penalty_function(theta['particle'][pidx]['x'], szx) * szx**2
            d2dt2_oob_2d[(pidx-1)*5 + 7][(pidx-1)*5 + 7] = d2dt2_position_penalty_function(theta['particle'][pidx]['y'], szy) * szy**2

    return d2dt2_oob_2d

# Maximum Likelihood Estimation of Hk                       
def modified_neg_loglikelihood_fn(norm_flat_trimmed_theta, hypothesis_index, roi_image, roi_min, roi_max, min_model_xy, psf_sigma, szx, szy):
    """ Calculate the modified negative log-likelihood function.
    
    Args:
        norm_flat_trimmed_theta (ndarray): The normalized flattened trimmed theta.
        hypothesis_index (int): The index of the hypothesis.
        roi_image (ndarray): The region of interest image.
        roi_min (float): The minimum value of the region of interest.
        roi_max (float): The maximum value of the region of interest.
        min_model_xy (float): The minimum value of the model coordinates.
        psf_sigma (float): The standard deviation of the point spread function.
        szx (int): The size of the x-axis.
        szy (int): The size of the y-axis.
        
    Returns:
        float: The modified negative log-likelihood value.
    """
    # # Force-fix negative values
    # norm_flat_trimmed_theta[norm_flat_trimmed_theta < 0] = 0
    # # Force-fix infinite values
    # norm_flat_trimmed_theta[np.isinf(norm_flat_trimmed_theta)] = 1
    # Denormalize theta to calculate model_xy
    theta = denormalize(norm_flat_trimmed_theta, hypothesis_index, roi_min, roi_max, psf_sigma, szx, szy)
    # Calculate the model value at each pixel position
    model_xy, _, _ = calculate_modelxy_ipsfx_ipsfy(theta, np.arange(szx), np.arange(szy), hypothesis_index, min_model_xy, psf_sigma)
    if isinstance(roi_min, (int, np.integer, float)) and isinstance(roi_max, (int, np.integer, float)): # Case: grayscale image
        modified_neg_loglikelihood = np.sum(model_xy - roi_image * np.log(model_xy))
    else: # Case: rgb image
        if roi_image.shape[0] != 3:
            roi_image = np.transpose(roi_image, (2, 0, 1))
        modified_neg_loglikelihood = np.sum(model_xy - roi_image * np.log(model_xy))

    modified_neg_loglikelihood += out_of_bounds_particle_penalty(theta, szx, szy) 

    return modified_neg_loglikelihood 


def calculate_modelxy_ipsfx_ipsfy(theta, xx, yy, hypothesis_index, min_model_xy, psf_sigma):
    ''' Calculate the model intensity at a given position (xx, yy) based on the given parameters.

    Parameters:
        theta (list): Either a list/ndarray (grayscale image) or a dictionary (case: rgb image)
        A list of particle parameters. Each element in the list represents a particle and contains the following information:
                    - Particle intensity (i)
                    - PSF x-coordinate (psf_x)
                    - PSF y-coordinate (psf_y)
        xx (float or numpy array): The x-coordinates to evaluate intensity.
        yy (float or numpy array): The y-coordinates to evaluate intensity.
        hypothesis_index (int): The index of the hypothesis being tested.
        min_model_xy (float): The minimum model intensity for (xx, yy).
        psf_sigma (float): The standard deviation of the PSF.

    Returns:
        tuple: A tuple containing the following values:
            - The model intensity at (xx, yy)
            - An array of integrated PSF x-coordinates
            - An array of integrated PSF y-coordinates
    '''

    if isinstance(theta, (np.ndarray, list, float)): # Case: grayscale image
        if hypothesis_index == 0: # The hypothesis and model assumes background only
            modelhk_at_xxyy = theta
            if modelhk_at_xxyy <= 0:
                modelhk_at_xxyy = min_model_xy
            integrated_psf_x = integrated_psf_y = np.nan

        else: # The hypothesis and model assumes background and particles
            # Initialize the psf_x and psf_y arrays
            integrated_psf_x = np.zeros((hypothesis_index + 1, 1 if isinstance(xx, int) else len(xx)))
            integrated_psf_y = np.zeros((hypothesis_index + 1, 1 if isinstance(yy, int) else len(yy)))
            # integrated_psf_x[0] is nan as it is not used. integrated_psf_x[1] is the psf_x of particle 1.
            integrated_psf_x[0, :] = np.nan
            # integrated_psf_y[0] is nan as it is not used. integrated_psf_y[1] is the psf_y of particle 1.
            integrated_psf_y[0, :] = np.nan
            # modelhk_at_xxyy = background + (i_0 * psf_x_0 * psf_y_0) + (i_1 * psf_x_1 * psf_y_1) + ...
            modelhk_at_xxyy = 0
            if isinstance(xx, int) and isinstance(yy, int):
                modelhk_at_xxyy = theta[0][0]
            else:
                modelhk_at_xxyy += theta[0][0] * np.ones((len(yy), len(xx))) # Add the background contribution to the model intensity

            for particle_index in range(1, hypothesis_index + 1):
                # Calculate the integral (over the pixel) of the normalized 1D psf function for x and y each, for the xx-th column and the yy-th row, respectivly.
                integrated_psf_x[particle_index, :] = integrate_gauss_1d(xx, theta[particle_index][1], psf_sigma)
                integrated_psf_y[particle_index, :] = integrate_gauss_1d(yy, theta[particle_index][2], psf_sigma)

                # update the particles contributions to the modelhk_at_xxyy value
                modelhk_at_xxyy += theta[particle_index][0] * np.outer(integrated_psf_y[particle_index],
                                                                    integrated_psf_x[particle_index]) 
                
                # If the model intensity is negative, set it to the minimum model intensity to ensure physicality
                modelhk_at_xxyy[modelhk_at_xxyy <= 0] = min_model_xy

        return np.squeeze(modelhk_at_xxyy), integrated_psf_x, integrated_psf_y
    
    elif isinstance(theta, dict): # Case: rgb image
        if hypothesis_index == 0: # The hypothesis and model assumes background only 
            assert len(theta['bg']) == 3, "Error: The background should be a list of 3 elements (RGB)."
            model = theta['bg']
            if model <= 0:
                model = min_model_xy
            integrated_psf_x = integrated_psf_y = np.nan

        else: # The hypothesis and model assumes background and particles
            integrated_psf_x = np.zeros((hypothesis_index + 1, 1 if isinstance(xx, int) else len(xx)))
            integrated_psf_y = np.zeros((hypothesis_index + 1, 1 if isinstance(yy, int) else len(yy)))

            # Calculate the model intensities
            model_rgb = np.zeros((3, len(xx), len(yy)))
            # Add the background contribution to the model intensity
            model_rgb += [theta['bg'][ch] * np.ones((len(yy), len(xx))) for ch in range(3)]
            # Add the contributions of the particles to the model intensity
            for particle_index in range(1, hypothesis_index + 1):
                # Calculate the integral (over the pixel) of the normalized 1D psf function for x and y each, for the xx-th column and the yy-th row, respectivly.
                integrated_psf_x = integrate_gauss_1d(xx, theta['particle'][particle_index-1]['x'], psf_sigma)
                integrated_psf_y = integrate_gauss_1d(yy, theta['particle'][particle_index-1]['y'], psf_sigma)
                model_rgb += [np.maximum(theta['particle'][particle_index-1]['I'][ch] * np.outer(integrated_psf_y, integrated_psf_x), min_model_xy) for ch in range(3)]
        return model_rgb, integrated_psf_x, integrated_psf_y
    else:
        print("Error: theta should be either a list/ndarray or a dictionary. Check required.")

def jacobian_fn(norm_flat_trimmed_theta, hypothesis_index, roi_image, roi_min, roi_max, min_model_xy, psf_sigma, szx, szy):
    """ Calculate the Jacobian matrix for the modified negative log-likelihood function.

    Args:
        norm_flat_trimmed_theta (ndarray): The normalized flattened trimmed parameter array.
        hypothesis_index (int): The index of the hypothesis.
        roi_image (ndarray): The region of interest image.
        roi_min (float): The minimum value of the region of interest.
        roi_max (float): The maximum value of the region of interest.
        min_model_xy (float): The minimum value of the model coordinates.
        psf_sigma (float): The standard deviation of the point spread function.
        szx (int): The size of the x-axis.
        szy (int): The size of the y-axis.

    Returns:
        ndarray: The Jacobian matrix.
    """
    theta = denormalize(norm_flat_trimmed_theta, hypothesis_index, roi_min, roi_max, psf_sigma, szx, szy)

    # Precalculate intensity and derivatives 
    # Model_xy: 3 (rgb) x 2D (szy x szx) array with row: y, col: x, Integrated_psf_x: 1d array following x-pos, Integrated_psf_y: 1d array following y-pos
    Model_xy, Integrated_psf_x, Integrated_psf_y = calculate_modelxy_ipsfx_ipsfy(theta, np.arange(szx), np.arange(szy), hypothesis_index, min_model_xy, psf_sigma)
    
    # derivative of the negative log-likelihood (ddt_nll) with respect to the parameters
    if isinstance(theta, (list, np.ndarray)): # Case: grayscale image
        ddt_nll = np.zeros((hypothesis_index + 1, 3)) 
        ddt_nll[0][1] = ddt_nll[0][2] = np.nan
        # Refer to the explanation written inside ddt_integrated_psf_1d function for better understanding of the below two lines.
        Ddt_integrated_psf_1d_x = np.array([ddt_integrated_psf_1d(np.arange(szx), theta[p_idx][1], psf_sigma) for p_idx in range(1, hypothesis_index + 1)]) # 2d array with row: pindex, col: x-position
        Ddt_integrated_psf_1d_y = np.array([ddt_integrated_psf_1d(np.arange(szy), theta[p_idx][2], psf_sigma) for p_idx in range(1, hypothesis_index + 1)]) # 2d array with row: pindex, col: y-position
    else: # Case: rgb image
        if roi_image.shape[0] != 3:
            roi_image = np.transpose(roi_image, (2, 0, 1))
        # Neg log likelihood is
        # sum( [model[ch] * roi[ch] * log(model[ch] for ch in range(3)] )
        # While model is a 3x2D array, NLL is a scalar. 
        ddt_nll_rgb = np.zeros((hypothesis_index + 1, 5))  # row: particle index (starts from 1 while 0 is for background), col: I_r, I_g, I_b, x, y
        ddt_nll_rgb[0][3] = ddt_nll_rgb[0][4] = np.nan
        # Refer to the explanation written inside ddt_integrated_psf_1d function for better understanding of the below two lines.
        Ddt_integrated_psf_1d_x = np.array([ddt_integrated_psf_1d(np.arange(szx), theta['particle'][p_idx]['x'], psf_sigma) for p_idx in range(0, hypothesis_index)]) # 2d array with row: pindex, col: x-position
        Ddt_integrated_psf_1d_y = np.array([ddt_integrated_psf_1d(np.arange(szy), theta['particle'][p_idx]['y'], psf_sigma) for p_idx in range(0, hypothesis_index)]) # 2d array with row: pindex, col: y-position

    # Add extra entry at beginning so indices match pidx  - both for grayscale and rgb case
    Ddt_integrated_psf_1d_x = np.insert(Ddt_integrated_psf_1d_x, 0, None, axis=0)
    Ddt_integrated_psf_1d_y = np.insert(Ddt_integrated_psf_1d_y, 0, None, axis=0)

    if isinstance(theta, (list, np.ndarray)): # Case: grayscale image. If it was RGB, theta would have been a dictionary.

        # Pre-calculate the ratio of the image to the model intensity
        one_minus_image_over_model = (1 - roi_image / Model_xy) # This will be used many times in the following calculations.

        # We need to calculate the derivatives of the modified negative log-likelihood function with respect to the normalized parameters 
        # - These derivative will be the derivatives with respect to unnormalized parameters times the "normalization factor"
        ddt_nll[0][0] = np.sum(one_minus_image_over_model) * roi_max # roi_max is the "normalization factor" for the intensity

        for p_idx in range(1, hypothesis_index + 1):
            ddt_nll[p_idx][0] = np.sum(one_minus_image_over_model * np.outer(Integrated_psf_y[p_idx], Integrated_psf_x[p_idx]) * (roi_max - roi_min) * 2 * np.pi * psf_sigma**2) # (roi_max - roi_min) * 2 * np.pi * psf_sigma**2 is the normalization factor for particle intensity
            ddt_nll[p_idx][1] = np.sum(one_minus_image_over_model * np.outer(Integrated_psf_y[p_idx], Ddt_integrated_psf_1d_x[p_idx]) * theta[p_idx][0] * szx) # szx is the normalization factor for the x-coordinate
            ddt_nll[p_idx][2] = np.sum(one_minus_image_over_model * np.outer(Ddt_integrated_psf_1d_y[p_idx], Integrated_psf_x[p_idx]) * theta[p_idx][0] * szy) # szy is the normalization factor for the y-coordinate

        jacobian = ddt_nll.flatten()
        jacobian = jacobian[~np.isnan(jacobian)]
    
    else: # Case: rgb image
        one_minus_image_over_model_rgb = np.array([(1 - roi_image[ch] / Model_xy[ch]) for ch in range(3)]) # This will be used many times in the following calculations. "1" will act as a matrix of ones.
        # ddt_nll_rgb[0][:3] = np.sum([one_minus_image_over_model_rgb * roi_max[ch] for ch in range(3)], axis=(1, 2, 3))
        ddt_nll_rgb[0][:3] = np.sum(one_minus_image_over_model_rgb * np.array([np.ones((szy, szx)) * 1 * roi_max[ch] for ch in range(3)]), axis=(1, 2))
        # Check: is the following the same as the above? : ddt_nll_rgb[0][:3] = np.sum([one_minus_image_over_model_rgb * roi_max[ch] for ch in range(3)]), axis=(1, 2))

        for p_idx in range(1, hypothesis_index + 1):
            ddt_nll_rgb[p_idx][:3] = [np.sum(one_minus_image_over_model_rgb[ch] * np.outer(Integrated_psf_y[p_idx], Integrated_psf_x[p_idx]) * (roi_max[ch] - roi_min[ch]) * 2 * np.pi * psf_sigma**2) for ch in range(3)] # (roi_max - roi_min) * 2 * np.pi * psf_sigma**2 is the normalization factor for particle intensity
            # Check really carefully here:
            ddt_nll_rgb[p_idx][3] = np.sum([one_minus_image_over_model_rgb[ch] * np.outer(Integrated_psf_y[p_idx], Ddt_integrated_psf_1d_x[p_idx]) * theta['particle'][p_idx - 1]['I'][ch] * szx for ch in range(3)]) # szx is the normalization factor for the x-coordinate
            ddt_nll_rgb[p_idx][4] = np.sum([one_minus_image_over_model_rgb[ch] * np.outer(Ddt_integrated_psf_1d_y[p_idx], Integrated_psf_x[p_idx]) * theta['particle'][p_idx - 1]['I'][ch] * szy for ch in range(3)]) # szy is the normalization factor for the y-coordinate
        
        jacobian = np.array(ddt_nll_rgb[~np.isnan(ddt_nll_rgb)]).flatten()
        

    # Check the shape of the gradient
    if jacobian.shape != norm_flat_trimmed_theta.shape:
        print("Warning: the shape of the jacobian is not the same as the shape of the parameters. Check required")
        # Reshape the gradient to have the same shape as norm_flat_trimmed_theta
        jacobian = jacobian.reshape(norm_flat_trimmed_theta.shape)

    # Add the out-of-bounds penalty to the Jacobian
    ddt_oob = jac_oob_penalty(theta, szx, szy, roi_max, roi_min, psf_sigma) 
    jac_oob = ddt_oob.flatten()
    jac_oob = jac_oob[~np.isnan(jac_oob)]
    jacobian += jac_oob

    return jacobian


def hessian_fn(norm_flat_trimmed_theta, hypothesis_index, roi_image, roi_min, roi_max, min_model_xy, psf_sigma, szx, szy):
    """
    Calculate the Hessian matrix for the negative log-likelihood function.

    Parameters:
    - norm_flat_trimmed_theta (array-like): Normalized and flattened theta values.
    - hypothesis_index (int): Number of hypotheses.
    - roi_image (array-like): Region of interest image.
    - roi_min (float): Minimum value of the region of interest.
    - roi_max (float): Maximum value of the region of interest.
    - min_model_xy (float): Minimum value of the model.
    - psf_sigma (float): Standard deviation of the point spread function.
    - szx (int): Size of the x-axis.
    - szy (int): Size of the y-axis.

    Returns:
    - d2dt2_nll_2d (array-like): Hessian matrix for the negative log-likelihood function.
    """
    theta = denormalize(norm_flat_trimmed_theta, hypothesis_index, roi_min, roi_max, psf_sigma, szx, szy)

    # Precalculate intensity and derivatives
    Model_xy, Integrated_psf_x, Integrated_psf_y = calculate_modelxy_ipsfx_ipsfy(theta, np.arange(szx), np.arange(szy), hypothesis_index, min_model_xy, psf_sigma)

    if isinstance(theta, (list, np.ndarray)): # Case: grayscale image
        # nll: negloglikelihood
        d2dt2_nll_2d = np.zeros((hypothesis_index * 3 + 1, hypothesis_index * 3 + 1))

        Ddt_integrated_psf_1d_x = np.array([ddt_integrated_psf_1d(np.arange(szx), theta[p_idx][1], psf_sigma) for p_idx in range(1, hypothesis_index + 1)])
        Ddt_integrated_psf_1d_y = np.array([ddt_integrated_psf_1d(np.arange(szy), theta[p_idx][2], psf_sigma) for p_idx in range(1, hypothesis_index + 1)])
        D2dt2_integrated_psf_1d_x = np.array([d2dt2_integrated_psf_1d(np.arange(szx), theta[p_idx][1], psf_sigma) for p_idx in range(1, hypothesis_index + 1)])
        D2dt2_integrated_psf_1d_y = np.array([d2dt2_integrated_psf_1d(np.arange(szy), theta[p_idx][2], psf_sigma) for p_idx in range(1, hypothesis_index + 1)])
    
    else: # Case: rgb image
        # Neg log likelihood is
        # sum( [model[ch] * roi[ch] * log(model[ch] for ch in range(3)] )
        # While model is a 3x2D array, NLL is a scalar. 
        # ddt_nll_rgb = np.zeros((hypothesis_index * 5 + 3, hypothesis_index * 5 + 3))  
        # ddt_nll_rgb[0][3] = ddt_nll_rgb[0][4] = np.nan

        d2dt2_nll_2d = np.zeros((hypothesis_index * 5 + 3, hypothesis_index * 5 + 3))
        
        # Refer to the explanation written inside ddt_integrated_psf_1d function for better understanding of the below two lines.
        Ddt_integrated_psf_1d_x = np.array([ddt_integrated_psf_1d(np.arange(szx), theta['particle'][p_idx]['x'], psf_sigma) for p_idx in range(0, hypothesis_index)]) # 2d array with row: pindex, col: x-position
        Ddt_integrated_psf_1d_y = np.array([ddt_integrated_psf_1d(np.arange(szy), theta['particle'][p_idx]['y'], psf_sigma) for p_idx in range(0, hypothesis_index)]) # 2d array with row: pindex, col: y-position
        D2dt2_integrated_psf_1d_x = np.array([d2dt2_integrated_psf_1d(np.arange(szx), theta['particle'][p_idx]['x'], psf_sigma) for p_idx in range(0, hypothesis_index)]) # 2d array with row: pindex, col: x-position
        D2dt2_integrated_psf_1d_y = np.array([d2dt2_integrated_psf_1d(np.arange(szy), theta['particle'][p_idx]['y'], psf_sigma) for p_idx in range(0, hypothesis_index)]) # 2d array with row: pindex, col: y-position
    
    # add extra entry at beginning so indices match pidx
    Ddt_integrated_psf_1d_x = np.insert(Ddt_integrated_psf_1d_x, 0, None, axis=0)
    Ddt_integrated_psf_1d_y = np.insert(Ddt_integrated_psf_1d_y, 0, None, axis=0)      
    D2dt2_integrated_psf_1d_x = np.insert(D2dt2_integrated_psf_1d_x, 0, None, axis=0)        
    D2dt2_integrated_psf_1d_y = np.insert(D2dt2_integrated_psf_1d_y, 0, None, axis=0)       

    if isinstance(theta, (list, np.ndarray)): # Case: grayscale image. If it was RGB, theta would have been a dictionary.
        # 4 + 3 + 2 + 1 = 10 combinations
        pixelval_over_model_squared = roi_image / Model_xy**2
        d2dt2_nll_00_00 = np.sum(pixelval_over_model_squared * (roi_max)**2)
        d2dt2_nll_2d[0][0] = d2dt2_nll_00_00
                
        for pidx in range(1, hypothesis_index + 1):
            # 2. i0, 00
            d2dt2_nll_i0_00 = np.sum(pixelval_over_model_squared * np.outer(Integrated_psf_y[pidx], Integrated_psf_x[pidx]) * (roi_max - roi_min) * (2 * np.pi * psf_sigma**2) * (roi_max) )
            d2dt2_nll_2d[0][(pidx - 1) * 3 + 1] = d2dt2_nll_i0_00
            d2dt2_nll_2d[(pidx - 1) * 3 + 1][0] = d2dt2_nll_i0_00
            # 3. i1, 00
            d2dt2_nll_i1_00 = np.sum(pixelval_over_model_squared * np.outer(Integrated_psf_y[pidx], Ddt_integrated_psf_1d_x[pidx]) * theta[pidx][0] * (szx) * (roi_max) )
            d2dt2_nll_2d[0][(pidx - 1) * 3 + 2] = d2dt2_nll_i1_00
            d2dt2_nll_2d[(pidx - 1) * 3 + 2][0] = d2dt2_nll_i1_00
            # 4. i2, 00
            d2dt2_nll_i2_00 = np.sum(pixelval_over_model_squared * np.outer(Ddt_integrated_psf_1d_y[pidx], Integrated_psf_x[pidx]) * theta[pidx][0] * (szy) * (roi_max) )
            d2dt2_nll_2d[0][(pidx - 1) * 3 + 3] = d2dt2_nll_i2_00
            d2dt2_nll_2d[(pidx - 1) * 3 + 3][0] = d2dt2_nll_i2_00
            # 5. i0, i0 ##
            d2dt2_nll_i0_i0 = np.sum(pixelval_over_model_squared * np.outer(Integrated_psf_y[pidx]**2, Integrated_psf_x[pidx]**2) * ((roi_max - roi_min) * 2 * np.pi * psf_sigma**2)**2 )
            d2dt2_nll_2d[(pidx - 1) * 3 + 1][(pidx - 1) * 3 + 1] = d2dt2_nll_i0_i0
            # 6. i1, i0
            d2dt2_nll_i1_i0 = np.sum((pixelval_over_model_squared * theta[pidx][0] * np.outer(Integrated_psf_y[pidx], Integrated_psf_x[pidx]) + (1 - roi_image / Model_xy)) \
                                    * np.outer(Integrated_psf_y[pidx], Ddt_integrated_psf_1d_x[pidx]) * szx * ((roi_max - roi_min) * 2 * np.pi * psf_sigma**2) )
            d2dt2_nll_2d[(pidx - 1) * 3 + 2][(pidx - 1) * 3 + 1] = d2dt2_nll_i1_i0
            d2dt2_nll_2d[(pidx - 1) * 3 + 1][(pidx - 1) * 3 + 2] = d2dt2_nll_i1_i0
            # 7. i2, i0
            d2dt2_nll_i2_i0 = np.sum((pixelval_over_model_squared * theta[pidx][0] * np.outer(Integrated_psf_y[pidx], Integrated_psf_x[pidx]) + (1 - roi_image / Model_xy)) \
                                    * np.outer(Ddt_integrated_psf_1d_y[pidx], Integrated_psf_x[pidx]) * szy * ((roi_max - roi_min) * 2 * np.pi * psf_sigma**2) )
            d2dt2_nll_2d[(pidx - 1) * 3 + 3][(pidx - 1) * 3 + 1] = d2dt2_nll_i2_i0
            d2dt2_nll_2d[(pidx - 1) * 3 + 1][(pidx - 1) * 3 + 3] = d2dt2_nll_i2_i0
            # 8. i1, i1 
            d2dt2_nll_i1_i1 = np.sum((pixelval_over_model_squared * theta[pidx][0] * np.outer(Integrated_psf_y[pidx]**2, Ddt_integrated_psf_1d_x[pidx] ** 2) \
                                    + (1 - roi_image / Model_xy) * np.outer(Integrated_psf_y[pidx], D2dt2_integrated_psf_1d_x[pidx])) * theta[pidx][0] * szx**2   )
            d2dt2_nll_2d[(pidx - 1) * 3 + 2][(pidx - 1) * 3 + 2] = d2dt2_nll_i1_i1
            # 9. i2, i1 
            d2dt2_nll_i2_i1 = np.sum((pixelval_over_model_squared * theta[pidx][0] * np.outer(Integrated_psf_y[pidx], Integrated_psf_x[pidx]) + (1 - roi_image / Model_xy)) * theta[pidx][0] \
                                    * np.outer(Ddt_integrated_psf_1d_y[pidx], Ddt_integrated_psf_1d_x[pidx]) * szx * szy )
            d2dt2_nll_2d[(pidx - 1) * 3 + 3][(pidx - 1) * 3 + 2] = d2dt2_nll_i2_i1
            d2dt2_nll_2d[(pidx - 1) * 3 + 2][(pidx - 1) * 3 + 3] = d2dt2_nll_i2_i1
            # 10. i2, i2 ##
            d2dt2_nll_i2_i2 = np.sum((pixelval_over_model_squared * theta[pidx][0] * np.outer(Ddt_integrated_psf_1d_y[pidx]** 2, Integrated_psf_x[pidx]**2) \
                                    + (1 - roi_image / Model_xy) * np.outer(D2dt2_integrated_psf_1d_y[pidx], Integrated_psf_x[pidx]))   * theta[pidx][0] * szy**2  )
            d2dt2_nll_2d[(pidx - 1) * 3 + 3][(pidx - 1) * 3 + 3] = d2dt2_nll_i2_i2

    else: # Case: rgb image
        if roi_image.shape[0] != 3:
            roi_image = np.transpose(roi_image, (2, 0, 1))
        # 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 = 36 combinations total.
        # 8: 00r-00r, 00r-00g, 00r-00b, 00r-i0r, 00r-i0g, 00r-i0b, 00r-ix, 00r-iy
        # 7:          00g-00g, 00g-00b, 00g-i0r, 00g-i0g, 00g-i0b, 00g-ix, 00g-iy
        # 6:                   00b-00b, 00b-i0r, 00b-i0g, 00b-i0b, 00b-ix, 00b-iy
        # 5:                            i0r-i0r, i0r-i0g, i0r-i0b, i0r-ix, i0r-iy
        # 4:                                     i0g-i0g, i0g-i0b, i0g-ix, i0g-iy
        # 3:                                              i0b-i0b, i0b-ix, i0b-iy
        # 2:                                                        ix-ix,  ix-iy
        # 1:                                                                iy-iy

        pixelval_over_model_squared = roi_image / Model_xy**2

        # 00r-related
        d2dt2_nll_00r_00r = np.sum([ pixelval_over_model_squared[0] * (roi_max[0])**2 ])
        # d2dt2_nll_00r_00g = d2dt2_nll_00r_00b = 0

        # 00g-related
        d2dt2_nll_00g_00g = np.sum([ pixelval_over_model_squared[1] * (roi_max[1])**2 ])
        # d2dt2_nll_00g_00b = 0

        # 00b-related
        d2dt2_nll_00b_00b = np.sum([ pixelval_over_model_squared[2] * (roi_max[2])**2 ])

        # Assign to the relevant places in the Hessian matrix.
        d2dt2_nll_2d[0][0] = d2dt2_nll_00r_00r # 00r takes the 0th index
        d2dt2_nll_2d[1][1] = d2dt2_nll_00g_00g # 00g takes the 1st index
        d2dt2_nll_2d[2][2] = d2dt2_nll_00b_00b # 00b takes the 2nd indej

        for pidx in range(1, hypothesis_index + 1):

            # 00r-related
            d2dt2_nll_00r_i0r = np.sum([ pixelval_over_model_squared[0] * np.outer(Integrated_psf_x[pidx], Integrated_psf_y[pidx]) * (roi_max[0] - roi_min[0]) * (2 * np.pi * psf_sigma**2) * (roi_max[0]) ])
            # d2dt2_nll_00r_i0g = d2dt2_nll_00r_i0b = 0
            d2dt2_nll_00r_ix  = np.sum([ pixelval_over_model_squared[0] * np.outer(Integrated_psf_y[pidx], Ddt_integrated_psf_1d_x[pidx]) * theta['particle'][pidx-1]['I'][0] * szx * roi_max[0] ])
            d2dt2_nll_00r_iy  = np.sum([ pixelval_over_model_squared[0] * np.outer(Ddt_integrated_psf_1d_y[pidx], Integrated_psf_x[pidx]) * theta['particle'][pidx-1]['I'][0] * szy * roi_max[0] ])

            # 00g-related
            # d2dt2_nll_00g_i0r = 0 
            d2dt2_nll_00g_i0g = np.sum([ pixelval_over_model_squared[1] * np.outer(Integrated_psf_x[pidx], Integrated_psf_y[pidx]) * (roi_max[1] - roi_min[1]) * (2 * np.pi * psf_sigma**2) * (roi_max[1]) ])
            # d2dt2_nll_00g_i0b =  0
            d2dt2_nll_00g_ix  = np.sum([ pixelval_over_model_squared[1] * np.outer(Integrated_psf_y[pidx], Ddt_integrated_psf_1d_x[pidx]) * theta['particle'][pidx-1]['I'][1] * szx * roi_max[1] ])
            d2dt2_nll_00g_iy  = np.sum([ pixelval_over_model_squared[1] * np.outer(Ddt_integrated_psf_1d_y[pidx], Integrated_psf_x[pidx]) * theta['particle'][pidx-1]['I'][1] * szy * roi_max[1] ])

            # 00b-related
            # d2dt2_nll_00b_i0r = d2dt2_nll_00b_i0g = 0
            d2dt2_nll_00b_i0b = np.sum([ pixelval_over_model_squared[2] * np.outer(Integrated_psf_x[pidx], Integrated_psf_y[pidx]) * (roi_max[2] - roi_min[2]) * (2 * np.pi * psf_sigma**2) * (roi_max[2]) ])
            d2dt2_nll_00b_ix  = np.sum([ pixelval_over_model_squared[2] * np.outer(Integrated_psf_y[pidx], Ddt_integrated_psf_1d_x[pidx]) * theta['particle'][pidx-1]['I'][2] * szx * roi_max[2] ])
            d2dt2_nll_00b_iy  = np.sum([ pixelval_over_model_squared[2] * np.outer(Ddt_integrated_psf_1d_y[pidx], Integrated_psf_x[pidx]) * theta['particle'][pidx-1]['I'][2] * szy * roi_max[2] ])


            # i0r-related
            d2dt2_nll_i0r_i0r = np.sum(pixelval_over_model_squared[0] * np.outer(Integrated_psf_y[pidx]**2, Integrated_psf_x[pidx]**2) * (roi_max[0] - roi_min[0]) * (2 * np.pi * psf_sigma**2)**2) 
            # d2dt2_nll_i0r_i0g = d2dt2_nll_i0r_i0b = 0
            d2dt2_nll_i0r_ix  = np.sum( ( pixelval_over_model_squared[0] * theta['particle'][pidx-1]['I'][0] * np.outer(Integrated_psf_y[pidx], Integrated_psf_x[pidx]) + (1 - roi_image[0] / Model_xy[0])) * np.outer(Integrated_psf_y[pidx], Ddt_integrated_psf_1d_x[pidx]) * szx * (roi_max[0] - roi_min[0]) * 2 * np.pi * psf_sigma**2 )
            d2dt2_nll_i0r_iy  = np.sum( ( pixelval_over_model_squared[0] * theta['particle'][pidx-1]['I'][0] * np.outer(Integrated_psf_y[pidx], Integrated_psf_x[pidx]) + (1 - roi_image[0] / Model_xy[0])) * np.outer(Ddt_integrated_psf_1d_y[pidx], Integrated_psf_x[pidx]) * szy * (roi_max[0] - roi_min[0]) * 2 * np.pi * psf_sigma**2 )

            # i0g-related
            d2dt2_nll_i0g_i0g = np.sum(pixelval_over_model_squared[1] * np.outer(Integrated_psf_y[pidx]**2, Integrated_psf_x[pidx]**2) * (roi_max[1] - roi_min[1]) * (2 * np.pi * psf_sigma**2)**2)
            # d2dt2_nll_i0g_i0b = 0
            d2dt2_nll_i0g_ix  = np.sum( ( pixelval_over_model_squared[1] * theta['particle'][pidx-1]['I'][1] * np.outer(Integrated_psf_y[pidx], Integrated_psf_x[pidx]) + (1 - roi_image[1] / Model_xy[1])) * np.outer(Integrated_psf_y[pidx], Ddt_integrated_psf_1d_x[pidx]) * szx * (roi_max[1] - roi_min[1]) * 2 * np.pi * psf_sigma**2 )
            d2dt2_nll_i0g_iy =  np.sum( ( pixelval_over_model_squared[1] * theta['particle'][pidx-1]['I'][1] * np.outer(Integrated_psf_y[pidx], Integrated_psf_x[pidx]) + (1 - roi_image[1] / Model_xy[1])) * np.outer(Ddt_integrated_psf_1d_y[pidx], Integrated_psf_x[pidx]) * szy * (roi_max[1] - roi_min[1]) * 2 * np.pi * psf_sigma**2 )

            # i0b-related
            d2dt2_nll_i0b_i0b = np.sum(pixelval_over_model_squared[2] * np.outer(Integrated_psf_y[pidx]**2, Integrated_psf_x[pidx]**2) * (roi_max[2] - roi_min[2]) * (2 * np.pi * psf_sigma**2)**2)
            d2dt2_nll_i0b_ix  = np.sum(pixelval_over_model_squared[2] * np.outer(Integrated_psf_y[pidx], Ddt_integrated_psf_1d_x[pidx]) * theta['particle'][pidx-1]['I'][2] * szx * (roi_max[2] - roi_min[2]) * 2 * np.pi * psf_sigma**2)
            d2dt2_nll_i0b_iy  = np.sum(pixelval_over_model_squared[2] * np.outer(Ddt_integrated_psf_1d_y[pidx], Integrated_psf_x[pidx]) * theta['particle'][pidx-1]['I'][2] * szy * (roi_max[2] - roi_min[2]) * 2 * np.pi * psf_sigma**2)
            
            # ix-related
            d2dt2_nll_ix_ix   = np.sum([(pixelval_over_model_squared[ch] * theta['particle'][pidx-1]['I'][ch] * np.outer(Ddt_integrated_psf_1d_x[pidx]**2, Ddt_integrated_psf_1d_x[pidx]**2) + (1 - roi_image[ch] / Model_xy[ch]) * np.outer(Integrated_psf_y[pidx], D2dt2_integrated_psf_1d_x[pidx])) * theta['particle'][pidx-1]['I'][ch] * szx**2 for ch in range(3)])
            d2dt2_nll_ix_iy   = np.sum([(pixelval_over_model_squared[ch] * theta['particle'][pidx-1]['I'][ch] * np.outer(Integrated_psf_x[pidx], Integrated_psf_y[pidx]) + (1 - roi_image[ch] / Model_xy[ch])) * theta['particle'][pidx-1]['I'][ch] * np.outer(Ddt_integrated_psf_1d_y[pidx], Ddt_integrated_psf_1d_x[pidx]) * szx * szy for ch in range(3)])

            # ix-related
            d2dt2_nll_iy_iy   = np.sum([(pixelval_over_model_squared[ch] * theta['particle'][pidx-1]['I'][ch] * np.outer(Ddt_integrated_psf_1d_y[pidx]**2, Ddt_integrated_psf_1d_y[pidx]**2) + (1 - roi_image[ch] / Model_xy[ch]) * np.outer(D2dt2_integrated_psf_1d_y[pidx], Integrated_psf_x[pidx])) * theta['particle'][pidx-1]['I'][ch] * szy**2 for ch in range(3)])

            # Assign to the relevant places in the Hessian matrix.
            d2dt2_nll_2d[0][(pidx-1)*5 + 3] = d2dt2_nll_2d[(pidx-1)*5 + 3][0] = d2dt2_nll_00r_i0r # 00r takes the 0th index, i0r takes the '3 + (pidx - 1) * 5'th index
            # d2dt2_nll_2d[0][(pidx-1)*5 + 4] and its transpose element is 0. # 00r takes the 0th index, i0g takes the '4 + (pidx - 1) * 5'th index
            # d2dt2_nll_2d[0][(pidx-1)*5 + 5] and its transpose element is 0. # 00r takes the 0th index, i0b takes the '5 + (pidx - 1) * 5'th index
            d2dt2_nll_2d[0][(pidx-1)*5 + 6] = d2dt2_nll_2d[(pidx-1)*5 + 6][0] = d2dt2_nll_00r_ix  # 00r takes the 0th index, ix  takes the '6 + (pidx - 1) * 5'th index
            d2dt2_nll_2d[0][(pidx-1)*5 + 7] = d2dt2_nll_2d[(pidx-1)*5 + 7][0] = d2dt2_nll_00r_iy  # 00r takes the 0th index, iy  takes the '7 + (pidx - 1) * 5'th index

            # d2dt2_nll_2d[1][(pidx-1)*5 + 3] and its transpose element is 0. # 00g takes the 0th index, i0r takes the '4 + (pidx - 1) * 5'th index
            d2dt2_nll_2d[1][(pidx-1)*5 + 4] = d2dt2_nll_2d[(pidx-1)*5 + 4][1] = d2dt2_nll_00g_i0g # 00g takes the 1th index, i0g takes the '4 + (pidx - 1) * 5'th index
            # d2dt2_nll_2d[1][(pidx-1)*5 + 5] and its transpose element is 0. # 00g takes the 0th index, i0b takes the '5 + (pidx - 1) * 5'th index
            d2dt2_nll_2d[1][(pidx-1)*5 + 6] = d2dt2_nll_2d[(pidx-1)*5 + 6][1] = d2dt2_nll_00g_ix  # 00g takes the 1th index, ix  takes the '6 + (pidx - 1) * 5'th index
            d2dt2_nll_2d[1][(pidx-1)*5 + 7] = d2dt2_nll_2d[(pidx-1)*5 + 7][1] = d2dt2_nll_00g_iy  # 00g takes the 1th index, iy  takes the '7 + (pidx - 1) * 5'th index

            # d2dt2_nll_2d[2][(pidx-1)*5 + 3] and its transpose element is 0. # 00b takes the 2th index, i0r takes the '3 + (pidx - 1) * 5'th index
            # d2dt2_nll_2d[2][(pidx-1)*5 + 4] and its transpose element is 0. # 00b takes the 2th index, i0g takes the '4 + (pidx - 1) * 5'th index
            d2dt2_nll_2d[2][(pidx-1)*5 + 5] = d2dt2_nll_2d[(pidx-1)*5 + 3][2] = d2dt2_nll_00b_i0b # 00b takes the 1th index, i0b takes the '5 + (pidx - 1) * 5'th index
            d2dt2_nll_2d[2][(pidx-1)*5 + 6] = d2dt2_nll_2d[(pidx-1)*5 + 6][2] = d2dt2_nll_00b_ix  # 00b takes the 1th index, ix  takes the '6 + (pidx - 1) * 5'th index
            d2dt2_nll_2d[2][(pidx-1)*5 + 7] = d2dt2_nll_2d[(pidx-1)*5 + 7][2] = d2dt2_nll_00b_iy  # 00b takes the 1th index, iy  takes the '7 + (pidx - 1) * 5'th index

            d2dt2_nll_2d[(pidx-1)*5 + 3][(pidx-1)*5 + 3] = d2dt2_nll_i0r_i0r # i0r takes the '3 + (pidx - 1) * 5'th index
            d2dt2_nll_2d[(pidx-1)*5 + 3][(pidx-1)*5 + 6] = d2dt2_nll_2d[(pidx-1)*5 + 6][(pidx-1)*5 + 3] = d2dt2_nll_i0r_ix  # ix  takes the '6 + (pidx - 1) * 5'th index
            d2dt2_nll_2d[(pidx-1)*5 + 3][(pidx-1)*5 + 7] = d2dt2_nll_2d[(pidx-1)*5 + 7][(pidx-1)*5 + 3] = d2dt2_nll_i0r_iy  # iy  takes the '7 + (pidx - 1) * 5'th index

            d2dt2_nll_2d[(pidx-1)*5 + 4][(pidx-1)*5 + 4] = d2dt2_nll_i0g_i0g # i0g takes the '4 + (pidx - 1) * 5'th index
            d2dt2_nll_2d[(pidx-1)*5 + 4][(pidx-1)*5 + 6] = d2dt2_nll_2d[(pidx-1)*5 + 6][(pidx-1)*5 + 4] = d2dt2_nll_i0g_ix  # ix  takes the '6 + (pidx - 1) * 5'th index
            d2dt2_nll_2d[(pidx-1)*5 + 4][(pidx-1)*5 + 7] = d2dt2_nll_2d[(pidx-1)*5 + 7][(pidx-1)*5 + 4] = d2dt2_nll_i0g_iy  # iy  takes the '7 + (pidx - 1) * 5'th index
        
            d2dt2_nll_2d[(pidx-1)*5 + 5][(pidx-1)*5 + 5] = d2dt2_nll_i0b_i0b # i0b takes the '5 + (pidx - 1) * 5'th index
            d2dt2_nll_2d[(pidx-1)*5 + 5][(pidx-1)*5 + 6] = d2dt2_nll_2d[(pidx-1)*5 + 6][(pidx-1)*5 + 5] = d2dt2_nll_i0b_ix  # ix  takes the '6 + (pidx - 1) * 5'th index
            d2dt2_nll_2d[(pidx-1)*5 + 5][(pidx-1)*5 + 7] = d2dt2_nll_2d[(pidx-1)*5 + 7][(pidx-1)*5 + 5] = d2dt2_nll_i0b_iy  # iy  takes the '7 + (pidx - 1) * 5'th index

            d2dt2_nll_2d[(pidx-1)*5 + 6][(pidx-1)*5 + 6] = d2dt2_nll_ix_ix
            d2dt2_nll_2d[(pidx-1)*5 + 6][(pidx-1)*5 + 7] = d2dt2_nll_2d[(pidx-1)*5 + 7][(pidx-1)*5 + 6] = d2dt2_nll_ix_iy

            d2dt2_nll_2d[(pidx-1)*5 + 7][(pidx-1)*5 + 7] = d2dt2_nll_iy_iy

    d2dt2_nll_2d += hess_oob_penalty(theta, szx, szy, roi_max, roi_min, psf_sigma) # newly added line (2nd July 2024)

    return d2dt2_nll_2d


def getbox(input_image, ii, sz, x_positions, y_positions):
    """ Returns the specified subregion of input_image along with the left and top coordinate.
    Args:
        input_image: The original 2D image to crop from.
        ii: The index of the point to copy.
        sz: The size of the subregion to copy.
        x_positions: X coordinates of the center of the subregions.
        y_positions: Y coordinates of the center of the subregions.
    Returns:
        np.array: The cropped subregion.
        int: The left coordinate of the subregion.
        int: The top coordinate of the subregion.
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

    # Create subregions around each point
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
    kernel_length = len(one_d_kernel)

    # Create a full 2D kernel from the 1D kernel
    full_kernel = np.outer(one_d_kernel, one_d_kernel)

    # Calculate padding based on the desired origin
    pad_before = origin - 1
    pad_after = kernel_length - origin

    # Apply padding to create an adjusted kernel
    adjusted_kernel = np.pad(full_kernel, ((pad_before, pad_after), (pad_before, pad_after)), mode='constant')

    return adjusted_kernel

def get_tentative_peaks(image, min_distance=1,):
    """ Returns the tentative peak coordinates of the image.
    Args:
        image: The 2D image to process.
        min_distance: The minimum distance between peaks.
    Returns:
        np.array: The tentative peak coordinates.
    """

    # Define filters
    h2 = 1/16
    h1 = 1/4
    h0 = 3/8
    g0 = np.array([h2, h1, h0, h1, h2])
    g1 = np.array([h2, 0, h1, 0, h0, 0, h1, 0, h2])
    k0 = create_separable_filter(g0, 3)
    dip_image = dip.Image(image)
    # Filter image
    v0 = dip.Convolution(dip_image, k0, method="best")
    k1 = create_separable_filter(g1, 5)
    v1 = dip.Convolution(v0, k1, method="best")
    filtered_image = np.asarray(v0 - v1)
    filtered_image = filtered_image - np.min(filtered_image)
    tentative_peak_coordinates = peak_local_max(filtered_image, min_distance=min_distance)
    return tentative_peak_coordinates


def integrate_gauss_1d(i, x, sigma):
    """ Compute the integral of the 1D Gaussian.
    Args:
        i (int or numpy array of ints): Pixel index.
        x (float): Center position of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.
    Returns:
        float: Integral of the Gaussian from i-0.5 to i+0.5.
    """
    norm = 1/2/sigma**2
    # Below is the same as integral(from i-0.5 to i+0.5) [1/2sqrt(pi)*exp(-norm*(t-x)**2) dt]
    return 0.5*(erf((i-x+0.5)*np.sqrt(norm))-erf((i-x-0.5)*np.sqrt(norm)))


def ddt_integrated_psf_1d(i, t, sigma):
    '''
    Calculate the derivative of the integrated PSF with respect to the (estimated) particle location t.
    (the derivative of the integral of the 1D Gaussian from i-0.5 to i+0.5 with respect to t)
    (In my note, this corresponds to d/d((theta_i1) [I_x^i]

    Parameters:
    i (float or numpy array): The x or y coordinate (or an array of coordinates) to evaluate derivative of integrated PSF.
    t (float): The (estimated) particle location.
    sigma (float): The width of the PSF.

    Returns:
        - The derivative of integrated PSF x or y coordinate (or an array of values at x or y coordinate, given i is an array).
    '''
    a = np.exp(-(i + 0.5 - t)**2 / (2 * sigma**2)) # This corresponds to g_x^i(+) in my note
    b = np.exp(-(i - 0.5 - t)**2 / (2 * sigma**2)) # This corresponds to g_x^i(-) in my note
    return -1 / (np.sqrt(2 * np.pi) * sigma) * (a - b)

def d2dt2_integrated_psf_1d(i, t, sigma):
    """ Calculate the second derivative of the integrated PSF with respect to the (estimated) particle location t.
    (the second derivative of the integral of the 1D Gaussian from i-0.5 to i+0.5 with respect to t)
    (In my note, this corresponds to d^2/d((theta_i1)^2) [I_x^i]
    
    Args:
        i (float or numpy array): The x or y coordinate (or an array of coordinates) to evaluate the second derivative of integrated PSF.
        t (float): The (estimated) particle location.
        sigma (float): The width of the PSF.
        
    Returns:
        - The second derivative of integrated PSF x or y coordinate (or an array of values at x or y coordinate, given i is an array).
    """
    a = np.exp(-0.5 * ((i + 0.5 - t) / sigma)**2)
    b = np.exp(-0.5 * ((i - 0.5 - t) / sigma)**2)
    return -1 / np.sqrt(2 * np.pi) / sigma**3 * ((i + 0.5 - t) * a - (i - 0.5 - t) * b)


def merge_coincident_particles(image, tile_dicts, psf, display_merged_locations=True):
    """ If an image was subdivided into tiles, this function merges the coincident particles in the overlapping regions of the tiles.

    Parameters:
        image (np.ndarray): The image.
        tile_dicts_array (np.ndarray): The array of tile dictionaries.
        - tile_dicts_array[x_index][y_index] = {'x_low_end': x_low_end, 'y_low_end': y_low_end, 'image_slice': image[y_low_end:y_high_end, x_low_end:x_high_end], 'particle_locations': []}
        psf (float): The point spread function's sigma (width) in pixels.

    Returns:
        merged_locations (list): The list of merged particle locations.
    """

    # Display the merged locations if display_merged_locations is True
    if display_merged_locations:
        _, axs = plt.subplots(2, 1, figsize=(5,10))
        markers = ['1', '2', '|',  '_', '+', 'x',] * 100
        palette = sns.color_palette('Paired', len(tile_dicts.flatten()))
        plt.sca(axs[0])
        plt.imshow(image, cmap='gray')     
        count_before_resolution = sum([len(tile_dict['particle_locations']) for tile_dict in tile_dicts.flatten()])
        plt.title(f'Particle count before resolution: {count_before_resolution}')
        ax = plt.gca()

        # Display tile boundaries and each tile's particle locations
        for particle_marker_idx, tile_dict in enumerate(tile_dicts.flatten()):
            locations = tile_dict['particle_locations']
            rectangle = plt.Rectangle((tile_dict['x_low_end'], tile_dict['y_low_end']), tile_dict['image_slice'].shape[1], tile_dict['image_slice'].shape[0], edgecolor=palette[particle_marker_idx], facecolor='none', linewidth=1, )
            ax.add_patch(rectangle)
            for loc in locations:
                plt.scatter(loc[0] + tile_dict['x_low_end'], loc[1] + tile_dict['y_low_end'], marker=markers[particle_marker_idx], s=300, color=palette[particle_marker_idx], linewidths=2)

    # For each tile 
    for ref_col in range(tile_dicts.shape[0]):
        for ref_row in range(tile_dicts.shape[1]):
    
            # Set the reference tile as the current tile.
            ref_tile = tile_dicts[ref_col][ref_row]
            # List all particle indices of the reference tile.
            all_pidx = list(range(len(ref_tile['particle_locations'])))
    
            if ref_col < tile_dicts.shape[0] - 1:  # If the tile is NOT the rightmost tile.

                # Get the tile to the right.
                right_tile = tile_dicts[ref_col + 1][ref_row]

                del_pidx = [] # If determined to be the same particle, the particle index (as referenced in the reference tile) will be added to this list.

                for ref_pidx in all_pidx: # For each particle's location recorded for the reference tile:

                    ref_loc = ref_tile['particle_locations'][ref_pidx] # This location is relative to the reference tile.

                    for right_loc in right_tile['particle_locations']: # For each particle's location relative to the right tile:
    
                        # Calculate the absolute locations of the particles.
                        abs_ref_loc = ref_loc + np.array([ref_tile['x_low_end'], ref_tile['y_low_end']])
                        abs_right_loc = right_loc + np.array([right_tile['x_low_end'], right_tile['y_low_end']])
    
                        # If the distance between the two locations is less than psf, then consider them as the same particle.
                        if np.sum((abs_ref_loc - abs_right_loc)**2) < psf**2:
                            # These particles will be deleted from the reference tile. (One could also average the particle location, but such implementation needs more careful consideration.)
                            del_pidx.append(ref_pidx)

                # From the reference tile, delete the particles that are determined to be the same particle. It's important to delete from the ref tile only, and not from the right tile.
                ref_tile['particle_locations'] = [loc for i, loc in enumerate(ref_tile['particle_locations']) if i not in del_pidx]

            # List all particle indices of the reference tile again, as the indices may have changed.
            all_pidx = list(range(len(ref_tile['particle_locations'])))

            if ref_row < tile_dicts.shape[1] - 1: # If the tile is NOT the bottommost tile:

                # Get the tile below.
                bottom_tile = tile_dicts[ref_col][ref_row + 1]

                # Initialize the list of particle indices to be deleted (as referenced in the reference tile).
                del_pidx = []

                for ref_pidx in all_pidx: # For each particle's location recorded for the reference tile:

                    ref_loc = ref_tile['particle_locations'][ref_pidx] # This location is relative to the reference tile.
                    
                    for bottom_loc in bottom_tile['particle_locations']: # For each particle's location relative to the bottom tile:

                        # Calculate the absolute locations of the particles.
                        abs_ref_loc = ref_loc + np.array([ref_tile['x_low_end'], ref_tile['y_low_end']])
                        abs_bottom_loc = bottom_loc + np.array([bottom_tile['x_low_end'], bottom_tile['y_low_end']])

                        # If the distance between the two locations is less than psf, then consider them as the same particle.
                        if np.sum((abs_ref_loc - abs_bottom_loc)**2) < psf**2:
                            del_pidx.append(ref_pidx)

                # From the reference tile, delete the particles that are determined to be the same particle. It's important to delete from the ref tile only, and not from the bottom tile.
                ref_tile['particle_locations'] = [loc for i, loc in enumerate(ref_tile['particle_locations']) if i not in del_pidx]

    result_locations = []
    for tile_dict in tile_dicts.flatten():
        for loc in tile_dict['particle_locations']:
            absolute_loc = loc + np.array([tile_dict['x_low_end'], tile_dict['y_low_end']])
            result_locations.append(absolute_loc)

    if display_merged_locations:
        plt.sca(axs[1])
        ax = plt.gca()
        plt.title(f'Same locations merged (count:{len(result_locations)})')
        plt.imshow(image, cmap='gray')     
        for loc in result_locations:
            plt.scatter(loc[0], loc[1], marker=markers[particle_marker_idx], s=200, color='red', linewidths=1)
        plt.show()

    return result_locations

def generalized_maximum_likelihood_rule(roi_image, psf_sigma, last_h_index=5, random_seed=0, display_fit_results=False, display_xi_graph=False, use_exit_condi=False):
#   generalized_maximum_likelihood_rule(tile_dict['image_slice'], psf_sigma, last_h_index, analysis_rand_seed_per_image, use_exit_condi=use_exit_condi, display_fit_results=display_fit_results, display_xi_graph=display_xi_graph)
    # Set the random seed
    np.random.seed(random_seed)

    # Check the input image
    if roi_image.ndim == 3:
        szy, szx = roi_image.shape[1], roi_image.shape[2]
    elif roi_image.ndim == 2:
        szy, szx = roi_image.shape 
    else:
        print('Invalid input image shape.')
        return
    
    """ Indexing rules
    - hypothesis_index: 0, 1, 2, ...    (H0, H1, H2, ...)
    - particle_index: 1, 2, 3, ...     (particle 1, particle 2, particle 3, ...)
    - param_type_index: 0, 1, 2        (intensity, x-coordinate, y-coordinate)
    """ 

    # Find tentative peaks
    tentative_peaks = get_tentative_peaks(roi_image, min_distance=1)
    rough_peaks_xy = [peak[::-1] for peak in tentative_peaks]

    # Set the minimum model at any x, y coordinate to avoid dividing by zero.
    min_model_xy = 1e-2
    # Set the method to use with scipy.optimize.minimize for the MLE estimation.
    method = 'trust-exact'
    
    # MLE estimation of H1, H2, 
    n_hk_params_per_particle = 3
    n_h0_params = 1
    
    # Initialize test scores
    xi = [] # Which will be lli - penalty
    lli = [] # log likelihood
    penalty = [] # penalty term
    
    fisher_info = [] # Fisher Information Matrix

    # roi_max and roi_min will be used to initialize background and particle intensities estimations.
    roi_max, roi_min = np.max(roi_image), np.min(roi_image) 

    # Figure showing parameter estimation results for all tested hypotheses.
    if display_fit_results:
        _, ax_main = plt.subplots(2, last_h_index + 1, figsize=(2 * (last_h_index + 1), 4))
        # Create a colormap instance
        cmap = plt.get_cmap('turbo')# Create a colormap instance for tentative peak coordinates presentation.
        
        # As an exception to the other axes, use ax_main[1][0] to show tentative peak locations, since there's is not much to show for a simple background estimation.
        for i, coord in enumerate(rough_peaks_xy):
            x, y = coord # Check whether this is correct.
            color = cmap(i / len(rough_peaks_xy))  # Use turbo colormap
            ax_main[1][0].text(x, y, f'{i}', fontsize=6, color=color) 
        ax_main[1][0].set_xlim(0-.5, szx-.5)
        ax_main[1][0].set_ylim(szy-.5, 0-.5) 
        ax_main[1][0].set_aspect('equal')
        ax_main[1][0].set_title('Tentative Peak Coordinates', fontsize=8)
        ax_main[0][0].imshow(roi_image)
        plt.show(block=False)
    
    # xi_drop_count is used for exit condition of the loop.
    xi_drop_count = 0
    
    # Initialize the fit results
    fit_results = [] 

    for hypothesis_index in range(last_h_index + 1): # hypothesis_index is also the number of particles. 

        # print('Testing: Hypothesis Index ', hypothesis_index)
        # Initialization
        n_hk_params = n_h0_params + hypothesis_index * (n_hk_params_per_particle) #H0: 1, H1: 5, H2: 8, ...
        

        # Initialize the theta (parameter) vector
        # theta[1][0] will be the estimated scattering strength of particle 1.
        # theta[1][1], theta[1][2] will be the estimated x and y coordinate of particle 1, etc.
        # theta[0][0] will be the estimated background intensity. (However, if hypothesis_index == 0, theta will just be a scalar, and equal the background intensity.)
        # Since background intensity is the only parameter for H0, theta[0][1] and theta[0][2] will be nan and, importantly, not be passed for optimization.
        if hypothesis_index == 0:
            # initialize theta as a scalar value for background intensity.
            theta = 0.0
        else:
            theta = np.zeros((hypothesis_index + 1, n_hk_params_per_particle)) 

        # Starting values
        if hypothesis_index == 0:
            assert isinstance(theta, (int, float))
            theta = roi_image.sum() / szx / szy

        else: # Initializing estimated particle_intensities
            assert theta.ndim == 2

            # Set the background first
            theta[0][0] = roi_min
            theta[0][1] = theta[0][2] = np.nan

            try:
                for particle_index in range(1, hypothesis_index + 1): # Note that the particle index starts from 1, not 0. 
                    # Initialize estimated particle intensities to the maximum value of the Gaussian roi image.
                    theta[particle_index][0] = (roi_max - roi_min) * 2 * np.pi * psf_sigma**2

                # Initialize particle coordinates according to the tentative peaks found.
                for particle_index in range(1, hypothesis_index + 1):
                    if len(rough_peaks_xy) <= 0:
                        print('No tentative peaks found.')
                        break
                    if particle_index <= len(rough_peaks_xy):
                        theta[particle_index][1] = rough_peaks_xy[particle_index-1][0]
                        theta[particle_index][2] = rough_peaks_xy[particle_index-1][1]
                    else:
                        # assign random positions. 
                        theta[particle_index][1] = random.random() * (szx - 1)
                        theta[particle_index][2] = random.random() * (szy - 1)
            except Exception as e:
                print(f"Error occurred during initialization of theta inside gmlr(): {e}")
                print(f"theta: {theta}")

        # Only do the MLE if k > 0
        if hypothesis_index == 0:
            assert n_hk_params == 1
            convergence_list = [True]
        else:
            # Normazlize the parameters before passing on to neg_loglikelihood_function
            norm_flat_trimmed_theta = normalize(theta, hypothesis_index, roi_min, roi_max, psf_sigma, szx, szy)

            # Initialize storage for the jacobian and hessian snapshots
            # jac_snapshots = []
            gradientnorm_snapshots = []
            # hess_snapshots = []
            fn_snapshots = []
            theta_snapshots = []
            denormflat_theta_snapshots = []

            # Define the callback function as a nested function
            def callback_fn(xk, *args):
                jac = jacobian_fn(xk, hypothesis_index, roi_image, roi_min, roi_max, min_model_xy, psf_sigma, szx, szy)
                gradientnorm = np.linalg.norm(jac)
                fn = modified_neg_loglikelihood_fn(xk, hypothesis_index, roi_image, roi_min, roi_max, min_model_xy, psf_sigma, szx, szy)
                gradientnorm_snapshots.append(gradientnorm)
                fn_snapshots.append(fn)                
                theta_snapshots.append(xk)
                denormflat_theta_snapshots.append(denormalize(xk, hypothesis_index, roi_min, roi_max, psf_sigma, szx, szy).flatten())

            # Now, let's update the parameters using scipy.optimize.minimize
            if np.isnan(norm_flat_trimmed_theta).any() or np.isinf(norm_flat_trimmed_theta).any():  # Check if the array contains NaN or inf values
                print("norm_flat_trimmed_theta contains NaN or inf values.")

            # print(f"Starting parameter vector (denormalized): \n{denormalize(norm_flat_trimmed_theta)}")
            try:
                minimization_result = minimize(modified_neg_loglikelihood_fn, norm_flat_trimmed_theta, args=(hypothesis_index, roi_image, roi_min, roi_max, min_model_xy, psf_sigma, szx, szy),
                                method=method, jac=jacobian_fn, hess=hessian_fn, callback=callback_fn, options={'gtol': 100})
            except Exception as e:
                print(f"Error occurred during optimization: {e}")
                # print("Here is the last (denorm) theta snapshot:")
                # print(denormalize(theta_snapshots[-1], hypothesis_index, roi_min, roi_max, psf_sigma, szx, szy))

                

            # print(f'H{hypothesis_index} converged?: {result.success}')
            # print(f'Last gradientnorm: {gradientnorm_snapshots[-1]:.0f}')
            snapshot_length = len(fn_snapshots)
            convergence = minimization_result.success
            norm_theta = minimization_result.x

            convergence_list.append(convergence)

            # Retrieve the estimated parameters.
            theta = denormalize(norm_theta, hypothesis_index, roi_min, roi_max, psf_sigma, szx, szy)           
                            
        # Store fit results
        if hypothesis_index == 0:
            current_hypothesis_fit_result = {
                'hypothesis_index': hypothesis_index,
                'theta': theta,
                'convergence': True,
            }
        else:
            current_hypothesis_fit_result = {
                'hypothesis_index': hypothesis_index,
                'theta': theta,
                'convergence': convergence,
            }
        # Append the fit result to fit_results
        fit_results.append(current_hypothesis_fit_result)

        if display_fit_results and hypothesis_index > 0:
            # ax_main[0].cla()
            # _, ax_main = plt.subplots(2, 1, figsize=(2 * (1), 5))
            ax_main[0][hypothesis_index].set_title(f"H{hypothesis_index} - convgd: {convergence_list[hypothesis_index]}\nbg: {theta[0][0]:.1f}", fontsize=8)
            for particle_index in range(1, hypothesis_index + 1):
                # print(f"theta[ {particle_index} ]: {theta[particle_index][0]:.3f}\t{theta[particle_index][1]:.3f}\t{theta[particle_index][2]:.3f}")
                ax_main[0][hypothesis_index].imshow(roi_image)
                red = random.randint(200, 255)
                green = random.randint(0, 100)
                blue = random.randint(0, 50)
                color_code = '#%02X%02X%02X' % (red, green, blue)
                ax_main[0][hypothesis_index].scatter(theta[particle_index][1], theta[particle_index][2], s=10, color=color_code, marker='x')
                ax_main[0][hypothesis_index].text(theta[particle_index][1] + np.random.rand() * 1.5, theta[particle_index][2] + (np.random.rand() - 0.5) * 4,
                                            f'  {theta[particle_index][0]:.1f}', color=color_code, fontsize=10,) 
            ax_main[1][hypothesis_index].set_title(f'Gradient norm\nFinal func val: {fn_snapshots[-1]:.04e}', fontsize=8)
            ax_main[1][hypothesis_index].plot(np.arange(snapshot_length), gradientnorm_snapshots, '-o', color='black', markersize=2, label='Gradient norm')
            ax_main[1][hypothesis_index].set_ylim(bottom=0)
            plt.tight_layout()
            plt.show(block=False)
            pass

        ## -- One attempt at vectorization. Didn't work. Need checking. -- ##
        # norm_fisher_mat = np.zeros((n_hk_params, n_hk_params))
        # norm_flat_trimmed_theta = normalize(theta)
        # if hypothesis_index == 0:
        #     fisher_mat[0,0] = 1 / max(norm_flat_trimmed_theta, min_model_xy / roi_max) # theta (which is model_h0) is the constant background across the image (thus independent of x and y coordinates)
        #     assert fisher_mat.shape == (1,1)
        # else:
        #     Modelhk, Integrated_psf_x, Integrated_psf_y = calculate_modelxy_ipsfx_ipsfy(theta, range(szx), range(szy), hypothesis_index, min_model_xy, psf_sigma)
        #     # Calculate the first derivatives of the model (row: particle index (0 means background), col: [intensity, x, y])
        #     Ddt_modelhk = np.zeros((hypothesis_index + 1, n_hk_params_per_particle, szy, szx))
        #     # Ddt_modelhk_at_xxyy[0][0] = 1.0 # differentiation w.r.t. background intensity is 1.0.
        #     Ddt_modelhk[0][0] = np.ones((szy, szx), dtype=float) # differentiation w.r.t. background intensity is 1.0.
        #     # Ddt_modelhk_at_xxyy[0][1] = Ddt_modelhk_at_xxyy[0][2] = np.nan  # No x and y coordinates for background intensity.
        #     Ddt_modelhk[0, :] = np.full((szy, szx), np.nan) # differentiation w.r.t. x and y coordinates are 0.0.
        #     # Calculate the first derivatives of the model w.r.t intensity, x, and y coordinates for each particle.
        #     Ddt_integrated_psf_1d_x = np.append(np.array(np.nan), np.array([ddt_integrated_psf_1d(np.arange(szx), theta[p_idx][1], psf_sigma) for p_idx in range(1, hypothesis_index + 1)]))
        #     Ddt_integrated_psf_1d_y = np.append(np.array(np.nan), np.array([ddt_integrated_psf_1d(np.arange(szy), theta[p_idx][2], psf_sigma) for p_idx in range(1, hypothesis_index + 1)]))

        fisher_mat = np.zeros((n_hk_params, n_hk_params)) # Fisher Information Matrix
        fisher_mat_original = np.zeros(fisher_mat.shape) # Fisher Information Matrix
        # All iterations finished. Now, let's calculate the Fisher Information Matrix (FIM) under Hk.
        if hypothesis_index == 0:
            fisher_mat_original[0,0] = 1 / max(theta, min_model_xy) # theta (which is model_h0) is the constant background across the image (thus independent of x and y coordinates)
            assert fisher_mat_original.shape == (1,1)
        else: # hypothesis_index > 0
            #####--------Below vectorization didn't work. Will fix later. (2024.07.09, Neil) --------#####
            # # # Ddt_modelhk_at_xxyy is a 2D array with row: particle_index, col: param_type_index
            # Modelhk, Integrated_psf_x, Integrated_psf_y = calculate_modelxy_ipsfx_ipsfy(theta, range(szx), range(szy), hypothesis_index, min_model_xy, psf_sigma)
            # # Create the 2D array for x-position
            # # # Calculate the first derivatives of the model w.r.t. x, y, N, and bg
            # Ddt_modelhk = np.zeros((hypothesis_index + 1, n_hk_params_per_particle, szy, szx))
            # # Ddt_modelhk_at_xxyy[0][0] = 1.0 # differentiation w.r.t. background intensity is 1.0.
            # Ddt_modelhk[0][0] = np.ones((szy, szx), dtype=float) # differentiation w.r.t. background intensity is 1.0.
            # # Ddt_modelhk_at_xxyy[0][1] = Ddt_modelhk_at_xxyy[0][2] = np.nan  # No x and y coordinates for background intensity.
            # Ddt_modelhk[0, :] = np.full((szy, szx), np.nan) # differentiation w.r.t. x and y coordinates are 0.0.
            # # Calculate the first derivatives of the model w.r.t intensity, x, and y coordinates for each particle.
            # Ddt_integrated_psf_1d_x = np.append(np.array(np.nan), np.array([ddt_integrated_psf_1d(np.arange(szx), theta[p_idx][1], psf_sigma) for p_idx in range(1, hypothesis_index + 1)]))
            # Ddt_integrated_psf_1d_y = np.append(np.array(np.nan), np.array([ddt_integrated_psf_1d(np.arange(szy), theta[p_idx][2], psf_sigma) for p_idx in range(1, hypothesis_index + 1)]))
            # for particle_index in range(1, hypothesis_index + 1):
            #     Ddt_modelhk[particle_index][0] = np.outer(Integrated_psf_x[particle_index], Integrated_psf_y[particle_index])
            #     Ddt_modelhk[particle_index][1] = theta[particle_index][0] * np.outer(Integrated_psf_y[particle_index], Ddt_integrated_psf_1d_x[particle_index])
            #     Ddt_modelhk[particle_index][2] = theta[particle_index][0] * np.outer(Ddt_integrated_psf_1d_y[particle_index], Integrated_psf_x[particle_index])
            # assert fisher_mat.shape == (n_hk_params, n_hk_params)
            # fisher_mat[0,0] = np.sum(1 / Modelhk_at_xxyy) # this is np.sum(ddt_modelhk_at_xxyy[0][0] / Modelhk_at_xx_yy), and we know that ddt_modelhk_at_xxyy[0][0] == 1.0, becuase differentiation w.r.t. background intensity is 1.0.
            # for p_index1 in range(1, hypothesis_index + 1):
            #     for param_type1 in range(n_hk_params_per_particle):
            #         for p_index2 in range(1, hypothesis_index + 1):
            #             for param_type2 in range(n_hk_params_per_particle):
            #                 element = np.sum(1 / Modelhk * Ddt_modelhk[p_index1][param_type1] * Ddt_modelhk[p_index2][param_type2])
            #                 row = (p_index1 - 1) * n_hk_params_per_particle + param_type1 + 1
            #                 col = (p_index2 - 1) * n_hk_params_per_particle + param_type2 + 1
            #                 if element == 0 and row == col:
            #                     print(f"element is zero at row == col, where p_index1: {p_index1}, param_type1: {param_type1}, p_index2: {p_index2}, param_type2: {param_type2}")
            #                     pass
            #                 fisher_mat[row, col] = element
            #                 fisher_mat[col, row] = element

            for xx in range(szy):
                for yy in range(szy):
                    # pixel_val = roi_image[yy, xx]
                    # Initialize the first derivatives (for calulculating FIM, there is no need for second derivatives)
                    ddt_modelhk_at_xxyy = np.zeros((hypothesis_index + 1, n_hk_params_per_particle))
                    modelhk_at_xxyy, integrated_psf_x, integrated_psf_y = calculate_modelxy_ipsfx_ipsfy(theta, xx, yy, hypothesis_index, min_model_xy, psf_sigma)
                        
                    # Now, let's calculate the derivatives 
                    # -- Below are special treatmenst for the [0]'s index (the background intensity)
                    # Derivtive w.r.t background (the first index [0] refers to background and the last index [0] the intensity)
                    ddt_modelhk_at_xxyy[0][0] = 1.0
                    # Below are set as nan, as they are not used. (background does not have x and y coordinates)
                    ddt_modelhk_at_xxyy[0][1] = ddt_modelhk_at_xxyy[0][2] = np.nan

                    try:
                        # -- Below are special treatment for the [1]'s index and beyond (related to the particle intensities and coordinates)
                        for particle_index in range(1, hypothesis_index + 1):
                            # model = background + (i_0 * psf_x_0 * psf_y_0) + (i_1 * psf_x_1 * psf_y_1) + ...
                            # Calculate derivatives w.r.t particle[particle_index]'s intensity
                            ddt_modelhk_at_xxyy[particle_index][0] = integrated_psf_x[particle_index] * integrated_psf_y[particle_index]
                            # Calculate derivatives w.r.t particle[particle_index]'s x coordinate
                            ddt_modelhk_at_xxyy[particle_index][1] = ddt_integrated_psf_1d(xx, theta[particle_index][1], psf_sigma) * theta[particle_index][0] * integrated_psf_y[particle_index]
                            # Calculate derivatives w.r.t particle[particle_index]'s y coordinate
                            ddt_modelhk_at_xxyy[particle_index][2] = ddt_integrated_psf_1d(yy, theta[particle_index][2], psf_sigma) * theta[particle_index][0] * integrated_psf_x[particle_index]
                    except Exception as e:
                        print(f"Error occurred during the calculation of derivatives inside gmlr(): {e}")
                        
                    # Calculate the Fisher Information Matrix (FIM) under Hk.
                    assert fisher_mat_original.shape == (n_hk_params, n_hk_params)

                    # Building the Fisher Information Matrix regarding Hk.
                    # - Calculation with regards to the background 
                    fisher_mat_original[0, 0] += ddt_modelhk_at_xxyy[0][0] ** 2 / modelhk_at_xxyy * 1 * 1 ####################### 

                    # - Calculation with regards to the background and the particles
                    for kk in range(1, hypothesis_index * n_hk_params_per_particle + 1):
                        # convert kk to particle_index and param_type
                        particle_index = (kk - 1) // n_hk_params_per_particle + 1
                        param_type = (kk - 1) % n_hk_params_per_particle
                        # Using Poisson pdf for likelihood function, the following formula is derived. (Ref: Smith et al. 2010, nmeth, SI eq (9)).
                        fisher_mat_original[0, kk] += ddt_modelhk_at_xxyy[0][0] * ddt_modelhk_at_xxyy[particle_index][param_type] / modelhk_at_xxyy  * 1 * 1
                        fisher_mat_original[kk, 0] = fisher_mat_original[0, kk] # The FIM is symmetric.

                    # - Calculation with regards to the particles
                    for kk in range(1, hypothesis_index * n_hk_params_per_particle + 1):
                        # convert kk to particle_index and param_type
                        particle_index_kk = (kk - 1) // n_hk_params_per_particle + 1
                        param_type_kk = (kk - 1) % n_hk_params_per_particle
                        for ll in range(kk, hypothesis_index * n_hk_params_per_particle + 1):
                            # convert kk to particle_index and param_type
                            particle_index_ll = (ll - 1) // n_hk_params_per_particle  + 1
                            param_type_ll = (ll - 1) % n_hk_params_per_particle
                            fisher_mat_original[kk, ll] += ddt_modelhk_at_xxyy[particle_index_kk][param_type_kk] * ddt_modelhk_at_xxyy[particle_index_ll][param_type_ll] / modelhk_at_xxyy * 1 * 1 #######################
                            fisher_mat_original[ll, kk] = fisher_mat_original[kk, ll] # The FIM is symmetric.

            # Check diagonal zeros
            zero_diag_count = np.sum(np.diag(fisher_mat_original) == 0) 
            if zero_diag_count > 0:
                print(" ******************* Warning: Diagonal zeros detected in the FIM. ******************* ")
                print(f"fisher_mat_original: \n{fisher_mat_original}")
                # print(f"fisher_mat: \n{fisher_mat}")
                fisher_mat = fisher_mat_original.copy()
                        
            # print(f"fisher_mat_original: \n{fisher_mat_original}")
            # Compare fisher_mat with fisher_mat_original
            # visualize_fim = True
            # if visualize_fim:
            #     cmap = plt.cm.viridis
            #     cmap.set_bad(color='red')

            #     masked_fisher_mat_original = np.ma.masked_where(fisher_mat_original == 0, fisher_mat_original)

            #     fig, axs = plt.subplots(1,1, figsize=(5,4))
            #     ax = axs
            #     cax2= ax.imshow(masked_fisher_mat_original, cmap=cmap, interpolation='none')
            #     ax.set_title(f'Original Fisher Information Matrix (fisher_mat_original)\n{non_zero_diag_count=}')
            #     ax.set_xlabel('param_idx1')
            #     ax.set_ylabel('param_idx2')
            #     fig.colorbar(cax2, ax=ax, label='Value')
            #     plt.savefig(f'figure_hypothesis_{hypothesis_index}.png')
            #     pass
            #     plt.close(fig)
            #     if hypothesis_index == 5:
            #         plt.subplots()
            #         plt.imshow(roi_image)
            #         pass
                    
                
        weighted_fisher_mat = fisher_mat_original.copy()
        if hypothesis_index == 0:
            weighted_fisher_mat *= roi_max**2
            # weighted_fisher_mat *= theta**2
        else:
            # norm_norm_fisher_mat = np.zeros(norm_fisher_mat.shape)
            scales = np.array([(roi_max - roi_min) * 2 * np.pi * psf_sigma**2, szx, szy]) # 0: particle_intensity, 1: x-coordinate, 2: y-coordinate
            for row in range(weighted_fisher_mat.shape[0]):
                if row == 0:
                    weighted_fisher_mat[row,:] *= roi_max
                    # weighted_fisher_mat[row,:] *= theta[0][0]
                else:
                    # particle_index = (row - 1) // n_hk_params_per_particle
                    param_type = (row - 1) % n_hk_params_per_particle
                    # weighted_fisher_mat[row,:] *= theta[particle_index][param_type]
                    weighted_fisher_mat[row,:] *= scales[param_type]
            for col in range(weighted_fisher_mat.shape[1]):
                if col == 0:
                    weighted_fisher_mat[:,col] *= roi_max
                    # weighted_fisher_mat[:,col] *= theta[0][0]
                else:
                    # particle_index = (col - 1) // n_hk_params_per_particle
                    param_type = (col - 1) % n_hk_params_per_particle
                    # weighted_fisher_mat[row,:] *= theta[particle_index][param_type]
                    weighted_fisher_mat[:,col] *= scales[param_type]

            # Compare fisher_mat with fisher_mat_original
        visualize_fim = False 
        if visualize_fim:
            cmap = plt.cm.viridis
            cmap.set_bad(color='red')

            masked_fisher_mat_original = np.ma.masked_where(fisher_mat_original == 0, fisher_mat_original)
            masked_fisher_mat_weighted = np.ma.masked_where(weighted_fisher_mat == 0, weighted_fisher_mat)

            fig, axs = plt.subplots(1,2, figsize=(10,4))
            ax = axs[0]
            cax= ax.imshow(masked_fisher_mat_original, cmap=cmap, interpolation='none')
            if hypothesis_index == 0:
                ax.text(0, 0, f'{fisher_mat_original[0,0]:.5f}', ha='center', va='center', color='white')
            non_zero_diag_count = np.sum(np.diag(fisher_mat_original) == 0) 
            ax.set_title(f'Fisher Information Matrix\n{non_zero_diag_count=}')
            ax.set_xlabel('param_idx1')
            ax.set_ylabel('param_idx2')
            fig.colorbar(cax, ax=ax, label='Value')

            ax = axs[1]
            cax = ax.imshow(masked_fisher_mat_weighted, cmap=cmap, interpolation='none')
            if hypothesis_index == 0:
                ax.text(0, 0, f'{weighted_fisher_mat[0,0]:.5f}', ha='center', va='center', color='white')
            non_zero_diag_count = np.sum(np.diag(weighted_fisher_mat) == 0) 
            ax.set_title(f'Weighted Fisher Information Matrix\n{non_zero_diag_count=}')
            ax.set_xlabel('param_idx1')
            ax.set_ylabel('param_idx2')
            fig.colorbar(cax, ax=ax, label='Value')
            
            plt.savefig(f'normal vs weighted FIM hypothesis_{hypothesis_index}.png')
            pass
            plt.close(fig)
            if hypothesis_index == 5:
                plt.subplots()
                plt.imshow(roi_image)
                pass

        # visualize_fim = False
        # if visualize_fim:
        #     cmap = plt.cm.viridis
        #     cmap.set_bad(color='red')
        #     masked_weighted_fisher_mat = np.ma.masked_where(weighted_fisher_mat== 0, weighted_fisher_mat)
        #     fig, axs = plt.subplots(1,1, figsize=(5,4))
        #     ax = axs
        #     cax2= ax.imshow(masked_weighted_fisher_mat, cmap=cmap, interpolation='none')
        #     non_zero_diag_count = np.sum(np.diag(weighted_fisher_mat) == 0) 
        #     ax.set_title(f'Weighted Fisher Information Matrix\n{non_zero_diag_count=}')
        #     if hypothesis_index == 0:
        #         ax.text(0, 0, f'{weighted_fisher_mat[0,0]:.5f}', ha='center', va='center', color='white')
        #     ax.set_xlabel('param_idx1')
        #     ax.set_ylabel('param_idx2')
        #     fig.colorbar(cax2, ax=ax, label='Value')
        #     plt.savefig(f'weighted_figure_hypothesis_{hypothesis_index}.png')
        #     plt.close(fig)
        fisher_mat = weighted_fisher_mat.copy()

        # Now I got the FIM under Hk. Let's use this to calculate the Xi_k (GMLR criterion)
        # Xi[k] = log(likelihood(data; MLE params under Hk)) - 1/2 * log(det(FIM under Hk))

        # -- Let's calculate the first term of the Xi_k (GMLR criterion)
        # sum_loglikelihood is the sum of loglikelihoods of all pixels
        sum_loglikelihood = 0.0
        Modelhk_at_xxyy, _, _ = calculate_modelxy_ipsfx_ipsfy(theta, range(szx), range(szy), hypothesis_index, min_model_xy, psf_sigma)
        sum_loglikelihood = np.sum(roi_image * np.log(np.maximum(Modelhk_at_xxyy, 1e-2)) - Modelhk_at_xxyy - gammaln(roi_image + 1))
        
        # Before vectorization
        # for yy in range(szy):
        #     for xx in range(szx):
        #         # Let's get the actual pixel value
        #         pixel_val = roi_image[yy, xx]
        #         modelhk_at_xxyy, _, _ = calculate_modelxy_ipsfx_ipsfy(theta, xx, yy, hypothesis_index, min_model_xy, psf_sigma)
        #         # We now have the model value at (xx, yy) under Hk. Let's calculate the loglikelihood.
        #         loglikelihood = pixel_val * np.log(max(modelhk_at_xxyy, 1e-2)) - modelhk_at_xxyy - gammaln(pixel_val + 1) 
        #         sum_loglikelihood += loglikelihood
        
        # Let's calculate the second term of the Xi_k (GMLR criterion), which is -1/2 * log(det(FIM under Hk))
        if hypothesis_index == 0:
            # weighted_fisher_mat = fisher_mat * roi_max**2
            weighted_fisher_mat = fisher_mat * theta**2
            fisher_mat = weighted_fisher_mat.copy()

        _, log_det_fisher_mat = np.linalg.slogdet(fisher_mat)

        prev_xi_assigned = False
        if len(xi) > 0:
            prev_xi = xi[-1]
            prev_xi_assigned = True

        if hypothesis_index != 0 and zero_diag_count > 0:
            penalty += [1e10]
        else:
            penalty += [0.5 * log_det_fisher_mat]
        if np.isinf(penalty[-1]):
            penalty[-1] = np.nan
        lli += [sum_loglikelihood]
        if np.isinf(lli[-1]):
            lli[-1] = np.nan
        if penalty[hypothesis_index] < 0 or hypothesis_index == 0:
            xi += [lli[-1]]
        else:
            xi += [lli[-1] - penalty[-1]]
            # print(f'Warning: penalty < 0. {hypothesis_index=} assigning "xi = lli", instead of "xi = lli - penalty".')
            # break

        if prev_xi_assigned and prev_xi > xi[-1]:
            xi_drop_count += 1
            # print(f'xi_drop_count: {xi_drop_count}')
         
        if use_exit_condi and xi_drop_count >= 2:
            print('drop count >= 2. No higher order hypothesis will be tested for this image.                                        ')
            break

        fisher_info.append(fisher_mat)

    # Store xi, lli and penalty to test_metric
    test_metrics = {
        'xi': xi,
        'lli': lli,
        'penalty': penalty,
        'fisher_info': fisher_info,
    }

    display_xi_graph = False 
    if display_xi_graph:
        max_xi_index = np.nanargmax(xi)
        _, axs = plt.subplots(3,1, figsize=(4.2,3.9))
        ax = axs[0]
        hypothesis_list_length = len(xi)
        ax.plot(range(hypothesis_list_length), xi, 'o-', color='purple')              
        ax.set_ylabel('xi\n(logL- penalty)')
        ax.axvline(x=max_xi_index, color='gray', linestyle='--')
        ax = axs[1]
        ax.plot(range(hypothesis_list_length), lli, 'o-', color='navy')
        ax.set_ylabel('loglikelihood')
        ax = axs[2]
        ax.axhline(y=0, color='black', linestyle='--')
        # ax.plot(range(last_h_index + 1), np.exp(penalty * 2), 'o-', color='crimson') 
        ax.plot(range(hypothesis_list_length), penalty, 'o-', color='crimson') 
        ax.set_ylabel('penalty')
        ax.set_xlabel('hypothesis_index')
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(f'scores.png')
        pass

    # Determine the most likely hypothesis
    if np.any(np.isnan(xi)):
        pass
    estimated_num_particles = np.nanargmax(xi)
    # estimated_num_particles = np.argmax(xi)
    # input("End of a single image test - Press any key to continue...")
    if plt.get_fignums():
        plt.close('all')

    return estimated_num_particles, fit_results, test_metrics 

    

def generalized_maximum_likelihood_rule_on_rgb(roi_image, psf_sigma, last_h_index=5, random_seed=0, display_fit_results=False, display_xi_graph=False, use_exit_condi=True):
    """ Perform the generalized maximum likelihood rule on the RGB image.
    Args:
        roi_image (np.array): The 3D RGB image to process.
        psf_sigma (float): The standard deviation of the PSF.
        last_h_index (int): The last hypothesis index to test.
        random_seed (int): The random seed to use.
        display_fit_results (bool): Whether to display the fit results.
        display_xi_graph (bool): Whether to display the xi graph.
        use_exit_condi (bool): Whether to use the exit condition.
    Returns:
        dict: The fit results.
    """
    # Set the random seed
    np.random.seed(random_seed)

    # Check the input image (C, H, W)
    assert roi_image.ndim == 3
    if roi_image.shape[0] != 3:
        roi_image = np.transpose(roi_image, (1, 2, 0))
    _, szy, szx = roi_image.shape
    """ Indexing rules
    - hypothesis_index: 0, 1, 2, ...   (H0, H1, H2, ...)
    - particle_index: 1, 2, 3, ...     (particle 1, particle 2, particle 3, ...)
    - param_type_index: 0, 1, 2        (intensity, x-coordinate, y-coordinate)
    """ 
    # x, y, BG_r, BG_g, BG_b, (I_r, I_g, I_b) * n_particles, 
    # num of parameters for hypothesis n:
    # Hn: 3 (BG_r, BG_g, BG_b) + 2 (x, y) * n + 3 (I_r, I_g, I_b) * n 
    #   = 3 + 5n

    # Find tentative peaks
    roi_image_grayscale = np.mean(roi_image, axis=0)
    tentative_peaks = get_tentative_peaks(roi_image_grayscale, min_distance=1)
    rough_peaks_xy = [peak[::-1] for peak in tentative_peaks]

    # Set the minimum model at any x, y coordinate to avoid dividing by zero.
    min_model_xy = 1e-2
    # Set the method to use with scipy.optimize.minimize for the MLE estimation.
    method = 'trust-exact'

    # Initialize test scores
    xi = [] # Which will be lli - penalty
    lli = [] # log likelihood
    penalty = [] # penalty term

    fisher_info = [] # Fisher Information Matrix

    # get minimum and maximum values of each R, G, B channel of the image.
    roi_max_rgb = [np.max(roi_image[i]) for i in range(3)]
    roi_min_rgb = [np.min(roi_image[i]) for i in range(3)]

    # Initialize the fit results
    fit_results = [] 
    
    # Figure showing parameter estimation results for all tested hypotheses.
    if display_fit_results:
        _, ax_main = plt.subplots(2, last_h_index + 1, figsize=(2 * (last_h_index + 1), 4))
        # Create a colormap instance
        cmap = plt.get_cmap('turbo')# Create a colormap instance for tentative peak coordinates presentation.
        
        # As an exception to the other axes, use ax_main[1][0] to show tentative peak locations, since there's is not much to show for a simple background estimation.
        for i, coord in enumerate(rough_peaks_xy):
            x, y = coord # Check whether this is correct.
            color = cmap(i / len(rough_peaks_xy))  # Use turbo colormap
            ax_main[1][0].text(x, y, f'{i}', fontsize=6, color=color) 
        ax_main[1][0].set_xlim(0-.5, szx-.5)
        ax_main[1][0].set_ylim(szy-.5, 0-.5) 
        ax_main[1][0].set_aspect('equal')
        ax_main[1][0].set_title('Tentative Peak Coordinates', fontsize=8)
        if roi_image.shape[0] == 3:
            ax_main[0][0].imshow(np.transpose(roi_image, (1, 2, 0)))
        else:
            ax_main[0][0].imshow(roi_image)
        plt.show(block=False)

    # Theta will be a dict
    # bg -> r, g, b
    # particles -> x, y, I_r, I_g, I_b

    for hypothesis_index in range(last_h_index + 1): # hypothesis_index is also the number of particles. 

        # Initialization
        # n_hk_params = 3 + 5 * hypothesis_index # Number of parameters for each hypothesis
        
        # Initialize the theta (parameter) dict
        theta = {}

        # Starting values
        if hypothesis_index == 0:
            theta['bg'] = [channel_img.sum() / szx / szy for channel_img in roi_image]
        else: 
            # Initialize estimated particle_intensities
            theta['bg'] = [np.min(channel_img) for channel_img in roi_image]

            # Initialization for all particles
            try:
                theta['particles'] = []
                for particle_index in range(1, hypothesis_index + 1): # Note that the particle index starts from 1, not 0. 

                    # Initialize estimated particle intensities to the maximum value of the Gaussian roi image.
                    intensity_rgb = [(roi_max_rgb[i] - roi_min_rgb[i]) * 2 * np.pi * psf_sigma**2 for i in range(3)]

                    if len(rough_peaks_xy) <= 0:
                        print('No tentative peaks found.')
                        break
                    if particle_index <= len(rough_peaks_xy):
                        rough_x = rough_peaks_xy[particle_index - 1][0]
                        rough_y = rough_peaks_xy[particle_index - 1][1]
                    else:
                        # assign random positions. 
                        rough_x = random.random() * (szx - 1)
                        rough_y = random.random() * (szy - 1)

                    theta['particles'] += [{'I':intensity_rgb, 'x':rough_x, 'y':rough_y}] # Initialize the particle dict
                    # theta['particles'][0]['I'] = [(roi_max_rgb[i] - roi_min_rgb[i]) * 2 * np.pi * psf_sigma**2 for i in range(3)]
                    # theta['particles'][particle_index]['I'] = [(roi_max_rgb[i] - roi_min_rgb[i]) * 2 * np.pi * psf_sigma**2 for i in range(3)]

            except Exception as e:
                print(f"Error occurred during initialization of theta inside gmlr_rgb(): {e}")
                print(f"theta: {theta}")

        # Only do the MLE if hypothesis_index > 0
        if hypothesis_index == 0:
            # assert n_hk_params == 3
            convergence_list = [True] # For H0, convergence is always True, as it is simple averaging.
        else:
            # Normazlize the parameters before passing on to neg_loglikelihood_function
            norm_flat_theta = normalize(theta, hypothesis_index, roi_min_rgb, roi_max_rgb, psf_sigma, szx, szy)

            # # Initialize storage for snapshots
            gradientnorm_snapshots = []
            fn_snapshots = []
            theta_snapshots = []
            denormflat_theta_snapshots = []

            def callback_fn(xk, *args):
                jac = jacobian_fn(xk, hypothesis_index, roi_image, roi_min_rgb, roi_max_rgb, min_model_xy, psf_sigma, szx, szy)
                gradientnorm = np.linalg.norm(jac)
                fn = modified_neg_loglikelihood_fn(xk, hypothesis_index, roi_image, roi_min_rgb, roi_max_rgb, min_model_xy, psf_sigma, szx, szy)
                gradientnorm_snapshots.append(gradientnorm)
                fn_snapshots.append(fn)                
                theta_snapshots.append(xk)
                # denormflat_theta_snapshots.append(denormalize(xk, hypothesis_index, roi_min_rgb, roi_max_rgb, psf_sigma, szx, szy).flatten())

            # # Define the callback function as a nested function
            # def callback_fn(xk, *args):
            #     jac = jacobian_fn(xk, hypothesis_index, roi_image, roi_min_rgb, roi_max_rgb, min_model_xy, psf_sigma, szx, szy)
            #     gradientnorm = np.linalg.norm(jac)
            #     fn = modified_neg_loglikelihood_fn(xk, hypothesis_index, roi_image, roi_min, roi_max, min_model_xy, psf_sigma, szx, szy)
            #     gradientnorm_snapshots.append(gradientnorm)
            #     fn_snapshots.append(fn)                
            #     theta_snapshots.append(xk)
            #     denormflat_theta_snapshots.append(denormalize(xk, hypothesis_index, roi_min, roi_max, psf_sigma, szx, szy).flatten())

            # Now, let's update the parameters using scipy.optimize.minimize
            try:
                minimization_result = minimize(modified_neg_loglikelihood_fn, norm_flat_theta, args=(hypothesis_index, roi_image, roi_min_rgb, roi_max_rgb, min_model_xy, psf_sigma, szx, szy),
                                method=method, jac=jacobian_fn, hess=hessian_fn, callback=callback_fn, options={'gtol': 100})
            except Exception as e:
                print(f"Error occurred during optimization: {e}")
                # print("Here is the last (denorm) theta snapshot:")
                # print(denormalize(theta_snapshots[-1], hypothesis_index, roi_min, roi_max, psf_sigma, szx, szy))

                

            # print(f'H{hypothesis_index} converged?: {result.success}')
            # print(f'Last gradientnorm: {gradientnorm_snapshots[-1]:.0f}')
            snapshot_length = len(fn_snapshots)
            convergence = minimization_result.success
            norm_theta = minimization_result.x

            convergence_list.append(convergence)

            # Retrieve the estimated parameters.
            theta = denormalize(norm_theta, hypothesis_index, roi_min_rgb, roi_max_rgb, psf_sigma, szx, szy)           
                            
        # Store fit results
        if hypothesis_index == 0:
            current_hypothesis_fit_result = {
                'hypothesis_index': hypothesis_index,
                'theta': theta,
                'convergence': True,
            }
        else:
            current_hypothesis_fit_result = {
                'hypothesis_index': hypothesis_index,
                'theta': theta,
                'convergence': convergence,
            }
        # Append the fit result to fit_results
        fit_results.append(current_hypothesis_fit_result)

        if display_fit_results and hypothesis_index > 0:
            bg_values = f"R:{theta['bg'][0]:.1f}, G:{theta['bg'][1]:.1f}, B:{theta['bg'][2]:.1f}"
            ax_main[0][0].set_title(f'Bg = {bg_values}', fontsize=8)
            for particle_index in range(1, hypothesis_index + 1):
                if roi_image.shape[0] == 3:
                    ax_main[0][hypothesis_index].imshow(np.transpose(roi_image, (1, 2, 0)))
                else:
                    ax_main[0][hypothesis_index].imshow(roi_image)

                red = random.randint(200, 255)
                green = random.randint(0, 100)
                blue = random.randint(0, 50)
                color_code = '#%02X%02X%02X' % (red, green, blue)

                if roi_image.ndim == 3:
                    # ax_main[0][hypothesis_index].set_title(f"H{hypothesis_index} - convgd: {convergence_list[hypothesis_index]}\nbg: {theta['bg']:.1f}", fontsize=8)
                    bg_values = ", ".join([f"{value:.1f}" for value in theta['bg']])
                    ax_main[0][hypothesis_index].set_title(f'Bg: {bg_values}', fontsize=8)
                    ax_main[0][hypothesis_index].scatter(theta['particle'][particle_index - 1]['x'], theta['particle'][particle_index-1]['y'], s=10, color=color_code, marker='x')
                    ax_main[0][hypothesis_index].text(theta['particle'][particle_index - 1]['x'] + np.random.rand() * 1.5,       theta['particle'][particle_index - 1]['y'] + (np.random.rand() - 0.5) * 4,
                                                f"R:{theta['particle'][particle_index - 1]['I'][0]:.1f}\nG:{theta['particle'][particle_index - 1]['I'][1]:.1f}\nB:{theta['particle'][particle_index - 1]['I'][2]:.1f}", color='white', fontsize=6,) 
                else:

                    ax_main[0][hypothesis_index].set_title(f"H{hypothesis_index} - convgd: {convergence_list[hypothesis_index]}\nbg: {theta[0][0]:.1f}", fontsize=8)
                    ax_main[0][hypothesis_index].scatter(theta[particle_index][1], theta[particle_index][2], s=10, color=color_code, marker='x')
                    ax_main[0][hypothesis_index].text(theta[particle_index][1] + np.random.rand() * 1.5, theta[particle_index][2] + (np.random.rand() - 0.5) * 4,
                                                f'  {theta[particle_index][0]:.1f}', color=color_code, fontsize=10,) 
            ax_main[1][hypothesis_index].set_title(f'Gradient norm\nFinal func val: {fn_snapshots[-1]:.04e}', fontsize=8)
            ax_main[1][hypothesis_index].plot(np.arange(snapshot_length), gradientnorm_snapshots, '-o', color='black', markersize=2, label='Gradient norm')
            ax_main[1][hypothesis_index].set_ylim(bottom=0)
            plt.tight_layout()
            plt.show(block=False)
            pass

    return fit_results

# test_norm_flat_theta = normalize(test_theta, h_index, roi_min, roi_max, 1, 10, 10)
# jacobian_fn(test_norm_flat_theta, h_index, np.ones((3, 10, 10)), roi_min, roi_max, 1e-2, 1, 10, 10)
# hessian_fn(test_norm_flat_theta, h_index, np.ones((3, 10, 10)), roi_min, roi_max, 1e-2, 1, 10, 10) 

# test_gray_theta = np.array([[500, np.nan, np.nan], [1000, 5.6, 7.2], [1200, 3.2, 5], [900, 2, 4]])
# test_gray_theta_norm = normalize(test_gray_theta, h_index, 10, 10000, 1, 10, 10)
