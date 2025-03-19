import warnings
import seaborn as sns
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import erf, gammaln
from scipy.stats import norm
from skimage.feature import peak_local_max
import diplib as dip

# Define penalty factor constants for the out-of-bounds particles.
POSITION_PENALTY_FACTOR = 1e5


# Print numpy arrays with 3 decimal points
np.set_printoptions(precision=4, formatter={'float': '{:0.6f}'.format}, linewidth=np.inf)


def normalize(th, hypothesis_index, roi_max, szx, szy, alpha, color_mode='grayscale'):
    """ Normalize the theta values for the optimization process.
    Args:
        th (list): Short for theta. The list of particle parameters, un-normalized and fully structured.
        hypothesis_index (int): The index of the hypothesis being tested. (it matches the number of particles each hypothesis assumes)
        roi_max (float): The maximum value of the region of interest.
        psf_sigma (float): The standard deviation of the point spread function.
        szx (int): The size of the x-axis.
        szy (int): The size of the y-axis.
        alpha (float or list): The normalization factor for the particle intensity. If the image is grayscale, alpha should be a float. If the image is RGB, alpha should be a list of 3 floats.
    Returns:
        ndarray: The normalized and flattened theta values.
    """

    if color_mode == 'grayscale': # If the image is a grayscale image, then roi_max and roi_min should be a single value.
        # Note: alpha == (roi_max - roi_min) * 2 * np.pi * psf_sigma**2

        if hypothesis_index == 0: 
            return th / roi_max

        else:
            # First, normalize theta into n_th (normalized theta) 
            n_th = np.zeros((hypothesis_index + 1, 3)) # 3 is for intensity, x, and y
            n_th[0][0] = th[0][0] / roi_max # background level
            n_th[0][1] = np.nan # x-coordinate for background does not exist
            n_th[0][2] = np.nan # y-coordinate for background does not exist

            # Normalize the particle intensity and position values by dividing them by the corresponding normalization factors
            for particle_index in range(1, hypothesis_index + 1):
                n_th[particle_index][0] = th[particle_index][0] / alpha 
                n_th[particle_index][1] = th[particle_index][1] / szx
                n_th[particle_index][2] = th[particle_index][2] / szy
                
            # Modify theta into the format that can be passed on to scipy.optimize.minimize
            nf_th = n_th.flatten() # nf_th (normalized and flattened theta)
            nft_th = nf_th[~np.isnan(nf_th)] # ntf_th (normalized, flattened, and trimmed theta)

            return nft_th # Return the normalized, flattened, and trimmed theta values

    elif color_mode == 'rgb': # If the image is an RGB image, then roi_max and roi_min should be a list of 3 values.
        # Note: alpha = [(roi_max[ch] - roi_min[ch]) * 2 * np.pi * psf_sigma**2 for ch in range(3)] 
        # Note: Format of nf_th: [Bg_r, Bg_g, Bg_b, i_r1, i_g1, i_b1, x1, y1, ...] - There is nothing to trim when dealing with RGB images due to the way the theta values are structured.

        if hypothesis_index == 0:
            return [th['bg'][ch] / roi_max[ch] for ch in range(3)]

        else:

            nf_th = [] # normalized and flattened theta

            # Append the background values (rgb)
            nf_th.append([th['bg'][ch] / roi_max[ch] for ch in range(3)])

            # Append the particle intensity (rgb) and position values 
            for particle_index in range(hypothesis_index):
                nf_th.append([th['particles'][particle_index]['I'][ch] / alpha[ch] for ch in range(3)])
                nf_th.append([th['particles'][particle_index]['x'] / szx, th['particles'][particle_index]['y'] /szy])
            
            # Flatten the list of lists
            nf_th = np.array([item for sublist in nf_th for item in sublist])
            
            return nf_th # Return the normalized, flattened (nothing to trim) theta values

    else: # If the image is neither grayscale nor RGB, raise an error
        raise ValueError("Error: color_mode should be either 'grayscale' or 'rgb'. Check required. location: normalize()")

def denormalize(norm_flat_trimmed_theta, hypothesis_index, roi_max, szx, szy, alpha, color_mode='grayscale'):
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

    if color_mode == 'grayscale': # If the image is a grayscale image, then roi_max and roi_min should be a single value.

        if hypothesis_index == 0:
            return np.array([[norm_flat_trimmed_theta[0] * roi_max, np.nan, np.nan]])

        else:
            # Note: alpha = (roi_max - roi_min) * 2 * np.pi * psf_sigma**2

            num_parameters_per_particle = 3 # Number of parameters per particle (intensity, x, y)

            nan_padded_nft_theta = np.insert(norm_flat_trimmed_theta, [1, 1], np.nan) # Insert nan values to the 1st and 2nd indices of the normalized and flattened theta values
            structured_norm_theta_gray = np.reshape(nan_padded_nft_theta, (-1, num_parameters_per_particle)) # Reshape the normalized and flattened theta values into a 2D array with 3 columns

            # Initialize the theta array
            theta = np.zeros((hypothesis_index + 1, num_parameters_per_particle))

            # Assign the background terms
            theta[0][0] = structured_norm_theta_gray[0][0] * roi_max
            theta[0][1] = theta[0][2] = np.nan

            # Assign the particle terms (by denormalizing the intensity and position values)
            for particle_index in range(1, hypothesis_index + 1):
                theta[particle_index][0] = structured_norm_theta_gray[particle_index][0] * alpha
                theta[particle_index][1] = structured_norm_theta_gray[particle_index][1] * szx
                theta[particle_index][2] = structured_norm_theta_gray[particle_index][2] * szy

            return theta
        
    elif color_mode == 'rgb': # If the image is an RGB image, then roi_max and roi_min should be a list of 3 values.
        # Note: alpha = [(roi_max[ch] - roi_min[ch]) * 2 * np.pi * psf_sigma**2 for ch in range(3)]

        if hypothesis_index == 0:
            return {'bg': [norm_flat_trimmed_theta[ch] * roi_max[ch] for ch in range(3)], 'particle': []}

        else:

            num_parameters_per_particle = 5 # Number of parameters per particle (intensity_r, intensity_g, intensity_b, x, y)
            num_parameters_for_background = 3
            rgb_slice_length = 3
            xy_slice_length = 2
            
            # Format of norm_flat_trimmed_theta: [Bg_r, Bg_g, Bg_b, i_r1, i_g1, i_b1, x1, y1, ...]
            theta = {'bg': [0, 0, 0], 'particle': [{} for _ in range(hypothesis_index)]}

            # First, denormalize the background values (rgb) and save it to the structured_theta dictionary
            bg_rgb = norm_flat_trimmed_theta[:num_parameters_for_background]

            # Denormalize the background values (rgb)
            theta['bg'] = [norm_bg_ch * roi_max[ch] for ch, norm_bg_ch in enumerate(bg_rgb)]

            # Denormalize the particle intensity (rgb) and position values and save it to the structured_theta dictionary
            for particle_index in range(1, hypothesis_index + 1):
                # Example: Accessing a specific parameter: theta['particle'][particle_index]['x']

                # Extract the normalized particle intensity (rgb) values
                index_of_r_intensity_of_current_particle = num_parameters_for_background + num_parameters_per_particle * (particle_index - 1) # 3, 8, 13... are the indices of the first element (R) of each particle's intensity values
                normalized_rgb_slice_of_current_particle = norm_flat_trimmed_theta[index_of_r_intensity_of_current_particle:(index_of_r_intensity_of_current_particle + rgb_slice_length)]

                # Extract the normalized x and y position values
                index_of_x_position_of_current_particle = 6 + 5 * (particle_index - 1) # 6, 11, 16... are the indices of the first element of each particle's position values
                norm_position = norm_flat_trimmed_theta[index_of_x_position_of_current_particle:(index_of_x_position_of_current_particle + xy_slice_length)]

                # Denormalize the particle intensity (rgb) and position values
                theta['particle'][particle_index - 1] = {'I': [normalized_rgb_slice_of_current_particle[ch] * alpha[ch] for ch in range(rgb_slice_length)],
                                                        'x': norm_position[0] * szx, 
                                                        'y': norm_position[1] * szy}
            return theta

    else: # If the image is neither grayscale nor RGB, raise an error
        raise ValueError("Error: color_mode should be either 'grayscale' or 'rgb'. Check required. location: denormalize()")

def position_penalty_function(particle_coordinate_value, roi_width):
    """ Returns a penalty term for a particle that is out of bounds in one dimension.
    Note: To be rigorous, one has to pass in the normalized theta values to this function. 
    - However, since the doing so will only result in a simple scale difference in the return value, which can be tuned with a constant factor (POSITION_PENALTY_FACTOR), 
    - we simply pass in the unnormalized theta values to this function for simplicity.
    """
    return -POSITION_PENALTY_FACTOR * (particle_coordinate_value + .5)**3 if particle_coordinate_value < -.5 else (POSITION_PENALTY_FACTOR * (particle_coordinate_value - roi_width + .5)**3 if particle_coordinate_value >= roi_width - .5 else 0)
    
def ddt_position_penalty_function(particle_coordinate_value, roi_width):
    """ Returns the first derivative of the penalty term for a particle that is out of bounds in one dimension.
    Note: To be rigorous, one has to pass in the normalized theta values to this function. 
    - However, since the doing so will only result in a simple scale difference in the return value, which can be tuned with a constant factor (POSITION_PENALTY_FACTOR), 
    - we simply pass in the unnormalized theta values to this function for simplicity.
    """
    return -3 * POSITION_PENALTY_FACTOR * (particle_coordinate_value + .5)**2 if particle_coordinate_value < -.5 else (3 * POSITION_PENALTY_FACTOR * (particle_coordinate_value - roi_width + .5)**2 if particle_coordinate_value >= roi_width - .5 else 0)
    
def d2dt2_position_penalty_function(particle_coordinate_value, roi_width):
    """ Returns the first derivative of the penalty term for a particle that is out of bounds in one dimension.
    Note: To be rigorous, one has to pass in the normalized theta values to this function. 
    - However, since the doing so will only result in a simple scale difference in the return value, which can be tuned with a constant factor (POSITION_PENALTY_FACTOR), 
    - we simply pass in the unnormalized theta values to this function for simplicity.
    """
    return -6 * POSITION_PENALTY_FACTOR * (particle_coordinate_value + .5) if particle_coordinate_value < -.5 else (6 * POSITION_PENALTY_FACTOR * (particle_coordinate_value - roi_width + .5) if particle_coordinate_value >= roi_width - .5 else 0)


def out_of_bounds_particle_penalty(theta, szx, szy, color_mode='grayscale'):
    """ Returns a penalty for particles that are out of bounds.
    Note: To be rigorous, one has to pass in the normalized theta values to this function. 
    - However, since the doing so will only result in a simple scale difference in the return value, which can be tuned with a constant factor (POSITION_PENALTY_FACTOR), 
    - we simply pass in the unnormalized theta values to this function for simplicity.
    Args:
        theta (list): The list of particle parameters.
    Returns:
        float: The summed penalties for particles that are out of bounds.
    """
    penalty = 0

    if color_mode == 'grayscale': 
        for i in range(1, len(theta)):
            x_term = position_penalty_function(theta[i][1], szx)
            y_term = position_penalty_function(theta[i][2], szy)
            penalty += x_term + y_term
    elif color_mode == 'rgb':
        for i in range(len(theta['particle'])):
            x_term = position_penalty_function(theta['particle'][i]['x'], szx)
            y_term = position_penalty_function(theta['particle'][i]['y'], szy)
            penalty += x_term + y_term
    else:
        raise ValueError("Error: color_mode should be either 'grayscale' or 'rgb'. Check required. location: out_of_bounds_particle_penalty()")

    return penalty

def jac_oob_penalty(theta, szx, szy, alpha, color_mode='grayscale'):
    """ Returns the derivative of the out of bounds penalty."""
    num_parameters_per_particle = 3

    if color_mode == 'grayscale':
        ddt_oob = np.zeros((len(theta), num_parameters_per_particle)) # 3 is for intensity, x, and y

        # Treat the background terms: 
        # -> ddt_oob[0][0] is already zero as there is no penalty for the background. 
        # -> Set ddt_oob[0][1] and ddt_oob[0][2] to nan as they are not used.
        ddt_oob[0][1] = ddt_oob[0][2] = np.nan

        # Treat the particle terms
        for i in range(1, len(theta)):
            # -> ddt_oob[i][0]'s are already zeros as there is no penalty for the particle intensity
            ddt_oob[i][1] = ddt_position_penalty_function(theta[i][1], szx) * szx # szx is the normalization factor for the x-coordinate
            ddt_oob[i][2] = ddt_position_penalty_function(theta[i][2], szy) * szy # szy is the normalization factor for the y-coordinate

        return ddt_oob 

    elif color_mode == 'rgb':
        # REF - Format of nf_th: [Bg_r, Bg_g, Bg_b, i_r1, i_g1, i_b1, x1, y1, ...]
        # REF - ddt_nll_rgb = np.zeros((hypothesis_index + 1, 5))  # row: particle index (starts from 1 while 0 is for background), col: I_r, I_g, I_b, x, y
        ddt_oob = np.zeros((len(theta['particle'])+1, 5))

        # Treat the background terms: 
        # -> ddt_oob[0][0] is already zero as there is no penalty for the background. 
        # -> Set ddt_oob[0][1] and ddt_oob[0][2] to nan as they are not used.
        ddt_oob[0][3] = ddt_oob[0][4] = np.nan

        # Treat the particle terms
        for i in range(len(theta['particle'])):
            # -> ddt_oob[i][0:3] are already zeros as there is no penalty for the particle intensity
            ddt_oob[i][3] = ddt_intensity_penalty_function(theta['particle'][i]['x']) * szx
            ddt_oob[i][4] = ddt_intensity_penalty_function(theta['particle'][i]['y']) * szy
    
        return ddt_oob 

    else: 
        raise ValueError("Error: color_mode should be either 'grayscale' or 'rgb'. Check required. location: jac_oob_penalty()")


def hess_oob_penalty(theta, szx, szy, alpha, color_mode='grayscale'):
    """ Returns the Hessian of the out of bounds penalty."""
    if color_mode == 'grayscale':
        # Note: Format of nf_th (normalized and flattened theta): [Bg, i1, x1, y1, i2, x2, y2, ...]
        # Note: Shape of d2dt2_nll_2d (the 2d formatted double derivatives of the negative log likelihood): (hypothesis_index * 3 - 2, hypothesis_index * 3 - 2)
        
        # Initialize the Hessian matrix - Note that its side length is smaller than the length of theta, because here we do not include the [0][1] and [0][2] terms (that has no meanings and thus were assigned nan) found in the theta.
        d2dt2_oob_2d = np.zeros((len(theta)* 3 - 2, len(theta)* 3 - 2)) # Note: Only some of the diagonal terms are non-zero, because differentiating the penalty function with respect to two unrelated variables results in zero.

        # Treat the background terms (d2dt2_oob_2d[0,:], and d2dt2_oob_2b[:,0]) - They are all zeros as there is no penalty for the background
        d2dt2_oob_2d[0, :] = d2dt2_oob_2d[:, 0] = 0 

        # Treat the particle terms
        for pidx in range(1, len(theta)):
            # -> Of the diagonal terms, the ones differentiated twice with respect to the particle intensity (d2dt2_oob_2d[(pidx - 1) * 3 + 1][(pidx - 1) * 3 + 1]) are already zero as there is no penalty for the particle intensity

            # Diagonal term related to particle x-position (i1, i1), is differentiated twice with respect to the x-position
            d2dt2_oob_2d[(pidx - 1) * 3 + 2][(pidx - 1) * 3 + 2] = d2dt2_position_penalty_function(theta[pidx][1], szx) * szx**2

            # Diagonal term related to particle y-position (i2, i2), is differentiated twice with respect to the y-position
            d2dt2_oob_2d[(pidx - 1) * 3 + 3][(pidx - 1) * 3 + 3] = d2dt2_position_penalty_function(theta[pidx][2], szy) * szy**2 

        return d2dt2_oob_2d

    elif color_mode == 'rgb':
        # Note: Format of nf_th (normalized and flattened theta): [Bg_r, Bg_g, Bg_b, i_r1, i_g1, i_b1, x1, y1, ...]
        # Note: Shape of d2dt2_nll_2d (the 2d formatted double derivatives of the negative log likelihood): (hypothesis_index * 5 + 3, hypothesis_index * 5 + 3)

        # Initialize the Hessian matrix
        d2dt2_oob_2d = np.zeros((len(theta['particle']) * 5 + 3, len(theta['particle']) * 5 + 3)) # Note: Only some of the diagonal terms are non-zero, because differentiating the penalty function with respect to two unrelated variables results in zero.

        # -> Treat the background terms, R (d2dt2_oob_2d[0,:], and d2dt2_oob_2b[:,0]) - They are all zeros as there is no penalty for the background
        # -> Treat the background terms, G (d2dt2_oob_2d[1,:], and d2dt2_oob_2b[:,1]) - They are all zeros as there is no penalty for the background
        # -> Treat the background terms, B (d2dt2_oob_2d[2,:], and d2dt2_oob_2b[:,2]) - They are all zeros as there is no penalty for the background

        # Treat the particle terms
        for pidx in range(len(theta['particle'])):
            # -> Of the diagonal terms, the ones differentiated twice with respect to the particle intensity R, G, or B (d2dt2_oob_2d[(pidx - 1) * 5 + c][(pidx - 1) * 5 + c]'s, where c is 3, 4, or 5) are already zero as there is no penalty for the particle intensity
            d2dt2_oob_2d[(pidx-1)*5 + 6][(pidx-1)*5 + 6] = d2dt2_position_penalty_function(theta['particle'][pidx]['x'], szx) * szx**2
            d2dt2_oob_2d[(pidx-1)*5 + 7][(pidx-1)*5 + 7] = d2dt2_position_penalty_function(theta['particle'][pidx]['y'], szy) * szy**2

        return d2dt2_oob_2d
    
    else:
        raise ValueError("Error: color_mode should be either 'grayscale' or 'rgb'. Check required. location: hess_oob_penalty()")

def modified_neg_loglikelihood_fn(norm_flat_trimmed_theta, hypothesis_index, roi_image, roi_min, roi_max, minimum_model_value, psf_sigma, szx, szy, alpha, color_mode='grayscale'):
    """ Calculate the modified negative log-likelihood function.
    
    Args:
        norm_flat_trimmed_theta (ndarray): The normalized flattened trimmed theta.
        hypothesis_index (int): The index of the hypothesis.
        roi_image (ndarray): The region of interest image.
        roi_min (float): This value is not used, but it is kept for consistency with the other functions.
        roi_max (float): The maximum value of the region of interest.
        minimum_model_value (float): The minimum value of the model coordinates.
        psf_sigma (float): The standard deviation of the point spread function.
        szx (int): The size of the x-axis.
        szy (int): The size of the y-axis.
        
    Returns:
        float: The modified negative log-likelihood value.
    """
    # Denormalize theta to calculate roi_model
    theta = denormalize(norm_flat_trimmed_theta, hypothesis_index, roi_max, szx, szy, alpha, color_mode=color_mode)

    # Calculate the ROI model (2d array) 
    roi_model, _, _ = calculate_model_ipsfx_ipsfy(theta, np.arange(szx), np.arange(szy), hypothesis_index, minimum_model_value, psf_sigma, color_mode=color_mode)

    # Calculate the modified negative log-likelihood value (which drops the constant terms)
    modified_neg_loglikelihood = np.sum(roi_model - roi_image * np.log(roi_model))

    # Add the penalty for out-of-bounds particles
    modified_neg_loglikelihood += out_of_bounds_particle_penalty(theta, szx, szy) 

    return modified_neg_loglikelihood 


# calculate_model_ipsfx_
def calculate_model_ipsfx_ipsfy(theta, xx, yy, hypothesis_index, minimum_model_value, psf_sigma, color_mode='grayscale'):
    ''' Calculate the model intensity at a given position (xx, yy) based on the given parameters.
    Note: This function is a rare case where the theta can passed in as unnormalized without any issues.
    - Most other functions in this code should be called with normalized parameters to make the likelihood-space and the fisher information-space the same.
    - This function simply returns values of the model intensity and the integrated PSF x and y coordinates.

    Parameters:
        theta (list): Either a list/ndarray (grayscale image) or a dictionary (case: rgb image)
        A list of particle parameters. Each element in the list represents a particle and contains the following information:
                    - Particle intensity (i)
                    - PSF x-coordinate (psf_x)
                    - PSF y-coordinate (psf_y)
        xx (float or numpy array): The x-coordinates to evaluate intensity.
        yy (float or numpy array): The y-coordinates to evaluate intensity.
        hypothesis_index (int): The index of the hypothesis being tested.
        minimum_model_value (float): The minimum model intensity for (xx, yy).
        psf_sigma (float): The standard deviation of the PSF.

    Returns:
        tuple: A tuple containing the following values:
            - The model intensity at (xx, yy)
            - An array of integrated PSF x-coordinates
            - An array of integrated PSF y-coordinates
    '''

    if hypothesis_index == 0: # hypothesis_index == 0 case should not have called this function.
        raise ValueError("Error: hypothesis_index should be greater than 0 to call this function. Check required. location: calculate_model_ipsfx_ipsfy()")

    # Initialize the arrays to store the factors to be multiplied to each corresponding pixel positions in the x and y direction, coming from the point spread function (PSF) of each paritcle.
    psf_factors_for_pixels_in_x = np.zeros((hypothesis_index + 1, 1 if isinstance(xx, int) else len(xx)))
    psf_factors_for_pixels_in_y = np.zeros((hypothesis_index + 1, 1 if isinstance(yy, int) else len(yy)))

    # psf_factors_for_pixels_in_x[0] and psf_factors_for_pixels_in_y[0] are not used (as [0] corresponds to the 0-index particle, which does not exist; 0-index is for background).
    psf_factors_for_pixels_in_x[0, :] = np.nan
    psf_factors_for_pixels_in_y[0, :] = np.nan

    if color_mode == 'grayscale':

        # Initialize the model intensity at (xx, yy) to zeros
        roi_model = np.zeros((len(yy), len(xx))) 

        # Note: roi_model = background + (i_0 * psf_x_0 * psf_y_0) + (i_1 * psf_x_1 * psf_y_1) + ...
        # -> Add the background contribution to the roi model 
        roi_model += np.abs(theta[0][0]) * np.ones((len(yy), len(xx))) 

        # -> Add the contributions of each particle to the roi model
        for particle_index in range(1, hypothesis_index + 1):
            # Calculate the integral (over each pixel) of the normalized 1D psf function (i.e., factors to be multipled to each corresponding pixel positions) for x and y directions, for the xx-th column and the yy-th row, respectivly.
            psf_factors_for_pixels_in_x[particle_index, :] = normal_gaussian_integrated_within_each_pixel(xx, theta[particle_index][1], psf_sigma)
            psf_factors_for_pixels_in_y[particle_index, :] = normal_gaussian_integrated_within_each_pixel(yy, theta[particle_index][2], psf_sigma)

            # update the particles contributions to the roi_model value
            roi_model += np.abs(theta[particle_index][0]) * (np.outer(psf_factors_for_pixels_in_y[particle_index], psf_factors_for_pixels_in_x[particle_index]) )
            
            # If the model intensity is negative, set it to the minimum model intensity to ensure physicality
            roi_model[roi_model <= minimum_model_value] = minimum_model_value
    
    elif color_mode == 'rgb':

        # Initialize the model in RGB format (C x H x W)
        num_channels = 3
        roi_model = np.zeros((num_channels, len(xx), len(yy)))

        # Add the background contribution to the model intensity
        roi_model += [theta['bg'][ch] * np.ones((len(yy), len(xx))) for ch in range(num_channels)]

        # Add the contributions of the particles to the model intensity
        for particle_index in range(1, hypothesis_index + 1):
            # Calculate the integral (over each pixel) of the normalized 1D psf function (i.e., factors to be multipled to each corresponding pixel positions) for x and y directions, for the xx-th column and the yy-th row, respectivly.
            psf_factors_for_pixels_in_x = normal_gaussian_integrated_within_each_pixel(xx, theta['particle'][particle_index-1]['x'], psf_sigma)
            psf_factors_for_pixels_in_y = normal_gaussian_integrated_within_each_pixel(yy, theta['particle'][particle_index-1]['y'], psf_sigma)

            # Update the particles contributions to the model intensity with ensuring the minimum model intensity
            roi_model += [np.maximum(theta['particle'][particle_index-1]['I'][ch] * np.outer(psf_factors_for_pixels_in_y, psf_factors_for_pixels_in_x), minimum_model_value) for ch in range(num_channels)]

    else:
        print("Error: theta should be either a list/ndarray or a dictionary. Check required.")

    return roi_model, psf_factors_for_pixels_in_x, psf_factors_for_pixels_in_y


def jacobian_fn(norm_flat_trimmed_theta, hypothesis_index, roi_image, roi_min, roi_max, minimum_model_value, psf_sigma, szx, szy, alpha, color_mode='grayscale'):
    """ Calculate the Jacobian matrix for the modified negative log-likelihood function.

    Args:
        norm_flat_trimmed_theta (ndarray): The normalized flattened trimmed parameter array.
        hypothesis_index (int): The index of the hypothesis.
        roi_image (ndarray): The region of interest image.
        roi_min (float): The minimum value of the region of interest.
        roi_max (float): The maximum value of the region of interest.
        minimum_model_value (float): The minimum value of the model coordinates.
        psf_sigma (float): The standard deviation of the point spread function.
        szx (int): The size of the x-axis.
        szy (int): The size of the y-axis.

    Returns:
        ndarray: The Jacobian matrix.
    """
    # Denormalize theta to calculate roi_model
    theta = denormalize(norm_flat_trimmed_theta, hypothesis_index, roi_max, szx, szy, alpha, color_mode=color_mode)

    # Precalculate intensity and derivatives 
    # roi_model: 3 (rgb) x 2D (szy x szx) array with row: y, col: x, psf_factors_for_pixels_in_x: 1d array following x-pos, psf_factors_for_pixels_in_y: 1d array following y-pos
    roi_model, psf_factors_for_pixels_in_x, psf_factors_for_pixels_in_y = calculate_model_ipsfx_ipsfy(theta, np.arange(szx), np.arange(szy), hypothesis_index, minimum_model_value, psf_sigma, color_mode=color_mode)
    
    # derivative of the negative log-likelihood (ddt_nll) with respect to the parameters
    # Calculate the ingredients for the derivatives
    if color_mode == 'grayscale':
        # Initialize the first derivatives of the negative log-likelihood function with respect to the normalized parameters
        first_derivatives_of_negative_log_likelihood = np.zeros((hypothesis_index + 1, 3)) 
        first_derivatives_of_negative_log_likelihood[0][1] = first_derivatives_of_negative_log_likelihood[0][2] = np.nan
        # Refer to the explanation written inside first_derivatives_of_normal_gaussian_integrated_within_each_pixel function for better understanding of the below two lines.
        first_derivatives_of_psf_factors_for_pixels_in_x = np.array([first_derivatives_of_normal_gaussian_integrated_within_each_pixel(np.arange(szx), theta[p_idx][1], psf_sigma) for p_idx in range(1, hypothesis_index + 1)]) # 2d array with row: pindex, col: x-position
        first_derivatives_of_psf_factors_for_pixels_in_y = np.array([first_derivatives_of_normal_gaussian_integrated_within_each_pixel(np.arange(szy), theta[p_idx][2], psf_sigma) for p_idx in range(1, hypothesis_index + 1)]) # 2d array with row: pindex, col: y-position
    else: # Case: rgb image
        if roi_image.shape[0] != 3:
            roi_image = np.transpose(roi_image, (2, 0, 1))
        # Neg log likelihood is
        # sum( [model[ch] * roi[ch] * log(model[ch] for ch in range(3)] )
        # While model is a 3x2D array, NLL is a scalar. 
        first_derivatives_of_negative_log_likelihood_rgb = np.zeros((hypothesis_index + 1, 5))  # row: particle index (starts from 1 while 0 is for background), col: I_r, I_g, I_b, x, y
        first_derivatives_of_negative_log_likelihood_rgb[0][3] = first_derivatives_of_negative_log_likelihood_rgb[0][4] = np.nan
        # Refer to the explanation written inside first_derivatives_of_normal_gaussian_integrated_within_each_pixel function for better understanding of the below two lines.
        # psf_factors_for_pixels_in_x
        first_derivatives_of_psf_factors_for_pixels_in_x = np.array([first_derivatives_of_normal_gaussian_integrated_within_each_pixel(np.arange(szx), theta['particle'][p_idx]['x'], psf_sigma) for p_idx in range(0, hypothesis_index)]) # 2d array with row: pindex, col: x-position
        first_derivatives_of_psf_factors_for_pixels_in_y = np.array([first_derivatives_of_normal_gaussian_integrated_within_each_pixel(np.arange(szy), theta['particle'][p_idx]['y'], psf_sigma) for p_idx in range(0, hypothesis_index)]) # 2d array with row: pindex, col: y-position

    # Add extra entry at beginning so indices match pidx  - both for grayscale and rgb case
    first_derivatives_of_psf_factors_for_pixels_in_x = np.insert(first_derivatives_of_psf_factors_for_pixels_in_x, 0, None, axis=0)
    first_derivatives_of_psf_factors_for_pixels_in_y = np.insert(first_derivatives_of_psf_factors_for_pixels_in_y, 0, None, axis=0)


    if isinstance(theta, (list, np.ndarray)): # Case: grayscale image. If it was RGB, theta would have been a dictionary.

        # Pre-calculate the ratio of the image to the model intensity
        one_minus_image_over_model = (1 - roi_image / roi_model) # This will be used many times in the following calculations.

        # We need to calculate the derivatives of the modified negative log-likelihood function with respect to the normalized parameters 
        # - These derivative will be the derivatives with respect to unnormalized parameters times the "normalization factor"
        first_derivatives_of_negative_log_likelihood[0][0] = np.sum(one_minus_image_over_model * np.sign(theta[0][0])) * roi_max # roi_max is the "normalization factor" for the intensity


        for p_idx in range(1, hypothesis_index + 1):
            first_derivatives_of_negative_log_likelihood[p_idx][0] = np.sum(one_minus_image_over_model * np.sign(theta[p_idx][0]) * np.outer(psf_factors_for_pixels_in_y[p_idx], psf_factors_for_pixels_in_x[p_idx]) * alpha) # (roi_max - roi_min) * 2 * np.pi * psf_sigma**2 is the normalization factor for particle intensity
            first_derivatives_of_negative_log_likelihood[p_idx][1] = np.sum(one_minus_image_over_model * np.outer(psf_factors_for_pixels_in_y[p_idx], first_derivatives_of_psf_factors_for_pixels_in_x[p_idx]) * np.abs(theta[p_idx][0]) * szx) # szx is the normalization factor for the x-coordinate
            first_derivatives_of_negative_log_likelihood[p_idx][2] = np.sum(one_minus_image_over_model * np.outer(first_derivatives_of_psf_factors_for_pixels_in_y[p_idx], psf_factors_for_pixels_in_x[p_idx]) * np.abs(theta[p_idx][0]) * szy) # szy is the normalization factor for the y-coordinate

        jacobian = first_derivatives_of_negative_log_likelihood.flatten()
        jacobian = jacobian[~np.isnan(jacobian)]
    
    else: # Case: rgb image - rgb code not updated to work with the absolute values of particle intensities yet. (2024/11/29)
        # one_minus_image_over_model_rgb = np.array([(1 - roi_image[ch] / roi_model[ch]) for ch in range(3)]) # This will be used many times in the following calculations. "1" will act as a matrix of ones.
        # # first_derivatives_of_negative_log_likelihood_rgb[0][:3] = np.sum([one_minus_image_over_model_rgb * roi_max[ch] for ch in range(3)], axis=(1, 2, 3))
        # first_derivatives_of_negative_log_likelihood_rgb[0][:3] = np.sum(one_minus_image_over_model_rgb * np.array([np.ones((szy, szx)) * 1 * roi_max[ch] for ch in range(3)]), axis=(1, 2))
        # # Check: is the following the same as the above? : first_derivatives_of_negative_log_likelihood_rgb[0][:3] = np.sum([one_minus_image_over_model_rgb * roi_max[ch] for ch in range(3)]), axis=(1, 2))

        # for p_idx in range(1, hypothesis_index + 1):
        #     first_derivatives_of_negative_log_likelihood_rgb[p_idx][:3] = [np.sum(one_minus_image_over_model_rgb[ch] * np.outer(psf_factors_for_pixels_in_y[p_idx], psf_factors_for_pixels_in_x[p_idx]) * alpha[ch]) for ch in range(3)] # (roi_max - roi_min) * 2 * np.pi * psf_sigma**2 is the normalization factor for particle intensity
        #     # Check really carefully here:
        #     first_derivatives_of_negative_log_likelihood_rgb[p_idx][3] = np.sum([one_minus_image_over_model_rgb[ch] * np.outer(psf_factors_for_pixels_in_y[p_idx], first_derivatives_of_psf_factors_for_pixels_in_x[p_idx]) * theta['particle'][p_idx - 1]['I'][ch] * szx for ch in range(3)]) # szx is the normalization factor for the x-coordinate
        #     first_derivatives_of_negative_log_likelihood_rgb[p_idx][4] = np.sum([one_minus_image_over_model_rgb[ch] * np.outer(first_derivatives_of_psf_factors_for_pixels_in_y[p_idx], psf_factors_for_pixels_in_x[p_idx]) * theta['particle'][p_idx - 1]['I'][ch] * szy for ch in range(3)]) # szy is the normalization factor for the y-coordinate
        
        # jacobian = np.array(first_derivatives_of_negative_log_likelihood_rgb[~np.isnan(first_derivatives_of_negative_log_likelihood_rgb)]).flatten()
        pass
        

    # Check the shape of the gradient
    if jacobian.shape != norm_flat_trimmed_theta.shape:
        print("Warning: the shape of the jacobian is not the same as the shape of the parameters. Check required")
        # Reshape the gradient to have the same shape as norm_flat_trimmed_theta
        jacobian = jacobian.reshape(norm_flat_trimmed_theta.shape)

    # Add the out-of-bounds penalty to the Jacobian
    ddt_oob = jac_oob_penalty(theta, szx, szy, alpha)  # This will not work well for rgb... needs checking (11/26/2024)
    jac_oob = ddt_oob.flatten()
    jac_oob = jac_oob[~np.isnan(jac_oob)]
    jacobian += jac_oob

    return jacobian


def hessian_fn(norm_flat_trimmed_theta, hypothesis_index, roi_image, roi_min, roi_max, minimum_model_value, psf_sigma, szx, szy, alpha, color_mode='grayscale'):
    """
    Calculate the Hessian matrix for the negative log-likelihood function.

    Parameters:
    - norm_flat_trimmed_theta (array-like): Normalized and flattened theta values.
    - hypothesis_index (int): Number of hypotheses.
    - roi_image (array-like): Region of interest image.
    - roi_min (float): Minimum value of the region of interest.
    - roi_max (float): Maximum value of the region of interest.
    - minimum_model_value (float): Minimum value of the model.
    - psf_sigma (float): Standard deviation of the point spread function.
    - szx (int): Size of the x-axis.
    - szy (int): Size of the y-axis.
    - alpha (float or list): Intensity Normalization factor.

    Returns:
    - d2dt2_nll_2d (array-like): Hessian matrix for the negative log-likelihood function.
    """
    # Denormalize theta to calculate roi_model
    theta = denormalize(norm_flat_trimmed_theta, hypothesis_index, roi_max, szx, szy, alpha, color_mode=color_mode)

    # Precalculate intensity and derivatives
    roi_model, psf_factors_for_pixels_in_x, psf_factors_for_pixels_in_y = calculate_model_ipsfx_ipsfy(theta, np.arange(szx), np.arange(szy), hypothesis_index, minimum_model_value, psf_sigma, color_mode=color_mode)

    # Calculate ingredients for the Hessian matrix
    if color_mode == 'grayscale':
        # nll: negloglikelihood
        d2dt2_nll_2d = np.zeros((hypothesis_index * 3 + 1, hypothesis_index * 3 + 1))

        first_derivatives_of_psf_factors_for_pixels_in_x = np.array([first_derivatives_of_normal_gaussian_integrated_within_each_pixel(np.arange(szx), theta[p_idx][1], psf_sigma) for p_idx in range(1, hypothesis_index + 1)])
        first_derivatives_of_psf_factors_for_pixels_in_y = np.array([first_derivatives_of_normal_gaussian_integrated_within_each_pixel(np.arange(szy), theta[p_idx][2], psf_sigma) for p_idx in range(1, hypothesis_index + 1)])
        second_derivatives_of_psf_factors_for_pixels_in_x = np.array([second_derivatives_of_normal_gaussian_integrated_within_each_pixel(np.arange(szx), theta[p_idx][1], psf_sigma) for p_idx in range(1, hypothesis_index + 1)])
        second_derivatives_of_psf_factors_for_pixels_in_y = np.array([second_derivatives_of_normal_gaussian_integrated_within_each_pixel(np.arange(szy), theta[p_idx][2], psf_sigma) for p_idx in range(1, hypothesis_index + 1)])
    
    elif color_mode == 'rgb': # Case: rgb image - not updated to work with the absolute values of particle intensities yet. (2024/11/29) [TODO]
        # # Neg log likelihood is
        # # sum( [model[ch] * roi[ch] * log(model[ch] for ch in range(3)] )
        # # While model is a 3x2D array, NLL is a scalar. 
        # # ddt_nll_rgb = np.zeros((hypothesis_index * 5 + 3, hypothesis_index * 5 + 3))  
        # # ddt_nll_rgb[0][3] = ddt_nll_rgb[0][4] = np.nan

        # d2dt2_nll_2d = np.zeros((hypothesis_index * 5 + 3, hypothesis_index * 5 + 3))
        
        # # Refer to the explanation written inside first_derivatives_of_normal_gaussian_integrated_within_each_pixel function for better understanding of the below two lines.
        # first_derivatives_of_psf_factors_for_pixels_in_x = np.array([first_derivatives_of_normal_gaussian_integrated_within_each_pixel(np.arange(szx), theta['particle'][p_idx]['x'], psf_sigma) for p_idx in range(0, hypothesis_index)]) # 2d array with row: pindex, col: x-position
        # first_derivatives_of_psf_factors_for_pixels_in_y = np.array([first_derivatives_of_normal_gaussian_integrated_within_each_pixel(np.arange(szy), theta['particle'][p_idx]['y'], psf_sigma) for p_idx in range(0, hypothesis_index)]) # 2d array with row: pindex, col: y-position
        # second_derivatives_of_psf_factors_for_pixels_in_x = np.array([second_derivatives_of_normal_gaussian_integrated_within_each_pixel(np.arange(szx), theta['particle'][p_idx]['x'], psf_sigma) for p_idx in range(0, hypothesis_index)]) # 2d array with row: pindex, col: x-position
        # second_derivatives_of_psf_factors_for_pixels_in_y = np.array([second_derivatives_of_normal_gaussian_integrated_within_each_pixel(np.arange(szy), theta['particle'][p_idx]['y'], psf_sigma) for p_idx in range(0, hypothesis_index)]) # 2d array with row: pindex, col: y-position
        pass

    else:
        raise ValueError("Error: color_mode should be either 'grayscale' or 'rgb'. Check required. location: hessian_fn()")
    
    # add extra entry at beginning so indices match particle_index
    first_derivatives_of_psf_factors_for_pixels_in_x = np.insert(first_derivatives_of_psf_factors_for_pixels_in_x, 0, None, axis=0)
    first_derivatives_of_psf_factors_for_pixels_in_y = np.insert(first_derivatives_of_psf_factors_for_pixels_in_y, 0, None, axis=0)      
    second_derivatives_of_psf_factors_for_pixels_in_x = np.insert(second_derivatives_of_psf_factors_for_pixels_in_x, 0, None, axis=0)        
    second_derivatives_of_psf_factors_for_pixels_in_y = np.insert(second_derivatives_of_psf_factors_for_pixels_in_y, 0, None, axis=0)       

    if color_mode == 'grayscale':
        # Note: There are 4 + 3 + 2 + 1 = 10 combinations of possible second derivatives of the negative log-likelihood function with respect to the parameters

        # Calculate pixelval_over_model_squared (used in all the calculations)
        pixelval_over_model_squared = roi_image / roi_model**2

        # 1. Differentiated with respect to theta of the following indices: 00, 00
        d2dt2_nll_00_00 = np.sum(pixelval_over_model_squared * (roi_max)**2)
        d2dt2_nll_2d[0][0] = d2dt2_nll_00_00
                
        for pidx in range(1, hypothesis_index + 1):

            original_sign_of_theta_pidx_0 = np.sign(theta[pidx][0])
            theta[pidx][0] = np.abs(theta[pidx][0])

            # 2. Differentiated with respect to theta of the following indices: i0, 00
            d2dt2_nll_i0_00 = np.sum(pixelval_over_model_squared * np.outer(psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) * alpha * (roi_max) )
            # -> Treatment for theta_pidx_0 -> |theta_pidx_0|
            d2dt2_nll_i0_00 *= original_sign_of_theta_pidx_0
            d2dt2_nll_2d[0][(pidx - 1) * 3 + 1] = d2dt2_nll_2d[(pidx - 1) * 3 + 1][0] = d2dt2_nll_i0_00

            # 3. Differentiated with respect to theta of the following indices: i1, 00
            d2dt2_nll_i1_00 = np.sum(pixelval_over_model_squared * np.outer(psf_factors_for_pixels_in_y[pidx], first_derivatives_of_psf_factors_for_pixels_in_x[pidx]) * theta[pidx][0] * (szx) * (roi_max) )
            d2dt2_nll_2d[0][(pidx - 1) * 3 + 2] = d2dt2_nll_2d[(pidx - 1) * 3 + 2][0] = d2dt2_nll_i1_00

            # 4. Differentiated with respect to theta of the following indices: i2, 00
            d2dt2_nll_i2_00 = np.sum(pixelval_over_model_squared * np.outer(first_derivatives_of_psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) * theta[pidx][0] * (szy) * (roi_max) )
            d2dt2_nll_2d[0][(pidx - 1) * 3 + 3] = d2dt2_nll_2d[(pidx - 1) * 3 + 3][0] = d2dt2_nll_i2_00

            # 5. Differentiated with respect to theta of the following indices: i0, i0
            d2dt2_nll_i0_i0 = np.sum(pixelval_over_model_squared * np.outer(psf_factors_for_pixels_in_y[pidx]**2, psf_factors_for_pixels_in_x[pidx]**2) * alpha**2 )
            d2dt2_nll_2d[(pidx - 1) * 3 + 1][(pidx - 1) * 3 + 1] = d2dt2_nll_i0_i0

            # 6. Differentiated with respect to theta of the following indices: i1, i0
            d2dt2_nll_i1_i0 = np.sum((pixelval_over_model_squared * theta[pidx][0] * np.outer(psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) + (1 - roi_image / roi_model)) \
                                    * np.outer(psf_factors_for_pixels_in_y[pidx], first_derivatives_of_psf_factors_for_pixels_in_x[pidx]) * szx * (alpha) )
            # -> Treatment for theta_pidx_0 -> |theta_pidx_0|
            d2dt2_nll_i1_i0 *= original_sign_of_theta_pidx_0
            d2dt2_nll_2d[(pidx - 1) * 3 + 2][(pidx - 1) * 3 + 1] = d2dt2_nll_2d[(pidx - 1) * 3 + 1][(pidx - 1) * 3 + 2] = d2dt2_nll_i1_i0

            # 7. Differentiated with respect to theta of the following indices: i2, i0
            d2dt2_nll_i2_i0 = np.sum((pixelval_over_model_squared * theta[pidx][0] * np.outer(psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) + (1 - roi_image / roi_model)) \
                                    * np.outer(first_derivatives_of_psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) * szy * (alpha) )
            # -> Treatment for theta_pidx_0 -> |theta_pidx_0|
            d2dt2_nll_i2_i0 *= original_sign_of_theta_pidx_0
            d2dt2_nll_2d[(pidx - 1) * 3 + 3][(pidx - 1) * 3 + 1] = d2dt2_nll_2d[(pidx - 1) * 3 + 1][(pidx - 1) * 3 + 3] = d2dt2_nll_i2_i0

            # 8. Differentiated with respect to theta of the following indices: i1, i1 
            d2dt2_nll_i1_i1 = np.sum((pixelval_over_model_squared * theta[pidx][0] * np.outer(psf_factors_for_pixels_in_y[pidx]**2, first_derivatives_of_psf_factors_for_pixels_in_x[pidx] ** 2) \
                                    + (1 - roi_image / roi_model) * np.outer(psf_factors_for_pixels_in_y[pidx], second_derivatives_of_psf_factors_for_pixels_in_x[pidx])) * theta[pidx][0] * szx**2   )
            d2dt2_nll_2d[(pidx - 1) * 3 + 2][(pidx - 1) * 3 + 2] = d2dt2_nll_i1_i1

            # 9. Differentiated with respect to theta of the following indices: i2, i1 
            d2dt2_nll_i2_i1 = np.sum((pixelval_over_model_squared * theta[pidx][0] * np.outer(psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) + (1 - roi_image / roi_model)) * theta[pidx][0] \
                                    * np.outer(first_derivatives_of_psf_factors_for_pixels_in_y[pidx], first_derivatives_of_psf_factors_for_pixels_in_x[pidx]) * szx * szy )
            d2dt2_nll_2d[(pidx - 1) * 3 + 3][(pidx - 1) * 3 + 2] = d2dt2_nll_2d[(pidx - 1) * 3 + 2][(pidx - 1) * 3 + 3] = d2dt2_nll_i2_i1

            # 10. Differentiated with respect to theta of the following indices: i2, i2 ##
            d2dt2_nll_i2_i2 = np.sum((pixelval_over_model_squared * theta[pidx][0] * np.outer(first_derivatives_of_psf_factors_for_pixels_in_y[pidx]** 2, psf_factors_for_pixels_in_x[pidx]**2) \
                                    + (1 - roi_image / roi_model) * np.outer(second_derivatives_of_psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]))   * theta[pidx][0] * szy**2  )
            d2dt2_nll_2d[(pidx - 1) * 3 + 3][(pidx - 1) * 3 + 3] = d2dt2_nll_i2_i2

    else: # Case: rgb image - not updated to work with the absolute values of particle intensities yet. (2024/11/29)
        # if roi_image.shape[0] != 3:
        #     roi_image = np.transpose(roi_image, (2, 0, 1))
        # # 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 = 36 combinations total.
        # # 8: 00r-00r, 00r-00g, 00r-00b, 00r-i0r, 00r-i0g, 00r-i0b, 00r-ix, 00r-iy
        # # 7:          00g-00g, 00g-00b, 00g-i0r, 00g-i0g, 00g-i0b, 00g-ix, 00g-iy
        # # 6:                   00b-00b, 00b-i0r, 00b-i0g, 00b-i0b, 00b-ix, 00b-iy
        # # 5:                            i0r-i0r, i0r-i0g, i0r-i0b, i0r-ix, i0r-iy
        # # 4:                                     i0g-i0g, i0g-i0b, i0g-ix, i0g-iy
        # # 3:                                              i0b-i0b, i0b-ix, i0b-iy
        # # 2:                                                        ix-ix,  ix-iy
        # # 1:                                                                iy-iy

        # pixelval_over_model_squared = roi_image / roi_model**2

        # # 00r-related
        # d2dt2_nll_00r_00r = np.sum([ pixelval_over_model_squared[0] * (roi_max[0])**2 ])
        # # d2dt2_nll_00r_00g = d2dt2_nll_00r_00b = 0

        # # 00g-related
        # d2dt2_nll_00g_00g = np.sum([ pixelval_over_model_squared[1] * (roi_max[1])**2 ])
        # # d2dt2_nll_00g_00b = 0

        # # 00b-related
        # d2dt2_nll_00b_00b = np.sum([ pixelval_over_model_squared[2] * (roi_max[2])**2 ])

        # # Assign to the relevant places in the Hessian matrix.
        # d2dt2_nll_2d[0][0] = d2dt2_nll_00r_00r # 00r takes the 0th index
        # d2dt2_nll_2d[1][1] = d2dt2_nll_00g_00g # 00g takes the 1st index
        # d2dt2_nll_2d[2][2] = d2dt2_nll_00b_00b # 00b takes the 2nd indej

        # for pidx in range(1, hypothesis_index + 1):

        #     # 00r-related
        #     d2dt2_nll_00r_i0r = np.sum([ pixelval_over_model_squared[0] * np.outer(psf_factors_for_pixels_in_x[pidx], psf_factors_for_pixels_in_y[pidx]) * (alpha[0]) * (roi_max[0]) ])
        #     # d2dt2_nll_00r_i0g = d2dt2_nll_00r_i0b = 0
        #     d2dt2_nll_00r_ix  = np.sum([ pixelval_over_model_squared[0] * np.outer(psf_factors_for_pixels_in_y[pidx], first_derivatives_of_psf_factors_for_pixels_in_x[pidx]) * theta['particle'][pidx-1]['I'][0] * szx * roi_max[0] ])
        #     d2dt2_nll_00r_iy  = np.sum([ pixelval_over_model_squared[0] * np.outer(first_derivatives_of_psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) * theta['particle'][pidx-1]['I'][0] * szy * roi_max[0] ])

        #     # 00g-related
        #     # d2dt2_nll_00g_i0r = 0 
        #     d2dt2_nll_00g_i0g = np.sum([ pixelval_over_model_squared[1] * np.outer(psf_factors_for_pixels_in_x[pidx], psf_factors_for_pixels_in_y[pidx]) * (alpha[1]) * (roi_max[1]) ])
        #     # d2dt2_nll_00g_i0b =  0
        #     d2dt2_nll_00g_ix  = np.sum([ pixelval_over_model_squared[1] * np.outer(psf_factors_for_pixels_in_y[pidx], first_derivatives_of_psf_factors_for_pixels_in_x[pidx]) * theta['particle'][pidx-1]['I'][1] * szx * roi_max[1] ])
        #     d2dt2_nll_00g_iy  = np.sum([ pixelval_over_model_squared[1] * np.outer(first_derivatives_of_psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) * theta['particle'][pidx-1]['I'][1] * szy * roi_max[1] ])

        #     # 00b-related
        #     # d2dt2_nll_00b_i0r = d2dt2_nll_00b_i0g = 0
        #     d2dt2_nll_00b_i0b = np.sum([ pixelval_over_model_squared[2] * np.outer(psf_factors_for_pixels_in_x[pidx], psf_factors_for_pixels_in_y[pidx]) * (alpha[2]) * (roi_max[2]) ])
        #     d2dt2_nll_00b_ix  = np.sum([ pixelval_over_model_squared[2] * np.outer(psf_factors_for_pixels_in_y[pidx], first_derivatives_of_psf_factors_for_pixels_in_x[pidx]) * theta['particle'][pidx-1]['I'][2] * szx * roi_max[2] ])
        #     d2dt2_nll_00b_iy  = np.sum([ pixelval_over_model_squared[2] * np.outer(first_derivatives_of_psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) * theta['particle'][pidx-1]['I'][2] * szy * roi_max[2] ])


        #     # i0r-related
        #     d2dt2_nll_i0r_i0r = np.sum(pixelval_over_model_squared[0] * np.outer(psf_factors_for_pixels_in_y[pidx]**2, psf_factors_for_pixels_in_x[pidx]**2) * alpha[0]**2) 
        #     # d2dt2_nll_i0r_i0g = d2dt2_nll_i0r_i0b = 0
        #     d2dt2_nll_i0r_ix  = np.sum( ( pixelval_over_model_squared[0] * theta['particle'][pidx-1]['I'][0] * np.outer(psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) + (1 - roi_image[0] / roi_model[0])) * np.outer(psf_factors_for_pixels_in_y[pidx], first_derivatives_of_psf_factors_for_pixels_in_x[pidx]) * szx * (roi_max[0] - roi_min[0]) * 2 * np.pi * psf_sigma**2 )
        #     d2dt2_nll_i0r_iy  = np.sum( ( pixelval_over_model_squared[0] * theta['particle'][pidx-1]['I'][0] * np.outer(psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) + (1 - roi_image[0] / roi_model[0])) * np.outer(first_derivatives_of_psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) * szy * (roi_max[0] - roi_min[0]) * 2 * np.pi * psf_sigma**2 )

        #     # i0g-related
        #     d2dt2_nll_i0g_i0g = np.sum(pixelval_over_model_squared[1] * np.outer(psf_factors_for_pixels_in_y[pidx]**2, psf_factors_for_pixels_in_x[pidx]**2) * alpha[1]**2)
        #     # d2dt2_nll_i0g_i0b = 0
        #     d2dt2_nll_i0g_ix  = np.sum( ( pixelval_over_model_squared[1] * theta['particle'][pidx-1]['I'][1] * np.outer(psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) + (1 - roi_image[1] / roi_model[1])) * np.outer(psf_factors_for_pixels_in_y[pidx], first_derivatives_of_psf_factors_for_pixels_in_x[pidx]) * szx * (roi_max[1] - roi_min[1]) * 2 * np.pi * psf_sigma**2 )
        #     d2dt2_nll_i0g_iy =  np.sum( ( pixelval_over_model_squared[1] * theta['particle'][pidx-1]['I'][1] * np.outer(psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) + (1 - roi_image[1] / roi_model[1])) * np.outer(first_derivatives_of_psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) * szy * (roi_max[1] - roi_min[1]) * 2 * np.pi * psf_sigma**2 )

        #     # i0b-related
        #     d2dt2_nll_i0b_i0b = np.sum(pixelval_over_model_squared[2] * np.outer(psf_factors_for_pixels_in_y[pidx]**2, psf_factors_for_pixels_in_x[pidx]**2) * alpha[2]**2)
        #     d2dt2_nll_i0b_ix  = np.sum(pixelval_over_model_squared[2] * np.outer(psf_factors_for_pixels_in_y[pidx], first_derivatives_of_psf_factors_for_pixels_in_x[pidx]) * theta['particle'][pidx-1]['I'][2] * szx * (roi_max[2] - roi_min[2]) * 2 * np.pi * psf_sigma**2)
        #     d2dt2_nll_i0b_iy  = np.sum(pixelval_over_model_squared[2] * np.outer(first_derivatives_of_psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx]) * theta['particle'][pidx-1]['I'][2] * szy * (roi_max[2] - roi_min[2]) * 2 * np.pi * psf_sigma**2)
            
        #     # ix-related
        #     d2dt2_nll_ix_ix   = np.sum([(pixelval_over_model_squared[ch] * theta['particle'][pidx-1]['I'][ch] * np.outer(first_derivatives_of_psf_factors_for_pixels_in_x[pidx]**2, first_derivatives_of_psf_factors_for_pixels_in_x[pidx]**2) + (1 - roi_image[ch] / roi_model[ch]) * np.outer(psf_factors_for_pixels_in_y[pidx], second_derivatives_of_psf_factors_for_pixels_in_x[pidx])) * theta['particle'][pidx-1]['I'][ch] * szx**2 for ch in range(3)])
        #     d2dt2_nll_ix_iy   = np.sum([(pixelval_over_model_squared[ch] * theta['particle'][pidx-1]['I'][ch] * np.outer(psf_factors_for_pixels_in_x[pidx], psf_factors_for_pixels_in_y[pidx]) + (1 - roi_image[ch] / roi_model[ch])) * theta['particle'][pidx-1]['I'][ch] * np.outer(first_derivatives_of_psf_factors_for_pixels_in_y[pidx], first_derivatives_of_psf_factors_for_pixels_in_x[pidx]) * szx * szy for ch in range(3)])

        #     # ix-related
        #     d2dt2_nll_iy_iy   = np.sum([(pixelval_over_model_squared[ch] * theta['particle'][pidx-1]['I'][ch] * np.outer(first_derivatives_of_psf_factors_for_pixels_in_y[pidx]**2, first_derivatives_of_psf_factors_for_pixels_in_y[pidx]**2) + (1 - roi_image[ch] / roi_model[ch]) * np.outer(second_derivatives_of_psf_factors_for_pixels_in_y[pidx], psf_factors_for_pixels_in_x[pidx])) * theta['particle'][pidx-1]['I'][ch] * szy**2 for ch in range(3)])

        #     # Assign to the relevant places in the Hessian matrix.
        #     d2dt2_nll_2d[0][(pidx-1)*5 + 3] = d2dt2_nll_2d[(pidx-1)*5 + 3][0] = d2dt2_nll_00r_i0r # 00r takes the 0th index, i0r takes the '3 + (pidx - 1) * 5'th index
        #     # d2dt2_nll_2d[0][(pidx-1)*5 + 4] and its transpose element is 0. # 00r takes the 0th index, i0g takes the '4 + (pidx - 1) * 5'th index
        #     # d2dt2_nll_2d[0][(pidx-1)*5 + 5] and its transpose element is 0. # 00r takes the 0th index, i0b takes the '5 + (pidx - 1) * 5'th index
        #     d2dt2_nll_2d[0][(pidx-1)*5 + 6] = d2dt2_nll_2d[(pidx-1)*5 + 6][0] = d2dt2_nll_00r_ix  # 00r takes the 0th index, ix  takes the '6 + (pidx - 1) * 5'th index
        #     d2dt2_nll_2d[0][(pidx-1)*5 + 7] = d2dt2_nll_2d[(pidx-1)*5 + 7][0] = d2dt2_nll_00r_iy  # 00r takes the 0th index, iy  takes the '7 + (pidx - 1) * 5'th index

        #     # d2dt2_nll_2d[1][(pidx-1)*5 + 3] and its transpose element is 0. # 00g takes the 0th index, i0r takes the '4 + (pidx - 1) * 5'th index
        #     d2dt2_nll_2d[1][(pidx-1)*5 + 4] = d2dt2_nll_2d[(pidx-1)*5 + 4][1] = d2dt2_nll_00g_i0g # 00g takes the 1th index, i0g takes the '4 + (pidx - 1) * 5'th index
        #     # d2dt2_nll_2d[1][(pidx-1)*5 + 5] and its transpose element is 0. # 00g takes the 0th index, i0b takes the '5 + (pidx - 1) * 5'th index
        #     d2dt2_nll_2d[1][(pidx-1)*5 + 6] = d2dt2_nll_2d[(pidx-1)*5 + 6][1] = d2dt2_nll_00g_ix  # 00g takes the 1th index, ix  takes the '6 + (pidx - 1) * 5'th index
        #     d2dt2_nll_2d[1][(pidx-1)*5 + 7] = d2dt2_nll_2d[(pidx-1)*5 + 7][1] = d2dt2_nll_00g_iy  # 00g takes the 1th index, iy  takes the '7 + (pidx - 1) * 5'th index

        #     # d2dt2_nll_2d[2][(pidx-1)*5 + 3] and its transpose element is 0. # 00b takes the 2th index, i0r takes the '3 + (pidx - 1) * 5'th index
        #     # d2dt2_nll_2d[2][(pidx-1)*5 + 4] and its transpose element is 0. # 00b takes the 2th index, i0g takes the '4 + (pidx - 1) * 5'th index
        #     d2dt2_nll_2d[2][(pidx-1)*5 + 5] = d2dt2_nll_2d[(pidx-1)*5 + 3][2] = d2dt2_nll_00b_i0b # 00b takes the 1th index, i0b takes the '5 + (pidx - 1) * 5'th index
        #     d2dt2_nll_2d[2][(pidx-1)*5 + 6] = d2dt2_nll_2d[(pidx-1)*5 + 6][2] = d2dt2_nll_00b_ix  # 00b takes the 1th index, ix  takes the '6 + (pidx - 1) * 5'th index
        #     d2dt2_nll_2d[2][(pidx-1)*5 + 7] = d2dt2_nll_2d[(pidx-1)*5 + 7][2] = d2dt2_nll_00b_iy  # 00b takes the 1th index, iy  takes the '7 + (pidx - 1) * 5'th index

        #     d2dt2_nll_2d[(pidx-1)*5 + 3][(pidx-1)*5 + 3] = d2dt2_nll_i0r_i0r # i0r takes the '3 + (pidx - 1) * 5'th index
        #     d2dt2_nll_2d[(pidx-1)*5 + 3][(pidx-1)*5 + 6] = d2dt2_nll_2d[(pidx-1)*5 + 6][(pidx-1)*5 + 3] = d2dt2_nll_i0r_ix  # ix  takes the '6 + (pidx - 1) * 5'th index
        #     d2dt2_nll_2d[(pidx-1)*5 + 3][(pidx-1)*5 + 7] = d2dt2_nll_2d[(pidx-1)*5 + 7][(pidx-1)*5 + 3] = d2dt2_nll_i0r_iy  # iy  takes the '7 + (pidx - 1) * 5'th index

        #     d2dt2_nll_2d[(pidx-1)*5 + 4][(pidx-1)*5 + 4] = d2dt2_nll_i0g_i0g # i0g takes the '4 + (pidx - 1) * 5'th index
        #     d2dt2_nll_2d[(pidx-1)*5 + 4][(pidx-1)*5 + 6] = d2dt2_nll_2d[(pidx-1)*5 + 6][(pidx-1)*5 + 4] = d2dt2_nll_i0g_ix  # ix  takes the '6 + (pidx - 1) * 5'th index
        #     d2dt2_nll_2d[(pidx-1)*5 + 4][(pidx-1)*5 + 7] = d2dt2_nll_2d[(pidx-1)*5 + 7][(pidx-1)*5 + 4] = d2dt2_nll_i0g_iy  # iy  takes the '7 + (pidx - 1) * 5'th index
        
        #     d2dt2_nll_2d[(pidx-1)*5 + 5][(pidx-1)*5 + 5] = d2dt2_nll_i0b_i0b # i0b takes the '5 + (pidx - 1) * 5'th index
        #     d2dt2_nll_2d[(pidx-1)*5 + 5][(pidx-1)*5 + 6] = d2dt2_nll_2d[(pidx-1)*5 + 6][(pidx-1)*5 + 5] = d2dt2_nll_i0b_ix  # ix  takes the '6 + (pidx - 1) * 5'th index
        #     d2dt2_nll_2d[(pidx-1)*5 + 5][(pidx-1)*5 + 7] = d2dt2_nll_2d[(pidx-1)*5 + 7][(pidx-1)*5 + 5] = d2dt2_nll_i0b_iy  # iy  takes the '7 + (pidx - 1) * 5'th index

        #     d2dt2_nll_2d[(pidx-1)*5 + 6][(pidx-1)*5 + 6] = d2dt2_nll_ix_ix
        #     d2dt2_nll_2d[(pidx-1)*5 + 6][(pidx-1)*5 + 7] = d2dt2_nll_2d[(pidx-1)*5 + 7][(pidx-1)*5 + 6] = d2dt2_nll_ix_iy

        #     d2dt2_nll_2d[(pidx-1)*5 + 7][(pidx-1)*5 + 7] = d2dt2_nll_iy_iy
        pass

    d2dt2_nll_2d += hess_oob_penalty(theta, szx, szy, psf_sigma, color_mode=color_mode) # newly added line (2nd July 2024)

    return d2dt2_nll_2d

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

def get_tentative_peaks(image, min_distance=1,color_mode='grayscale'): # How does it perform for the case of RGB image? Not yet tested thoroughly (11/26/2024) [TODO]
    """ Returns the tentative peak coordinates of the image.
    Args:
        image: The 2D image to process.
        min_distance: The minimum distance between peaks.
    Returns:
        np.array: The tentative peak coordinates.
    """

    if color_mode == 'rgb':
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Not yet tested [TODO]
    elif color_mode == 'grayscale':
        grayscale_image = image
    else:
        raise ValueError("color_mode must be either 'grayscale' or 'rgb'. location: get_tentative_peaks()")
    
    # Define filters
    h2 = 1/16
    h1 = 1/4
    h0 = 3/8
    g0 = np.array([h2, h1, h0, h1, h2])
    g1 = np.array([h2, 0, h1, 0, h0, 0, h1, 0, h2])
    k0 = create_separable_filter(g0, 3)
    dip_image = dip.Image(grayscale_image)

    # Filter image
    v0 = dip.Convolution(dip_image, k0, method="best")
    k1 = create_separable_filter(g1, 5)
    v1 = dip.Convolution(v0, k1, method="best")
    filtered_image = np.asarray(v0 - v1)
    filtered_image = filtered_image - np.min(filtered_image)
    tentative_peak_coordinates = peak_local_max(filtered_image, min_distance=min_distance)
    return tentative_peak_coordinates


def normal_gaussian_integrated_within_each_pixel(i, x, sigma):
    """ Compute the integral of the 1D Gaussian.
    Args:
        i (numpy array of ints or int): Pixel index. (e.g., [0, 1, 2, 3, 4, ..., 99])
        x (float): Center position of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.
    Returns:
        float: Integral of the Gaussian from i-0.5 to i+0.5.
    """
    norm = 1/2/sigma**2
    # Below is the same as integral(from i-0.5 to i+0.5) [1/2sqrt(pi)*exp(-normalization_factor*(t-x)**2) dt]
    return 0.5*(erf((i-x+0.5)*np.sqrt(norm))-erf((i-x-0.5)*np.sqrt(norm)))


def first_derivatives_of_normal_gaussian_integrated_within_each_pixel(i, t, sigma):
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


def second_derivatives_of_normal_gaussian_integrated_within_each_pixel(i, t, sigma):
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


def merge_coincident_particles(entire_image, tile_dicts, psf, display_merged_locations=True):
    """ If an entire_image was subdivided into tiles, this function merges the coincident particles in the overlapping regions of the tiles.

    Parameters:
        entire_image (np.ndarray): The entire_image.
        tile_dicts_array (np.ndarray): The array of tile dictionaries.
        - tile_dicts_array[x_index][y_index] = {'x_low_end': x_low_end, 'y_low_end': y_low_end, 'image_slice': entire_image[y_low_end:y_high_end, x_low_end:x_high_end], 'particle_locations': []}
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
        plt.imshow(entire_image, cmap='gray')     
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
                ref_tile['particle_intensities'] = [intensity for i, intensity in enumerate(ref_tile['particle_intensities']) if i not in del_pidx]

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
                ref_tile['particle_intensities'] = [intensity for i, intensity in enumerate(ref_tile['particle_intensities']) if i not in del_pidx]

    # Initialize the resulting locations and intensities.
    resulting_locations = []
    result_intensities = []
    
    for tile_dict in tile_dicts.flatten():
        for loc in tile_dict['particle_locations']:
            absolute_loc = loc + np.array([tile_dict['x_low_end'], tile_dict['y_low_end']])
            resulting_locations.append(absolute_loc)
        for intensity in tile_dict['particle_intensities']:
            resulting_intensities.append(intensity)

    if display_merged_locations:
        plt.sca(axs[1])
        ax = plt.gca()
        plt.title(f'Same locations merged (count:{len(resulting_locations)})')
        plt.imshow(entire_image, cmap='gray')     
        for loc in resulting_locations:
            plt.scatter(loc[0], loc[1], marker=markers[particle_marker_idx], s=200, color='red', linewidths=1)
        plt.show()

    return resulting_locations, resulting_intensities

def calculate_fisher_information_matrix_vectorized(norm_flat_trimmed_theta, szy, szx, hypothesis_index, minimum_model_value, psf_sigma, roi_max, alpha, color_mode='grayscale'):
    """
    Vectorized calculation of Fisher Information Matrix for particle detection.
    TODO: This is not yet realized for RGB images. 
    """

    if color_mode == 'grayscale':
        number_of_parameters_per_particle = 3
    elif color_mode == 'rgb':
        number_of_parameters_per_particle = 5
    else:
        raise ValueError('Invalid color_mode. Please choose either "rgb" or "grayscale". Location: calculate_fisher_information_matrix_vectorized()')

    if hypothesis_index == 0:
        assert isinstance(norm_flat_trimmed_theta, float) # "For hypothesis_index == 0, theta must be a scalar."
        fisher_mat = np.zeros((1, 1))
        fisher_mat[0,0] = 1 / max(norm_flat_trimmed_theta, minimum_model_value / roi_max) * roi_max
    else:
        # Calculate model and PSF pixel factors for all positions at once
        theta = denormalize(norm_flat_trimmed_theta, hypothesis_index, roi_max, szx, szy, alpha, color_mode=color_mode)
        xx, yy = np.arange(szx), np.arange(szy)
        model, psf_factors_for_pixels_in_x, psf_factors_for_pixels_in_y = calculate_model_ipsfx_ipsfy(
            theta, xx, yy, hypothesis_index, minimum_model_value, psf_sigma
        )
        
        # Initialize the derivatives array with shape (hypothesis_index + 1, number_of_parameters_per_particle, szy, szx)
        ddt_model_norm = np.zeros((hypothesis_index + 1, number_of_parameters_per_particle, szy, szx))
        
        # Set the derivatives with respect to the background
        ddt_model_norm[0, 0] = np.ones((szy, szx)) * roi_max  # d/d(normalized background) = d/d(bg/roi_max) = roi_max
        ddt_model_norm[0, 1:] = np.nan              # No x,y coordinates for background - ddt_model[0, 1] and ddt_model[0, 2] become (szy, szx) filled with np.nan
        
        # Calculate derivatives for each particle
        for particle_index in range(1, hypothesis_index + 1):
            # Intensity derivatives
            ddt_model_norm[particle_index, 0] = (
                psf_factors_for_pixels_in_y[particle_index].reshape(-1, 1) @ 
                psf_factors_for_pixels_in_x[particle_index].reshape(1, -1)
            ) * alpha  # alpha is the scaling factor for the intensity
            
            # X coordinate derivatives
            dpsf_x = first_derivatives_of_normal_gaussian_integrated_within_each_pixel(xx, theta[particle_index][1], psf_sigma)
            ddt_model_norm[particle_index, 1] = (
                theta[particle_index][0] * 
                psf_factors_for_pixels_in_y[particle_index].reshape(-1, 1) @ 
                dpsf_x.reshape(1, -1)
            ) * szx # szx is the scaling factor for the x-coordinate
            
            # Y coordinate derivatives
            dpsf_y = first_derivatives_of_normal_gaussian_integrated_within_each_pixel(yy, theta[particle_index][2], psf_sigma)
            ddt_model_norm[particle_index, 2] = (
                theta[particle_index][0] * 
                dpsf_y.reshape(-1, 1) @ 
                psf_factors_for_pixels_in_x[particle_index].reshape(1, -1)
            ) * szy # szy is the scaling factor for the y-coordinate
        
        # Initialize Fisher Information Matrix
        num_param_for_background = 1
        num_parameters = num_param_for_background + hypothesis_index * number_of_parameters_per_particle
        fisher_mat = np.zeros((num_parameters, num_parameters))
        
        # Background-Background term
        fisher_mat[0, 0] = np.sum(1.0 / model) * roi_max**2 # roi_max is the scaling factor for the background
        
        # Background-Particle terms
        for particle_index in range(1, hypothesis_index + 1):
            for param_type in range(number_of_parameters_per_particle):
                param_idx = (particle_index - 1) * number_of_parameters_per_particle + param_type + 1
                fisher_mat[0, param_idx] = fisher_mat[param_idx, 0] = np.sum( ddt_model_norm[particle_index, param_type] * 1 / model) * roi_max # roi_max is the scaling factor for the background 
        
        # Particle-Particle terms
        for p1 in range(1, hypothesis_index + 1):
            for param1 in range(number_of_parameters_per_particle):
                idx1 = (p1 - 1) * number_of_parameters_per_particle + param1 + 1
                for p2 in range(p1, hypothesis_index + 1):
                    for param2 in range(number_of_parameters_per_particle):
                        idx2 = (p2 - 1) * number_of_parameters_per_particle + param2 + 1
                        fisher_mat[idx1, idx2] = fisher_mat[idx2, idx1] = np.sum( ddt_model_norm[p1, param1] * ddt_model_norm[p2, param2] / model)
                        if idx1 == idx2 and fisher_mat[idx1, idx2] == 0:
                            raise ValueError('Fisher Information Matrix is singular. Please check the input parameters. Location: calculate_fisher_information_matrix_vectorized()')
    
    return fisher_mat

def generalized_maximum_likelihood_rule(roi_image, psf_sigma, last_h_index=5, random_seed=0, display_fit_results=False, display_xi_graph=False, use_exit_condition=False, roi_name=None, color_mode='grayscale'):
    # Convert the input image to float32. 
    roi_image = roi_image.astype(np.float32)

    # Set the random seed
    np.random.seed(random_seed)

    # Check the input image
    if color_mode == 'grayscale':
        szy, szx = roi_image.shape 
        number_of_parameters_per_particle = 3
        num_parameters_for_background = 1
    elif color_mode == 'rgb':
        szy, szx = roi_image.shape[1], roi_image.shape[2]
        number_of_parameters_per_particle = 5
        num_parameters_for_background = 3
    else:
        raise ValueError('Invalid color_mode. Please choose either "rgb" or "grayscale". Location: generalized_maximum_likelihood_rule()')
    
    """ Indexing rules
    - hypothesis_index: 0, 1, 2, ... (H0, H1, H2, ...)
    - particle_index: 1, 2, 3, ... particle 1, particle 2, particle 3, ...)
    - param_type_index (grayscale): 0, 1, 2 (intensity, x-coordinate, y-coordinate)
    - param_type_index (RGB): 0, 1, 2, 3, 4 (intensity_R, intensity_G, intensity_B, x-coordinate, y-coordinate)
    """ 

    # Find tentative peaks
    tentative_peaks = get_tentative_peaks(roi_image, min_distance=1, color_mode=color_mode)
    number_of_peaks_to_find = 7
    rough_peaks_xy = [peak[::-1] for peak in tentative_peaks[:number_of_peaks_to_find]] # Convert (y, x) to (x, y) format.

    # Set the minimum model at any x, y coordinate to avoid dividing by zero.
    minimum_model_value = 1e-2

    # Set the method to use with scipy.optimize.minimize for the MLE estimation.
    method = 'trust-exact'
    
    # Initialize test scores
    xi = [] # Which will be lli - penalty
    xi_aic = [] # AIC score
    xi_bic = [] # BIC score
    lli = [] # log likelihood
    penalty_laplace = [] # penalty_laplace term
    penalty_aic = []
    penalty_bic = []
    
    # Initialize the Fisher Information Matrix 
    fisher_info = [] 

    # Extract roi_max and roi_min 
    roi_max, roi_min = np.max(roi_image), np.min(roi_image) 

    # Calculate alpha factor
    alpha = (roi_max - roi_min) * 2 * np.pi * psf_sigma**2

    # Create figure showing parameter estimation results for all tested hypotheses.
    if display_fit_results:
        # Create a figure to show the results of the fit.
        _, ax_main = plt.subplots(2, last_h_index + 1, figsize=(2 * (last_h_index + 1), 4))

        # Create a colormap instance
        cmap = plt.get_cmap('turbo')# Create a colormap instance for tentative peak coordinates presentation.
        
        # As an exception to the other axes, use ax_main[1][0] to show tentative peak locations, since there's is not much to show for a simple background estimation.
        for i, coord in enumerate(rough_peaks_xy):
            x, y = coord 
            color = cmap(i / len(rough_peaks_xy))  # Use turbo colormap
            ax_main[1][0].text(x, y, f'{i}', fontsize=6, color=color) 
        ax_main[1][0].set_xlim(0-.5, szx-.5)
        ax_main[1][0].set_ylim(szy-.5, 0-.5) 
        ax_main[1][0].set_aspect('equal')
        ax_main[1][0].set_title('Tentative Peak Coordinates', fontsize=8)
        ax_main[0][0].imshow(roi_image)
        plt.show(block=False)
    
    # if use_exit_condition == True and xi_drop_count reaches a certain number, then the algorithm will exit the loop. 
    xi_drop_count = 0
    
    # Initialize the fit results
    fit_results = [] 
    
    # Loop through all hypotheses and the corresponding models
    for hypothesis_index in range(last_h_index + 1): # hypothesis_index is also the number of particles. 

        # Initialization
        num_parameters_for_current_model = num_parameters_for_background + hypothesis_index * (number_of_parameters_per_particle) #H0: 1, H1: 5, H2: 8, ...
        
        # Initialize the theta (parameter) vector
        # theta[0][0] will be the estimated background intensity. (However, if hypothesis_index == 0, theta will just be a scalar, and equal the background intensity.)
        # Since background intensity is the only parameter for H0, theta[0][1] and theta[0][2] will be nan and, importantly, not be passed for optimization.
        # theta[1][0] will be the estimated scattering strength of particle 1.
        # theta[1][1], theta[1][2] will be the estimated x and y coordinate of particle 1, etc.
        if hypothesis_index == 0:
            # initialize theta as a scalar value for background intensity.
            theta = roi_image.sum() / szx / szy
            convergence = True
            convergence_record = [convergence]

        # Only do the MLE (maximum likelihood estimation) for the hypothesis_index >= 1
        else:
            # Initialize theta as a matrix for the parameters of the particles and the background.
            theta = np.zeros((hypothesis_index + 1, number_of_parameters_per_particle)) 

            # Set the background first
            theta[0][0] = roi_min
            theta[0][1] = theta[0][2] = np.nan

            # Set initial guess of all particle intensities as alpha where alpha = (roi_max - roi_min) * 2 * np.pi * psf_sigma**2 is a reasonable initial guess for the intensity of the particle.
            theta[1:, 0] = alpha

            # Initialize particle coordinates according to the tentative peaks found.
            for particle_index in range(1, hypothesis_index + 1):
                if particle_index <= len(rough_peaks_xy):
                    theta[particle_index][1] = rough_peaks_xy[particle_index-1][0]
                    theta[particle_index][2] = rough_peaks_xy[particle_index-1][1]
                else:
                    # assign random positions. 
                    theta[particle_index][1] = random.random() * (szx - 1)
                    theta[particle_index][2] = random.random() * (szy - 1)

            # Normazlize the parameters before passing on to neg_loglikelihood_function
            norm_flat_trimmed_theta = normalize(theta, hypothesis_index, roi_max, szx, szy, alpha, color_mode=color_mode) # for color_mode=='grayscale', the returned theta will be also flat and trimmed.

            # Define the callback function as a nested function - to store the snapshots of the optimization process. Intended for debugging purposes.
            def callback_fn(xk, *args):
                jac = jacobian_fn(xk, hypothesis_index, roi_image, roi_min, roi_max, minimum_model_value, psf_sigma, szx, szy, alpha, color_mode)
                gradientnorm = np.linalg.norm(jac)
                fn = modified_neg_loglikelihood_fn(xk, hypothesis_index, roi_image, roi_min, roi_max, minimum_model_value, psf_sigma, szx, szy, alpha, color_mode)
                gradientnorm_snapshots.append(gradientnorm)
                fn_snapshots.append(fn)                
                theta_snapshots.append(xk)
                denormflat_theta_snapshots.append(denormalize(xk, hypothesis_index, roi_max, szx, szy, alpha, color_mode).flatten())

            # Address the possibility of the minimization failing due error (e.g., passing negative of inf value to np.sqrt())
            num_minimize_trials = 10 # Arbitrary number of trials to attempt to avoid the error in the minimization - Thus far num_minimize_trials = 2 has been sufficient.
            for minimizing_trial_index in range(num_minimize_trials):
                # Initialize storage for minimization snapshots if record_snapshots is True
                record_snapshots = False
                if record_snapshots:
                    gradientnorm_snapshots = []
                    fn_snapshots = []
                    theta_snapshots = []
                    denormflat_theta_snapshots = []

                # Now, let's update the parameters using scipy.optimize.minimize
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('error', category=RuntimeWarning)
                        if record_snapshots:
                            minimization_result = minimize(modified_neg_loglikelihood_fn, norm_flat_trimmed_theta, args=(hypothesis_index, roi_image, roi_min, roi_max, minimum_model_value, psf_sigma, szx, szy, alpha), method=method, jac=jacobian_fn, hess=hessian_fn, callback=callback_fn, options={'gtol': 100})
                        else:
                            minimization_result = minimize(modified_neg_loglikelihood_fn, norm_flat_trimmed_theta, args=(hypothesis_index, roi_image, roi_min, roi_max, minimum_model_value, psf_sigma, szx, szy, alpha), method=method, jac=jacobian_fn, hess=hessian_fn, options={'gtol': 100})
                        break # If minimization is successful, exit the loop without trying again.

                except RuntimeWarning as e:
                    print(f"\nRunTimeWarning occurred during optimization: {e}, roi_name: {roi_name}, hypothesis_index: {hypothesis_index}. \n-- Adjusting the initial guess and 'trying again.' Trial number: {i+1}/{num_minimize_trials}\n")

                    # Adjust the initial guess or options and try again
                    adjusted_norm_flat_trimmed_theta = np.abs(norm_flat_trimmed_theta) * (1 + 0.05 * (np.random.rand() - 0.5))  
                    norm_flat_trimmed_theta = adjusted_norm_flat_trimmed_theta

            if minimizing_trial_index > 0:
                if minimizing_trial_index == num_minimize_trials - 1:
                    print(f"\nRuntimeerror in the minimization could not be avoided (failed) even with perturbing the initial guess and trying again {num_minimize_trials} times. roi_name: {roi_name}, hypothesis_index: {hypothesis_index}\n")
                else:
                    print(f"\nRuntimeerror in the minimization was avoided (success!) by perturbing the initial guess and trying again. {i} times. roi_name: {roi_name}, hypothesis_index: {hypothesis_index}\n")

            # calculate the length of the snapshots for plotting purposes
            if record_snapshots:
                snapshot_length = len(fn_snapshots)
            else:
                snapshot_length = 0

            # Store the fitted parameters and the convergence status
            convergence = minimization_result.success
            norm_theta = minimization_result.x
            convergence_record.append(convergence)

            # Retrieve the estimated parameters by denormalizing the normalized parameters
            theta = denormalize(norm_theta, hypothesis_index, roi_max, szx, szy, alpha, color_mode=color_mode)
        
            # If the estimated particle intensity is negative, take its positive value and store as particle intensity parameter.
            for pidx in range(1, hypothesis_index + 1):
                theta[pidx][0] = np.abs(theta[pidx][0]) # Note: The minimization algorithm may have converged to a negative particle intensity all involved landscapes are a mirror image of the positive particle intensity.

            # If the converged background intensity is negative (although such case is highly unlikely), take its positive value and store as background intensity parameter.
            theta[0][0] = np.abs(theta[0][0])
        
        # Store fit results
        current_hypothesis_fit_result = {
            'hypothesis_index': hypothesis_index,
            'theta': theta,
            'convergence': convergence,
        }

        # Append the fit result to fit_results
        fit_results.append(current_hypothesis_fit_result)

        if display_fit_results and hypothesis_index > 0:
            ax_main[0][hypothesis_index].set_title(f"H{hypothesis_index} - convgd: {convergence_record[hypothesis_index]}\nbg: {theta[0][0]:.1f}", fontsize=8)
            for particle_index in range(1, hypothesis_index + 1):
                ax_main[0][hypothesis_index].imshow(roi_image)
                red = random.randint(200, 255)
                green = random.randint(0, 100)
                blue = random.randint(0, 50)
                color_code = '#%02X%02X%02X' % (red, green, blue)
                ax_main[0][hypothesis_index].scatter(theta[particle_index][1], theta[particle_index][2], s=10, color=color_code, marker='x')
                ax_main[0][hypothesis_index].text(theta[particle_index][1] + np.random.rand() * 1.5, theta[particle_index][2] + (np.random.rand() - 0.5) * 4,
                                            f'  {theta[particle_index][0]:.1f}', color=color_code, fontsize=10,) 
            if snapshot_length > 0:
                ax_main[1][hypothesis_index].set_title(f'Gradient norm\nFinal func val: {fn_snapshots[-1]:.04e}', fontsize=8)
                ax_main[1][hypothesis_index].plot(np.arange(snapshot_length), gradientnorm_snapshots, '-o', color='black', markersize=2, label='Gradient norm')
                ax_main[1][hypothesis_index].set_ylim(bottom=0)
            else:
                ax_main[1][hypothesis_index].set_title('Snapshots not recorded', fontsize=8)
            plt.tight_layout()
            plt.show(block=False)

        # Calculate the Fisher Information Matrix
        norm_flat_trimmed_theta = normalize(theta, hypothesis_index, roi_max, szx, szy, alpha, color_mode=color_mode)
        fisher_mat = calculate_fisher_information_matrix_vectorized(norm_flat_trimmed_theta, szy, szx, hypothesis_index, minimum_model_value, psf_sigma, roi_max, alpha, color_mode=color_mode)

        # Note: Xi[k]= log(likelihood(data; hypothesis)) - 1/2 * log(det(fisher_mat)) - We pick the maximum Xi to determine which hypothesis to choose.
        # -> Calculate the sum_loglikelihood (the sum of loglikelihoods of all pixels) and append it to lli (the loglikelihood list)
        if hypothesis_index == 0:
            roi_model = theta * np.ones((szy, szx))
        else:
            roi_model, _, _ = calculate_model_ipsfx_ipsfy(theta, range(szx), range(szy), hypothesis_index, minimum_model_value, psf_sigma)
        sum_loglikelihood = np.sum(roi_image * np.log(np.maximum(roi_model, 1e-2)) - roi_model - gammaln(roi_image + 1)) # gammaln(k+1) == ln(k!) (where ! denotes factorial).
        lli += [sum_loglikelihood] # Adding to the log-likelihood list
        # If the loglikelihood is infinite, assign np.nan to it.
        if np.isinf(lli[-1]):
            lli[-1] = np.nan
        
        # -> Calculate the log determinant of the Fisher Information Matrix
        try:
            _, log_det_fisher_mat = np.linalg.slogdet(fisher_mat)
        except np.linalg.LinAlgError as e:
            print(f"Error occurred during the calculation of log determinant of the Fisher Information Matrix: {e}, Assigning np.nan to the log_det_fisher_mat. roi_name: {roi_name}. Location: generalized_maximum_likelihood_rule()")
            log_det_fisher_mat = np.nan

        # Calculate the penalty term - using the log determinant of the Fisher Information Matrix
        penalty_laplace += [0.5 * log_det_fisher_mat] # Adding to the list
        # If the penalty term is infinite, assign np.nan to it.
        if np.isinf(penalty_laplace[-1]):
            penalty_laplace[-1] = np.nan
        # Calculate the AIC and BIC penalties (not likely to be used)
        penalty_aic += [2*num_parameters_for_current_model] 
        penalty_bic += [2*num_parameters_for_current_model * np.log(szy*szx)] 

        # Append the xi value to the list
        xi += [lli[-1] - penalty_laplace[-1]]
        xi_aic += [2 * lli[-1] - penalty_aic[-1]] # Not likely to be used
        xi_bic += [2 * lli[-1] - penalty_bic[-1]] # Not likely to be used
        # TODO: check if below is necessary
        # if penalty_laplace[hypothesis_index] < 0 and hypothesis_index == 0:
        #     xi += [lli[-1]]

        # Update the max_xi and xi_drop_count (used for premature exit condition)
        if len(xi) > 1 and xi[-1] < xi[-2]:
            xi_drop_count += 1
         
        if use_exit_condition and xi_drop_count >= 2:
            break

        fisher_info.append(fisher_mat)

    # -- End of the loop through all hypotheses

    # Store xi, lli and penalty_laplace to test_metric
    test_metrics = {
        'xi': xi,
        'xi_aic': xi_aic,
        'xi_bic': xi_bic,
        'lli': lli,
        'penalty': penalty_laplace,
        'penalty_aic': penalty_aic,
        'penalty_bic': penalty_bic,
        'fisher_info': fisher_info,
    }

    display_xi_graph = False 
    if display_xi_graph:
        max_xi_index = np.nanargmax(xi)
        _, axs = plt.subplots(3,1, figsize=(4.2,3.9))
        ax = axs[0]
        hypothesis_list_length = len(xi)
        ax.plot(range(hypothesis_list_length), xi, 'o-', color='purple')              
        ax.set_ylabel('xi\n(logL- penalty_laplace)')
        ax.axvline(x=max_xi_index, color='gray', linestyle='--')
        ax = axs[1]
        ax.plot(range(hypothesis_list_length), lli, 'o-', color='navy')
        ax.set_ylabel('loglikelihood')
        ax = axs[2]
        ax.axhline(y=0, color='black', linestyle='--')
        # ax.plot(range(last_h_index + 1), np.exp(penalty_laplace * 2), 'o-', color='crimson') 
        ax.plot(range(hypothesis_list_length), penalty_laplace, 'o-', color='crimson') 
        ax.set_ylabel('penalty_laplace')
        ax.set_xlabel('hypothesis_index')
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(f'scores.png')
        pass

    # Find the estimated number of particles
    qualifying_xi_indices = [i for i in range(len(xi))]
    if qualifying_xi_indices:
        estimated_num_particles = qualifying_xi_indices[np.nanargmax([xi[i] for i in qualifying_xi_indices])]
    else:
        estimated_num_particles = -1
    if plt.get_fignums():
        plt.close('all')

    return estimated_num_particles, fit_results, test_metrics 

    

def generalized_maximum_likelihood_rule_on_rgb(roi_image, psf_sigma, last_h_index=5, random_seed=0, display_fit_results=False, display_xi_graph=False, use_exit_condition=True):
    """ Perform the generalized maximum likelihood rule on the RGB image.
    Args:
        roi_image (np.array): The 3D RGB image to process.
        psf_sigma (float): The standard deviation of the PSF.
        last_h_index (int): The last hypothesis index to test.
        random_seed (int): The random seed to use.
        display_fit_results (bool): Whether to display the fit results.
        display_xi_graph (bool): Whether to display the xi graph.
        use_exit_condition (bool): Whether to use the exit condition.
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
    minimum_model_value = 1e-2
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

    # alpha
    alpha = [(roi_max_rgb[i] - roi_min_rgb[i]) * 2 * np.pi * psf_sigma**2 for i in range(3)]

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
        # num_parameters_for_current_model = 3 + 5 * hypothesis_index # Number of parameters for each hypothesis
        
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
            # assert num_parameters_for_current_model == 3
            convergence = True
            convergence_record = [True] # For H0, convergence is always True, as it is simple averaging.
        else:
            # Normazlize the parameters before passing on to neg_loglikelihood_function
            norm_flat_theta = normalize(theta, hypothesis_index, roi_max_rgb, szx, szy, alpha)

            # # Initialize storage for snapshots
            gradientnorm_snapshots = []
            fn_snapshots = []
            theta_snapshots = []
            denormflat_theta_snapshots = []

            def callback_fn(xk, *args):
                jac = jacobian_fn(xk, hypothesis_index, roi_image, roi_min_rgb, roi_max_rgb, minimum_model_value, psf_sigma, szx, szy, alpha)
                gradientnorm = np.linalg.norm(jac)
                fn = modified_neg_loglikelihood_fn(xk, hypothesis_index, roi_image, roi_min_rgb, roi_max_rgb, minimum_model_value, psf_sigma, szx, szy)
                gradientnorm_snapshots.append(gradientnorm)
                fn_snapshots.append(fn)                
                theta_snapshots.append(xk)

            # Update the parameters using scipy.optimize.minimize
            try:
                minimization_result = minimize(modified_neg_loglikelihood_fn, norm_flat_theta, args=(hypothesis_index, roi_image, roi_min_rgb, roi_max_rgb, minimum_model_value, psf_sigma, szx, szy), method=method, jac=jacobian_fn, hess=hessian_fn, callback=callback_fn, options={'gtol': 100})
            except Exception as e:
                print(f"Error occurred during optimization: {e}")

            snapshot_length = len(fn_snapshots)
            convergence = minimization_result.success
            norm_theta = minimization_result.x

            convergence_record.append(convergence)

            # Retrieve the estimated parameters.
            theta = denormalize(norm_theta, hypothesis_index, roi_max_rgb, szx, szy, alpha)           
                            
        # Store fit results
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
                    # ax_main[0][hypothesis_index].set_title(f"H{hypothesis_index} - convgd: {convergence_record[hypothesis_index]}\nbg: {theta['bg']:.1f}", fontsize=8)
                    bg_values = ", ".join([f"{value:.1f}" for value in theta['bg']])
                    ax_main[0][hypothesis_index].set_title(f'Bg: {bg_values}', fontsize=8)
                    ax_main[0][hypothesis_index].scatter(theta['particle'][particle_index - 1]['x'], theta['particle'][particle_index-1]['y'], s=10, color=color_code, marker='x')
                    ax_main[0][hypothesis_index].text(theta['particle'][particle_index - 1]['x'] + np.random.rand() * 1.5,       theta['particle'][particle_index - 1]['y'] + (np.random.rand() - 0.5) * 4,
                                                f"R:{theta['particle'][particle_index - 1]['I'][0]:.1f}\nG:{theta['particle'][particle_index - 1]['I'][1]:.1f}\nB:{theta['particle'][particle_index - 1]['I'][2]:.1f}", color='white', fontsize=6,) 
                else:

                    ax_main[0][hypothesis_index].set_title(f"H{hypothesis_index} - convgd: {convergence_record[hypothesis_index]}\nbg: {theta[0][0]:.1f}", fontsize=8)
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
