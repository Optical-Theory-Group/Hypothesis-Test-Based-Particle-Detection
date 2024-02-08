import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from scipy.stats import norm
from scipy.special import gammaln
from scipy.optimize import minimize
import random

def gaussianblur_max_min_2d(data, sigma):
    """ Returns the maximum and minimum values of the 2D Gaussian blurred image.
    Args:
        data (np.array): 2D numpy array representing the image.
        sigma (float): Standard deviation of the Gaussian.
    Returns:
        float: Maximum value of the Gaussian blurred image.
        float: Minimum value of the Gaussian blurred image.
    """

    assert data.shape[0] == data.shape[1]
    sz = data.shape[0]
        
    filteredpixel = 0 
    sum_val = 0  # renamed to avoid shadowing built-in sum function
    max_i = 0
    min_bg = 10e10
    norm = 1/2/sigma**2

    # The following two for loops are used to calculate the Gaussian blur of the image.
    for filter_center_x in range(sz):
        for filter_center_y in range(sz):
            filteredpixel = 0
            sum_val = 0
            for contributing_pos1 in range(sz):
                for contributing_pos2 in range(sz):
                    weight_contributing_pos1 = np.exp(-((contributing_pos1 - filter_center_x) ** 2) * norm)
                    weight_contributing_pos2 = np.exp(-((contributing_pos2 - filter_center_y) ** 2) * norm)
                    filteredpixel += weight_contributing_pos1 * weight_contributing_pos2 * data[contributing_pos1, contributing_pos2]
                    sum_val += weight_contributing_pos1 * weight_contributing_pos2

            filteredpixel /= sum_val
            
            # max_i is the maximum value of the Gaussian blurred image.
            max_i = max(max_i, filteredpixel)
            # min_bg is the minimum value of the Gaussian blurred image.
            min_bg = min(min_bg, filteredpixel)

    return max_i, min_bg

def integrate_gauss_1d(i, x, sigma):
    """ Compute the integral of the 1D Gaussian.
    Args:
        i (int): Pixel index.
        x (float): Mean value of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.
    Returns:
        float: Integral of the Gaussian from i-0.5 to i+0.5.
    """
    norm = 1/2/sigma**2
    # Below is the same as integral(from i-0.5 to i+0.5) [1/2sqrt(pi)*exp(-norm*(t-x)**2) dt]
    return 0.5*(erf((i-x+0.5)*np.sqrt(norm))-erf((i-x-0.5)*np.sqrt(norm)))

def derivative_int_gauss_1d(i, x, sigma, N, PSFy):
    """ Compute the derivative of the 1D Gaussian.
    Args:
        i (int): Pixel index.
        x (float): Mean value of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.
        N (float): Amplitude of the Gaussian.
        PSFy (float): Scaling factor along y.
    Returns:
        float: Derivative of the Gaussian with respect to x at i.
        float: Second derivative of the Gaussian with respect to x at i (optional).
    """
    a = np.exp(-0.5 * ((i + 0.5 - x) / sigma)**2)
    b = np.exp(-0.5 * ((i - 0.5 - x) / sigma)**2)

    dmodel_dt = -N / np.sqrt(2 * np.pi) / sigma * (a - b) * PSFy
    d2model_dt2 = -N / np.sqrt(2 * np.pi) / sigma**3 * ((i + 0.5 - x) * a - (i - 0.5 - x) * b) * PSFy

    return dmodel_dt, d2model_dt2

def center_of_mass_2d(data):
    """ Compute the 2D center of mass of a subregion.
    Args:
        data (np.array): 2D numpy array representing the image.
    Returns:
        float: x-coordinate of the center of mass.
        float: y-coordinate of the center of mass.
    """
    # The following two lines are used to calculate the center of mass of the image.
    i, j = np.indices(data.shape)
    tmpx = np.sum((data) * i)
    tmpy = np.sum((data) * j)
    tmpsum = np.sum(data)
    
    # Avoid division by zero
    if tmpsum == 0:
        return np.nan, np.nan
    
    # x and y are the x and y coordinates of the center of mass of the image.
    x = tmpx / tmpsum
    y = tmpy / tmpsum
    
    return x, y

def calculate_pfa(t_g):
    """ Returns the probability of false alarm (pfa) of deciding H1 when in fact H0 is true.
    Args:
        t_g (float): GRLT statistic.
    Returns:
        float: Probability of false alarm (pfa).
    """
    pfa = 2 * norm.cdf(-np.sqrt(max(t_g, 0.0)), 0, 1) # 0 and 1 are the mean and standard deviation of the normal distribution, respectively.

    return pfa

def fdr_bh(pvals, q=0.05, method='dep', report='no'):
    """ Returns the indices of the p-values that are significant using the Benjamini-Hochberg procedure.
    Args:
        pvals (ndarray): Array of p-values.
        q (float): False discovery rate.
        method (str): Method for the Benjamini-Hochberg procedure. 'pdep' for independent or positively dependent tests and 'dep' for any dependency structure.
        report (str): Whether to print the number of significant tests.
    Returns:
        tuple: A tuple containing the indices of the p-values that are significant, the critical p-value, and the adjusted p-values.
            - h (ndarray): Array of indices of the p-values that are significant.
            - crit_p (float): Critical p-value.
            - adj_p (ndarray): Array of adjusted p-values.
    """
    if np.any(pvals < 0):
        raise ValueError("Some p-values are less than 0.")
    if np.any(pvals > 1):
        raise ValueError("Some p-values are greater than 1.")

    pvals_shape = pvals.shape
    p_sorted = np.sort(pvals, axis=None)
    sort_ids = np.argsort(pvals, axis=None)
    m = len(p_sorted)

    if method == 'pdep':
        # BH procedure for independence or positive dependence
        thresh = (np.arange(1, m + 1) * q) / m
        wtd_p = m * p_sorted / np.arange(1, m + 1)
    elif method == 'dep':
        # BH procedure for any dependency structure
        denom = m * np.sum(1.0 / np.arange(1, m + 1))
        thresh = (np.arange(1, m + 1) * q) / denom
        wtd_p = denom * p_sorted / np.arange(1, m + 1)
    else:
        raise ValueError("Argument 'method' needs to be 'pdep' or 'dep'.")

    adj_p = np.full(m, np.nan)
    wtd_p_sorted = np.sort(wtd_p)
    wtd_p_sindex = np.argsort(wtd_p)
    nextfill = 0
    for k in range(m):
        if wtd_p_sindex[k] >= nextfill:
            adj_p[nextfill:wtd_p_sindex[k] + 1] = wtd_p_sorted[k]
            nextfill = wtd_p_sindex[k] + 1
            if nextfill >= m:
                break

    adj_p = adj_p[np.argsort(sort_ids)].reshape(pvals_shape)

    rej = p_sorted <= thresh
    max_id = np.max(np.where(rej)[0]) if np.any(rej) else -1

    if max_id == -1:
        crit_p = 0
        h = np.zeros_like(pvals)
    else:
        crit_p = p_sorted[max_id]
        h = pvals <= crit_p

    if report == 'yes':
        n_sig = np.sum(p_sorted <= crit_p)
        print(f'Out of {m} tests, {n_sig} are significant using a false discovery rate of {q}.')
        if method == 'pdep':
            print('FDR procedure used is guaranteed valid for independent or positively dependent tests.')
        else:
            print('FDR procedure used is guaranteed valid for independent or dependent tests.')

    return h, crit_p, adj_p


def glrt2(image, psf_sd, sz, iterations):
    """ Returns parameters, Cramer-Rao Lower Bounds (CRLBs), and statistics for the Gaussian Maximum Likelihood Estimation (MLE) test with two parameters of H1 (theta_h1[0] and theta_h1[1]).
    Args:
        image (ndarray): Input image.
        psf_sd (float): Standard deviation of the Point Spread Function (PSF).
        sz (int): Width of the image.
        iterations (int): Number of iterations for the MLE estimation.
    Returns:
        tuple: A tuple containing the estimated parameters, CRLBs, and statistics.
            - theta_h0 (ndarray): Estimated parameters for H0.
            - theta_h1 (ndarray): Estimated parameters for H1.
            - crlbs (ndarray): Cramer-Rao Lower Bounds for the estimated parameters.
            - pfa (float): Probability of false alarm (pfa) of deciding H1 when in fact H0 is true.
    """
    # Initialization
    n_h0_params = 1
    n_h1_params = 2 
    
    fisher_mat = np.zeros((n_h1_params, n_h1_params))
    inv_fisher_mat = np.zeros((n_h1_params, n_h1_params))

    theta_h1 = np.zeros(n_h1_params)
    theta_h0 = np.zeros(n_h0_params)

    # Starting values
    blurred_max, blurred_min = gaussianblur_max_min_2d(image, psf_sd)
    theta_h1[0] = (blurred_max - blurred_min) * 2 * np.pi * psf_sd**2
    theta_h1[0] = max(theta_h1[0], blurred_min)
    theta_h1[1] = blurred_min

    # Maximum Likelihood Estimation of H1
    for _ in range(iterations):
        # nr stands for Newton-Raphson
        nr_numerator = np.zeros(n_h1_params)
        nr_denominator = np.zeros(n_h1_params)
        for ii in range(sz): # sz: width of the image
            for jj in range(sz):
                # Calculates the integral of the normalized 1D psf function for x ranging in the ii-th column.
                psf_x = integrate_gauss_1d(ii, (sz-1) / 2, psf_sd) 
                # Calculates the integral of the normalized 1D psf function for y ranging in the jj-th row.
                psf_y = integrate_gauss_1d(jj, (sz-1) / 2, psf_sd)
                
                # modelh1 is the "model" value of the data at (ii, jj) under H1
                modelh1 = theta_h1[1] + theta_h1[0] * psf_x * psf_y 

                # Below is the "actual" pixel value of the image at (ii, jj)
                pixel_val = image[jj, ii]

                # first derivatives
                ddt_modelh1 = np.zeros(n_h1_params)
                # second derivatives
                d2dt2_modelh1 = np.zeros(n_h1_params)

                ddt_modelh1[0] = psf_x * psf_y  # Because, ddtheta_h1[0] modelh1 == psf_x * psf_y
                d2dt2_modelh1[0] = 0.0
                ddt_modelh1[1] = 1.0 
                d2dt2_modelh1[1] = 0.0

                # See https://doi.org/10.1364/OPEX.13.010503, Section 2.5 to better understand cf and df
                cf = 0.0
                df = 0.0
                if modelh1 > 0:
                    # TODO: Later check whether modelh1 values actually goes to zero or negative.
                    cf = pixel_val / modelh1 - 1
                    df = pixel_val / modelh1 ** 2
                else:
                    cf = 1e5
                    df = 1e5

                # Newton-Raphson update denominators and numerators
                for ll in range(n_h1_params): 
                    nr_numerator[ll] += ddt_modelh1[ll] * cf
                    nr_denominator[ll] += d2dt2_modelh1[ll] * cf - ddt_modelh1[ll] ** 2 * df

        # Parameter update
        theta_h1[0] -= min(max(nr_numerator[0] / nr_denominator[0] / 2, -theta_h1[0]), theta_h1[0]/2)
        theta_h1[0] = max(theta_h1[0], blurred_max/2)

        theta_h1[1] -= nr_numerator[1] / nr_denominator[1]
        theta_h1[1] = max(theta_h1[1], 0.01)

    # Maximum likelihood estimate of H0
    theta_h0[0] = image.sum() / sz**2

    # Calculate the t_g (GRLT statistic) and the Fisher Information Matrix
    t_g = 0.0
    for ii in range(sz):
        for jj in range(sz):
            # Calculates the integral of the normalized 1D psf function for x ranging in the ii-th column.
            psf_x = integrate_gauss_1d(ii, (sz-1) / 2, psf_sd)
            # Calculates the integral of the normalized 1D psf function for y ranging in the jj-th row.
            psf_y = integrate_gauss_1d(jj, (sz-1) / 2, psf_sd)

            # modelh1 is the "model" value of the data at (ii, jj) under H1
            modelh1 = theta_h1[1] + theta_h1[0] * psf_x * psf_y
            # modelh0 is the "model" value of the data at (ii, jj) under H0
            pixel_val = image[jj, ii]

            # Only the first derivatives are required for constructing the Fisher information matrix (FIM).
            ddt_modelh1 = np.zeros(n_h1_params)
            ddt_modelh1[0] = psf_x * psf_y
            ddt_modelh1[1] = 1.0

            # Building the Fisher Information Matrix regarding H1.
            for kk in range(n_h1_params):
                for ll in range(kk, n_h1_params):
                    # Using Poisson pdf for likelihood function, the following formula is derived.
                    # Ref: Smith et al. 2010, nmeth, SI eq (9).
                    fisher_mat[kk, ll] += ddt_modelh1[ll] * ddt_modelh1[kk] / modelh1
                    # The FIM is symmetric.
                    fisher_mat[ll, kk] = fisher_mat[kk, ll]

            # Estimated model value ratio
            modelh0 = theta_h0[0]
            ratio = modelh1 / modelh0
            if ratio > 0 and pixel_val > 0: 
                # Because the likelihood function is Poisson pdf, the following formula can be derived.
                # log likelihood ratio = pixel_val * ( log(model_h1/model_h0) - model_h1 + model_h0)
                t_g += 2 * (pixel_val * np.log(max(ratio, 1e-2)) - modelh1 + modelh0) 

    # Matrix inverse (CRLB=F^-1)
    inv_fisher_mat = np.linalg.inv(fisher_mat)
    crlbs = np.diag(inv_fisher_mat)
    
    pfa = 2 * norm.cdf(-np.sqrt(max(t_g, 0.0)), 0, 1) # 0 and 1 are the mean and standard deviation of the normal distribution, respectively.

    return theta_h0, theta_h1, crlbs, pfa

def glrt4(image, psf_sd, sz, iterations):
    """Returns parameters, Cramer-Rao Lower Bounds (CRLBs), and statistics for the Gaussian Maximum Likelihood Estimation (MLE) test with four parameters of H1 (theta_h1[0], theta_h1[1], theta_h1[2], theta_h1[3]).
    Args:
        image (ndarray): Input image.
        psf_sd (float): Standard deviation of the Point Spread Function (PSF).
        sz (int): Width of the image.
        iterations (int): Number of iterations for the MLE estimation.
    Returns:
        tuple: A tuple containing the estimated parameters, CRLBs, and statistics.
            - theta_h0 (ndarray): Estimated parameters for H0.
            - theta_h1 (ndarray): Estimated parameters for H1.
            - crlbs (ndarray): Cramer-Rao Lower Bounds for the estimated parameters.
            - pfa (float): Probability of false alarm (pfa) of deciding H1 when in fact H0 is true.
    """
    
    # initialization
    n_h0_params = 1
    n_h1_params = 4 

    fisher_mat = np.zeros((n_h1_params, n_h1_params))
    inv_fisher_mat = np.zeros((n_h1_params, n_h1_params))

    theta_h1 = np.zeros(n_h1_params)
    theta_h0 = np.zeros(n_h0_params)

    maxjump = np.array([1.0, 1.0, 100.0, 2.0])
    
    # Strating values 
    theta_h1[0], theta_h1[1] = center_of_mass_2d(image)
    blurred_max, blurred_min = gaussianblur_max_min_2d(image, psf_sd)
    theta_h1[2] = (blurred_max - blurred_min) * 2 * np.pi*psf_sd**2
    theta_h1[2] = max(theta_h1[2], blurred_min)
    theta_h1[3] = blurred_min

    # Maximum Likelihood Estimation of H1
    for _ in range(iterations):
        nr_numerator = np.zeros(n_h1_params)
        nr_denominator = np.zeros(n_h1_params)

        for ii in range(sz):
            for jj in range(sz):
                # Calculating the integral of the normalized 1D psf function for x ranging in the ii-th column.
                psf_x = integrate_gauss_1d(ii, theta_h1[0], psf_sd)
                # Calculating the integral of the normalized 1D psf function for y ranging in the jj-th row.
                psf_y = integrate_gauss_1d(jj, theta_h1[1], psf_sd)

                # Calculating the model value at (ii, jj) under H1
                modelh1 = theta_h1[3] + theta_h1[2] * psf_x * psf_y
                # Retrieving the "actual" pixel value of the image at (ii, jj)
                pixel_val = image[jj, ii]

                # Calculating derivatives
                ddt_modelh1 = np.zeros(n_h1_params)
                d2dt2_modelh1 = np.zeros(n_h1_params)

                # First and second derivatives of the model w.r.t. x, y, N, and bg
                ddt_modelh1[0], d2dt2_modelh1[0] = derivative_int_gauss_1d(ii, theta_h1[0], psf_sd, theta_h1[2], psf_y) 
                ddt_modelh1[1], d2dt2_modelh1[1] = derivative_int_gauss_1d(jj, theta_h1[1], psf_sd, theta_h1[2], psf_x) 
                ddt_modelh1[2] = psf_x * psf_y  # First derivative of model w.r.t. N
                d2dt2_modelh1[2] = 0.0 # Second derivative of model w.r.t. N
                ddt_modelh1[3] = 1.0  # First derivative of model w.r.t. bg
                d2dt2_modelh1[3] = 0.0 # Second derivative of model w.r.t. bg

                # See https://doi.org/10.1364/OPEX.13.010503, Section 2.5 to better understand cf and df
                cf = 0.0
                df = 0.0
                # if model > 10e-3:
                if modelh1 > 0:
                    cf = pixel_val / modelh1 - 1
                    df = pixel_val / modelh1 ** 2
                else:
                    cf = 1e5
                    df = 1e5

                # Newton-Raphson update denominators and numerators
                for ll in range(n_h1_params):
                    nr_numerator[ll] += ddt_modelh1[ll] * cf
                    nr_denominator[ll] += d2dt2_modelh1[ll] * cf - ddt_modelh1[ll]**2 * df

        # Parameter update, with maxjump control 
        for ll in range(n_h1_params):
            # theta_h1[ll] -= np.clip(nr_numerator[ll] / (nr_denominator[ll]), -maxjump[ll], maxjump[ll])
            theta_h1[ll] -= nr_numerator[ll] / (nr_denominator[ll])

        # Any other constraints
        theta_h1[2] = max(theta_h1[2], 1e-5) 
        theta_h1[3] = max(theta_h1[3], 1e-5)

    # Maximum likelihood estimate of background model
    theta_h0[0] = 0.0
    for ii in range(sz):
        for jj in range(sz):
            theta_h0[0] += image[jj, ii]
    theta_h0[0] = theta_h0[0] / sz**2

    # Calculate the CRLB and LogLikelihood
    t_g = 0.0
    for ii in range(sz):
        for jj in range(sz):
            # Calculating the integral of the normalized 1D psf function for x ranging in the ii-th column.
            psf_x = integrate_gauss_1d(ii, theta_h1[0], psf_sd)
            # Calculating the integral of the normalized 1D psf function for y ranging in the jj-th row.
            psf_y = integrate_gauss_1d(jj, theta_h1[1], psf_sd)

            # Calculating the model value at (ii, jj) under H1
            modelh1 = theta_h1[3] + theta_h1[2] * psf_x * psf_y
            # Retrieving the "actual" pixel value of the image at (ii, jj)
            pixel_val = image[jj, ii]

            # First derivatives of the model w.r.t. x, y, N, and bg
            ddt_modelh1[0], _ = derivative_int_gauss_1d(ii, theta_h1[0], psf_sd, theta_h1[2], psf_y) 
            ddt_modelh1[1], _ = derivative_int_gauss_1d(jj, theta_h1[1], psf_sd, theta_h1[2], psf_x) 
            ddt_modelh1[2] = psf_x * psf_y  
            ddt_modelh1[3] = 1.0  

            # Fisher Information Matrix calculation regarding H1
            for kk in range(n_h1_params):
                for ll in range(kk, n_h1_params):
                    # Using Poisson pdf for likelihood function, the following formula is derived.
                    # Ref: Smith et al. 2010, nmeth, SI eq (9).
                    fisher_mat[kk, ll] += ddt_modelh1[ll] * ddt_modelh1[kk] / modelh1
                    # The FIM is symmetric.
                    fisher_mat[ll, kk] = fisher_mat[kk, ll]

            # model
            # Estimated model value ratio
            modelh0 = theta_h0[0] 
            ratio = modelh1 / modelh0 
            if ratio > 0 and pixel_val > 0: 
                # Because the likelihood function is Poisson pdf, the following formula can be derived.
                # log likelihood ratio = pixel_val * ( log(model_h1/model_h0) - model_h1 + model_h0)
                t_g += 2 * (pixel_val * np.log(max(ratio, 1e-2)) - modelh1 + modelh0) 

    # Compute the CRLB as the inverse of the Fisher Information Matrix
    inv_fisher_mat = np.linalg.inv(fisher_mat)
    crlbs = np.diag(inv_fisher_mat)

    pfa = 2 * norm.cdf(-np.sqrt(max(t_g, 0.0)), 0, 1) # 0 and 1 are the mean and standard deviation of the normal distribution, respectively.

    return theta_h0, theta_h1, crlbs, pfa

def generalized_likelihood_ratio_test(roi_image, psf_sd, iterations=8, fittype=0):
    """ Returns parameters, Cramer-Rao Lower Bounds (CRLBs), and statistics for the Generalized Likelihood Ratio Test (GLRT).
    Args:
        roi_image (ndarray): Input image.
        psf_sd (float): Standard deviation of the Point Spread Function (PSF).
        iterations (int): Number of iterations for the MLE estimation.
        fittype (int): Type of fit. 0 for (Photons, Bg) and 1 for (x, y, Bg, Photons).
    Returns:
        tuple: A tuple containing the estimated parameters, CRLBs, and statistics.
            - est_params (ndarray): Estimated parameters.
            - crlbs (ndarray): Cramer-Rao Lower Bounds for the estimated parameters.
            - statistics (ndarray): Array of statistics calculated using the likelihood ratio test.
            - p_fa (ndarray): Array of p_fa's (probability of false alarm of deciding H1 when in fact H0 is true) calculated using the likelihood ratio test.
    """
    assert roi_image.ndim == 2
    sz = roi_image.shape[0] # sz_y == sz_x always.

    if fittype == 0:
        # fittype == 0:  Fits (particle_intensity,background_intensity) under H1 and (Bg) under H0 given psf_sd. 
        # est_params: theta_h1[0], theta_h1[1], theta_h0[0]
        h0_params, h1_params, crlbs, pfa = glrt2(roi_image, psf_sd, sz, iterations) 
    elif fittype == 1:
        # fittype=1:  Fits (x,y,bg,Photons) under H1 and (Bg) under H0 given psf_sd. 
        # est_params: theta_h1[0], theta_h1[1], theta_h1[2], theta_h1[3], theta_h0[0]
        h0_params, h1_params, crlbs, pfa = glrt4(roi_image, psf_sd, sz, iterations) 
        pass
    else:   
        pass
    return h0_params, h1_params, crlbs, pfa


def generalized_maximum_likelihood_rule(roi_image, mask, psf_sd):

    assert roi_image.ndim == 2
    szy, szx = roi_image.shape 
    
    """ Indexing rules
    - hypothesis_index: 0, 1, 2, ...    (H0, H1, H2, ...)
    - particle_index: 1, 2, 3, ...     (particle 1, particle 2, particle 3, ...)
    - param_type_index: 0, 1, 2        (intensity, x-coordinate, y-coordinate)
    """ 
    
    # MLE estimation of H1, H2, ... (where should it end?) 
    n_hk_params_per_particle = 3
    n_h0_params = 1

    # Test up to H4 for now (1/14/2024 Mo, temporary)
    max_particles_plus1 = 5
    
    # Initialize xi, the criterion for H_k
    xi = np.zeros(max_particles_plus1) 
    lli = np.zeros(max_particles_plus1) # log likelihood
    penalty_i = np.zeros(max_particles_plus1) # penalty term

    # roi_max and roi_min will be used to set the starting points for background and particle intensities.
    roi_max, roi_min = np.max(roi_image), np.min(roi_image) # let's forget about blurring and work with simple max and min for now.
    # scaling is the integrated value of a normalized 2D Gaussian with sigma = psf_sd for the center pixel.
    scaling = (0.5 * (erf(0.5 / (psf_sd * np.sqrt(2))) - erf(-0.5 / (psf_sd * np.sqrt(2)))))**2 

    # Figure showing parameter estimation results for each hypothesis.
    fig, ax = plt.subplots(1, max_particles_plus1, figsize=(2 * (max_particles_plus1), 2))
    plt.show(block=False)
    fig.suptitle('Parameter estimation results for each hypothesis')
    # variable to check whether the parameter estimation converged
    convergence = np.zeros(max_particles_plus1, dtype=bool)
    
    for hypothesis_index in range(max_particles_plus1): # hypothesis_index is also the number of particles. 
        # Initialization
        n_hk_params = n_h0_params + hypothesis_index * (n_hk_params_per_particle) #H0: 1, H1: 5, H2: 8, ...
        fisher_mat = np.zeros((n_hk_params, n_hk_params)) # Fisher Information Matrix

        # Initialize the theta (parameter) vector
        # theta_hk[0][0] will be the estimated background intensity. (However, if hypothesis_index == 0, theta_hk will just be a scalar, and equal the background intensity.)
        # theta_hk[1][0] will be the estimated center-pixel intensity of particle 1.
        # theta_hk[1][1], theta_hk[1][2] will be the estimated x and y coordinate of particle 1, etc.
        # Since background intensity is the only parameter for H0, theta_hk[0][1] and theta_hk[0][2] will be nan and, importantly, not be passed for optimization.
        if hypothesis_index == 0:
            theta_hk = 0.0
        else:
            theta_hk = np.zeros((hypothesis_index + 1, n_hk_params_per_particle)) 

        # Starting values
        if hypothesis_index == 0:
            assert isinstance(theta_hk, (int, float))
            theta_hk = roi_image.sum() / szx / szy

        else: # Initializing estimated particle_intensities
            assert theta_hk.ndim == 2

            # Set the background first
            theta_hk[0][0] = roi_min
            theta_hk[0][1] = theta_hk[0][2] = np.nan

            for particle_index in range(1, hypothesis_index + 1): # Note that the particle index starts from 1, not 0. 
                # Initialize estimated particle intensities to the maximum value of the Gaussian roi image.
                theta_hk[particle_index][0] = roi_max - roi_min
            
            com_x, com_y = center_of_mass_2d(roi_image)
            # Initialize all particle coordinates as the center of mass of the image.
            for particle_index in range(1, hypothesis_index + 1):
                theta_hk[particle_index][1] = np.clip(com_x + np.random.normal(0, szx / 5), 0, szx - 1)
                theta_hk[particle_index][2] = np.clip(com_y + np.random.normal(0, szy / 5), 0, szy - 1)
                
        # Only do the MLE if k > 0
        if hypothesis_index == 0:
            assert n_hk_params == 1
        else:
            # Maximum Likelihood Estimation of Hk
            def modified_neg_loglikelihood_fn(norm_flat_trimmed_theta):
                # Put nan back in
                normalized_theta_hk = np.insert(norm_flat_trimmed_theta, [1,1], np.nan)
                normalized_theta_hk = np.reshape(normalized_theta_hk, (-1, 3))
                theta_hk = np.zeros((hypothesis_index + 1, n_hk_params_per_particle))
                theta_hk[0][0] = normalized_theta_hk[0][0] * roi_min
                theta_hk[0][1] = theta_hk[0][2] = np.nan
                for particle_index in range(1, hypothesis_index + 1):
                    theta_hk[particle_index][0] = normalized_theta_hk[particle_index][0] * (roi_max - roi_min) + roi_min
                    theta_hk[particle_index][1] = normalized_theta_hk[particle_index][1] * szx
                    theta_hk[particle_index][2] = normalized_theta_hk[particle_index][2] * szy
                modified_neg_loglikelihood = 0.0
                for yy in range(szy):
                    for xx in range(szx):
                        # Initialize the psf_x and psf_y arrays
                        integrated_psf_x = np.zeros(hypothesis_index + 1)
                        g_y = np.zeros(hypothesis_index + 1)
                        # integrated_psf_x[0] and psf_yx[0] are nans as they are not used. 
                        integrated_psf_x[0] = g_y[0] = np.nan
                        # model_xy = bg + (i_0 * psf_x_0 * psf_y_0) + (i_1 * psf_x_1 * psf_y_1) + ...
                        model_xy = theta_hk[0][0]
                        for particle_index in range(1, hypothesis_index + 1):
                            # Calculate the integral of the normalized 1D psf function for x and y, for the xx-th column and the yy-th row.
                            integrated_psf_x[particle_index] = integrate_gauss_1d(xx, theta_hk[particle_index][1], psf_sd)
                            g_y[particle_index] = integrate_gauss_1d(yy, theta_hk[particle_index][2], psf_sd)
                            # update the particles contributions to the model_xy value
                            model_xy += theta_hk[particle_index][0] / scaling * integrated_psf_x[particle_index] * g_y[particle_index] 
                        modified_neg_loglikelihood += model_xy - roi_image[yy,xx] * np.log(model_xy) # modified: omit log(roi_image[yy,xx]!)
                return modified_neg_loglikelihood

            # Now, let's update the parameters using scipy.optimize.minimize
            set_bounds = True
            if set_bounds:
                bounds = [(0, None)]
                for particle_index in range(1, hypothesis_index + 1):
                    bounds += [(0.0, None), (0, (szx-1)/szx), (0, (szy-1)/szy)]
                    
            # Normazlize the parameters before passing on to neg_loglikelihood_function
            norm_theta_hk = np.zeros((hypothesis_index + 1, n_hk_params_per_particle))
            for particle_index in range(0, hypothesis_index + 1):
                norm_theta_hk[particle_index][0] = (theta_hk[particle_index][0] - roi_min) / (roi_max - roi_min)
                if particle_index != 0:
                    norm_theta_hk[particle_index][1] = theta_hk[particle_index][1] / szx
                    norm_theta_hk[particle_index][2] = theta_hk[particle_index][2] / szy
                else:
                    norm_theta_hk[particle_index][1] = np.nan
                    norm_theta_hk[particle_index][2] = np.nan
                
            # Manipulate theta to use with scipy.optimize.minimize
            norm_flat_theta = norm_theta_hk.flatten()
            norm_flat_trimmed_theta = norm_flat_theta[~np.isnan(norm_flat_theta)]

            # What goes in front of the normalized 2d gaussian is intensity (e.g., 3000)
            # What gets recorded as theta_hk[particle_index][0] is the center pixel value, which is {intensity * scaling}. (e.g., 3000 * 0.08 = 240)
            # Then, what goes in front of the normalized 2d gaussian is {theta_hk[particle_index][0] / scaling}. (e.g., 240 / 0.08 = 3000)
          
            def jacobian_fn(norm_flat_trimmed_theta):
                """ """
                # The function to minimize is modified_neg_loglikelihood.
                # Thus the jacobian is the derivative of modified_neg_loglikelihood.
                # modified_neg_loglikelihood = (sum over xx and yy) [ model - pixel_val * log(model) ]
                #                   where model == theta_hk[0][0] + theta_hk[1][0] / scaling * integrated_psf_x[1] * g_y[1] + ...
                # ddt(negloglikelihood) [when t == norm_theta[0][0]] = d(nll)/d(t00) * d(t00) / d(norm_t00)
                #                            = (sum over xx and yy) [ (1 - pixel_val / model) * (roi_max - roi_min) ]       # (because, t00 = norm_t00 * (roi_max - roi_min) + roi_min)
                # ddt(negloglikelihood) [when t == norm_t10] = (sum over xx and yy) {1/scaling * psf_x[1] * psf_y[1] * (1 - pixel_val / model} * (roi_max - roi_min)
                # ddt(negloglikelihood) [when t == norm_t11] = (sum over xx and yy) [ derivative_int_gauss_1d(xx, theta_hk[1][1], psf_sd, theta_hk[1][0] / scaling, integrated_psf_y[1])[0] \
                    # * (1 - pixel_val / model) * szx          # ( Because, d(t11) / d(norm_t11) == szx )
                # ddt(negloglikelihood) [when t == norm_t12] = (sum over xx and yy) [ derivative_int_gauss_1d(yy, theta_hk[1][2], psf_sd, theta_hk[1][0] / scaling, integrated_psf_x[1])[0] \
                    # * (1 - pixel_val / model) * szy          # ( Because, d(t12) / d(norm_t12) == szy )
                flat_norm_theta_hk = np.insert(norm_flat_trimmed_theta, [1,1], np.nan)
                norm_theta_hk = np.reshape(flat_norm_theta_hk, (-1, 3))
                for particle_index in range(0, hypothesis_index + 1):
                    theta_hk[particle_index][0] = norm_theta_hk[particle_index][0] * (roi_max - roi_min) + roi_min
                    if particle_index != 0:
                        theta_hk[particle_index][1] = norm_theta_hk[particle_index][1] * szx
                        theta_hk[particle_index][2] = norm_theta_hk[particle_index][2] * szy
                # nll: negloglikelihood
                ddt_nll = np.zeros((hypothesis_index + 1, 3)) # 3 --> 1 for intensity, 2 for coordinates
                for yy in range(szy):
                    for xx in range(szx):
                        pixel_val = roi_image[yy, xx]
                        model_xy = 0.0
                        model_xy += theta_hk[0][0] 
                        integrated_psf_x = np.zeros(hypothesis_index + 1)
                        integrated_psf_y = np.zeros(hypothesis_index + 1)
                        # integrated_psf_x[0] and integrated_psf_y[0] are nan as they are not used. 
                        integrated_psf_x[0] = np.nan
                        integrated_psf_y[0] = np.nan
                        for particle_index in range(1, hypothesis_index + 1):
                            # Calculate the integral of the normalized 1D psf function for x and y, for the xx-th column and the yy-th row.
                            integrated_psf_x[particle_index] = integrate_gauss_1d(xx, theta_hk[particle_index][1], psf_sd)
                            integrated_psf_y[particle_index] = integrate_gauss_1d(yy, theta_hk[particle_index][2], psf_sd)
                            # update the particles contributions to the model_xy value
                            model_xy += theta_hk[particle_index][0] / scaling * integrated_psf_x[particle_index] * integrated_psf_y[particle_index] # particle intensity contribution at (xx, yy)
                        # Calculate the first derivatives of the negloglikelihood
                        ddt_nll[0][0] += (1 - pixel_val / model_xy) # * (roi_max - roi_min)
                        ddt_nll[0][1] = ddt_nll[0][2] = np.nan
                        for p_idx in range(1, hypothesis_index + 1):
                            ddt_nll[p_idx][0] += (1/scaling * integrated_psf_x[p_idx] * integrated_psf_y[p_idx]) * (1 - pixel_val / model_xy) * (roi_max - roi_min)
                            ddt_nll[p_idx][1] += derivative_int_gauss_1d(xx, theta_hk[p_idx][1], psf_sd, theta_hk[p_idx][0] / scaling, integrated_psf_y[p_idx])[0] * (1 - pixel_val / model_xy) * szx
                            ddt_nll[p_idx][2] += derivative_int_gauss_1d(yy, theta_hk[p_idx][2], psf_sd, theta_hk[p_idx][0] / scaling, integrated_psf_x[p_idx])[0] * (1 - pixel_val / model_xy) * szy
                jacobian = ddt_nll.flatten()
                jacobian = jacobian[~np.isnan(jacobian)]
                # Check the shape of the gradient
                if jacobian.shape != norm_flat_trimmed_theta.shape:
                    # Reshape the gradient to have the same shape as norm_flat_trimmed_theta
                    jacobian = jacobian.reshape(norm_flat_trimmed_theta.shape)
                return jacobian

            def hessian_fn(flat_norm_trimmed_theta):
                """"""
                theta_hk = np.zeros((hypothesis_index + 1, 3))
                norm_theta_hk = np.reshape(flat_norm_trimmed_theta, (-1, 3))
                for particle_index in range(0, hypothesis_index + 1):
                    theta_hk[particle_index][0] = norm_theta_hk[particle_index][0] * (roi_max - roi_min) + roi_min
                    if particle_index != 0:
                        theta_hk[particle_index][1] = norm_theta_hk[particle_index][1] * szx
                        theta_hk[particle_index][2] = norm_theta_hk[particle_index][2] * szy
                # nll: negloglikelihood
                d2dt2_nll_2d = np.zeros((hypothesis_index * 3 + 1), (hypothesis_index * 3 + 1))

                for yy in range(szy):
                    for xx in range(szx):
                        pixel_val = roi_image[yy, xx]
                        model_xy = 0.0
                        model_xy += theta_hk[0][0] 
                        integrated_psf_x = np.zeros(hypothesis_index + 1)
                        integrated_psf_y = np.zeros(hypothesis_index + 1)
                        # integrated_psf_x[0] and integrated_psf_y[0] are nan as they are not used. 
                        integrated_psf_x[0] = np.nan
                        integrated_psf_y[0] = np.nan
                        for particle_index in range(1, hypothesis_index + 1):
                            # Calculate the integral of the normalized 1D psf function for x and y, for the xx-th column and the yy-th row.
                            integrated_psf_x[particle_index] = integrate_gauss_1d(xx, theta_hk[particle_index][1], psf_sd)
                            integrated_psf_y[particle_index] = integrate_gauss_1d(yy, theta_hk[particle_index][2], psf_sd)
                            # update the particles contributions to the model_xy value
                            model_xy += theta_hk[particle_index][0] / scaling * integrated_psf_x[particle_index] * integrated_psf_y[particle_index] # particle intensity contribution at (xx, yy)

                        # Calculate the double derivatives of the negloglikelihood
                        # 1. 00,00 ##
                        d2dt2_nll_00_00 += pixel_val / model_xy**2 * (roi_max - roi_min)**2
                        d2dt2_nll_2d[0][0] = d2dt2_nll_00_00
                        for pidx in range(1, hypothesis_index + 1):
                            # 2. i0, 00
                            d2dt2_nll_i0_00 += pixel_val / model_xy**2 / scaling * integrated_psf_x[pidx] * integrated_psf_y[pidx] * (roi_max - roi_min)**2
                            d2dt2_nll_2d[0][(pidx - 1) * 3 + 1] = d2dt2_nll_i0_00
                            d2dt2_nll_2d[(pidx - 1) * 3 + 1][0] = d2dt2_nll_i0_00
                            # 3. i1, 00
                            d2dt2_nll_i1_00 += pixel_val / model_xy**2 * derivative_int_gauss_1d(xx, theta_hk[pidx][1], psf_sd, theta_hk[pidx][0] / scaling, integrated_psf_y[pidx])[0] * szx * (roi_max - roi_min)
                            d2dt2_nll_2d[0][(pidx - 1) * 3 + 2] = d2dt2_nll_i1_00
                            d2dt2_nll_2d[(pidx - 1) * 3 + 2][0] = d2dt2_nll_i1_00
                            # 4. i2, 00
                            d2dt2_nll_i2_00 += pixel_val / model_xy**2 * derivative_int_gauss_1d(yy, theta_hk[pidx][2], psf_sd, theta_hk[pidx][0] / scaling, integrated_psf_x[pidx])[0] * szy * (roi_max - roi_min)
                            d2dt2_nll_2d[0][(pidx - 1) * 3 + 3] = d2dt2_nll_i2_00
                            d2dt2_nll_2d[(pidx - 1) * 3 + 3][0] = d2dt2_nll_i2_00
                            # 5. i0, i0 ##
                            d2dt2_nll_i0_i0 += pixel_val / model_xy**2 * (1 / scaling * integrated_psf_x[pidx] * integrated_psf_y[pidx])**2 * (roi_max - roi_min)**2
                            d2dt2_nll_2d[(pidx - 1) * 3 + 1][(pidx - 1) * 3 + 1] = d2dt2_nll_i0_i0
                            # 6. i1, i0
                            d2dt2_nll_i1_i0 += (pixel_val / model_xy**2 * integrated_psf_x[pidx] * integrated_psf_y[pidx] / scaling + (1 - pixel_val / model_xy) / theta_hk[pidx][0]) * derivative_int_gauss_1d(xx, theta_hk[pidx][1], psf_sd, theta_hk[pidx][0] / scaling, integrated_psf_y[pidx])[0] * szx * (roi_max - roi_min)
                            d2dt2_nll_2d[(pidx - 1) * 3 + 2][(pidx - 1) * 3 + 1] = d2dt2_nll_i1_i0
                            d2dt2_nll_2d[(pidx - 1) * 3 + 1][(pidx - 1) * 3 + 2] = d2dt2_nll_i1_i0
                            # 7. i2, i0
                            d2dt2_nll_i2_i0 += (pixel_val / model_xy**2 * integrated_psf_x[pidx] * integrated_psf_y[pidx] / scaling + (1 - pixel_val / model_xy) / theta_hk[pidx][0]) * derivative_int_gauss_1d(yy, theta_hk[pidx][2], psf_sd, theta_hk[pidx][0] / scaling, integrated_psf_x[pidx])[0] * szy * (roi_max - roi_min)
                            d2dt2_nll_2d[(pidx - 1) * 3 + 3][(pidx - 1) * 3 + 1] = d2dt2_nll_i2_i0
                            d2dt2_nll_2d[(pidx - 1) * 3 + 1][(pidx - 1) * 3 + 3] = d2dt2_nll_i2_i0
                            # 8. i1, i1 ##
                            d2dt2_nll_i1_i1 += (1 - pixel_val / model_xy) * derivative_int_gauss_1d(xx, theta_hk[pidx][1], psf_sd, theta_hk[pidx][0] / scaling, integrated_psf_y[pidx])[1] * szx**2
                            d2dt2_nll_2d[(pidx - 1) * 3 + 2][(pidx - 1) * 3 + 2] = d2dt2_nll_i1_i1
                            # 9. i2, i1
                            d2dt2_nll_i2_i1 += (pixel_val / model_xy**2 + (1 - pixel_val / model_xy) * scaling / theta_hk[pidx][0] / integrated_psf_x[pidx] * integrated_psf_y[pidx]) * derivative_int_gauss_1d(xx, theta_hk[pidx][1], psf_sd, theta_hk[pidx][0] / scaling, integrated_psf_y[pidx])[0] * derivative_int_gauss_1d(yy, theta_hk[pidx][2], psf_sd, theta_hk[pidx][0] / scaling, integrated_psf_x[pidx])[0] * szx * szy
                            d2dt2_nll_2d[(pidx - 1) * 3 + 3][(pidx - 1) * 3 + 2] = d2dt2_nll_i2_i1
                            d2dt2_nll_2d[(pidx - 1) * 3 + 2][(pidx - 1) * 3 + 3] = d2dt2_nll_i2_i1
                            # 10. i2, i2 ##
                            d2dt2_nll_i2_i2 += (1 - pixel_val / model_xy) * derivative_int_gauss_1d(yy, theta_hk[pidx][2], psf_sd, theta_hk[pidx][0] / scaling, integrated_psf_x[pidx])[1] * szy**2
                            d2dt2_nll_2d[(pidx - 1) * 3 + 3][(pidx - 1) * 3 + 3] = d2dt2_nll_i2_i2
                return d2dt2_nll_2d

            # Now, let's update the parameters using scipy.optimize.minimize
            if set_bounds:
                if np.isnan(norm_flat_trimmed_theta[1]):
                    break
                result = minimize(modified_neg_loglikelihood_fn, norm_flat_trimmed_theta, method='SLSQP', bounds=bounds, jac=jacobian_fn, hess=hessian_fn)#, options={'ftol': 1e-9}) 
            else: 
                # result = minimize(modified_neg_loglikelihood_fn, norm_flat_trimmed_theta, method='L-BFGS-B')#, options={'disp':True})
                pass
            print(f'H{hypothesis_index} converged?: {result.success}')
            convergence[hypothesis_index] = result.success
            norm_theta_hk = result.x
            # Put back the np.nan's
            norm_theta_hk = np.insert(norm_theta_hk, [1,1], np.nan)
            norm_theta_hk = np.reshape(norm_theta_hk, (-1, 3))

            # Now, let's denormalize the parameters
            for particle_index in range(0, hypothesis_index + 1):
                theta_hk[particle_index][0] = norm_theta_hk[particle_index][0] * (roi_max - roi_min) + roi_min
                if particle_index != 0:
                    theta_hk[particle_index][1] = norm_theta_hk[particle_index][1] * szx
                    theta_hk[particle_index][2] = norm_theta_hk[particle_index][2] * szy
                            
        # Print the estimated parameters
        print(f'*** hypothesis_index: {hypothesis_index}')
        if hypothesis_index == 0 :
            print("theta_hk[ 0 ]: ", theta_hk)
            ax[0].imshow(roi_image)
            ax[0].set_title("H0")
            plt.show(block=False)
        else:
            print("theta_hk[ 0 ]: {:.3f}\tnan\tnan".format(theta_hk[particle_index][0]))
            # Marking each particle with a red cross
            for particle_index in range(1, hypothesis_index + 1):
                print(f"theta_hk[ {particle_index} ]: {theta_hk[particle_index][0]:.3f}\t{theta_hk[particle_index][1]:.3f}\t{theta_hk[particle_index][2]:.3f}")
                ax[hypothesis_index].imshow(roi_image)
                ax[hypothesis_index].set_title(f"H{hypothesis_index} - convgd: {convergence[hypothesis_index]}")
                red = random.randint(200, 255)
                green = random.randint(0, 100)
                blue = random.randint(0, 50)
                color_code = '#%02X%02X%02X' % (red, green, blue)
                ax[hypothesis_index].scatter(theta_hk[particle_index][1], theta_hk[particle_index][2], s=100, color=color_code, marker='x')
                ax[hypothesis_index].text(theta_hk[particle_index][1] + np.random.rand() * 1.5, theta_hk[particle_index][2] + (np.random.rand() - 0.5) * 4,
                                            f'  {theta_hk[particle_index][0]:.1f}', color=color_code, fontsize=10,) 
                plt.tight_layout()
                plt.pause(0.1)
                pass
        plt.show(block=False)

        # Calcuate the Fisher Information Matrix (FIM)
        # All iterations finished. Now, let's calculate the Fisher Information Matrix (FIM) under Hk.
        if hypothesis_index == 0:
            if np.var(roi_image) == 0:
                fisher_mat[0,0] = 1 / 1e-5
            else:
                fisher_mat[0,0] = 1 / np.var(roi_image)
            assert fisher_mat.shape == (1,1)
        else:
            for yy in range(szy):
                for xx in range(szx):
                    # Let's get the actual pixel value, again.
                    pixel_val = roi_image[yy, xx]
                    
                    # Initialize the first derivatives (for calulculating FIM, there is no need for second derivatives)
                    ddt_modelhk_at_xxyy = np.zeros((hypothesis_index + 1, n_hk_params_per_particle))

                    # Initialize the psf_x and psf_y arrays
                    integrated_psf_x = np.zeros(hypothesis_index + 1)
                    integrated_psf_y = np.zeros(hypothesis_index + 1)
                    # integrated_psf_x[0] is nan as it is not used. integrated_psf_x[1] is the psf_x of particle 1.
                    integrated_psf_x[0] = np.nan
                    # integrated_psf_y[0] is nan as it is not used. integrated_psf_y[1] is the psf_y of particle 1.
                    integrated_psf_y[0] = np.nan

                    # Calculating the model value at (xx, yy) under Hk.
                    # modelhk_at_xxyy = background + (i_0 * psf_x_0 * psf_y_0) + (i_1 * psf_x_1 * psf_y_1) + ...
                    modelhk_at_xxyy = 0.0
                    modelhk_at_xxyy += roi_min
                    
                    for particle_index in range(1, hypothesis_index + 1):
                        # Calculate the integral of the normalized 1D psf function for x and y, for the xx-th column and the yy-th row.
                        integrated_psf_x[particle_index] = integrate_gauss_1d(xx, theta_hk[particle_index][1], psf_sd)
                        integrated_psf_y[particle_index] = integrate_gauss_1d(yy, theta_hk[particle_index][2], psf_sd)

                        # update the particles contributions to the modelhk_at_xxyy value
                        modelhk_at_xxyy += theta_hk[particle_index][0] * integrated_psf_x[particle_index] * integrated_psf_y[particle_index] # particle intensity contribution at (xx, yy)
                        
                    # Now, let's calculate the derivatives 

                    # -- Below are special treatmenst for the [0]'s index (the background intensity)
                    # Derivtive w.r.t background (the first index [0] refers to background and the last index [0] the intensity)
                    ddt_modelhk_at_xxyy[0][0] = 1.0
                    # Below are set as nan, as they are not used. (background does not have x and y coordinates)
                    ddt_modelhk_at_xxyy[0][1] = ddt_modelhk_at_xxyy[0][2] = np.nan

                    # -- Below are special treatmenst for the [1]'s index and beyond (related to the particle intensities and coordinates)
                    for particle_index in range(1, hypothesis_index + 1):
                        
                        # Calculate derivatives w.r.t particle[particle_index]'s intensity
                        ddt_modelhk_at_xxyy[particle_index][0] = integrated_psf_x[particle_index] * integrated_psf_y[particle_index]

                        # Calculate derivatives w.r.t particle[particle_index]'s x coordinate
                        first, _ = derivative_int_gauss_1d(xx, theta_hk[particle_index][1], psf_sd, theta_hk[particle_index][0], integrated_psf_x[particle_index])
                        ddt_modelhk_at_xxyy[particle_index][1] = first

                        # Calculate derivatives w.r.t particle[particle_index]'s y coordinate
                        first, _ = derivative_int_gauss_1d(yy, theta_hk[particle_index][2], psf_sd, theta_hk[particle_index][0], integrated_psf_y[particle_index])
                        ddt_modelhk_at_xxyy[particle_index][2] = first
                        
                    # Calculate the Fisher Information Matrix (FIM) under Hk.
                    assert fisher_mat.shape == (n_hk_params, n_hk_params)

                    # Building the Fisher Information Matrix regarding Hk.
                    # - Calculation with regards to the background 
                    fisher_mat[0, 0] += ddt_modelhk_at_xxyy[0][0] ** 2 / modelhk_at_xxyy
                    for kk in range(1, hypothesis_index * n_hk_params_per_particle + 1):
                        # convert kk to particle_index and param_type
                        particle_index = (kk - 1) // n_hk_params_per_particle + 1
                        param_type = (kk - 1) % n_hk_params_per_particle
                        # Using Poisson pdf for likelihood function, the following formula is derived. (Ref: Smith et al. 2010, nmeth, SI eq (9)).
                        fisher_mat[0, kk] += ddt_modelhk_at_xxyy[0][0] * ddt_modelhk_at_xxyy[particle_index][param_type] / modelhk_at_xxyy
                        fisher_mat[kk, 0] = fisher_mat[0, kk] # The FIM is symmetric.

                    # - Calculation with regards to the particles
                    for kk in range(1, hypothesis_index * n_hk_params_per_particle + 1):
                        # convert kk to particle_index and param_type
                        particle_index_kk = (kk - 1) // n_hk_params_per_particle + 1
                        param_type_kk = (kk - 1) % n_hk_params_per_particle
                        for ll in range(1, hypothesis_index * n_hk_params_per_particle + 1):
                            # convert kk to particle_index and param_type
                            particle_index_ll = (ll - 1) // n_hk_params_per_particle  + 1
                            param_type_ll = (ll - 1) % n_hk_params_per_particle
                            fisher_mat[kk, ll] += ddt_modelhk_at_xxyy[particle_index_kk][param_type_kk] * ddt_modelhk_at_xxyy[particle_index_ll][param_type_ll] / modelhk_at_xxyy
                            fisher_mat[ll, kk] = fisher_mat[kk, ll] # The FIM is symmetric.

        # Now I got the FIM under Hk. Let's use this to calculate the Xi_k (GMLR criterion)
        # Xi[k] = log(likelihood(data; MLE params under Hk)) - 1/2 * log(det(FIM under Hk))

        # -- Let's calculate the first term of the Xi_k (GMLR criterion)
        # sum_loglikelihood is the sum of loglikelihoods of all pixels
        sum_loglikelihood = 0.0 
        for yy in range(szy):
            for xx in range(szx):
                # Let's get the actual pixel value
                pixel_val = roi_image[yy, xx]
                
                # modelhk_at_xxyy = background + (i_0 * psf_x_0 * psf_y_0) + (i_1 * psf_x_1 * psf_y_1) + ...
                modelhk_at_xxyy = 0.0
                modelhk_at_xxyy += roi_min

                for particle_index in range(1, hypothesis_index + 1):
                    # Calculate the integral of the normalized 1D psf function for x and y, for the xx-th column and the yy-th row.
                    psf_x = integrate_gauss_1d(xx, theta_hk[particle_index][1], psf_sd)
                    psf_y = integrate_gauss_1d(yy, theta_hk[particle_index][2], psf_sd)

                    # Update the particles contributions to the modelhk_at_xxyy value
                    modelhk_at_xxyy += theta_hk[particle_index][0] * psf_x * psf_y # particle intensity contribution at (xx, yy)

                # We not have the model value at (xx, yy) under Hk. Let's calculate the loglikelihood.
                # likelihood = modelhk_at_xxyy**pixel_val * exp(-modelhk_at_xxyy) / factorial(pixel_val)
                # loglikelihood = pixel_val * np.log(max(modelhk_at_xxyy, 1e-2)) - modelhk_at_xxyy - pixel_val * np.log(pixel_val) + pixel_val
                # modified_loglikelihood = pixel_val * np.log(max(modelhk_at_xxyy, 1e-2)) - modelhk_at_xxyy # - pixel_val * np.log(pixel_val)
                # loglikelihood = pixel_val * np.log(max(modelhk_at_xxyy, 1e-2)) - modelhk_at_xxyy - pixel_val * np.log(pixel_val) + pixel_val
                loglikelihood = pixel_val * np.log(max(modelhk_at_xxyy, 1e-2)) - modelhk_at_xxyy - gammaln(pixel_val + 1)
                # sum_loglikelihood += modified_loglikelihood
                sum_loglikelihood += loglikelihood
        
        # Let's calculate the second term of the Xi_k (GMLR criterion), which is -1/2 * log(det(FIM under Hk))
        _, log_det_fisher_mat = np.linalg.slogdet(fisher_mat)

        # xi[hypothesis_index] = sum_loglikelihood - 0.5 * log_det_fisher_mat 
        xi[hypothesis_index] = sum_loglikelihood - 0.5 * log_det_fisher_mat 
        lli[hypothesis_index] = sum_loglikelihood
        penalty_i[hypothesis_index] = 0.5 * log_det_fisher_mat

        print(f'{sum_loglikelihood=:.2f}, {log_det_fisher_mat=:.2f}')
        # xi[hypothesis_index] = sum_loglikelihood + 0.5 * log_det_fisher_mat * 100

    _, axs = plt.subplots(3,1, figsize=(4.2,3.5))
    ax = axs[0]
    ax.plot(range(max_particles_plus1), xi, 'o-', color='purple')              
    ax.set_ylabel('loglikelihood\n - penalty')
    ax = axs[1]
    ax.plot(range(max_particles_plus1), lli, 'o-', color='navy')
    ax.set_ylabel('loglikelihood')
    ax = axs[2]
    # ax.plot(range(max_particles_plus1), np.exp(penalty_i * 2), 'o-', color='crimson') 
    ax.plot(range(max_particles_plus1), penalty_i, 'o-', color='crimson') 
    ax.set_ylabel('penalty')
    ax.set_xlabel('hypothesis_index')
    plt.show(block=False)
    plt.tight_layout()
    pass