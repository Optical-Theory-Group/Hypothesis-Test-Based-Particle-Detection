import numpy as np
from scipy.special import erf
from scipy.stats import norm

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
    return 1.0/2.0*(erf((i-x+0.5)*np.sqrt(norm))-erf((i-x-0.5)*np.sqrt(norm)))

def calculate_pfa(t_g):
    """ Returns the probability of false alarm (pfa) of deciding H1 when in fact H0 is true.
    Args:
        t_g (float): GRLT statistic.
    Returns:
        float: Probability of false alarm (pfa).
    """
    pfa = 2 * norm.cdf(-np.sqrt(max(t_g, 0.0)), 0, 1) # 0 and 1 are the mean and standard deviation of the normal distribution, respectively.

    return pfa

def gaussian_mle_test_two_params(image, psf_sd, sz, iterations):
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
    numvar_h0 = 1
    numvar_h1 = 2 
    
    fisher_mat = np.zeros((numvar_h1, numvar_h1))
    inv_fisher_mat = np.zeros((numvar_h1, numvar_h1))

    theta_h1 = np.zeros(numvar_h1)
    theta_h0 = np.zeros(numvar_h0)

    # Starting values
    blurred_max, blurred_min = gaussianblur_max_min_2d(image, psf_sd)
    theta_h1[0] = (blurred_max - blurred_min) * 2 * np.pi * psf_sd**2
    theta_h1[0] = max(theta_h1[0], blurred_min)
    theta_h1[1] = blurred_min

    # Maximum Likelihood Estimation of H1
    for _ in range(iterations):
        # nr stands for Newton-Raphson
        nr_numerator = np.zeros(numvar_h1)
        nr_denominator = np.zeros(numvar_h1)
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
                ddt_modelh1 = np.zeros(numvar_h1)
                # second derivatives
                d2dt2_modelh1 = np.zeros(numvar_h1)

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
                for ll in range(numvar_h1): 
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
            ddt_modelh1 = np.zeros(numvar_h1)
            ddt_modelh1[0] = psf_x * psf_y
            ddt_modelh1[1] = 1.0

            # Building the Fisher Information Matrix regarding H1.
            for kk in range(numvar_h1):
                for ll in range(kk, numvar_h1):
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

def gaussian_mle_test_four_params(image, psf_sd, sz, iterations):
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
    numvar_h0 = 1
    numvar_h1 = 4 

    fisher_mat = np.zeros((numvar_h1, numvar_h1))
    inv_fisher_mat = np.zeros((numvar_h1, numvar_h1))

    theta_h1 = np.zeros(numvar_h1)
    theta_h0 = np.zeros(numvar_h0)

    maxjump = np.array([1.0, 1.0, 100.0, 2.0])
    
    # Strating values 
    theta_h1[0], theta_h1[1] = center_of_mass_2d(image)
    blurred_max, blurred_min = gaussianblur_max_min_2d(image, psf_sd)
    theta_h1[2] = (blurred_max - blurred_min) * 2 * np.pi*psf_sd**2
    theta_h1[2] = max(theta_h1[2], blurred_min)
    theta_h1[3] = blurred_min

    # Maximum Likelihood Estimation of H1
    for kk in range(iterations):
        nr_numerator = np.zeros(numvar_h1)
        nr_denominator = np.zeros(numvar_h1)

        for ii in range(sz):
            for jj in range(sz):
                # Calculating the integral of the normalized 1D psf function for x ranging in the ii-th column.
                psf_x = integrate_gauss_1d(ii, theta_h1[0], psf_sd)
                # Calculating the integral of the normalized 1D psf function for y ranging in the jj-th row.
                psf_y = integrate_gauss_1d(jj, theta_h1[1], psf_sd)

                # Calculating the model value of the data at (ii, jj) under H1
                modelh1 = theta_h1[3] + theta_h1[2] * psf_x * psf_y
                # Retrieving the "actual" pixel value of the image at (ii, jj)
                pixel_val = image[jj, ii]

                # Calculating derivatives
                ddt_modelh1 = np.zeros(numvar_h1)
                d2dt2_modelh1 = np.zeros(numvar_h1)

                # First and second derivatives of the model w.r.t. x, y, N, and bg
                ddt_modelh1[0], d2dt2_modelh1[0] = derivative_int_gauss_1d(ii, theta_h1[0], psf_sd, theta_h1[2], psf_x) 
                ddt_modelh1[1], d2dt2_modelh1[1] = derivative_int_gauss_1d(jj, theta_h1[1], psf_sd, theta_h1[2], psf_y) 
                ddt_modelh1[2] = psf_x * psf_y  # First derivative of model w.r.t. N
                d2dt2_modelh1[2] = 0.0 # Second derivative of model w.r.t. N
                ddt_modelh1[3] = 1.0  # First derivative of model w.r.t. bg
                d2dt2_modelh1[3] = 0.0 # Second derivative of model w.r.t. bg

                # See https://doi.org/10.1364/OPEX.13.010503, Section 2.5 to better understand cf and df
                cf = 0.0
                df = 0.0
                # if model > 10e-3:
                if model > 0:
                    cf = pixel_val / model - 1
                    df = pixel_val / model**2
                else:
                    cf = 1e5
                    df = 1e5

                # Newton-Raphson update denominators and numerators
                for ll in range(numvar_h1):
                    nr_numerator[ll] += ddt_modelh1[ll] * cf
                    nr_denominator[ll] += d2dt2_modelh1[ll] * cf - ddt_modelh1[ll]**2 * df

        # Parameter update, with maxjump control 
        for ll in range(numvar_h1):
            theta_h1[ll] -= np.clip(nr_numerator[ll] / (nr_denominator[ll]), -maxjump[ll], maxjump[ll])

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

            # Calculating the model value of the data at (ii, jj) under H1
            model = theta_h1[3] + theta_h1[2] * psf_x * psf_y
            # Retrieving the "actual" pixel value of the image at (ii, jj)
            pixel_val = image[jj, ii]

            # First derivatives of the model w.r.t. x, y, N, and bg
            ddt_modelh1[0], _ = derivative_int_gauss_1d(ii, theta_h1[0], psf_sd, theta_h1[2], psf_x) 
            ddt_modelh1[1], _ = derivative_int_gauss_1d(jj, theta_h1[1], psf_sd, theta_h1[2], psf_y) 
            ddt_modelh1[2] = psf_x * psf_y  
            ddt_modelh1[3] = 1.0  

            # Fisher Information Matrix calculation regarding H1
            for kk in range(numvar_h1):
                for ll in range(kk, numvar_h1):
                    # Using Poisson pdf for likelihood function, the following formula is derived.
                    # Ref: Smith et al. 2010, nmeth, SI eq (9).
                    fisher_mat[kk, ll] += ddt_modelh1[ll] * ddt_modelh1[kk] / model
                    # The FIM is symmetric.
                    fisher_mat[ll, kk] = fisher_mat[kk, ll]

            # Estimated model value ratio
            modelh0 = theta_h0[0] 
            ratio = modelh1/ modelh0 
            if log_model > 0 and pixel_val > 0:
                # Because the likelihood function is Poisson pdf, the following formula can be derived.
                # log likelihood ratio = pixel_val * ( log(model_h1/model_h0) - model_h1 + model_h0)
                t_g += 2 * (pixel_val * np.log(max(ratio, 1e-2)) - modelh1 + modelh0) 

    # Compute the CRLB as the inverse of the Fisher Information Matrix
    inv_fisher_mat = np.linalg.inv(fisher_mat)
    crlbs = np.diag(inv_fisher_mat)

    pfa = 2 * norm.cdf(-np.sqrt(max(t_g, 0.0)), 0, 1) # 0 and 1 are the mean and standard deviation of the normal distribution, respectively.

    return theta_h0, theta_h1, crlbs, pfa

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
        h0_params, h1_params, crlbs, pfa = gaussian_mle_test_two_params(roi_image, psf_sd, sz, iterations) 
    elif fittype == 1:
        # fittype=1:  Fits (x,y,bg,Photons) under H1 and (Bg) under H0 given psf_sd. 
        # est_params: theta_h1[0], theta_h1[1], theta_h1[2], theta_h1[3], theta_h0[0]
        h0_params, h1_params, crlbs, pfa = gaussian_mle_test_four_params(roi_image, psf_sd, sz, iterations) 
        pass
    else:   
        pass
    return h0_params, h1_params, crlbs, pfa
    
