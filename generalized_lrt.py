import numpy as np
from scipy.special import erf
from scipy.stats import norm
# from imagesimulation import makeimg

def gaussianblur_max_min_2d(data, sigma):

    assert data.shape[0] == data.shape[1]
    sz = data.shape[0]
        
    filteredpixel = 0 
    sum_val = 0  # renamed to avoid shadowing built-in sum function
    max_i = 0
    min_bg = 10e10
    norm = 1/2/sigma**2

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
            
            max_i = max(max_i, filteredpixel)
            min_bg = min(min_bg, filteredpixel)

    return max_i, min_bg

def derivative_int_gauss_1d(ii, x, sigma, N, PSFy):
    """
    Compute the derivative of the 1D Gaussian.
    
    Args:
        ii (int): Pixel index.
        x (float): Mean value of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.
        N (float): Amplitude of the Gaussian.
        PSFy (float): Scaling factor along y.
    
    Returns:
        float: Derivative of the Gaussian with respect to x at ii.
        float: Second derivative of the Gaussian with respect to x at ii (optional).
    """
    a = np.exp(-0.5 * ((ii + 0.5 - x) / sigma)**2)
    b = np.exp(-0.5 * ((ii - 0.5 - x) / sigma)**2)

    dmodel_dt = -N / np.sqrt(2 * np.pi) / sigma * (a - b) * PSFy
    d2model_dt2 = -N / np.sqrt(2 * np.pi) / sigma**3 * ((ii + 0.5 - x) * a - (ii - 0.5 - x) * b) * PSFy

    return dmodel_dt, d2model_dt2

def center_of_mass_2d(data, ax=0, ay=0):
    """
    Compute the 2D center of mass of a subregion.

    Parameters:
    - data: 2D numpy array representing the subregion to search
- ax: optional adjustment in x
    - ay: optional adjustment in y

    Returns:
    - x: x coordinate of the center of mass
    - y: y coordinate of the center of mass
    """
    # sz = data.shape[0]  # Assume data is square (sz x sz)
    ii, jj = np.indices(data.shape)
    adjusted_data = data - ax * ii - ay * jj 
    tmpx = np.sum((adjusted_data) * ii)
    tmpy = np.sum((adjusted_data) * jj)
    tmpsum = np.sum(adjusted_data)
    
    if tmpsum == 0:
        return np.nan, np.nan  # Avoid division by zero
    
    x = tmpx / tmpsum
    y = tmpy / tmpsum
    
    return x, y

def integrate_gauss_1d(ii, x, sigma):
    
    norm = 1/2/sigma**2
    return 1.0/2.0*(erf((ii-x+0.5)*np.sqrt(norm))-erf((ii-x-0.5)*np.sqrt(norm)))
    # the above, re-written using exponential function integrals is as below:
    # integral(from ii-0.5 to ii+0.5) [1/2sqrt(pi)*exp(-norm*(t-x)**2) dt]

def calc_llr_prop(cr, i_theta, t_g):
    """
    Returns probabilities corresponding to LLR.
    
    Returns:
    - llr: List of computed probabilities.
    """
    
    llr = np.zeros(6)
    llr[0] = t_g
    llr[1] = i_theta**2 / (cr + 1e-5)
    llr[2] = 2 * (1 - norm.cdf(np.sqrt(max(t_g, 0.0)), 0, 1))
    llr[3] = (1 - norm.cdf(np.sqrt(max(t_g, 0.0)), np.sqrt(llr[1]), 1)) + \
            (1 - norm.cdf(np.sqrt(max(t_g, 0.0)), -np.sqrt(llr[1]), 1))
    llr[4] = 2 * norm.pdf(np.sqrt(max(t_g, 0.0)), 0, 1)
    llr[5] = norm.pdf(np.sqrt(max(t_g, 0.0)), np.sqrt(llr[1]), 1) + \
            norm.pdf(np.sqrt(max(t_g, 0.0)), -np.sqrt(llr[1]), 1)
    
    # The following lines are commented out in the original code, but they can be included if needed.
    # llr[5] = llr[5] / (llr[4] + llr[5] + 1e-4)
    # llr[4] = llr[4] / (llr[4] + llr[5] + 1e-4)
    
    return llr

def gaussian_mle_test_two_params(data, psf_sigma, sz, iterations):
    """ Returns parameters, crlbs, and loglikelihoods

    Args:
        data (_type_): _description_
        psf_sigma (_type_): _description_
        sz_x (_type_): _description_
        iterations (_type_): _description_
        nfit (_type_): _description_
    """
    # Initialization
    nv_rh0 = 1
    nv_rh1 = 2 

    fisher_mat = np.zeros((nv_rh1, nv_rh1))
    inv_fisher_mat = np.zeros((nv_rh1, nv_rh1))

    theta_h1 = np.zeros(nv_rh1)
    theta_h0 = np.zeros(nv_rh0)

    # initial starting values
    max_estimate, theta_h1[1] = gaussianblur_max_min_2d(data, psf_sigma)
    theta_h1[0] = max(0.1, (max_estimate - theta_h1[1]) * 4 * np.pi * psf_sigma**2)

    for kk in range(iterations):
        nr_numerator = np.zeros(nv_rh1)
        nr_denominator = np.zeros(nv_rh1)
        for ii in range(sz): # sz: width of the image
            for jj in range(sz):
                psf_x = integrate_gauss_1d(ii, (sz-1) / 2, psf_sigma) # calculates the psf function's value at the corresponding ii's pixel.
                psf_y = integrate_gauss_1d(jj, (sz-1) / 2, psf_sigma)
                
                model = theta_h1[1] + theta_h1[0] * psf_x * psf_y 
                data_val = data[jj, ii]  

                dmodel_dt = np.zeros(nv_rh1)
                d2model_dt2 = np.zeros(nv_rh1)
                # dmodel_dt[0]: d/d(theta_h1[0]) [model] and so on.
                dmodel_dt[0] = psf_x * psf_y  # d (mode) / d (theta_h1[0])
                d2model_dt2[0] = 0.0
                dmodel_dt[1] = 1.0 
                d2model_dt2[1] = 0.0

                # See https://doi.org/10.1364/OPEX.13.010503, Section 2.5 to better understand cf and df
                cf = 0.0
                df = 0.0
                if model > 10e-3: 
                    cf = data_val / model - 1
                    df = data_val / model ** 2
                cf = min(cf, 10e4)
                df = min(df, 10e4)

                for ll in range(nv_rh1): # nv_rh1 == 2 (number of estimators under hypothesis 1)
                    nr_numerator[ll] += dmodel_dt[ll] * cf
                    nr_denominator[ll] += d2model_dt2[ll] * cf - dmodel_dt[ll] ** 2 * df

        theta_h1[0] -= min(max(nr_numerator[0] / nr_denominator[0] / 2, -theta_h1[0]), theta_h1[0]/2)
        theta_h1[0] = max(theta_h1[0], max_estimate/2)

        theta_h1[1] -= nr_numerator[1] / nr_denominator[1]
        theta_h1[1] = max(theta_h1[1], 0.01)

    # Maximum likelihood estimate of background model
    theta_h0[0] = 0.0
    for ii in range(sz):
        for jj in range(sz):
            theta_h0[0] += data[jj, ii]
    theta_h0[0] = theta_h0[0] / sz**2

    # Calculatibng the CRLB and LogLikelihoodRatio
    t_g = 0.0
    for ii in range(sz):
        for jj in range(sz):
            psf_x = integrate_gauss_1d(ii, (sz-1) / 2, psf_sigma)
            psf_y = integrate_gauss_1d(jj, (sz-1) / 2, psf_sigma)

            model = theta_h1[1] + theta_h1[0] * psf_x * psf_y
            data_val = data[jj, ii]

            dmodel_dt = np.zeros(nv_rh1)
            dmodel_dt[0] = psf_x * psf_y
            dmodel_dt[1] = 1.0

            # Building the Fisher Information Matrix - This Fisher info Mat is only related to H1.
            for kk in range(nv_rh1):
                for ll in range(kk, nv_rh1):
                    fisher_mat[kk, ll] += dmodel_dt[ll] * dmodel_dt[kk] / model # Note that fisher_mat_ij should be E[((d/dti log_model)(d/dtj log_model)d/dti log_model)(d/dtj log_model)]
                    # but d/dti model = d/dti log_model * model
                    # Thus (d/dti model)(d/dtj model) = (d/dti log_model)(d/dtj log_model) * model**2 

                    fisher_mat[ll, kk] = fisher_mat[kk, ll]

            # LogLikelihood
            likelihood_ratio = model / (theta_h0[0] + 1e-5) 
            if likelihood_ratio > 0 and data_val > 0:
                # log likelihood ratio = data_val ( log(model_h1/model_h0) - model_h1 + model_h0)
                t_g += 2 * (data_val * np.log(likelihood_ratio + 1e-5) - model + theta_h0[0]) # t_g is used for its property that it follows chi-sqrd distribution

    # Matrix inverse (CRLB=F^-1)
    inv_fisher_mat = np.linalg.inv(fisher_mat)
    crlbs = np.diag(inv_fisher_mat)
    
    # Calculate the return values
    llr = calc_llr_prop(crlbs[0], theta_h1[0], t_g)
    parameters = np.concatenate([theta_h1, theta_h0])

    return parameters, crlbs, llr

def gaussian_mle_test_four_params(data, psf_sigma, sz, iterations):
    """ Returns parameters, crlbs, and loglikelihoods
    Args:
        data (_type_): _description_
        psf_sigma (_type_): _description_
        sz_x (_type_): _description_
        iterations (_type_): _description_
    """
    # initialization
    nv_rh0 = 1
    nv_rh1 = 4 

    fisher_mat = np.zeros((nv_rh1, nv_rh1))
    inv_fisher_mat = np.zeros((nv_rh1, nv_rh1))

    theta_h1 = np.zeros(nv_rh1)
    theta_h0 = np.zeros(nv_rh0)

    maxjump = np.array([1.0, 1.0, 100.0, 2.0])
    gamma = np.array([1.0, 1.0, 0.5, 1.0])
    
    # initial strting values 
    theta_h1[0], theta_h1[1] = center_of_mass_2d(data)
    max_estimate, theta_h1[3] = gaussianblur_max_min_2d(data, psf_sigma)
    # theta_h1[2] = max(0.1, (max_estimate - theta_h1[3]) * 2 * np.pi*psf_sigma**2) - It's okay to rid this [6. 4:14pm 11/15/2023]
    theta_h1[2] = max(0, (max_estimate - theta_h1[3]) * 2 * np.pi*psf_sigma**2)

    for kk in range(iterations):
        nr_numerator = np.zeros(nv_rh1)
        nr_denominator = np.zeros(nv_rh1)

        for ii in range(sz):
            for jj in range(sz):
                psf_x = integrate_gauss_1d(ii, theta_h1[0], psf_sigma)
                psf_y = integrate_gauss_1d(jj, theta_h1[1], psf_sigma)

                model = theta_h1[3] + theta_h1[2] * psf_x * psf_y
                data_val = data[jj, ii]

                # Calculating derivatives
                dmodel_dt = np.zeros(nv_rh1)
                d2model_dt2 = np.zeros(nv_rh1)

                # Here you would calculate the derivatives. As an example, I'm using placeholders
                # Replace these with the actual derivative calculations
                dmodel_dt[0], d2model_dt2[0] = derivative_int_gauss_1d(ii, theta_h1[0], psf_sigma, theta_h1[2], psf_x) 
                dmodel_dt[1], d2model_dt2[1] = derivative_int_gauss_1d(jj, theta_h1[1], psf_sigma, theta_h1[2], psf_y) 
                dmodel_dt[2] = psf_x * psf_y  # derivative of model w.r.t. N
                d2model_dt2[2] = 0.0
                dmodel_dt[3] = 1.0  # derivative of model w.r.t. bg
                d2model_dt2[3] = 0.0

                # Correction factor and derivative factor
                cf = 0.0
                df = 0.0
                # if model > 10e-3:
                if model > 0:
                    cf = data_val / model - 1
                    df = data_val / model**2
                # cf = min(cf, 10e4) - It's okay to rid this [5. 4:09pm 11/15/2023]
                # df = min(df, 10e4) - it's okay to rid [6.]

                # Newton-Raphson update denominators and numerators
                for ll in range(nv_rh1):
                    nr_numerator[ll] += dmodel_dt[ll] * cf
                    nr_denominator[ll] += d2model_dt2[ll] * cf - dmodel_dt[ll]**2 * df

        # Parameter update, with gamma and maxjump to control the step size
        if kk < 2:
            for ll in range(nv_rh1):
                theta_h1[ll] -= gamma[ll] * np.clip(nr_numerator[ll] / nr_denominator[ll], -maxjump[ll], maxjump[ll])
        else:
            for ll in range(nv_rh1):
                theta_h1[ll] -= np.clip(nr_numerator[ll] / nr_denominator[ll], -maxjump[ll], maxjump[ll])

        # if kk < 2:  
        #     for ll in range(nv_rh1):
        #         update_step = nr_numerator[ll] / (nr_denominator[ll] + 1e-6)  # add a small constant to prevent division by zero
        #         theta_h1[ll] -= gamma[ll] * min(max(update_step, -maxjump[ll]), maxjump[ll])
        # else:
        #     for ll in range(nv_rh1):
        #         update_step = nr_numerator[ll] / (nr_denominator[ll] + 1e-6)  # add a small constant to prevent division by zero
        #         theta_h1[ll] -= min(max(update_step, -maxjump[ll]), maxjump[ll])
           
        # Any other constraints
        theta_h1[2] = max(theta_h1[2], 0)
        # theta_h1[2] = max(theta_h1[2], 1.0) - It's okay to turn 1.0 to 0 [4.]
        theta_h1[3] = max(theta_h1[3], 0)
        # theta_h1[3] = max(theta_h1[3], 0.01) - It's okay to turn 0.01 to 0. [3. 4:05pm 11/15/2023]

    # Maximum likelihood estimate of background model
    theta_h0[0] = 0.0
    for ii in range(sz):
        for jj in range(sz):
            theta_h0[0] += data[jj, ii]
    theta_h0[0] = theta_h0[0] / sz**2

    # Calculate the CRLB and LogLikelihood
    t_g = 0.0
    for ii in range(sz):
        for jj in range(sz):
            psf_x = integrate_gauss_1d(ii, theta_h1[0], psf_sigma)
            psf_y = integrate_gauss_1d(jj, theta_h1[1], psf_sigma)

            model = theta_h1[3] + theta_h1[2] * psf_x * psf_y
            data_val = data[jj, ii]

            dmodel_dt[0], _ = derivative_int_gauss_1d(ii, theta_h1[0], psf_sigma, theta_h1[2], psf_x) 
            dmodel_dt[1], _ = derivative_int_gauss_1d(jj, theta_h1[1], psf_sigma, theta_h1[2], psf_y) 
            dmodel_dt[2] = psf_x * psf_y  
            dmodel_dt[3] = 1.0  

            # Fisher Information Matrix calculation
            for kk in range(nv_rh1):
                for ll in range(kk, nv_rh1):
                    if kk == ll and 2 < ii < 6 and 2 < jj < 6:
                        pass
                    fisher_mat[kk, ll] += dmodel_dt[ll] * dmodel_dt[kk] / model
                    fisher_mat[ll, kk] = fisher_mat[kk, ll]

            # LogLikelihood calculation
            log_model = np.log(model / (theta_h0[0] )) 
            # log_model = np.log(model / (theta_h0[0] + 1e-5)) - It's okay to remove 1e-5 [1. 401pm 11/15/2023]
            if log_model > 0 and data_val > 0:
                t_g += 2 * (data_val * (log_model ) - model + theta_h0[0])
                # t_g += 2 * (data_val * (log_model + 1e-5) - model + theta_h0[0]) - It's okay to remove 1e-5 [2. 4:03pm 11/15/2023]

    # Compute the CRLB as the inverse of the Fisher Information Matrix
    inv_fisher_mat = np.linalg.inv(fisher_mat)
    crlbs = np.diag(inv_fisher_mat)

    # Output parameters
    parameters = np.concatenate((theta_h1, theta_h0))
    llr = calc_llr_prop(crlbs[2], theta_h1[2], t_g) 

    return parameters, crlbs, llr

def fdr_bh(pvals, q=0.05, method='pdep', report='no'):
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


def generalized_likelihood_ratio_test(roi_stack, psf_sigma, iterations=8, fittype=0, display='no'):
    """ Takes input image_roi stack and other parameters and outputs test statstics (currently p-values, but others wifilter_center_y be added later on.).
        This corresponds to the mexFunction found in mexFunction.cpp in the original repository.

    Args:
        roi_stack (np.array): Stack of cropped ROIs of the input image.
        psf_sigma (float): Standard deviation of the point spread function
        iterations (int): Newton-Raphson iteration number
        fittype (int, optional): Model type. Defaults to 0 (theta1[0]: I, theta1[1]: bg)
        display (str, optional): Output message. Defaults to 'no'.

    Returns:
        np.array: p_values per pixel for each image ROI
    """
    # Initialize variables
    data_shape = roi_stack.shape
    data_ndim = roi_stack.ndim
    
    if data_ndim == 2:
        sz_x = data_shape[0] # sz_y == sz_x always.
        # sz_z = 1 
    else:
       pass 

    if fittype == 0:
        # fittype=0:  Fits (Photons,Bg) under H1 and (Bg) under H0 given PSF_sigma. 
        # params: theta_h1[0], theta_h1[1], theta_h0[0]
        params, crlbs, llr = gaussian_mle_test_two_params(roi_stack, psf_sigma, sz_x, iterations) 
    elif fittype == 1:
        # fittype=1:  Fits (x,y,bg,Photons) under H1 and (Bg) under H0 given PSF_sigma. 
        # params: theta_h1[0], theta_h1[1], theta_h1[2], theta_h1[3], theta_h0[0]
        params, crlbs, llr = gaussian_mle_test_four_params(roi_stack, psf_sigma, sz_x, iterations) 
        pass
    else:   
        pass
   
    p_values = llr[2]
     
    # Don't worry about params and crlbs for now.
    return params, crlbs, p_values
    
