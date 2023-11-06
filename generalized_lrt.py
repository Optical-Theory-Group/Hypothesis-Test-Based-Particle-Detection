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

def integrate_gauss_1d(ii, x, sigma):
    
    norm = 1/2/sigma**2
    return 1.0/2.0*(erf((ii-x+0.5)*np.sqrt(norm))-erf((ii-x-0.5)*np.sqrt(norm)))

def kernel_calc_llr_prop(cr, i_theta, t_g):
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

def gaussian_mle_ratio_test(data, psf_sigma, sz_x, iterations):
    """ Returns parameters, crlbs, and loglikelihoods

    Args:
        data (_type_): _description_
        psf_sigma (_type_): _description_
        sz_x (_type_): _description_
        iterations (_type_): _description_
        nfit (_type_): _description_
    """
    nv_rh0 = 1
    nv_rh1 = 2 
    sz = sz_x
    fisher_mat = np.zeros((nv_rh1, nv_rh1))
    inv_fisher_mat = np.zeros((nv_rh1, nv_rh1))
    diag = np.zeros(nv_rh1)
    theta_h1 = np.zeros(nv_rh1)
    theta_h0 = np.zeros(nv_rh0)
    # Should the fofilter_center_yowing data be a picked subset of the whole roi_stack?
    i_max, theta_h1[1] = gaussianblur_max_min_2d(data, psf_sigma)
    theta_h1[0] = max(0.1, (i_max - theta_h1[1]) * 4 * np.pi * psf_sigma**2)  # It is not clear from the original code why 4 instead of 2.

    for kk in range(iterations):
        nr_numerator = np.zeros(nv_rh1)
        nr_denominator = np.zeros(nv_rh1)
        for ii in range(sz):
            for jj in range(sz):
                psf_x = integrate_gauss_1d(ii, (sz-1) / 2, psf_sigma) # calculates the psf function's value at the corresponding ii's pixel.
                psf_y = integrate_gauss_1d(jj, (sz-1) / 2, psf_sigma)
                
                model = theta_h1[1] + theta_h1[0] * psf_x * psf_y 
                data_val = data[jj, ii]  

                dudt = np.zeros(nv_rh1)
                d2udt2 = np.zeros(nv_rh1)
                # dudt[0]: d/d(theta_h1[0]) [model] and so on.
                dudt[0] = psf_x * psf_y 
                d2udt2[0] = 0.0
                dudt[1] = 1.0 
                d2udt2[1] = 0.0

                cf = 0.0
                df = 0.0
                if model > 10e-3: 
                    cf = data_val / model - 1
                    df = data_val / model ** 2
                cf = min(cf, 10e4)
                df = min(df, 10e4)

                # Fisher information matrix
                for ll in range(nv_rh1):
                    nr_numerator[ll] += dudt[ll] * cf
                    nr_denominator[ll] += d2udt2[ll] * cf - dudt[ll] ** 2 * df

        theta_h1[0] -= min(max(nr_numerator[0] / nr_denominator[0] / 2, -theta_h1[0]), theta_h1[0]/2)
        theta_h1[0] = max(theta_h1[0], i_max/2)

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

            dudt = np.zeros(nv_rh1)
            dudt[0] = psf_x * psf_y
            dudt[1] = 1.0

            # Building the Fisher Information Matrix
            for kk in range(nv_rh1):
                for ll in range(kk, nv_rh1):
                    fisher_mat[kk, ll] += dudt[ll] * dudt[kk] / model # dividing by model is averating, effectively? - yes.
                    fisher_mat[ll, kk] = fisher_mat[kk, ll]

            # LogLikelihood
            likelihood_ratio = model / (theta_h0[0] + 1e-5) 
            if likelihood_ratio > 0 and data_val > 0:
                # log likelihood ratio = data_val ( log(model_h1/model_h0) - model_h1 + model_h0)
                t_g += 2 * (data_val * np.log(likelihood_ratio + 1e-5) - model + theta_h0[0]) # 2 times log likelihood-ratio follows chi-sqrd distribution

    # Matrix inverse (CRLB=F^-1)
    inv_fisher_mat = np.linalg.inv(fisher_mat)
    diag = np.diag(inv_fisher_mat)
    
    # Calculate the return values
    loglikelihoods = kernel_calc_llr_prop(diag[0], theta_h1[0], t_g)
    parameters = np.concatenate([theta_h1, theta_h0])
    crlbs = np.diag(inv_fisher_mat)

    return parameters, crlbs, loglikelihoods

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
    
    if (data_ndim == 2):
        sz_x = data_shape[0] # sz_y == sz_x always.
        # sz_z = 1 
    else:
       pass 

    if (fittype == 0):
        # nfit = 2
        params, crlbs, loglikelihoods = gaussian_mle_ratio_test(roi_stack, psf_sigma, sz_x, iterations) 
    else:
        pass
   
    p_values = loglikelihoods[2]
     
    # Don't worry about params and crlbs for now.
    return params, crlbs, p_values
    
