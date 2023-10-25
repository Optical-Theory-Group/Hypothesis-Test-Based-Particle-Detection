import numpy as np
from scipy.stats import norm


def gaussMLE(data, PSFSigma, iterations, fittype):
    # Extract dimensions from data
    Z, Channels, szX, szY = data.shape
    sz = szX * szY
    
    # Create a placeholder for storing results
    # Assuming the output of mle_ratio() is a 1D array of length 2 for each processed data
    results = np.zeros((Z, Channels, 2))
    
    # Loop over the Z and Channels dimensions
    for i in range(Z):
        for j in range(Channels):
            flat_data = data[i, j].flatten()
            results[i, j] = mle_ratio(flat_data, PSFSigma, sz, iterations)
            
    return results

def gauss_f_max_min_2D(sz, sigma, data):
    # In the original GPUgaussLib.cuh, the outputs are 
    # MaxN: miximum pixel value,
    # MinBG: minimum background value.
    norm_val = -1.0 / (2.0 * sigma * sigma)
    max_filteredpixel = 0.0
    min_filteredpixel = float('inf')

    for kk in range(sz):
        for ll in range(sz):
            filteredpixel = 0.0
            summation = 0.0
            
            for ii in range(sz):
                for jj in range(sz):
                    weight = np.exp(((ii-kk)**2 + (jj-ll)**2) * norm_val)
                    filteredpixel += weight * data[ii*sz+jj]
                    summation += weight

            filteredpixel /= summation
            max_filteredpixel = max(max_filteredpixel, filteredpixel)
            min_filteredpixel = min(min_filteredpixel, filteredpixel)

    return max_filteredpixel, min_filteredpixel


def int_gauss_1D(ii, x, sigma):
    norm_val = np.sqrt(1.0 / (2.0 * sigma * sigma))
    return 0.5 * (norm.cdf((ii-x+0.5) * norm_val) - norm.cdf((ii-x-0.5) * norm_val))


def mle_ratio(data, psf_sigma, sz, iterations):
    NV_RH1 = 2 # Number of variables: I_particle and I_bg 
    s_data = data.copy()

    nmax, theta_h1_1 = gauss_f_max_min_2D(sz, psf_sigma, s_data)
    # nmax: maximum pixel value, theta_h1_1: background intensity (under hypothesis 1)
    theta_h1 = np.array([max(0.1, (nmax - theta_h1_1) * 4 * np.pi * psf_sigma**2), theta_h1_1])
    
    for _ in range(iterations):
    # NR: Newton-Raphson
        nr_numerator = np.zeros(NV_RH1)
        nr_denominator = np.zeros(NV_RH1)
        
        for ii in range(sz):
            for jj in range(sz):
                psfx = int_gauss_1D(ii, (sz-1)/2.0, psf_sigma)
                psfy = int_gauss_1D(jj, (sz-1)/2.0, psf_sigma)
                model = theta_h1[1] + theta_h1[0] * psfx * psfy
                datum = s_data[jj*sz + ii]

                cf = max(min((datum / model if model > 10e-3 else 0) - 1, 10e4), 0)
                df = max(min(datum / (model**2) if model > 10e-3 else 0, 10e4), 0)

                dudt = [psfx*psfy, 1.0]
                
                for ll in range(NV_RH1):
                    nr_numerator[ll] += dudt[ll] * cf
                    nr_denominator[ll] -= dudt[ll]**2 * df

        theta_h1[0] = max(theta_h1[0] - np.clip(nr_numerator[0] / (nr_denominator[0] / 2.0), -theta_h1[0], theta_h1[0] / 2.0), nmax/2.0)
        theta_h1[1] = max(theta_h1[1] - nr_numerator[1] / nr_denominator[1], 0.01)

    return theta_h1
    # In the original, the outputs are written to the global variabes: d_Parameters, d_CRLBs, d_LogLikelihood, and NFits)