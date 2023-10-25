import numpy as np
from cMakeSubregions import make_subregions
import diplib as dip
from scipy.ndimage import gaussian_filter
from mlefunctions import mle_ratio, gaussMLE

def LLRMapv2(process, PSFSigma=0, minPixels=0, compReduction=0, significance=0.05, iterations=8, split=True, maxFramesPerBlock=50):
# LLRMapv2 Detects single mocules according to the GLRT framework
# SYNOPSIS:
#  [coords,dectectionPar,cutProcess] = LLRMapv2(process,PSFSigma,minPixels,
#                               compReduction,iterations,split,maxCudaFits)
#
# PARAMETERS:
#     process: single molcule data (corrected for gain and dark counts)
#
#     PSFSigma: sigma of diffraction limited PSF in pixels
#
#     minPixels: minimal size of detection cluster. minPixels = [] obbits
#     calculation of detection features dectectionPar =[]
#
#     compReduction: computational complexity reduction level in standard
#     deviations from mean. Rejectregions with small change of single
#     molecules, based on wavelet filtering. compReduction = [] results in
#     computation of GLRT on all pixels.
#
#     significance: false discovery rate
#
#     iterations: number of iterations for MLE algorithm.
#
#     split: split clusters using watershed algorithm
#       ### [NK] This might become important
#
#     maxCudaFits: maximum numbers of cuda fits on single GPU device. If
#     n CUDA enables devices are ready 1xn vector is expected.
#
#
# DEFAULTS:
#     minPixels = (PSFSigma*1.5)^2;
#     compReduction = 2;
#     significanceSinglePixel = 0.05;
#     split = false;
#     maxCudaFits = 1000000;
#     maxFramesPerBlock = 50;
#
#
# OUTPUTS:
#   coords: matrix with detection coordinates.
#   dectectionPar
#         .circularity: circularity of cluster using P2A feature.
#                 .pH1: detection probability of cluster.
#         .clusterSize: cluster size
#                  .ll: labeled clusters
#                  .hh: binary image containing raw detections
#             .pfa_adj: false discovery rate corrected false positive
#                       probailty
#
#   cutProcess: cropped image with the actuall test pixels.
#
    if minPixels == 0:
        minPixels = np.floor((PSFSigma*1.5) ^ 2)

    dectectionPar = []

    # Stipping the edges.
    xbegin = round(1.5*(2*PSFSigma+1))
    xend = round(process.shape[0] - 1.5 * (2*PSFSigma+1))
    ybegin = round(1.5*(2*PSFSigma+1))
    yend = round(process.shape[1] - 1.5*(2*PSFSigma+1))
    # +1 because np.arange does not include the stop value
    szx = np.arange(xbegin, xend+1)
    szy = np.arange(ybegin, yend+1)
    # Make multiples of 4.
    szx = szx[:int(np.floor(szx.shape[0]/4)*4)]
    szy = szy[:int(np.floor(szy.shape[0]/4)*4)]

    # number of loops
    Nloops = int(np.ceil(process.shape[2] / maxFramesPerBlock))
    pfa = np.ones((len(szy), len(szx), process.shape[2]))
    hh = np.zeros((len(szy), len(szx), process.shape[2]), dtype=bool)
    print('Performing H1/H0 tests...')

    for nn in range(Nloops):
        st = nn * maxFramesPerBlock  # Starting frame number
        # Ending frame number
        en = min((nn + 1) * maxFramesPerBlock, process.shape[1]) 
        # +1 because np.arange does not include the stop value
        idx = np.arange(st, en)

        Xm, Ym, Zm = np.meshgrid(szx, szy, idx)

        XX = Xm.ravel()
        YY = Ym.ravel()
        ZZ = Zm.ravel()  # ravel() returns flattened 1D array.

        H2 = 1/16
        H1 = 1/4
        H0 = 3/8
        g = {}
        g[0] = [H2, H1, H0, H1, H2]
        g[1] = [H2, 0, H1, 0, H0, 0, H1, 0, H2]

        n_smooth = 2
        if len(process.shape) == 2:  
            cutProcess = process[np.ix_(szx, szy)]
            flags = [1, 1]
        elif len(process.shape) == 3:
            cutProcess = process[np.ix_(szx, szy, idx)]
            flags = [1, 1, 0]
        else:
            raise ValueError('Only 2D and 3D image stacks are supported')

    cutProcess = np.maximum(cutProcess - gaussian_filter(cutProcess,
                            [flag * (np.sqrt(n_smooth)*9) for flag in flags]), 0)
    locIm = np.ones(cutProcess.shape)

    if compReduction:
        kernel = [
            {'filter': g[0], 'origin': 2},
            {'filter': g[0], 'origin': 0}
        ]

        if cutProcess.ndim == 3:
                kernel.append({'filter': 1})  # The third dimension filter is a scalar 1

        # Convert filters to PyDIP Image and then apply convolution
        filters = [dip.Image(np.array(k['filter'])) for k in kernel]
        V1 = dip.Convolution(dip.Image(cutProcess), filters, method="separable")
        # or is it just - V1 = dip.separable_convolution(cutProcess, kernel, flags)

        # Update the filters for the second convolution operation for V2
        kernel[0]['filter'] = g[1]
        kernel[1]['filter'] = g[1]
        if cutProcess.ndim == 3:
            kernel[2]['filter'] = np.array([1])

        filters = [dip.Image(np.array(k['filter'])) for k in kernel]
        V2 = dip.Convolution(V1, filters, method="separable")

        # Compute the difference between V1 and V2
        W = dip.SubtractImages(V1, V2)
        # Originally this was: locIm =  W{2}	>mean(dip_image(W{2}),[],[1 2])+compReduction*std(dip_image(W{2}),[],[1 2]);
        locIm = W > np.mean(dip.Image(W), axis=(0,1)) + compReduction * np.std(dip.Image(W), axis=(0,1))

    if len(cutProcess.shape) > 2:
        idxIm = np.argwhere(np.transpose(locIm, (1, 0, 2))).nonzero()
    else:
        idxIm = np.argwhere(np.transpose(locIm, (1, 0))).nonzero()
        
        
    # Convert from MATLAB's 1-based indexing to Python's 0-based indexing
    YY, XX, ZZ = idxIm[0] - 1, idxIm[1] - 1, idxIm[2] - 1 if cutProcess.ndim == 3 else None
    ROIStack = make_subregions(YY, XX, ZZ, 3 * (2 * PSFSigma + 1), cutProcess.astype('float32'))

        
    YY_subregions = YY[idxIm[:, 0], idxIm[:, 1]].flatten()
    XX_subregions = XX[idxIm[:, 0], idxIm[:, 1]].flatten()
    if len(cutProcess.shape) > 2:
        ZZ_subregions = ZZ[idxIm[:, 0], idxIm[:, 1], idxIm[:, 2]].flatten()
    else:
        ZZ_subregions = None  # or some default value if needed

    if ZZ_subregions:
        ROIStack = make_subregions(YY_subregions, XX_subregions, ZZ_subregions)
    else:
        ROIStack = make_subregions(YY_subregions, XX_subregions)
    ROIStack_transposed = np.transpose(ROIStack, (1, 0, 2, 3)).astype(np.float32)
    LLr = gaussMLE(ROIStack_transposed, PSFSigma, iterations, 0)
    # anyways, the important algorithm is the gaussMLE !!!

    pass
