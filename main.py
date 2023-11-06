# import matplotlib.pyplot as plt
import numpy as np
from image_simul import psfvalueadditionmatrix # makeimg, Particle, 
from generalized_lrt import generalized_likelihood_ratio_test, fdr_bh 
# Let's make a simple 9x9 image with a particle at the center
test_image = np.zeros((9,9))
test_image = psfvalueadditionmatrix(particle_x=4, particle_y=4, psfmaxval=1000, particle_psf_r=1.39, imgwidth=9)

# plt.imshow(test_image)
# plt.show(block=False)
_,_,p_values_particle = generalized_likelihood_ratio_test(roi_stack=test_image, psf_sigma=1.39, iterations=8)

test_image = np.zeros((9,9))

_,_,p_values_nothing = generalized_likelihood_ratio_test(roi_stack=test_image, psf_sigma=1.39, iterations=8)


test_image = psfvalueadditionmatrix(particle_x=4, particle_y=4, psfmaxval=1000, particle_psf_r=1.39, imgwidth=9)
test_image += np.random.randint(0,9100,size=(9,9))

_,_,p_values_rand = generalized_likelihood_ratio_test(roi_stack=test_image, psf_sigma=1.39, iterations=8)

pfatemp = np.ones(9*9)
hhtemp = np.zeros(9*9, dtype=bool)
significance = 0.05
# _, _, pfa_adj = fdr_bh(np.reshape(LLr[2, :], (-1, 1)), significance, 'dep', 'no')
# hhtemp[idxIm] = pfa_adj <= significance

# process_shape = process.shape


pass