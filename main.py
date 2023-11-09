import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import numpy as np
from image_simul import psfvalueadditionmatrix # makeimg, Particle, 
from generalized_lrt import generalized_likelihood_ratio_test, fdr_bh 
# Let's make a simple 9x9 image with a particle at the center
# test_image = np.zeros((9,9))
# test_image = psfvalueadditionmatrix(particle_x=4, particle_y=4, psfmaxval=1000, particle_psf_r=1.39, imgwidth=9)

# plt.imshow(test_image)
# plt.show(block=False)
# _,_,p_values_particle = generalized_likelihood_ratio_test(roi_stack=test_image, psf_sigma=1.39, iterations=8)

# test_image = np.zeros((9,9))

# _,_,p_values_nothing = generalized_likelihood_ratio_test(roi_stack=test_image, psf_sigma=1.39, iterations=8)

x = 3.35
y = 4.69
sz = 9
test_image = psfvalueadditionmatrix(particle_x=x, particle_y=y, multiplying_constant=1000, particle_psf_r=1.39, imgwidth=sz)
rand_mat = np.random.normal(40, 10, size=(9,9))
test_image += rand_mat
plt.imshow(test_image)
plt.colorbar()
# params,crlb,p_values_rand = generalized_likelihood_ratio_test(roi_stack=test_image, psf_sigma=1.39, iterations=8, fittype=0)

fittype = 1
params,crlb,p_values_rand = generalized_likelihood_ratio_test(roi_stack=test_image, psf_sigma=1.39, iterations=8, fittype=fittype)

# print(f'{params=}')
for i, p in enumerate(params):
    if i != 1: 
        print(f"{p:.2f}")
    else:
        print(f'{sz-p:.2f}')
print('-----') 

if fittype == 1:
    plt.plot(x,sz-y,'ro', label='true center')
    
plt.legend()
plt.show(block=False)
for i, v in enumerate(crlb):
    print(f'{np.sqrt(v):.1e}')
print('----')
print(f'{p_values_rand=}')

if fittype == 1:
    plt.plot(x,sz-y,'ro', label='true center')
    
plt.legend()
plt.show(block=False)
pfatemp = np.ones(9*9)
hhtemp = np.zeros(9*9, dtype=bool)
significance = 0.05
# _, _, pfa_adj = fdr_bh(np.reshape(LLr[2, :], (-1, 1)), significance, 'dep', 'no')
# hhtemp[idxIm] = pfa_adj <= significance

# process_shape = process.shape


pass