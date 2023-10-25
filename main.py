import matplotlib.pyplot as plt
import numpy as np
from imagesimulation import makeimg, Particle, psfvalueadditionmatrix
from hypothesistest import hypothesis_test_statistics
# from detection import LLRMapv2

# # Read or simulate an image to pass on to the following procedure
# dataCam1 = makeimg(imgwidth=100, 
#        pparticlepixel=.0003, 
#        particlepsfr=3, 
#        particlepsfmaxval=1000,
#        bgfreq=1/400, 
#        bgamp=300,
#        clusterp=0.0, 
#        avgclustersz=0, 
#        particledistancecluster=0.00, 
#        dustpsfr=10, 
#        dustpsfmaxval=200, 
#        pdustpixel=0.0001, 
#        )

# PSFSigma = 1.39
# [coordsCam1,detParCam1,cutProcessCam1] = LLRMapv2(dataCam1,PSFSigma,[],-2)
# # def LLRMapv2(process, PSFSigma, minPixels=0, compReduction=0, significance=0.05, iterations=8, split=True, maxFramesPerBlock=50):

# Let's make a simple 9x9 image with a particle at the center
test_image = np.zeros((9,9))
test_image = psfvalueadditionmatrix(particle_x=4, particle_y=4, psfmaxval=1000, particle_psf_r=1.39, imgwidth=9)

# plt.imshow(test_image)
# plt.show(block=False)
params, crlbs, p_values = hypothesis_test_statistics(roi_stack=test_image, psf_sigma=1.39, iterations=100)

pass