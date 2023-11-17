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
def main(x=3.35, y=6.69, sz=9, intensity=10, bg = 4):
    image = psfvalueadditionmatrix(particle_x=x, particle_y=y, multiplying_constant=intensity, particle_psf_r=1.39, imgwidth=sz)
    # Adding background
    image += np.ones(image.shape)*bg
    image = np.random.poisson(image, size=(image.shape))
    
    # rand_mat = np.random.normal(4, 0, size=(9,9))
    plt.imshow(image)
    plt.colorbar()
    # params,crlb,p_value = generalized_likelihood_ratio_test(roi_stack=test_image, psf_sigma=1.39, iterations=8, fittype=0)

    fittype = 1
    params,crlb,p_value = generalized_likelihood_ratio_test(roi_stack=image, psf_sigma=1.39, iterations=16, fittype=fittype)

    # print(f'{params=}')
    for i, p in enumerate(params):
        if i == len(params) - 1:
            print('-----')
        if i != 1: 
            print(f"{p:.2f}")
        else:
            print(f'{sz-p:.2f}')
    print('-----') 

    if fittype == 1:
        plt.plot(x,y,'ro', markersize=15, label='true center')
        plt.plot(params[0],params[0],'*', markersize=15, label='guessed center', markerfacecolor='aqua')
        # plt.plot(params[0],sz-params[0]+1,'*', markersize=15, label='guessed center', markerfacecolor='aqua')


        
    plt.legend()
    # plt.show(block=False)
    crlbs = []
    for i, v in enumerate(crlb):
        # print(f'{np.sqrt(v):.1e}')
        print(f'{np.sqrt(v):.2f}')
        crlbs += np.sqrt(v)
    print('----')
    print(f'{p_value=}')

    titlestr = f"Ground truth: x={x:.2f}, y={y:.2f}, intensity={intensity}, bg={bg}\n"\
        f"Fitted: {params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f}, {params[3]:.2f}\n"\
        f"CRLB: {crlb[0]:.4f}, {crlb[1]:.4f}, {crlb[2]:.2f}, {crlb[3]:.2f}\n"\
        f"p-value: {p_value:.3e}"
    plt.title(titlestr)
    # plt.title(titlestr)

        
    # plt.legend(fontsize="20")
    plt.show(block=False)
    plt.tight_layout()
    pfatemp = np.ones(9*9)
    hhtemp = np.zeros(9*9, dtype=bool)
    significance = 0.05
    # _, _, pfa_adj = fdr_bh(np.reshape(LLr[2, :], (-1, 1)), significance, 'dep', 'no')
    # hhtemp[idxIm] = pfa_adj <= significance

    # process_shape = process.shape

    return p_value
    pass
# bgs = [0, 100, 500, 1000, 2000]
# bgs += [3000, 4000, 5000, 6000]
bgs = [6000, 7000, 10000, 15000, 20000, 25000, 30000]

# bgs = [2000]
intensity = 1000
ps = []
for i, bg in enumerate(bgs):
# for i in range(8):
    x = np.random.rand()*4 + 2 
    y = np.random.rand()*4 + 2
    # intensity = np.random.randint(500, 1000)
    # bg = np.random.randint(10, 500)
    p = main(x=x, y=y, intensity=intensity, bg = bg)
    # plt.savefig(f'./Figures/{i}.png')
    plt.savefig(f'./Figures/{intensity=} {bg=}.png')
    plt.figure()
    ps.append(p)

# plt.figure()
# plt.plot(bgs, ps, 'o--')
# plt.axhline(y=0.05, color = 'r')
# plt.show(block=False)
pass