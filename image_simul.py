from scipy.ndimage import binary_closing, label
from scipy.signal import convolve
import matplotlib.colors as colors
import numpy as np
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import random
from PIL import Image
import nmmn.plots

parula = nmmn.plots.parulacmap()


class Particle:
    def __init__(self, x, y, peak_val, particle_psf_r, clustered):
        """ The following values are inherent properties of a particle. psf_sd is not. """
        self.x = x
        self.y = y
        self.peak_val = peak_val
        self.particle_psf_r = particle_psf_r
        self.clustered = clustered


class Dust:
    def __init__(self, x, y, peak_val, particle_psf_r):
        """ The following values are inherent properties of a particle. psf_sd is not. """
        self.x = x
        self.y = y
        self.peak_val = peak_val
        self.particle_psf_r = particle_psf_r

# I'm trying to decide whether to make the psfconvolimg generation process per particle object to be the objects method or not.


def psfvalueadditionmatrix(particle_x, particle_y, psfmaxval, particle_psf_r, imgwidth):
    """Returns the pixel values to be added to the image

    Args:
        particle_x (float): _description_
        particle_y (float): _description_
        psfmaxval (float): _description_
        psfsd (float): _description_
        imgwidth (int): _description_

    Returns:
        numpy.array(dtype=int): imgwidth x imgwidth array of psf pixel values to be added to image
    """
    output = np.zeros((imgwidth, imgwidth))
    psfsd_squared = particle_psf_r ** 2

    for x in range(imgwidth):
        for y in range(imgwidth):
            distance_squared = (x - particle_x) ** 2 + (y - particle_y) ** 2
            pixel_value = psfmaxval * \
                np.exp(-(distance_squared / (2.0 * psfsd_squared)))
            output[y, x] = pixel_value
    return output


def create_hex_positions(n, particledistancecluster):
    rotmat = R.from_euler('z', 60, degrees=True).as_matrix()[:2, :2]
    next_pos_dir = []
    positions = np.empty((0, 2), float)
    layer_idx = 0
    n_sides = 6

    for i in range(n_sides - 1):
        next_pos_dir.append(
            np.array([.5, np.sqrt(3)/2]) @ matrix_power(rotmat, i))

    while len(positions) < n:
        layer_idx += 1
        # First position in layer:
        positions = np.append(positions, np.array(
            [[layer_idx * particledistancecluster * -1, 0]]), axis=0)
        # Following positions in the layeR:
        for i in range(len(next_pos_dir)):
            for _ in range(layer_idx):
                next_pos = positions[-1] + \
                    next_pos_dir[i] * particledistancecluster
                positions = np.append(positions, [next_pos], axis=0)
        # After that, randomly delete positions from the last shell
        if len(positions) > n:
            n_del = len(positions) - n
            del_idxs = random.sample(range(n, n + n_del), n_del)
            positions = np.delete(positions, del_idxs, axis=0)

    # Rotate the whole thing
    angle = random.randrange(360)
    rotmat = R.from_euler('z', angle, degrees=True).as_matrix()[:2, :2]
    positions = positions @ rotmat
    assert n == len(positions)
    return positions


def lowfreq_background(imgwidth, bgfreq, amplitude=100):
    center_x, center_y = random.randrange(imgwidth), random.randrange(imgwidth)
    outputimg = np.zeros((imgwidth, imgwidth))
    theta = np.random.random() * 2 * np.pi

    for x in range(imgwidth):
        for y in range(imgwidth):
            X_rot = (x - center_x) * np.cos(theta) - \
                     (y - center_y) * np.sin(theta)
            Y_rot = (x - center_x) * np.sin(theta) + \
                     (y - center_y) * np.cos(theta)
            outputimg[y, x] = amplitude / 2 * \
                (np.cos(2 * np.pi * bgfreq * X_rot) +
                 np.cos(2 * np.pi * bgfreq * Y_rot))
    return outputimg


def create_particle_list(imgwidth, p_particlepixel, clusterp, avgclustersz, particledistancecluster, particle_psf_r, particlepsfmaxval):
    particle_list = []
    n_particles_to_be_created = round(imgwidth ** 2 * p_particlepixel)
    for i in range(n_particles_to_be_created):
        x, y = np.random.rand() * imgwidth, np.random.rand() * imgwidth
        if np.random.random() < clusterp:
            particle_list.append(
                Particle(x, y, particlepsfmaxval, particle_psf_r, True))
            # Add particles in its vicinity
            new_cluster_size = round(np.random.exponential(avgclustersz))
            if new_cluster_size > 1:
                relativepositions = create_hex_positions(
                    n=new_cluster_size, particledistancecluster=particledistancecluster)
                for rel_x, rel_y in relativepositions:
                    if x + rel_x > 0 and y + rel_y > 0:
                        particle_list.append(
                            Particle(x + rel_x, y + rel_y, particlepsfmaxval, particle_psf_r, True))
        else:
            particle_list.append(
                Particle(x, y, particlepsfmaxval, particle_psf_r, False))

    return particle_list


def create_dust_list(imgwidth, p_dustpixel, particle_psf_r, dustpsfmaxval):
    dust_list = []
    n_dust_to_be_created = round(imgwidth ** 2 * p_dustpixel)
    for _ in range(n_dust_to_be_created):
        x, y = np.random.rand() * imgwidth, np.random.rand() * imgwidth
        dust_list.append(Dust(x, y, dustpsfmaxval, 2 * particle_psf_r))
    return dust_list


def create_positions_img(imgwidth, particle_list, ):
    positionsimg = np.zeros((imgwidth, imgwidth))
    for pt in particle_list:
        if round(pt.x) < imgwidth and round(pt.y) < imgwidth:
            positionsimg[round(pt.y), round(pt.x)] += 1
    return positionsimg


def makeimg(imgwidth=256, pparticlepixel=.1, particlepsfmaxval=1000, particle_psf_r=2, bgfreq=1/256, bgamp=50, clusterp=0.2, avgclustersz=20, particledistancecluster=0.5,
            pdustpixel=0.005, dustpsfr=10, dustpsfmaxval=500, vignettesd=128, ):  # background coherent speckle to be added
    """ (Note: All lengths are in units of pixels)
        * Note on development:
         - Take a step-by-step approach.
         - Don't realize all parameter usage all at once.
    """
    # Create Particle objects and store them in a list
    particle_list = create_particle_list(
        imgwidth, pparticlepixel, clusterp, avgclustersz, particledistancecluster, particle_psf_r, particlepsfmaxval)

    positionsimg = create_positions_img(imgwidth, particle_list)

    # Create psf convolved image
    psfconvolimg = np.zeros((imgwidth, imgwidth))

    # Create psf image for each particle
    for pt in particle_list:
        psfconvolimg += psfvalueadditionmatrix(pt.x,
                                               pt.y, pt.peak_val, pt.particle_psf_r, imgwidth)

    # Add low-frequency background
    lowfreq_background_img = lowfreq_background(imgwidth, bgfreq, bgamp)

    # Add dust particles
    dust_list = create_dust_list(
        imgwidth, pdustpixel, particle_psf_r, dustpsfmaxval)
    dustimg = np.zeros((imgwidth, imgwidth))
    for dust in dust_list:
        dustimg += psfvalueadditionmatrix(dust.x, dust.y,
                                          dust.peak_val, pt.particle_psf_r, imgwidth)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    plt.suptitle(f'{pparticlepixel=}, {particlepsfmaxval=}, {particle_psf_r=}, \n {clusterp=}, {avgclustersz=}, {particledistancecluster=}, \n  {pdustpixel=}, {dustpsfmaxval=}, {dustpsfr=}, ')

    # -- Figure layout --
    # Positions | Lowfreq BG effect | Dust effect
    # PSF Convoluted | Lowfreq BG added | Dust added
    ax = axes[0][0]
    ax.set_title('Particle Positions')
    ax.scatter([pt.x for pt in particle_list], [
               pt.y for pt in particle_list], marker='x')
    ax.set_ylim([imgwidth, 0])
    ax.set_xlim([0, imgwidth])
    # plt.show(block=False)

    ax = axes[1][0]
    pcm = ax.imshow(psfconvolimg, cmap=parula)
    ax.set_title('PSF convoluted')
    fig.colorbar(pcm)

    ax = axes[0][1]
    pcm = ax.imshow(lowfreq_background_img, cmap='gnuplot2')
    ax.set_title('Low Freq Background signal')
    fig.colorbar(pcm)

    ax = axes[1][1]
    pcm = ax.imshow(psfconvolimg + lowfreq_background_img, cmap=parula)
    ax.set_title('Low Freq BG added')
    fig.colorbar(pcm)

    ax = axes[0][2]
    pcm_last = ax.imshow(dustimg, cmap='gnuplot2')
    ax.set_title('Dust scatter signal')
    fig.colorbar(pcm_last)

    ax = axes[1][2]
    pcm_last = ax.imshow(
        psfconvolimg + lowfreq_background_img + dustimg, cmap=parula)
    ax.set_title('Dust scatter added')
    fig.colorbar(pcm_last)

    plt.tight_layout()
    # plt.show(block=False)
    return psfconvolimg + lowfreq_background_img + dustimg


# dataCam1 = makeimg(imgwidth=100, 
    #    pparticlepixel=.0003, 
    #    particle_psf_r=3, 
    #    particlepsfmaxval=1000,
    #    bgfreq=1/400, 
    #    bgamp=300,
    #    clusterp=0.0, 
    #    avgclustersz=0, 
    #    particledistancecluster=0.00, 
    #    dustpsfr=10, 
    #    dustpsfmaxval=200, 
    #    pdustpixel=0.0001, 
    #    )

# PSFSigma = 1.39
# [coordsCam1,detParCam1,cutProcessCam1] = LLRMapv2(dataCam1,PSFSigma,[],-2)