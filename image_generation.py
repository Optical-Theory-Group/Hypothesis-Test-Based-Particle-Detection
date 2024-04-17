import numpy as np
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import random
import nmmn.plots
from process_algorithms import integrate_gauss_1d

parula = nmmn.plots.parulacmap()

class Particle:
    def __init__(self, x, y, peak_val, psf_sd, clustered):
        """ The following values are inherent properties of a particle. psf_sd is not. """
        self.x = x
        self.y = y
        self.peak_val = peak_val
        self.psf_sd = psf_sd
        self.clustered = clustered

class Dust:
    def __init__(self, x, y, peak_val, psf_sd):
        """ The following values are inherent properties of a particle. psf_sd is not. """
        self.x = x
        self.y = y
        self.peak_val = peak_val
        self.psf_sd = psf_sd

# I'm trying to decide whether to make the psfconvolimg generation process per particle object to be the objects method or not.
        
def psfconvolution(particle_x, particle_y, multiplying_constant, psf_sd, imgwidth):
    """returns the pixel values to be added to the image based on psf convolution."""
    output = np.zeros((imgwidth, imgwidth))

    for x in range(imgwidth):
        for y in range(imgwidth):
            # integrate the psf over the pixel area for both dimensions
            integral_x = integrate_gauss_1d(x, particle_x, psf_sd)
            integral_y = integrate_gauss_1d(y, particle_y, psf_sd)
            
            # calculate the pixel value as the product of the 1d integrals in x and y, scaled by the multiplying constant
            pixel_value = multiplying_constant * integral_x * integral_y
            output[y, x] = pixel_value  # note: [y, x] for row, column indexing

    return output  
    # """Returns the pixel values to be added to the image

    # Args:
    #     particle_x (float): x coordinate of particle
    #     particle_y (float): y coordinate of particle
    #     multiplying_constant (float): constant to multiply the normalized 2D gaussian with
    #     psf_sd (float): standard deviation of the psf (radially symmetrical)
    #     imgwidth (int): width of the image (in pixels)

    # Returns:
    #     numpy.array(dtype=int): imgwidth x imgwidth array of the psf_convoluted image
    # """
    # output = np.zeros((imgwidth, imgwidth))
    # psfsd_squared = psf_sd ** 2

    # for x in range(imgwidth):
    #     for y in range(imgwidth):
    #         distance_squared = (x - particle_x) ** 2 + (y - particle_y) ** 2
    #         pixel_value = multiplying_constant / (4*np.pi) * np.exp(-(distance_squared / (2.0 * psfsd_squared)))
    #         output[y, x] = pixel_value # Personal note: remember the indexing here is [y,x]

    # return output

def create_hex_positions(n, particledistancecluster):
    """ Creates a list of positions in a hexagonal arrangement
    Args:
        n (int): Number of particles to be created
        particledistancecluster (float): Distance between particles in a cluster
    Returns:
        numpy.array(dtype=float): n x 2 array of positions
    """
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
    """ Creates a low frequency background signal
    Args:
        imgwidth (int): width of the image (in pixels)
        bgfreq (float): frequency of the signal
        amplitude (float): amplitude of the signal
    Returns:
        numpy.array(dtype=float): imgwidth x imgwidth array of the background signal
    """
    
    # the signal is a 2D cosine wave with a random center and rotation.
    center_x, center_y = random.randrange(imgwidth), random.randrange(imgwidth)
    theta = np.random.random() * 2 * np.pi

    # Create the output image numpy array
    outputimg = np.zeros((imgwidth, imgwidth))

    # Fill the output image numpy array
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

def create_particle_list(imgwidth, p_particlepixel, clusterp, avgclustersz, particledistancecluster, psf_sd, particle_multiplying_constant):
    """ Creates a list of particles
    Args:
        imgwidth (int): width of the image (in pixels)
        p_particlepixel (float): probability of a particle being present in a pixel
        clusterp (float): probability of a particle being clustered
        avgclustersz (int): average cluster size
        particledistancecluster (float): distance between particles in a cluster
        psf_sd (float): standard deviation of the psf (radially symmetrical)
        particle_multiplying_constant (float): constant to multiply the normalized 2D gaussian with
    Returns:
        list of Particle objects
    """

    particle_list = []
    n_particles_to_be_created = round(imgwidth ** 2 * p_particlepixel)
    for i in range(n_particles_to_be_created):
        x, y = np.random.rand() * imgwidth, np.random.rand() * imgwidth
        if np.random.random() < clusterp:
            particle_list.append(
                Particle(x, y, particle_multiplying_constant, psf_sd, True))
            # Add particles in its vicinity
            new_cluster_size = round(np.random.exponential(avgclustersz))
            if new_cluster_size > 1:
                relativepositions = create_hex_positions(
                    n=new_cluster_size, particledistancecluster=particledistancecluster)
                for rel_x, rel_y in relativepositions:
                    if x + rel_x > 0 and y + rel_y > 0:
                        particle_list.append(
                            Particle(x + rel_x, y + rel_y, particle_multiplying_constant, psf_sd, True))
        else:
            particle_list.append(
                Particle(x, y, particle_multiplying_constant, psf_sd, False))

    return particle_list


def create_dust_list(imgwidth, p_dustpixel, psf_sd, dust_multiplying_constant):
    """ Creates a list of dust particles
    Args:  
        imgwidth (int): width of the image (in pixels)
        p_dustpixel (float): probability of a dust particle being present in a pixel
        psf_sd (float): standard deviation of the psf (radially symmetrical)
        dust_multiplying_constant (float): constant to multiply the normalized 2D gaussian with
    Returns:
        list of Dust objects
    """

    dust_list = []
    n_dust_to_be_created = round(imgwidth ** 2 * p_dustpixel)
    for _ in range(n_dust_to_be_created):
        x, y = np.random.rand() * imgwidth, np.random.rand() * imgwidth
        dust_list.append(Dust(x, y, dust_multiplying_constant, 2 * psf_sd))
    return dust_list


def create_positions_img(imgwidth, particle_list, ):
    """ Creates an image of the particle positions
    Args:
        imgwidth (int): width of the image (in pixels)
        particle_list (list of Particle objects): list of particles
    Returns:
        numpy.array(dtype=int): imgwidth x imgwidth array of the particle positions (1 if a particle is present in a pixel, 0 otherwise)
    """
    
    positionsimg = np.zeros((imgwidth, imgwidth))
    for pt in particle_list:
        if round(pt.x) < imgwidth and round(pt.y) < imgwidth:
            positionsimg[round(pt.y), round(pt.x)] += 1
    return positionsimg


def makeimg(imgwidth=256, pparticlepixel=.1, particle_multiplying_constant=1000, psf_sd=2, bgfreq=1/256,
            bgamp=50, clusterp=0.2, avgclustersz=20, particledistancecluster=0.5,
            pdustpixel=0.005, dustpsfr=10, dust_multiplying_constant=500, vignettesd=128, imshow=False):  # background coherent speckle to be added
    """ Creates an image of the particle positions
    Args:   
        imgwidth (int): width of the image (in pixels)
        pparticlepixel (float): probability of a particle being present in a pixel
        particle_multiplying_constant (float): constant to multiply the normalized 2D gaussian with for particles
        psf_sd (float): standard deviation of the psf (radially symmetrical)
        bgfreq (float): frequency of the background signal
        bgamp (float): amplitude of the background signal
        clusterp (float): probability of a particle being clustered
        avgclustersz (int): average cluster size
        particledistancecluster (float): distance between particles in a cluster
        pdustpixel (float): probability of a dust particle being present in a pixel
        dustpsfr (float): psf_sd of the dust particle (radially symmetrical)
        dust_multiplying_constant (float): constant to multiply to the normalized 2D gaussian with for dusts
        vignettesd (float): standard deviation of the vignette
        imshow (bool): whether to show the image or not
    Returns:
        numpy.array(dtype=float): 
            imgwidth x imgwidth array of the produced image
    """
    # Create Particle objects and store them in a list
    particle_list = create_particle_list(
        imgwidth, pparticlepixel, clusterp, avgclustersz, particledistancecluster, 
        psf_sd, particle_multiplying_constant)

    # positionsimg = create_positions_img(imgwidth, particle_list)

    # Create psf convolved image
    psfconvolimg = np.zeros((imgwidth, imgwidth))

    # Create psf image for each particle
    for pt in particle_list:
        psfconvolimg += psfconvolution(pt.x, pt.y, pt.peak_val, pt.psf_sd, imgwidth)

    # Add low-frequency background
    lowfreq_background_img = lowfreq_background(imgwidth, bgfreq, bgamp)

    # Add dust particles
    dust_list = create_dust_list(
        imgwidth, pdustpixel, psf_sd, dust_multiplying_constant)
    dustimg = np.zeros((imgwidth, imgwidth))
    for dust in dust_list:
        dustimg += psfconvolution(dust.x, dust.y, dust.peak_val, pt.psf_sd, imgwidth)

    if imshow:
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        plt.suptitle(f'{pparticlepixel=}, {particle_multiplying_constant=}, {psf_sd=},'
                 '\n {clusterp=}, {avgclustersz=}, {particledistancecluster=},'
                 '\n {pdustpixel=}, {dust_multiplying_constant=}, {dustpsfr=}, ')

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
        plt.show(block=False)

    return psfconvolimg + lowfreq_background_img + dustimg

def run1():
    """This is what I want to run from inside this file (usually for testing)"""
    sz = 100 
    intensity=700
    image = makeimg(imgwidth=sz, pparticlepixel=3/6400, particle_multiplying_constant=intensity, 
                    psf_sd=1.39, bgfreq=0.002, bgamp=100, clusterp=0, avgclustersz=0, 
                    particledistancecluster=20/6400, pdustpixel=10/6400, dustpsfr=9, 
                    dust_multiplying_constant=intensity*0, vignettesd=0, ) 
    image += 500 * np.ones((sz, sz))
    image = np.random.poisson(image, size=(image.shape))
    plt.imshow(image, cmap='gray')
    plt.yticks([])
    plt.xticks([])
    plt.show(block=False)
    pass

if __name__ == "__main__":
    run1()