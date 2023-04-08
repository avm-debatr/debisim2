#!/usr/bin/env python

# ------------------------------------------------------------------------------
"""ctlib: Module containing functions useful for handling, processing and 
          analyzing ct image/sinogram data."""

__author__      = "Ankit Manerikar"
__copyright__   = "Copyright (C) 2020, Robot Vision Lab"
__date__        = "12th January, 2021"
__credits__     = ["Ankit Manerikar", "Fangda Li"]
__license__     = "Public Domain"
__version__     = "2.0.0"
__maintainer__  = ["Ankit Manerikar", "Fangda Li"]
__email__       = ["amanerik@purdue.edu", "li1208@purdue.edu"]
__status__      = "Prototype"
# ------------------------------------------------------------------------------

import time
from numpy import *
import skimage.transform as tr
import scipy.ndimage.filters as ndfilt


def read_sinogram_da_file(file_location):
    """
    ----------------------------------------------------------------------------
    Program to read .da sinogram files

    :param file_location:       location of .da file
    :return: 2D numpy array containing the sinogram
    ----------------------------------------------------------------------------
    """
    # read header from binary file
    header =  fromfile(file_location, dtype= dtype('>H'), count=256)

    # get row and column values from header
    col = header[0]
    row = header[1]

    # calculate number of blocks per column
    block = int( ceil(col * 4.0 / 512))

    # read file
    sino =  fromfile(file_location, dtype= dtype('>f'))
    # exclude header and reshape
    sino = sino[-block*128*row:].reshape(row, block*128)

    return sino[:row, :col]
# ------------------------------------------------------------------------------


def read_image_mi_file(file_location):
    """
    ----------------------------------------------------------------------------
    Program to read .mi image file

    :param file_location:   location of .mi file
    :return: CT image in Hounsfeld Units representing mu at 65 keV
    ----------------------------------------------------------------------------
    """

    OFF = 0
    PARSE_MI_LOC_MAGIC = 0  # location of magic number
    PARSE_MI_LOC_N = 1      # location of number of images
    PARSE_MI_LOC_FIRST = 2  # location of first block

    PARSE_MI_MAGIC_65K = 12431     # magic number for 65K files

    # pointers in image header
    PARSE_MI_LOC_NCOL = 40     # location of ncol
    PARSE_MI_LOC_NROW = 41     # location of nrow

    header_line_length = 512

    bin_data =  fromfile(file_location, dtype= dtype('>H'))

    if bin_data[OFF+PARSE_MI_LOC_MAGIC] != PARSE_MI_MAGIC_65K:
        print("Invalid or corrupted .mi file!")
        return False

    header = bin_data[:header_line_length]
    byte_num = header[PARSE_MI_LOC_N+OFF]

    cols = header[OFF+header_line_length/2+PARSE_MI_LOC_NCOL]
    rows = header[OFF+header_line_length/2+PARSE_MI_LOC_NROW]
    tmp = bin_data[header_line_length:]
    image = tmp.reshape(cols, rows)
    return image.T
# ------------------------------------------------------------------------------


def klein_nishina(e):
    """
    ----------------------------------------------------------------------------
    Calculate the Klein-Nishina basis function

    :param e - energy level
    :return Klein-Nishina cs value
    ----------------------------------------------------------------------------
    """

    a = e * 1.0 / 510.975
    opa = 1 + a
    op2a = 1 + 2 * a
    op3a = 1 + 3 * a
    t1 = 2 * opa / op2a
    t2 = 1 / a *  log(op2a)
    t3 = t2 / 2
    t4 = op3a / op2a ** 2

    return opa / (a ** 2) * (t1 - t2) + t3 - t4
# ------------------------------------------------------------------------------


def photoelectric(e):
    """
    ----------------------------------------------------------------------------
    Calculate the Photoelectric basis function

    :param e:   e - energy level
    :return: PE basis value
    ----------------------------------------------------------------------------
    """
    return (e * 1.0) ** -3

# ------------------------------------------------------------------------------


# Vectorization of the PE and Klein-nishina basis functions
compton_basis =  vectorize(klein_nishina)
pe_basis      =  vectorize(photoelectric)


def load_spectrum(spectrum_file_path):
    """
    ----------------------------------------------------------------------------
    Load the spectrum file from the given path

    :param spectrum_file_path:
    :return:   1D-array covering normalized spectrum values from 10keV to X keV
               where X is the kV rating of the source for which the spectrum is
               obtained.
    ----------------------------------------------------------------------------
    """
    spec_data =  loadtxt(spectrum_file_path)
    spectrum  = spec_data[:,1]
    startkV   = spec_data[0, 0]
    endkV     = spec_data[-1, 0]

    return spectrum, startkV, endkV
# ------------------------------------------------------------------------------


def effective_atomic_number(img_p, img_c):
    """
    ----------------------------------------------------------------------------
    Generate Z_eff image

    :param  img_p   - photoelectric image
    :param  img_c   - compton image
    :return img_z   - Z_eff image
    ---------------------------------------------------------------------"""

    Kp = 1 / 2.585
    n = 3.5

    if  isscalar(img_p) and  isscalar(img_c):
        return Kp * (img_p / img_c) ** (1 / n)
    else:
        img_p = abs(img_p)
        img_c = abs(img_c)
        img_z =  zeros(img_c.shape)
        nz = img_c.nonzero()
        img_z[nz] = Kp * (img_p[nz] / img_c[nz]) ** (1 / n)
        return img_z
# --------------------------------------------------------------------------


def combine_poly_energetic_sinograms(sino_array, keV_spectrum,
                                     system_gain=0.00025954,
                                     startkV=10, endkV=95,
                                     num_photons=1.8e5):
    """
    ----------------------------------------------------------------------------
    Code for combining poly energetic sinograms into a single sinogram

    :param sino_array:      array of polyenergetic sinograms
    :param keV_spectrum:    spectrum of the X-ray source
    :param system_gain:     DC gain of the system
    :param lowkV:           low end of the spectrum
    :param highkV:          high end of the spectrum
    :param num_photons:     photon count

    :return: sinogram_ideal, sinogram_noisy
    ----------------------------------------------------------------------------
    """
    # print sino_array.shape
    energy_weights =  arange(startkV, endkV+1)*system_gain

    # Energy-wise photon distribution
    photon_dist = num_photons*keV_spectrum
    flat_field_photon_count =  multiply(photon_dist, energy_weights)

    # 3D array of sinogram projections for each energy
    sino_projections =  multiply( exp(-sino_array),
                                   photon_dist)

    # Calculate polyenergetic sinogram projection
    # as a sum of monoenergetic sinograms
    sino_ideal =  sum( multiply(sino_projections,
                                    energy_weights), axis=2)

    # Generate polyenergetic sinogram projection with poisson noise
    sino_noisy =  sum( multiply( random.poisson(sino_projections),
                                    energy_weights),
                                    axis=2)

    # Add system shot noise to the sinograms
    u =  sqrt(20*65*system_gain)* random.normal(size=sino_noisy.shape)
    sino_noisy = sino_noisy + u

    # remove illegal values generated from noise
    sino_noisy[sino_noisy<1] = 1

    # get sinogram from projections
    sino_ln_ideal = - log(sino_ideal*1.0/ sum(flat_field_photon_count))
    sino_ln_noisy = - log(sino_noisy*1.0/ sum(flat_field_photon_count))

    # remove illegal values generated from noise
    sino_ln_ideal[sino_ideal==0]=0

    return sino_ln_ideal.T, sino_ln_noisy.T
# ------------------------------------------------------------------------------


def reconstruct_for_parallel_beam_geometry(sinogram, filter='ramp',
                                           blurring=[], i_scale=1.0,
                                           mhu_output=False):
    """
    ----------------------------------------------------------------------------
    Reconstruction for parallel beam geometry of a CT scanner.

    :param sinogram:        input sinogram
    :param filter:          filter to be used for back-projection
    :param size:            output image size
    :param blurring:        blurring filter to be added
    :param i_scale:         output image scale
    :param mhu_output:      set to true if output is to be in mhus

    :return: mu_image   - reocnstruct LAc image
    ----------------------------------------------------------------------------
    """
    mu_water = 0.202527
    if filter in ['ramp', 'shepp-logan', 'hann', 'cosine']:

        if blurring == []:
            mu_image = tr.iradon(sinogram, circle=True, filter='ramp').T
        else:
            filtered_sino = ndfilt.convolve1d(sinogram,
                                              blurring, axis=1)
            mu_image = tr.iradon(filtered_sino, circle=True, filter='ramp').T

    mu_image = i_scale * mu_image
    
    if mhu_output:
        mu_image = (mu_image * i_scale - mu_water) / mu_water * 1000 + 1000
        return mu_image
    else:
        mu_image[mu_image<0] = 0
        return mu_image
# ------------------------------------------------------------------------------


def ram_lak_filter(n_filt):
    """
    ----------------------------------------------------------------------------
    Ram-Lak filter for sinogram of resolution n_filt.

    :param n_filt:      resolution of sinogram
    :return: n_filt x1 filter array
    ----------------------------------------------------------------------------
    """
    w_right =  arange(1, n_filt, 2)
    ram_lak_filter =  zeros(2 * n_filt)
    ram_lak_filter[n_filt + w_right] = -1.0 / ( pi * w_right) ** 2
    ram_lak_filter = ram_lak_filter + ram_lak_filter[::-1]
    ram_lak_filter[n_filt] = 0.25
    return ram_lak_filter
# ------------------------------------------------------------------------------


def calculate_pe_compton_coeffs(kev_range, mu_spectrum, density=1.0):
    """
    ----------------------------------------------------------------------------
    Get a least squares estimates of Photoelectric and compton efficients for
    a given LAC curve.

    :param kev_range:       keV energy range
    :param mu_spectrum:     mu spectrum over the given energy range
    :param density:         density of material if MAC curve is provided.
    :return: compton_value, pe_value - Compton, PE coefficients
    ----------------------------------------------------------------------------
    """

    c_basis   = compton_basis(kev_range)
    p_basis   = pe_basis(kev_range)
    basis_mat =  vstack((c_basis, p_basis)).T
    results =  dot( linalg.pinv(basis_mat), mu_spectrum.T)
    results *= density
    return results
# ------------------------------------------------------------------------------
