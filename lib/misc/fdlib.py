#!/usr/bin/env python

# ------------------------------------------------------------------------------
"""
fdlib: Module containing functions useful for transforming and reconstructing
       ct data as well as for analyzing reconstructions/ projections.
"""
# ------------------------------------------------------------------------------

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


import matplotlib as mpl
# mpl.use('TkAgg')

import scipy.ndimage.filters as fi
import scipy.sparse as sp
import scipy.optimize as op
from scipy.fftpack import fft

from skimage import draw as dr
from skimage.transform import _warps
from skimage.morphology import reconstruction, square, disk, cube, ball
from sklearn import metrics as mt

import astra

from lib.misc.util import *
from lib.misc.ctlib import *


def cp_projection_to_hl_projection(A_c, A_p,
                                   spctrm_h, spctrm_l,
                                   pc_h, pc_l,
                                   neglog=True):
    """
    ---------------------------------------------------------------------------
    Given the pc sinograms, return the hl sinograms. Mostly used as part of
    the forward model.

    :param A_p - Photoelectric line integrals
    :param A_c - Compton line integrals
    :param neglog - True for logarithmic projection, False for photon count
    :return  sino_h - high energy projection/count sinogram
    :return  sino_l - low energy projection/count sinogram
    ---------------------------------------------------------------------------
    """

    flat = True
    if A_p.ndim > 1 and A_c.ndim > 1:
        # All numerical calculations are done with flat arrays
        flat = False
        A_p = A_p.flatten()
        A_c = A_c.flatten()
        shp = A_c.shape
    # Calculate projection values
    A_l = -outer(A_c, klein_nishina(spctrm_l[:, 0])) - \
          outer(A_p, photoelectric(spctrm_l[:, 0]))
    A_l = dot(exp(A_l), spctrm_l[:, 1])
    A_h = -outer(A_c, klein_nishina(spctrm_h[:, 0])) - \
          outer(A_p, photoelectric(spctrm_h[:, 0]))
    A_h = dot(exp(A_h), spctrm_h[:, 1])
    # Account for the actual photon count
    A_l *= pc_l
    A_h *= pc_h

    if neglog:
        # Convert photon count to attenuation (line integrals)
        A_h = maximum(A_h, 1)
        A_l = maximum(A_l, 1)
        A_l = -log(A_l) + log(pc_l)
        A_h = -log(A_h) + log(pc_h)

    if not flat:
        A_h = A_h.reshape(shp)
        A_l = A_l.reshape(shp)

    return A_h, A_l
# -----------------------------------------------------------------------------


def local_mean_filter(sino, k):
    """-------------------------------------------------------------------------
    Simply low pass filtering with kernel size of k.

    :param   sino_pc - input sinogram in photon counts
    :return  sino - output sinogram with destreaking
    -------------------------------------------------------------------------"""

    kernel = ones((k, k)) * 1.0 / k ** 2
    return fi.convolve(sino, kernel, mode='reflect')
# -----------------------------------------------------------------------------


def pc_to_logp(pc, pc0):
    """-------------------------------------------------------------------------
    Photon count to logarithmic projection

    :param      pc  - input photo count
    :param      pc0 - dose in number of photons
    :return:    p   - logarithmic projection
    -------------------------------------------------------------------------"""
    return -log(pc / pc0)

def logp_to_pc(p, pc0):
    """-------------------------------------------------------------------------
    Logarithmic projection to photon count

    :param     p   - logarithmic projection
    :param     pc0 - dose in number of photons
    :return    pc  - output photon count
    -------------------------------------------------------------------------"""
    return exp(-p) * pc0

def exp_seg(x):
    """-------------------------------------------------------------------------
    Desc.:      Exponential map used for segmentation
    Args.:      x       - input numpy array
    Returns:    m       - returned segmentation map
    -------------------------------------------------------------------------"""
    return exp(-abs(x - 1))


def mask_in_range(x, sig):
    """-------------------------------------------------------------------------
    Desc.:      Return a mask where the pixel values are in range (1-sig, 1+sig)
    Args.:      x       - input numpy array
                sig     - threshold value
    Returns:    m       - returned mask
    -------------------------------------------------------------------------"""
    return logical_and(x <= (1 + sig), x >= (1 - sig))


def precision_with_mask(x, mask, sig=None):
    """-------------------------------------------------------------------------
    Desc.:      Given the ground truth mask, evaluate the precision: the
                percentage of pixels in the range (1-sig, 1+sig) that actually
                resides in the masked region.
    Args.:      x       - input numpy array
                mask    - ground truth mask
                sig     - threshold value
    Returns:    prec    - calculated precision value
    -------------------------------------------------------------------------"""
    if x.dtype == bool:
        fgd_mask = x
    elif x.dtype == int:
        fgd_mask = x == 1
    else:
        fgd_mask = logical_and(x <= (1 + sig), x >= (1 - sig))  # Foreground mask
    nfgd = sum(fgd_mask)
    gt_fgd_mask = logical_and(fgd_mask, mask)
    ngt_fgd = sum(gt_fgd_mask)
    prec = ngt_fgd * 1.0 / nfgd
    return prec


def recall_with_mask(x, mask, sig=None):
    """-------------------------------------------------------------------------
    Desc.:      Given the ground truth mask, evaluate the recall: the
                percentage of pixels in the true mask that actually belongs
                to the range (1-sig, 1+sig) in the input image. Also could
                called true positive rate
    Args.:      x       - input numpy array
                mask    - ground truth mask
                sig     - threshold value
    Returns:    recl    - calculated recall value
    -------------------------------------------------------------------------"""
    if x.dtype == bool:
        fgd_mask = x
    elif x.dtype == int:
        fgd_mask = x == 1
    else:
        fgd_mask = logical_and(x <= (1 + sig), x >= (1 - sig))  # Foreground mask
    gt_fgd_mask = logical_and(fgd_mask, mask)
    ngt_fgd = sum(gt_fgd_mask)
    ngt = sum(mask)
    recl = ngt_fgd * 1.0 / ngt
    return recl


def pr_curve(x, v, gtmask):
    """-------------------------------------------------------------------------
    Desc.:      Given the ground truth mask and value, evaluate the precision-recall curve.
    Args.:      x       - input image numpy array
                v       - ground truth value
                gtmask  - ground truth mask
    Returns:    prec    - calculated precision value
                recl    - calculated recall value
    -------------------------------------------------------------------------"""
    sig_list = arange(0.05, 1, 0.05)
    prec = []
    recl = []
    for sig in sig_list:
        v_mask = mask_in_range(exp_seg(x / v), sig)
        prec.append(precision_with_mask(v_mask, gtmask, 0))
        recl.append(recall_with_mask(v_mask, gtmask, 0))
    return prec, recl


def pr_curve_cp(xc, xp, c, p, gtmask):
    """-------------------------------------------------------------------------
    Desc.:      Given the ground truth mask and value, evaluate the precision-recall curve.
    Args.:      xc      - input Compton numpy array
                xp      - input photoelectric numpy array
                c       - ground truth Compton value
                p       - ground truth photoelectric value
                gtmask  - ground truth mask
    Returns:    prec    - calculated precision value
                recl    - calculated recall value
    -------------------------------------------------------------------------"""
    sig_list = arange(0.05, 1, 0.05)
    pc_prec = []
    pc_recl = []
    for sig in sig_list:
        c_mask = mask_in_range(exp_seg(xc / c), sig)
        p_mask = mask_in_range(exp_seg(xp / p), sig)
        pc_mask = logical_and(c_mask, p_mask)
        pc_prec.append(precision_with_mask(pc_mask, gtmask, 0))
        pc_recl.append(recall_with_mask(pc_mask, gtmask, 0))
    return pc_prec, pc_recl


def filter_projection(radon_image, theta=None, filter="ramp"):
    """-------------------------------------------------------------------------
    Desc.:      Perform filtering on each column of the input radon_image. This
                function is specifically and internally used in variance
                propagation.
    Args.:
    Returns:
    -------------------------------------------------------------------------"""
    if radon_image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta is None:
        m, n = radon_image.shape
        nangs = n
    else:
        nangs = len(np.asarray(theta))

    # resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = \
        max(64, int(2 ** np.ceil(np.log2(2 * radon_image.shape[0]))))
    pad_width = ((0, projection_size_padded - radon_image.shape[0]), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)
    # print "img", img.shape
    # Construct the Fourier filter
    f = fftfreq(projection_size_padded).reshape(-1, 1)  # digital frequency
    omega = 2 * np.pi * f  # angular frequency
    fourier_filter = 2 * np.abs(f)  # ramp filter
    if filter == "ramp":
        pass
    elif filter == "shepp-logan":
        # Start from first element to avoid divide by zero
        fourier_filter[1:] = fourier_filter[1:] * np.sin(omega[1:]) / omega[1:]
    elif filter == "cosine":
        fourier_filter *= np.cos(omega)
    elif filter == "hamming":
        fourier_filter *= (0.54 + 0.46 * np.cos(omega / 2))
    elif filter == "hann":
        fourier_filter *= (1 + np.cos(omega / 2)) / 2
    elif filter is None:
        fourier_filter[:] = 1
    else:
        raise ValueError("Unknown filter: %s" % filter)
    # Apply filter in Fourier domain
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0))

    # Resize filtered image back to original size
    radon_filtered = radon_filtered[:radon_image.shape[0], :]

    # Same weird filter that iradon applies in the end
    return radon_filtered * np.pi / (2 * nangs)


def _radon_transpose(radon_image, theta, angs, width):
    """-------------------------------------------------------------------------
    Desc.:      Matrix-free implementation for the R^t.dot(v) operation,
                but the structure is specifically designed only for variance
                propagation. For other usage, the other radon transpose should
                be used.
    Args.:
    Returns:
    -------------------------------------------------------------------------"""
    # In radon_image, every column will be a projection, n_sino_pxls columns
    # in total
    img_shp = (width, width)
    n_img_pxls = width ** 2
    n_sino_pxls = radon_image.shape[1]
    # Output is basically the dense Rinv matrix
    out = np.zeros((n_img_pxls, n_sino_pxls), dtype=np.float32)
    # Initialize the padded image (sqrt(2) times bigger than original image)
    diagonal = np.sqrt(2) * max(img_shp)
    pad = [int(np.ceil(diagonal - s)) for s in img_shp]
    new_center = [(s + p) // 2 for s, p in zip(img_shp, pad)]
    old_center = [s // 2 for s in img_shp]
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
    padded_image = np.pad(np.zeros(img_shp), pad_width, mode='constant',
                          constant_values=0)
    # Rotate back image
    # pdd_shp = padded_image.shape
    center = padded_image.shape[0] // 2

    shift0 = np.array([[1, 0, -center],
                       [0, 1, -center],
                       [0, 0, 1]])
    shift1 = np.array([[1, 0, center],
                       [0, 1, center],
                       [0, 0, 1]])

    # build a rotation matrix around the center pixel
    def build_rotation(theta):
        T = np.deg2rad(theta)
        R = np.array([[np.cos(T), np.sin(T), 0],
                      [-np.sin(T), np.cos(T), 0],
                      [0, 0, 1]])
        return shift1.dot(R).dot(shift0)

    hw = width // 2
    for i in range(n_sino_pxls):
        # Smear back the projection vector for every angle
        ang = -theta[angs[i]]
        prj = radon_image[:, i].reshape(1, -1)
        rotated = repeat(prj, padded_image.shape[0], axis=0)
        rotated = _warps._warp_fast(rotated, build_rotation(ang))
        # Convert back to size of an unpadded image
        out[:, i] = rotated[new_center[0] - hw: new_center[0] + hw,
                    new_center[1] - hw: new_center[1] + hw].flatten()
    return out


def radon_transpose(sino, img_shp, sino_shp, circle):
    """-------------------------------------------------------------------------
    Desc.:  Matrix-free implementation for the R^t.dot(v) operation.
    Args.:  sino        - flattened sinogram vector
            sino_shp    - (nbins, nangs)
            img_shp     - (width, height)
            circle      - same as the circle argument in radon()
    Returns:out         - flattened image vector
    -------------------------------------------------------------------------"""
    width = img_shp[0]
    nangs = sino_shp[1]
    theta = linspace(0., 180.0, nangs, endpoint=False)
    sino = sino.reshape(sino_shp)
    # Output is basically the dense Rinv matrix
    out = np.zeros(img_shp, dtype=np.float32)
    # Initialize the padded image (sqrt(2) times bigger than original image)
    diagonal = np.sqrt(2) * max(img_shp)
    pad = [int(np.ceil(diagonal - s)) for s in img_shp]
    new_center = [(s + p) // 2 for s, p in zip(img_shp, pad)]
    old_center = [s // 2 for s in img_shp]
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
    padded_image = np.pad(np.zeros(img_shp), pad_width, mode='constant',
                          constant_values=0)
    # Rotate back image
    # pdd_shp = padded_image.shape
    center = padded_image.shape[0] // 2

    shift0 = np.array([[1, 0, -center],
                       [0, 1, -center],
                       [0, 0, 1]])
    shift1 = np.array([[1, 0, center],
                       [0, 1, center],
                       [0, 0, 1]])

    # build a rotation matrix around the center pixel
    def build_rotation(theta):
        T = np.deg2rad(theta)
        R = np.array([[np.cos(T), np.sin(T), 0],
                      [-np.sin(T), np.cos(T), 0],
                      [0, 0, 1]])
        return shift1.dot(R).dot(shift0)

    hw = width // 2
    for i in range(nangs):
        # Smear back the projection vector for every angle
        ang = -theta[i]
        prj = sino[:, i].reshape(1, -1)
        rotated = repeat(prj, padded_image.shape[0], axis=0)
        rotated = _warps._warp_fast(rotated, build_rotation(ang))
        # Convert back to size of an unpadded image
        if circle:
            out += rotated[new_center[0] - hw: new_center[0] + hw, :]
        else:
            out += rotated[new_center[0] - hw: new_center[0] + hw,
                   new_center[1] - hw: new_center[1] + hw]
    return out.flatten()


def lac_from_cp(e, c, p):
    """-------------------------------------------------------------------------
        Desc.:      Return Mass Attenuation Coefficients from Compton/PE values
        Args.:      e   - energy levels in keV
                    c   - Compton coefficient
                    p   - photoelectric coefficient
        Return:     mac - output array of MAC
    -------------------------------------------------------------------------"""
    return c * klein_nishina(e) + p * photoelectric(e)


def conj_grad(A, b, x0, maxiter=0, tol=0.01, M=None,
              callback=None, verbose=False):
    if not isinstance(A, sp.linalg.LinearOperator):
        A = sp.linalg.aslinearoperator(A)
    if not isinstance(M, sp.linalg.LinearOperator):
        M = sp.linalg.aslinearoperator(M)
    if maxiter == 0:
        maxiter = b.size
    i = 1
    r = b - A.matvec(x0)
    d = r
    delta_new = r.T.dot(r)
    delta_0 = delta_new
    x = x0
    while True:
        if verbose:
            print("CG: %3d %10.5E" % (i, delta_new))
        q = A.matvec(d)
        alpha = delta_new * 1.0 / (d.T.dot(q))
        x = x + alpha * d
        # if i > 0 and i % 50 == 0:
        #     r = b - A.matvec(x)
        # else:
        #     r = r - alpha * q
        r = r - alpha * q
        if M is not None:
            rtilde = M.matvec(r)
        else:
            rtilde = r
        delta_old = delta_new
        delta_new = r.T.dot(rtilde)
        # beta = -float((r.T * A * d) / float(d.T * A * d))
        beta = delta_new * 1.0 / delta_old
        d = rtilde + beta * d
        i += 1
        callback(x)
        if i > maxiter:
            if verbose:
                print("CG: Max iter has been reached.")
            return x, i
        if norm(r) <= maximum(tol * norm(b), tol):
            # if delta_new < tol * delta_0:
            if verbose:
                print("CG: dtol has been satisfied. d0, dnew = %.2f, %.2f" \
                      % (delta_0, delta_new))
            return x, 0


class wrapped_astra_projector(object):
    """-------------------------------------------------------------------------
        Desc.:      Wrapper class for using Astra projector as a linear operator
        Args.:      -
        Return:     -
    -------------------------------------------------------------------------"""

    def __init__(self, proj_type, proj_geom, vol_geom):
        self.proj_geom = proj_geom
        self.vol_geom = vol_geom
        self.proj_id = astra.create_projector(proj_type, proj_geom, vol_geom)
        self.shape = (
            proj_geom['DetectorCount'] * len(proj_geom['ProjectionAngles']),
            vol_geom['GridColCount'] * vol_geom['GridRowCount'])
        self.dtype = np.float

    def matvec(self, v):
        sid, s = astra.create_sino(np.reshape(v, (
            self.vol_geom['GridRowCount'], self.vol_geom['GridColCount'])),
                                   self.proj_id)
        astra.data2d.delete(sid)
        return s.transpose().ravel()

    def rmatvec(self, v):
        v = np.reshape(v, (self.proj_geom['DetectorCount'], len(
            self.proj_geom['ProjectionAngles']),)).transpose()
        bid, b = astra.create_backprojection(v, self.proj_id)
        astra.data2d.delete(bid)
        return b.ravel()

def ramp_filter(width, show_plots=False):
    projection_size_padded = \
        max(128, int(2 ** np.ceil(np.log2(2 * width))))
    f = fftfreq(projection_size_padded)
    f1, f2 = meshgrid(f, f)
    finv_frq = sqrt(f1 ** 2 + f2 ** 2)
    if show_plots:
        quick_imshow(1, 1, log(1 + abs(fftshift(finv_frq))),
                     'Inverse of Frequency Response')
    return finv_frq


def imimposemin(I, BW, conn=None, max_value=255):
    '''
    Python implementation of the imimposemin function in MATLAB.

    Reference: https://www.mathworks.com/help/images/ref/imimposemin.html
    '''
    if not I.ndim in (2, 3):
        raise Exception("'I' must be a 2-D or 3D array.")

    if BW.shape != I.shape:
        raise Exception("'I' and 'BW' must have the same shape.")

    if BW.dtype is not bool:
        BW = BW != 0

    # set default connectivity depending on whether the image is 2-D or 3-D
    if conn == None:
        if I.ndim == 3:
            conn = 26
        else:
            conn = 8
    else:
        if conn in (4, 8) and I.ndim == 3:
            raise Exception("'conn' is invalid for a 3-D image.")
        elif conn in (6, 18, 26) and I.ndim == 2:
            raise Exception("'conn' is invalid for a 2-D image.")

    # create structuring element depending on connectivity
    if conn == 4:
        selem = disk(1)
    elif conn == 8:
        selem = square(3)
    elif conn == 6:
        selem = ball(1)
    elif conn == 18:
        selem = ball(1)
        selem[:, 1, :] = 1
        selem[:, :, 1] = 1
        selem[1] = 1
    elif conn == 26:
        selem = cube(3)

    fm = I.astype(float)

    try:
        fm[BW]                 = -math.inf
        fm[np.logical_not(BW)] = math.inf
    except:
        fm[BW]                 = -float("inf")
        fm[np.logical_not(BW)] = float("inf")

    if I.dtype == float:
        I_range = np.amax(I) - np.amin(I)

        if I_range == 0:
            h = 0.1
        else:
            h = I_range*0.001
    else:
        h = 1

    fp1 = I + h

    g = np.minimum(fp1, fm)

    # perform reconstruction and get the image complement of the result
    if I.dtype == float:
        J = reconstruction(1 - fm, 1 - g, selem=selem)
        J = 1 - J
    else:
        J = reconstruction(255 - fm, 255 - g, method='dilation', selem=selem)
        J = 255 - J

    try:
        J[BW] = -math.inf
    except:
        J[BW] = -float("inf")

    return J


def intersection_over_union(binary_mask_1, binary_mask_2):
    intersection = logical_and(binary_mask_1, binary_mask_2)
    union = logical_or(binary_mask_1, binary_mask_2)
    iou = sum(intersection) * 1.0 / sum(union)
    return iou