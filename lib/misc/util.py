import matplotlib as mpl

# mpl.use('TkAgg')
from pylab import *
from numpy import *

import os, pickle, pydicom
import scipy.sparse as sp
import scipy.misc as misc

from matplotlib.patches import Ellipse
from astropy.io import fits as pyfits
from pydicom import uid
from skimage.measure import regionprops
from skimage.transform import rescale

from lib.__init__ import *
from lib.misc.multi_processor import *
from PIL import Image
import imageio


class Logger(object):
    """
    ---------------------------------------------------------------------------
    Module Description:
    This module is a tool for directing sys.out to both a file and printing
    in terminal. Example usage:

    sys.stdout = Logger(log_name)

    ---------------------------------------------------------------------------
    """

    def __init__(self, fname):
        """
        -----------------------------------------------------------------------
        Constructor

        :param fname: file name
        -----------------------------------------------------------------------
        """

        self.terminal = sys.stdout
        self.fname = fname

    def write(self, message):
        """
        -----------------------------------------------------------------------
        put message on the log file

        :param message:     message string to add
        :return:
        -----------------------------------------------------------------------
        """

        self.terminal.write(message)
        self.log = open(self.fname, "a")
        self.log.write(message)
        self.log.close()

    def flush(self):
        """
        -----------------------------------------------------------------------
        this flush method is needed for python 3 compatibility.
        this handles the flush command by doing nothing.
        you might want to specify some extra behavior here.

        :return:
        -----------------------------------------------------------------------
        """

        pass
# =============================================================================
# Class Ends
# =============================================================================


def get_logger(lname, logfile):
    """
    ---------------------------------------------------------------------------
    Function to create a python logger for a given class.

    :param lname:       name for logger - printed on terminal
    :param logfile:     log file path
    :return:
    ---------------------------------------------------------------------------
    """

    # Create logger object
    logger = logging.getLogger(lname)
    logger.setLevel(logging.INFO)

    # Create file handler
    f_handler = logging.FileHandler(logfile, mode='a')
    s_handler = logging.StreamHandler(sys.stdout)

    # Create formatter
    formatter = logging.Formatter(
                        '[%(asctime)s] [%(name)s] %(levelname)s: %(message)s')
    f_handler.setFormatter(formatter)
    s_handler.setFormatter(formatter)

    logger.addHandler(f_handler)
    logger.addHandler(s_handler)
    logger.propagate = False

    return logger
# -----------------------------------------------------------------------------


def quick_imshow(nrows, ncols=1,
                 images=None,
                 titles=None,
                 colorbar=True,
                 vmax=None,
                 vmin=None,
                 figsize=None,
                 figtitle=None,
                 visibleaxis=False,
                 colormap='jet',
                 saveas=''):
    """-------------------------------------------------------------------------
    Convenience function that make subplots of imshow

    :param  nrows - number of rows
    :param  ncols - number of cols
    :param  images - list of images
    :param  titles - list of titles
    :param  vmax - tuple of vmax for the colormap. If scalar, the same value is
                   used for all subplots. If one of the entries is None, no
                   colormap for that subplot will be drawn.
    :param  vmin - tuple of vmin

    :return: f - the figure handle
             axes - axes or array of axes objects
             caxes - tuple of axes image
    -------------------------------------------------------------------------"""

    if isinstance(nrows, ndarray):
        images = nrows
        nrows = 1
        ncols = 1

    if figsize == None:
        s = 3.5
        if figtitle:
            figsize = (s * ncols, s * nrows + 0.5)
        else:
            figsize = (s * ncols, s * nrows)

    if nrows == ncols == 1:
        f, ax = subplots(figsize=figsize)
        cax = ax.imshow(images, cmap=colormap, vmax=vmax, vmin=vmin)
        if colorbar:
            f.colorbar(cax, ax=ax)
        if titles != None:
            ax.set_title(titles)
        if figtitle != None:
            f.suptitle(figtitle)
        cax.axes.get_xaxis().set_visible(visibleaxis)
        cax.axes.get_yaxis().set_visible(visibleaxis)
        return f, ax, cax

    f, axes = subplots(nrows, ncols, figsize=figsize)
    caxes = []
    i = 0
    for ax, img in zip(axes.flat, images):
        if isinstance(vmax, tuple) and isinstance(vmin, tuple):
            if vmax[i] is not None and vmin[i] is not None:
                cax = ax.imshow(img, cmap=colormap, vmax=vmax[i], vmin=vmin[i])
            else:
                cax = ax.imshow(img, cmap=colormap)
        elif isinstance(vmax, tuple) and vmin is None:
            if vmax[i] is not None:
                cax = ax.imshow(img, cmap=colormap, vmax=vmax[i], vmin=0)
            else:
                cax = ax.imshow(img, cmap=colormap)
        elif vmax is None and vmin is None:
            cax = ax.imshow(img, cmap=colormap)
        else:
            cax = ax.imshow(img, cmap=colormap, vmax=vmax, vmin=vmin)
        if titles != None:
            ax.set_title(titles[i])
        if colorbar:
            f.colorbar(cax, ax=ax)
        caxes.append(cax)
        cax.axes.get_xaxis().set_visible(visibleaxis)
        cax.axes.get_yaxis().set_visible(visibleaxis)
        i = i + 1
    if figtitle != None:
        f.suptitle(figtitle)
    if saveas != '':
        f.savefig(saveas)
    return f, axes, tuple(caxes)
# ------------------------------------------------------------------------------


def update_subplots(images,
                    caxes,
                    f=None,
                    axes=None,
                    indices=(),
                    vmax=None,
                    vmin=None):
    """
    ----------------------------------------------------------------------------
    Update subplots in a figure

    :param images  - new images to plot
    :param caxes   - caxes returned at figure creation
    :param indices - specific indices of subplots to be updated

    :return
    ----------------------------------------------------------------------------
    """

    for i in range(len(images)):
        if len(indices) > 0:
            ind = indices[i]
        else:
            ind = i
        img = images[i]
        caxes[ind].set_data(img)
        cbar = caxes[ind].colorbar
        if isinstance(vmax, tuple) and isinstance(vmin, tuple):
            if vmax[i] is not None and vmin[i] is not None:
                cbar.set_clim([vmin[i], vmax[i]])
            else:
                cbar.set_clim([img.min(), img.max()])
        elif isinstance(vmax, tuple) and vmin is None:
            if vmax[i] is not None:
                cbar.set_clim([0, vmax[i]])
            else:
                cbar.set_clim([img.min(), img.max()])
        elif vmax is None and vmin is None:
            cbar.set_clim([img.min(), img.max()])
        else:
            cbar.set_clim([vmin, vmax])
        cbar.update_normal(caxes[ind])

    pause(0.01)
    tight_layout()
# ------------------------------------------------------------------------------


def slide_show(image, dt=0.01, vmax=None, vmin=None):
    """
    ---------------------------------------------------------------------------
    Slide show for visualizing an image volume. Image is (w, h, d)

    :param image: (w, h, d), slides are 2D images along the depth axis
    :param dt:      transition time
    :param vmax:    maximum cliiping value
    :param vmin:    minimum clipping value
    :return:
    ---------------------------------------------------------------------------
    """

    if image.dtype == bool:
        image *= 1.0
    if vmax is None:
        vmax = image.max()
    if vmin is None:
        vmin = image.min()
    plt.ion()
    plt.figure()
    for i in arange(image.shape[2]):
        plt.cla()
        cax = plt.imshow(image[:, :, i], cmap='jet', vmin=vmin, vmax=vmax)
        plt.title(str('Slice: %i' % i))
        if i == 0:
            cf = plt.gcf()
            ca = plt.gca()
            cf.colorbar(cax, ax=ca)
        plt.pause(dt)
        plt.draw()
# -----------------------------------------------------------------------------


def scatter_ellipse(X, labels, mu, R, figsize=(5, 5), s=0.01, alpha=0.1):
    """
    ---------------------------------------------------------------------------
    2D scatter plot with ellipse drawn based on mean and covariance.

    :param X: samples, (N, 2)
    :param labels: integer labels, (N,)
    :param mu: centroids, (k, 2)
    :param R: covariances, (k, 2, 2)
    :return:
    ---------------------------------------------------------------------------
    """
    k = len(unique(labels))

    f, ax = subplots(figsize=figsize)
    ax.scatter(X[:, 0], X[:, 1],
               s=s, c=labels, alpha=alpha, cmap='jet')

    for m in range(k):
        vals, vecs = eigh(R[m])
        x, y = vecs[:, 0]
        w, h = 2 * sqrt(vals)
        theta = degrees(arctan2(y, x))
        ax.add_artist(
            Ellipse(xy=mu[m],
                    width=w,
                    height=h,
                    angle=theta,
                    fill=False,
                    edgecolor='r'))

    return f, ax
# -----------------------------------------------------------------------------


def read_fits_data(input_file_name, field=1):
    """
    ---------------------------------------------------------------------------
    Loads a FITS image file

    :param input_file_name - file path
    :return image as a numpy ndarray
    ---------------------------------------------------------------------------
    """

    return  pyfits.open(input_file_name,
                        ignore_missing_end=True)[field].data
# -----------------------------------------------------------------------------


def save_fits_data(file_path, out_image, compress=False):
    """
    ---------------------------------------------------------------------------
    Save an image as a FITS file

    :param file_path:   path to the fits file
    :param out_image:   output image to be saved
    :return:
    ---------------------------------------------------------------------------
    """

    if os.path.exists(file_path):
        os.remove(file_path)

    imheader = pyfits.Header()

    if compress:
        hdu_list = pyfits.CompImageHDU(out_image, imheader)
    else:
        hdu_list = pyfits.PrimaryHDU(out_image, imheader)

    hdu_list.writeto(file_path)
# -----------------------------------------------------------------------------


def create_pil_collage(images, fpath, layout=None, vlims=None):
    """
    ---------------------------------------------------------------------------

    :param images:
    :param fpath:
    :param layout:
    :param vlims:
    :return:
    ---------------------------------------------------------------------------
    """

    if layout is None: layout = (len(images), 1)

    if vlims is not None:
        images = [np.clip(x, vlims[0], vlims[1])/vlims[1]*255 for x in images]
    else:
        images = [x/x.max()*255 for x in images]

    assert layout[0]*layout[1] == len(images)

    rows, cols = layout

    collage = np.vstack([np.hstack(images[x*cols:(x+1)*cols])
                         for x in range(rows)])
    im = Image.fromarray(collage)
    im = im.convert('L')
    im.save(fpath)
# -------------------------------------------------------------------------


def create_gif(fname, input_vol, stride=1, scale=None):
    """

    :param fname:
    :param input_vol:
    :param stride:
    :return:
    """

    input_vol = (input_vol-input_vol.min())/(input_vol.max()-input_vol.min())
    input_vol = (input_vol*255).astype(uint8)

    if scale is None:
        imageio.mimsave(fname,
                        [input_vol[:,:, z]
                         for z in range(0,input_vol.shape[2], stride)],
                        fps=5)
    else:
        imageio.mimsave(fname,
                        [rescale(input_vol[:,:, z], scale=scale, preserve_range=True)
                         for z in range(0,input_vol.shape[2], stride)],
                        fps=5)

