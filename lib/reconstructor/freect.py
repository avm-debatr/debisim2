#!/usr/bin/env python

# ------------------------------------------------------------------------------
"""
freect.py: Module for weighted filtered back-projection algorithm for
          reconstruction of 3D images from CT projection data using Spiral
          CBCT scanners. The reconstruction algorithm is based on the
          approximate wFBP technique by Stierstofer et al. and is used in
          the forward model to generate simulated reconstructions.
"""
# ------------------------------------------------------------------------------

__author__      = "Fangda Li"
__copyright__   = "Copyright (C) 2018, DEBATR Project"
__date__        = "11 February, 2018"
__credits__     = ["Fangda Li", "Ankit Manerikar", "Dr. Avinash Kak"]
__license__     = "Public Domain"
__version__     = "1.1"
__maintainer__  = ["Ankit Manerikar", "Fangda Li"]
__email__       = ["amanerik@purdue.edu", "li1208@purdue.edu"]
__status__      = "Prototype"
# ------------------------------------------------------------------------------

"""
--------------------------------------------------------------------------------
Module Description:
This module houses a library of functions that assist in executing 3D image 
reconstruction for Spiral Cone Beam CT scanners. The library of functions is 
essentially a Python wrapper for the FreeCT_wFBP CUDA library for 3D CT 
reconstruction [1].

FreeCT_wFBP is an open-source CUDA-based implementation of the weight filtered 
back-projection (wFBP) algorithm [2] by Stiestofer et al., which performs 3D 
reconstruction of X-ray images from CT projection data obtained from Spiral 
Cone Beam CT scanners with curved rectangular detector panels and a helical 
motion path for data acquisition. The CUDA implementation allows strong 
parallelization of the reconstruction operation allowinf fast reconstruction of 
large image volumes. 

This module provide support for using the FreeCT library in Python with 
functions allow feeding input projection data in the form of numpy ndarrays, 
tuning FreeCT scanner configurations and reconstruction parameters and reading 
FreeCT output data as Python numpy ndarrays.

[1]. J . Hoffman, S. Young, F. Noo, M. McNitt-Gray, Technical Note: FreeCT_wFBP: 
A robust, efficient, open-source implementation of weighted filtered backprojection 
for helical, fan-beam CT, Med. Phys., vol. 43, no. 3, pp. 1411-1420, Feb. 2016.

[2]. K. Stierstorfer, A. Rauscher, J. Boese, H. Bruder, S. Schaller, and T. Flohr, 
Weighted FBP - a simple approximate 3D FBP algorithm for multislice spiral CT with 
good dose usage for arbitrary pitch, Phys. Med. Biol., vol. 49, no. 11, pp. 
2209-2218, Jun.2004. 

* Functions:

fct_read_img_file()     - read binary image input file from FreeCT library
fct_read_prj_file()     - read binary projection input file from FreeCT library 
fct_write_prm_file()    - write scanner parameters to .prm file in FreeCT
fct_write_prj_file()    - save numpy array projection data as a .bin file 
fct_run()               - run FreeCT script on a projection data file to 
                          get reconstructed image output
                          
--------------------------------------------------------------------------------
"""

import sys, time, shutil
from lib.misc.fdlib import *
import subprocess

def fct_read_img_file(fname, w, h):
    """
    ----------------------------------------------------------------------------
    Read binary image input file from FreeCT library.

    :param fname:   binary image file name
    :param w:       image width
    :param h:       image height
    :return:    binary byte data as numpy 2D ndarrary
    ----------------------------------------------------------------------------
    """

    # print "Interpreting %s as stream of float32 values..." % fname
    bytes = fromfile(fname, dtype='float32')
    bytes = bytes.reshape(-1, w, h)
    print("Reconstructed Image Dimensions: ", bytes.shape)
    return bytes
# ------------------------------------------------------------------------------


def fct_read_prj_file(fname, nchnls, nrows, nviews):
    """
    ----------------------------------------------------------------------------
    Read binary projection input file from FreeCT library.

    :param fname:   binary projection input file
    :param nchnls:  number of channels
    :param nrows:   number of rows
    :param nviews:  number of views
    :return:  binary byte data as numpy 3D ndarrary
    ----------------------------------------------------------------------------
    """

    print("Interpreting %s as stream of float32 values..." % fname)
    bytes = fromfile(fname, dtype='float32')
    # Each row of data is one row
    bytes = bytes.reshape(nviews, nrows, nchnls)
    print("Projection shape is (nviews, nrows, nchnls) =", bytes.shape)
    return bytes
# ------------------------------------------------------------------------------


def fct_write_prm_file(fname, param_dict):
    """
    ----------------------------------------------------------------------------
    Write scanner parameters to .prm file in FreeCT.
    For information regarding the fields in the .prm file, check out FreeCT
    documentation at cvib.ucla.edu/freect/documentation.html

    :param fname:       .prm file path
    :param param_dict:  dictionary of fields in the .prm file
    :return:
    ----------------------------------------------------------------------------
    """

    f = open(fname, 'w')
    for k, v in param_dict.items():
        f.write("{}:\t{}\n".format(k, v))
    f.close()
    print("Generated %s" % fname)
# ------------------------------------------------------------------------------


def fct_write_prj_file(fname, proj):
    """
    ----------------------------------------------------------------------------
    Write out ndarray of projection data to the format fct understands.
    This does not take negative log nor separates the alternating channels, nor
    reverse the order of views.

    :param fname: projection file path
    :param proj: (nchnls, nrows, nviews)
    :return:
    ----------------------------------------------------------------------------
    """

    # Used to be K, but it did not work on simulated projections
    data = proj.flatten(order='F')

    with open(fname, 'wb') as f:
        data = bytearray(data.astype(np.float32))
        f.write(data)

    print("Written %s" % fname)
# ------------------------------------------------------------------------------


def fct_run(prm_fname, flags='-v --device=0'):
    """
    ----------------------------------------------------------------------------
    Run FreeCT script on a projection data file to get reconstructed image
    output.

    :param prm_fname:       .prm file path for FreeCT
    :param flags:           argument flags for executing FreeCT script,
                            for details, see FreeCT documentation,
                            cvib.ucla.edu/freect/documentation.html
    :return:
    ----------------------------------------------------------------------------
    """

    print("fct_wfbp started --->", end="")
    cmd = 'fct_wfbp %s %s' % (flags, prm_fname)
    t0 = time.time()

    o = subprocess.check_call(cmd,
                              shell=True)

    if o == 0:
        print("fct_wfbp finished ---> took %.2fs..." % (time.time() - t0))
    else:
        print("Error in running fct_wfbp...")
        raise RuntimeError
# ------------------------------------------------------------------------------
