#!/usr/bin/env python

# ------------------------------------------------------------------------------
"""template_siemens_sensation_32.py: Template for Siemens Sensation 32,
                                     contains dictionaries for machine /
                                     reconstruction geometry.

Source: Amin, AT Mohd, and AA Abd Rahni. "Modelling the Siemens SOMATOM
        Sensation 64 Multi-Slice CT (MSCT) Scanner." Journal of Physics:
        Conference Series. Vol. 851. No. 1. IOP Publishing, 2017.
"""
# ------------------------------------------------------------------------------

__author__      = "Ankit Manerikar"
__copyright__   = "Copyright (C) 2018, DEBATR Project"
__date__        = "12th February, 2020"
__credits__     = ["Ankit Manerikar", "Fangda Li"]
__license__     = "Public Domain"
__version__     = "1.2.0"
__maintainer__  = ["Ankit Manerikar", "Fangda Li"]
__email__       = ["amanerik@purdue.edu", "li1208@purdue.edu"]
__status__      = "Prototype"
# ------------------------------------------------------------------------------

import sys
from numpy import *

# =============================================================================
# Dictionary for Machine Geometry

# Initialize dictionary for Machine Geometry 
g = {}

# Annotate a name for the scanner - any log/prm files generated during 
# operation will be saved under this name 
g['scanner_name'] = 'siemens_sensation_32'

# Geometric dimensions of the scanner gantry setup - this can be obtained 
# from the manufacturer's datasheets
g['gantry_diameter'] = 600.0
g['source_origin']   = 570.0
g['origin_det']      = 470.0

# Dimensions of the rectangular detector panels
# sens_size_x, sens_size_y       are the x-y dimensions of one detector in mm
# sens_spacing_x, sens_spacing_y are the spacing between detector centers

g['sens_size_x'],    g['sens_size_y'] =      1.460, 1.02
g['sens_spacing_x'], g['sens_spacing_y'] =  1.440, 1.02

g['n_views_per_rot'] = 1160
g['n_sens_x'] = 672
g['anode_angle'] = 0.1222

# The keys det_row_count, det_col_count corresponds to the number of 
# detectors along the fan sweep / helical direction.

g['det_row_count'], g['det_col_count'] = 32, 672

machine_geometry = g.copy()
# =============================================================================

# =============================================================================
# Dictionary for Reconstruction Parameters

recon_params = {}

# These values must match with the corresponding values in machine_dict for
# proper reconstruction

recon_params['n_views']         = 1160
recon_params['pitch']           = 25.10 # 32*1.02*2/2.6
recon_params['slice_thickness'] = 1.5
recon_params['image_dims']      = (512, 512, 350)
recon_params['img_scale']       = 10
recon_params['mu_w']            = None
