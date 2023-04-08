#!/usr/bin/env python

# ------------------------------------------------------------------------------
"""template_siemens_force.py: Template for Siemens Force,
                              contains dictionaries for machine /
                              reconstruction geometry.
"""
# ------------------------------------------------------------------------------

__author__      = "Ankit Manerikar"
__copyright__   = "Copyright (C) 2018, DEBATR Project"
__date__        = "12th February, 2020"
__credits__     = ["Ankit Manerikar", "Fangda Li", "Dr. Avinash Kak"]
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
g['scanner_name'] = 'siemens_force'

# Geometric dimensions of the scanner gantry setup - this can be obtained 
# from the manufacturer's datasheets
g['gantry_diameter'] = 550
g['source_origin'] = 595.0
g['origin_det'] = 1085.6 - 595.0

# Dimensions of the detector for the CBCT scanner - the detector panels for 
# such scanner is typically a curved rectangular panel
# sens_size_x, sens_size_y are the x-y dimensions of one detector in mm
# sens_spacing_x, sens_spacing_y are the spacing between adjacent detector
# centers
g['sens_size_x'],    g['sens_size_y'] = 0.51, 0.53
g['sens_spacing_x'], g['sens_spacing_y'] = 0.51, 0.53

# The number of views must match the recorded number of views if real 
# scanner data is being used 
g['n_views_per_rot'] = 1400
g['n_sens_x'] = 920
g['anode_angle'] = 0.1222

# The keys det_row_count, det_col_count corresponds to the number of 
# detectors along the fan sweep / helical direction.

g['det_row_count'], g['det_col_count'] = 96, 920

machine_geometry = g.copy()
# =============================================================================

# =============================================================================
# Dictionary for Reconstruction Parameters

recon_params = {}

# These values must match with the corresponding values in machine_dict for
# proper reconstruction
recon_params['n_views']         = 1160
recon_params['pitch']           = 51.39
recon_params['slice_thickness'] = 0.53
# recon_params['fan_angle_inc'] = rad2deg(0.00094729)
# recon_params['fan_angle_0']   = -24.8175
recon_params['image_dims']      = (512, 512, 350)
recon_params['img_scale']       = 0.01
recon_params['mu_w']            = None