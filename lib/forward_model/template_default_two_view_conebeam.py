#!/usr/bin/env python

# ------------------------------------------------------------------------------
"""template_default_two_view_conebeam.py:
Template for Two-View CBCT X-ray Scanner, contains dictionaries for machine /
reconstruction geometry.

Here the scanner consists of a cone-beam source and detector setup with a
two-view setup but it is constructed as a CBCT setup with limited
number of views per rotation.
"""
# ------------------------------------------------------------------------------

__author__      = "Ankit Manerikar"
__copyright__   = "Copyright (C) 2018, DEBATR Project"
__date__        = "13th May, 2023"
__credits__     = ["Ankit Manerikar", "Fangda Li"]
__license__     = "Public Domain"
__version__     = "2.1.0"
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
g['scanner_name'] = 'two_view_cbct'

# Geometric dimensions of the scanner gantry setup - this can be obtained
# from the manufacturer's datasheets
g['gantry_diameter'] = 600.0
g['source_origin']   = 500.0
g['origin_det']      = 500.0

# Dimensions of the rectangular detector panels
# sens_size_x, sens_size_y       are the x-y dimensions of one detector in mm
# sens_spacing_x, sens_spacing_y are the spacing between detector centers

g['sens_size_x'],    g['sens_size_y']    = 1.460, 1.02
g['sens_spacing_x'], g['sens_spacing_y'] = 1.440, 1.02

g['n_views_per_rot'] = 4
g['n_sens_x'] = 1024
g['anode_angle'] = 0.1222

# The keys det_row_count, det_col_count corresponds to the number of
# detectors along the fan sweep / helical direction.

g['det_row_count'], g['det_col_count'] = 1024, 1024

machine_geometry = g.copy()
# =============================================================================

# =============================================================================
# Dictionary for Reconstruction Parameters

recon_params = {}

# These values must match with the corresponding values in machine_dict for
# proper reconstruction
# not for two-view scanner

recon_params['n_views']         = 4
recon_params['pitch']           = 25.10 # 32*1.02*2/2.6
recon_params['slice_thickness'] = 1.5
recon_params['image_dims']      = (512, 512, 350) # (512, 512, 350)
recon_params['img_scale']       = 10
recon_params['mu_w']            = None
