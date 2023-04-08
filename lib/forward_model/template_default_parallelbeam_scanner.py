#!/usr/bin/env python

# ------------------------------------------------------------------------------
"""template_default_parallelbeam_scanner.py:
        Template for a default scanner for rebinned parallel beam sinograms.
"""
# ------------------------------------------------------------------------------

__author__    = "Ankit Manerikar"
__copyright__ = "Copyright (C) 2023, Robot Vision Lab"
__date__      = "6th April, 2023"
__credits__   = ["Ankit Manerikar", "Fangda Li"]
__license__   = "Public Domain"
__version__   = "2.0.0"
__maintainer__= ["Ankit Manerikar", "Fangda Li"]
__email__     = ["amanerik@purdue.edu", "li1208@purdue.edu"]
__status__    = "Prototype"
# ------------------------------------------------------------------------------

from lib.__init__ import RAMLAK_FILTER_FILE

# ==============================================================================
# Dictionary for Machine Geometry

# Initialize dictionary for Machine Geometry
g = {}

# Annotate a name for the scanner - any log/prm files generated during
# operation will be saved under this name
g['scanner_name']     = 'default_parallelbeam'
g['gantry_diameter']  = 512

g['det_spacing_y'], g['det_spacing_x'] = 0.5, 1.0
g['det_col_count'], g['det_row_count'] = 1024, 350

machine_geometry = g.copy()
# =============================================================================

# =============================================================================
# Dictionary for Reconstruction Parameters

recon_params = {}

# These values must match with the corresponding values in machine_dict for
# proper reconstruction
recon_params['n_views']         = 720
recon_params['view_range']      = 180.
recon_params['slice_thickness'] = 1.0
recon_params['image_dims']      = (512, 512, 350)

recon_params['ramlak'] = RAMLAK_FILTER_FILE

recon_params['img_scale'] =  1./ machine_geometry['det_spacing_y']

#TODO: Crtical recheck this value for Dual Energy CT - will change with X-ray
# spectrum
recon_params['mu_w'] =  dict(lac_1=0.155,
                             lac_2=0.133,
                             compton=0.163,
                             pe=5058.233,
                             z=7.42,
                             density=1.000)
