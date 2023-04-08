#!/usr/bin/env python

# -----------------------------------------------------------------------------
"""create_acr_phantom_simulation.py:
Program for generating simulated reconstructions for the ACR Phantom using
DebiSim.
"""

__author__    = "Ankit Manerikar"
__copyright__ = "Copyright (C) 2023, Robot Vision Lab"
__date__      = "6th April, 2023"
__credits__   = ["Ankit Manerikar", "Fangda Li"]
__license__   = "Public Domain"
__version__   = "2.0.0"
__maintainer__= ["Ankit Manerikar", "Fangda Li"]
__email__     = ["amanerik@purdue.edu", "li1208@purdue.edu"]
__status__    = "Prototype"

# -----------------------------------------------------------------------------

import warnings
import argparse
warnings.filterwarnings('ignore')

from src.debisim_pipeline import *
import lib.forward_model.template_siemens_sensation_32 as siemens_sensation_32
import lib.forward_model.template_siemens_definition_as as siemens_definition_as
import lib.forward_model.template_siemens_force as siemens_force
import lib.forward_model.template_default_parallelbeam_scanner as default_scanner


parser = argparse.ArgumentParser(
            description="Create a CT simulation of the ACR phantom.")

parser.add_argument('--out_dir',
                    default='results/simulation_acr/',
                    help="Output simulation directory")
parser.add_argument('--scanner',
                    default='sensation_32',
                    choices=['sensation_32', 'definition_as', 'force', 'default'],
                    help="Scanner for simulation")

args = parser.parse_args()

scanner = {'sensation_32': siemens_sensation_32,
           'definition_as': siemens_definition_as,
           'force': siemens_force,
           'default': default_scanner}[args.scanner]

# add path of the shape list .pyc file
sl_file = os.path.join(EXAMPLE_DIR, 'acr_phantom.pyc')

assert os.path.exists(sl_file), 'SL file, acr_phantom.pyc not found ' \
                                '- please download the same from the link ' \
                                'provided in the GitHub repository'

# This offset is to fit the gt image at the center of the
# current scanner gantry - default image for the .pyc shape list files
# assumes a image dimension of (664,664,350). The offset adjust
# the locations for a image slice of dims. (600,600,350) for gantry
# dimensions of the Sensation32 scanner.
offset = -332+300

with open(sl_file, 'rb') as f:
    dict_acr = pickle.load(f, encoding='latin1')
    sl_acr = []
    for x in range(len(dict_acr.keys())):
        cobj = dict_acr['%i'%(x+1)]

        if cobj['shape'] in ['B', 'E']:
            cobj['geom']['center'][0] +=offset
            cobj['geom']['center'][1] +=offset

        if cobj['shape'] in ['C', 'Y']:
            cobj['geom']['apex'][0] +=offset
            cobj['geom']['apex'][1] +=offset

            cobj['geom']['base'][0] +=offset
            cobj['geom']['base'][1] +=offset

        sl_acr.append(cobj)

    f.close()

sim_dir = args.out_dir

# Scanner Model ---------------------------------------------------------------

scanner.recon_params['image_dims'] = (512, 512, 300)

# Uses default scanner (Siemens Sensation 32)
scanner_mdl = ScannerTemplate(
    geometry='cone' if args.scanner!='default' else 'parallel',
    scan='spiral' if args.scanner!='default' else 'circular',
    machine_dict=scanner.machine_geometry,
    recon='fbp',
    recon_dict=scanner.recon_params,
)
scanner_mdl.set_recon_geometry()

# -----------------------------------------------------------------------------

# X-ray Source Model ----------------------------------------------------------

xray_source_specs = dict(num_spectra=1,
                         kVp=140,
                         spectra=[os.path.join(SPECTRA_DIR,
                                               'example_spectrum_140kV.txt')],
    dosage= [2.2752e6]
)
# -----------------------------------------------------------------------------

simulator = DEBISimPipeline(sim_path=sim_dir,
                            scanner_model=scanner_mdl,
                            xray_source_model=xray_source_specs)

# -----------------------------------------------------------------------------
# Run reconstructor block

simulator.run_bag_generator(mode='manual',
                            sf_file=sl_acr)

slide_show(simulator.gt_image_3d.cpu().numpy(), vmin=0)

# -----------------------------------------------------------------------------
# Run forward X-ray modelling block

simulator.run_fwd_model(
    add_poisson_noise=True,     # Add Poisson noise to projections
    add_system_noise=True,      # Add detector system noise
    system_gain=1e2             # system gain
)
# -----------------------------------------------------------------------------
# Run Reconstructor block

simulator.run_reconstructor(
    img_type='HU'               # save data as Hounsfield units
)
# -----------------------------------------------------------------------------

img_1 = read_fits_data(os.path.join(simulator.f_loc['image_dir'],
                                    'recon_image_1.fits.gz'), 0)

slide_show(img_1, vmin=-1000, vmax=2000)
# -----------------------------------------------------------------------------
