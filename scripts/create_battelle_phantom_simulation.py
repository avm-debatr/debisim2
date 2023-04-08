#!/usr/bin/env python

# -----------------------------------------------------------------------------
"""create_battelle_phantom_simulation.py:
Program for generating simulated reconstructions for the Battelle Phantoms
(ANSI N42.45-2011) using DEBISim.
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
import lib.forward_model.template_siemens_definition_as as siemens_definition_as
import lib.forward_model.template_siemens_force as siemens_force
import lib.forward_model.template_siemens_sensation_32 as siemens_sensation_32
import lib.forward_model.template_default_parallelbeam_scanner as default_scanner


parser = argparse.ArgumentParser(
            description="Create a CT simulation of the Battelle phantom.")

parser.add_argument('--out_dir',
                    default='results/simulation_battelle/',
                    help="Output simulation directory")
parser.add_argument('--scanner',
                    default='sensation_32',
                    choices=['sensation_32', 'definition_as', 'force', 'default'],
                    help="Scanner for simulation")
parser.add_argument('--ptype',
                    default='a',
                    choices=['a', 'b'],
                    help="Battelle Phantom type: a or b")

args = parser.parse_args()

scanner = {'sensation_32': siemens_sensation_32,
           'definition_as': siemens_definition_as,
           'force': siemens_force,
           'default': default_scanner}[args.scanner]

# add path of the shape list .pyc file
phantom = args.ptype
sl_file = os.path.join(EXAMPLE_DIR, f'battelle_phantom_{phantom}.pyc')

# This offset is to fit the gt image at the center of the
# current scanner gantry - default image for the .pyc shape list files
# assumes a image dimension of (664,664,350). The offset adjust
# the locations for a image slice of dims. (600,600,350) for gantry
# dimensions of the Sensation32 scanner.
# offset = -332+600
offset = -332+300

with open(sl_file, 'rb') as f:
    dict_battelle = pickle.load(f, encoding='latin1')
    sl_battelle = []
    for x in range(len(dict_battelle.keys())):
        cobj = dict_battelle['%i'%(x+1)]

        if cobj['shape'] in ['B', 'E']:
            cobj['geom']['center'][0] +=offset
            cobj['geom']['center'][1] +=offset

        if cobj['shape'] in ['C', 'Y']:
            cobj['geom']['apex'][0] +=offset
            cobj['geom']['apex'][1] +=offset

            cobj['geom']['base'][0] +=offset
            cobj['geom']['base'][1] +=offset

        sl_battelle.append(cobj)

    f.close()

sim_dir = args.out_dir

# Scanner Model ---------------------------------------------------------------

scanner.recon_params['pitch'] = 25.10
scanner.recon_params['slice_thickness'] = 1.5

# WARNING!!!!!
# The Battelle phantom is about 0.8 m long hence simulating its projections
# for a small pitch with Sensation 32 result in a memory error - it is suggested
# to simulate it in parts, use a longer pitch or a different scanner

# to simulate full phantom
scanner.recon_params['image_dims'] = (512, 512, 800)

# to simulate half phantom at a time - to do this add z offset of 400
# to the sl_file and stitch the output images of the two simulations
# scanner.recon_params['image_dims'] = (512, 512, 400)

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
xray_source_specs = dict(
    num_spectra=1,
    kVp=140,
    spectra=[os.path.join(SPECTRA_DIR, 'example_spectrum_140kV.txt')],
    dosage= [2.2752e6]
)
# -----------------------------------------------------------------------------

simulator = DEBISimPipeline(
    sim_path=sim_dir,
    scanner_model=scanner_mdl,
    xray_source_model=xray_source_specs
)
# -----------------------------------------------------------------------------
# Run reconstructor block
simulator.run_bag_generator(mode='manual',
                            sf_file=sl_battelle)

slide_show(simulator.gt_image_3d.cpu().numpy(), vmin=0)
torch.cuda.empty_cache()

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
    img_type='HU'  # save data as Hounsfield units
)
# -----------------------------------------------------------------------------

img_1 = read_fits_data(os.path.join(simulator.f_loc['image_dir'],
                                    'recon_image_1.fits.gz'), 0)

slide_show(img_1, vmin=-1000, vmax=2000)
# -----------------------------------------------------------------------------
