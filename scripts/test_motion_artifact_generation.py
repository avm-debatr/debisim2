#!/usr/bin/env python

# -----------------------------------------------------------------------------

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


from src.debisim_pipeline import *
import argparse

parser = argparse.ArgumentParser(
            description="Test ring artifact simulation on an example sinogram")

parser.add_argument('--out_dir',
                    default='results/motion_artifact_sim',
                    help="Output simulation directory")

parser.add_argument('--scanner',
                    default='default',
                    choices=['sensation_32', 'definition_as', 'force', 'default'],
                    help="Scanner for simulation")

args = parser.parse_args()

t0 = time.time()

scanner = {'sensation_32': siemens_sensation_32,
           'definition_as': siemens_definition_as,
           'force': siemens_force,
           'default': default_scanner_parallel}[args.scanner]


# Define scanner model
id1 = ScannerTemplate(geometry='parallel' if args.scanner=='default' else 'cone',
                      scan='circular'  if args.scanner=='default' else 'spiral',
                      machine_dict=scanner.machine_geometry,
                      recon='fbp',
                      recon_dict=scanner.recon_params,
                      pscale=1.0
                      )
id1.set_recon_geometry()

# X-ray source specifications
xray_source_specs = dict(
    num_spectra=1,
    kVp=130,
    spectra=[os.path.join(SPECTRA_DIR, 'example_spectrum_130kV.txt')],
    dosage=[1.8e5]
)

mlist = ['ethanol',
         'Al', 'C', 'bakelite', 'pvc', 'bone',
         'pyrex', 'acrylic', 'polyethylene',
         'teflon', 'pvc', 'Si', 'polystyrene',
         'neoprene', 'acetal', 'nylon6'
         ]
lqd_list = ['water', 'ethanol']

material_pdf = [0.3] + [0.05] * 5 + [0.45 / 10.0] * 10
liquid_pdf = [1 / 2., 1 / 2.]

custom_objects = os.listdir(CUSTOM_SHAPES_DIR)

bag_creator_dict = dict(
    material_list=mlist,
    liquid_list=lqd_list,
    material_pdf=[1. / len(mlist)] * len(mlist),
    liquid_pdf=[1. / len(lqd_list)] * len(lqd_list),
    dim_range=(20, 70),
    number_of_objects=40,
    spawn_sheets=True,
    spawn_liquids=True,
    sheet_dim_list=range(2, 7),
    custom_objects=None
)

# Initialize simulator
simulator = DEBISimPipeline(
    sim_path=args.out_dir,
    scanner_model=id1,
    xray_source_model=xray_source_specs
)

simulator.run_bag_generator(mode='randomized',
                            bag_creator_dict=bag_creator_dict)

simulator.run_fwd_model_with_motion_artifacts(
    mode='objects',
    fwd_model_args=dict(add_poisson_noise=True,
                        add_system_noise=True,
                        system_gain=1.0)
)

print("Total Time:", time.time() - t0)

simulator.run_reconstructor()

