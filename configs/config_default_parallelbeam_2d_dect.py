# -----------------------------------------------------------------------------
"""
Default configuration file for:
    - a fan-beam parallel scanner geometry
    - dual energy CT setup
    - simulation of 2D baggage sections
"""
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

from lib.__init__ import *
from lib.forward_model.scanner_template import ScannerTemplate,\
                                               default_scanner_parallel

# -----------------------------------------------------------------------------
# Step 1: Specify dataset parameter:

bags_to_create = range(1, 10)                       # Number of bags to create
sim_dir        = 'results/example_parallelbeam_2d_dect/' # simulation directory
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Step 2: Specify Scanner Model using scanner_template.py

scanner_mdl = ScannerTemplate(
                geometry='parallel',
                scan='circular',
                machine_dict=default_scanner_parallel.machine_geometry,
                recon='fbp',
                recon_dict=default_scanner_parallel.recon_params,
                pscale=1.0
              )

scanner_mdl.set_recon_geometry()
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Step 3: Specify X-ray Source Model

# The X-ray Source Model is specified by a dictionary with the following
# key-value pairs:
# num_spectra  - No of X-ray sources/spectra
# kVp          - peak kV voltage for the X-ray source(s)
# spectra      - file paths for the each of the X-ray spectra.
#                The spectrum files are .txt files containing a N x 2
#                array with the keV values in the first column and
#                normalized photon distribution in the 2nd column.
#                See /include/spectra/ for examples to create your own
#                spectrum file.
# dosage       - dosage count for each of the sources

xray_source_specs = dict(num_spectra=2,
                         kVp=130,
                         spectra=[os.path.join(SPECTRA_DIR,
                                               'example_spectrum_130kV.txt'),
                                  os.path.join(SPECTRA_DIR,
                                               'example_spectrum_95kV.txt')
                                  ],
    dosage=[2e5, 1.85e5]
)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Step 4: Specify the arguments the BaggageCreator3D() Arguments - these
# arguments decide the nature of objects that can be spawned in the
# simulated bags.

# The list contains the list of materials that will be assigned to the
# objects in the bag - the material assignment is random but liquids need
# to be specified separately if liquid filled containers are to be spawned.

mlist    = ['ethanol',                                       # organic
            'Al', 'Ti', 'Fe',                                # metals
            'bakelite', 'pyrex','acrylic', 'Si',             # glass/
                                                             # ceramics
            'polyethylene',  'pvc', 'polystyrene', 'acetal', # plastics
            'neoprene',                                      # rubber
            'nylon6', 'teflon',                              # cloth
            ]
lqd_list = ['water', 'H2O2']                                 # liquids

# material selection probabilities
material_pdf = [0.3] + [0.05/3]*3 + [0.65/11]*11
liquid_pdf = [1/2., 1/2.]

# using custom shapes other than fixed geometries
custom_objects = [os.path.join(CUSTOM_SHAPES_DIR, s)
                  for s in os.listdir(CUSTOM_SHAPES_DIR)]

bag_creator_args = dict(
    # list of materials/liquids to simulate -----------------------------------
    material_list=mlist,
    liquid_list=lqd_list,
    # material selection probabilities - specify for each material ------------
    material_pdf=material_pdf,
    liquid_pdf=liquid_pdf,
    # params for deformable sheets/liquid-filled containers -------------------
    spawn_sheets=True,
    spawn_liquids=True,
    sheet_prob=0.2,       # probability of spawning a deformable sheet
    lqd_prob=0.3,         # probability of spawning a liquid-filled container
    sheet_dim_list=range(2, 7),  # range of sheet thicknesses
    # -------------------------------------------------------------------------
    # object shape specifications
    dim_range=(20,70),                   # min-max dims of simulated object
    number_of_objects=range(30, 40),     # number of objects in each bag
    custom_objects=custom_objects, # if custom objects are to be specified
    custom_obj_prob=0.3,  # probability of spawning a custom shape
    # -------------------------------------------------------------------------
    # specifications for metals / target objects
    metal_dict={'metal_amt':  1e2, 'metal_size': (3,5)},
    target_dict={'num_range': (1,3), 'is_liquid': False}
    # -------------------------------------------------------------------------
)
# -----------------------------------------------------------------------------
# Step 4 Specify the Dual Energy Decomposition Method

decomp_method = 'cdm' # constrained decomposition method for DECT

# default values - use gpus for faster 3d image processing
cdm_args = dict(cdm_solver='gpu',
                cdm_type='cpd',     # Compton-PE basis - default
                projector='cpu',
                init_val=(0, 0))

# -----------------------------------------------------------------------------
# Step 5: Forward Model Args

fwd_mdl_args = dict(add_poisson_noise=True,
                    add_system_noise=True,
                    system_gain=0.025949
                   )

# -----------------------------------------------------------------------------
# params to feed to the debisim pipeline
params = dict()

params['num_bags']          = bags_to_create
params['sim_dir']           = sim_dir
params['scanner']           = scanner_mdl
params['xray_src_mdl']      = xray_source_specs
params['bag_creator_args']  = bag_creator_args
params['fwd_mdl_args']      = fwd_mdl_args
params['save_sino']         = False
params['basis_fn']          = None
params['decomposer_args']   = cdm_args
params['recon_args']        = None
params['images_to_save']    = ['gt', 'lac_1', 'lac_2', 'compton', 'pe', 'zeff']
params['decomposer']        = decomp_method
params['slicewise']         = True # set if creating 2D cross-sections instead
                                   # of 3D bags
# -----------------------------------------------------------------------------
