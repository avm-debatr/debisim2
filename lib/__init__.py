from numpy import *
import os, warnings

warnings.filterwarnings('ignore')

# directory specifications

ROOT_DIR                    = os.path.dirname(os.path.dirname(__file__))

# module directories
CONFIG_DIR                  = os.path.join(ROOT_DIR, 'configs')
EXAMPLE_DIR                 = os.path.join(ROOT_DIR, 'examples')
INC_DIR                     = os.path.join(ROOT_DIR, 'include')
LIB_DIR                     = os.path.join(ROOT_DIR, 'lib')
RESULTS_DIR                 = os.path.join(ROOT_DIR, 'results')
SRC_DIR                     = os.path.join(ROOT_DIR, 'src')
DEPS_DIR                    = os.path.join(ROOT_DIR, 'deps')

MU_DIR                      = os.path.join(INC_DIR, 'mu')
SCANNER_DIR                 = os.path.join(INC_DIR, 'scanners')
SPECTRA_DIR                 = os.path.join(INC_DIR, 'spectra')

# material simulation data
MU_COMPOUNDS_DIR            = os.path.join(MU_DIR, 'compounds')
MU_ELEMENTS_DIR             = os.path.join(MU_DIR, 'elements')
MU_TARGETS_DIR              = os.path.join(MU_DIR, 'targets')
XCOM_DIR                    = os.path.join(MU_DIR, 'xcom')

MU_COMPOUNDS_DENSITY_FILE   = os.path.join(MU_DIR, 'compounds_density.txt')
MU_ELEMENTS_DENSITY_FILE    = os.path.join(MU_DIR, 'elements_density.txt')
MU_TARGETS_DENSITY_FILE     = os.path.join(MU_DIR, 'targets_density.txt')
MU_XCOM_DBASE_CMD           = os.path.join(XCOM_DIR, 'XCOM', 'a.out')

# scanner metadata, config + specifications
SCANNER_FILE_NAME           = os.path.join(SCANNER_DIR, '%s.scanner')
#TODO: CHANGE THIS TO DESIRED DIRECTORY
FCT_TMP_DIR                 = '/home/ubuntu/temp/'
RAMLAK_FILTER_FILE          = os.path.join(SCANNER_DIR, 'ramlak.mat')
SPECTRUM_FILE_NAME          = os.path.join(SCANNER_DIR, '%s_%ikV.txt')
CUSTOM_SHAPES_DIR           = os.path.join(EXAMPLE_DIR, 'custom_shapes')
PHANTOM_SL_DIR              = os.path.join(EXAMPLE_DIR, 'phantom_shape_lists')

# pipeline module directories
BAG_GEN_DIR                 = os.path.join(LIB_DIR, 'bag_generator')
DECOMPOSER_DIR              = os.path.join(LIB_DIR, 'decomposer')
FWD_MDL_DIR                 = os.path.join(LIB_DIR, 'forward_model')
RECON_DIR                   = os.path.join(LIB_DIR, 'reconstructor')
MISC_DIR                    = os.path.join(LIB_DIR, 'misc')

# gui modules
GUI_DIR                     = os.path.join(SRC_DIR, 'gui')
SCRIPTS_DIR                 = os.path.join(SRC_DIR, 'scripts')
SIMULATOR_DIR               = os.path.join(SRC_DIR, 'simulator')

# results directories
DEFAULT_SIM_DIR             = os.path.join(RESULTS_DIR, 'default_sim_dir')
