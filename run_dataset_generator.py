# -----------------------------------------------------------------------------
"""run_debisim_dataset_generator.py: Run the generator for creating randomized
                                     CT datasets using the DEBISim pipeline.
"""

__author__    = "Ankit Manerikar"
__copyright__ = "Copyright (C) 2021, Robot Vision Lab"
__date__      = "3rd February, 2021"
__credits__   = ["Ankit Manerikar", "Fangda Li"]
__license__   = "Public Domain"
__version__   = "2.0.0"
__maintainer__= ["Ankit Manerikar", "Fangda Li"]
__email__     = ["amanerik@purdue.edu", "li1208@purdue.edu"]
__status__    = "Prototype"
# -----------------------------------------------------------------------------

import warnings
warnings.filterwarnings('ignore')
import argparse
import importlib.util as config_loader

from src.debisim_dataset_generator import *


parser = argparse.ArgumentParser(
                description='Dataset Generator for DEBISim: \n'
                            '-----------\n'
                            'The script generates a simulated CT dataset'
                            ' of randomized baggage configurations. '
                            'The simulation parameters are specified using '
                            'a config.py file - these include setting up '
                            'the scanner + X-ray source/detector, '
                            'the types of objects of objects to be spawned '
                            'in the bag as well as the DE decomposition '
                            '+ reconstruction parameters.'
                            'Examples of config.py files are provided in '
                            'configs/ directory for different scanners '
                            'and scanner geometries')

parser.add_argument('--config',
                    default=os.path.join(CONFIG_DIR,
                                         'config_default_parallelbeam_2d.py'),
                    help='config file location',
                    dest='config'
                    )

parser.add_argument('--sim_dir',
                    default=os.path.join(RESULTS_DIR,
                                         'example_default_parallelbeam_2d/'),
                    help='simulation directory for saving output'
                    )

parser.add_argument('--num_bags',
                    default=1,
                    help='number of bags to simulate',
                    type=int
                    )

args = parser.parse_args()

spec = config_loader.spec_from_file_location("config.params",
                                             args.config)
config = config_loader.module_from_spec(spec)
spec.loader.exec_module(config)

config.params['sim_dir'] = args.sim_dir

config.params['num_bags'] = range(1, args.num_bags+1)
run_xray_dataset_generator(**config.params)
# ----------------------------------------------------------------------------
