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
                description='Run a single baggae simulation for custom shapes: \n'
                            '-----------\n'
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
                                         'config_default_parallelbeam_3d.py'),
                    help='config file location',
                    dest='config'
                    )

parser.add_argument('--sim_dir',
                    default=os.path.join(RESULTS_DIR,
                                         'results/test_1/'),
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

simulator = DEBISimPipeline(sim_path=config.params['sim_dir'],
                            scanner_model=config.params['scanner'],
                            xray_source_model=config.params['xray_src_mdl'],
                            compress_data=False)

mu = simulator.mu.material('water')

simulator.create_random_simulation_instance(config.params['bag_creator_args'],
                                            save_images=config.params['images_to_save'],
                                            slicewise=config.params['slicewise'])

# =============================================================================
# Assign Specific Material to Custom Shapes

CUSTOM_SHAPE_MATERIAL = 'bone'
cs_mat = simulator.mu.material(CUSTOM_SHAPE_MATERIAL)

for sf_obj in simulator.sf_obj_list:
    if sf_obj['shape']=='M':
        sf_obj.update(cs_mat)
        sf_obj['material'] = CUSTOM_SHAPE_MATERIAL

simulator.create_simulation_from_sl_file(shape_list_file=simulator.sf_obj_list,
                                         gt_image=simulator.gt_image_3d)
# =============================================================================

simulator.run_fwd_model()

if config.params['decomposer'] == 'none':
    pass
else:

    simulator.logger.info('\n' + '-' * 50 + "DECOMPOSER" + '-' * 50 + '\n')

    simulator.run_decomposer(type=config.params['decomposer'],
                             decomposer_args=config.params['decomposer_args'],
                             basis_fn=config.params['basis_fn'],
                             save_sino=config.params['save_sino'])

if config.params['recon_args'] is not None:
    simulator.run_reconstructor(**config.params['recon_args'])
else:
    simulator.run_reconstructor()




