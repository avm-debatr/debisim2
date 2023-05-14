# -----------------------------------------------------------------------------
"""run_debisim_dataset_generator.py: Run the generator for creating randomized
                                     CT datasets using the DEBISim pipeline.
"""

__author__    = "Ankit Manerikar"
__copyright__ = "Copyright (C) 2021, Robot Vision Lab"
__date__      = "12th May, 2023"
__credits__   = ["Ankit Manerikar", "Fangda Li"]
__license__   = "Public Domain"
__version__   = "2.1.0"
__maintainer__= ["Ankit Manerikar", "Fangda Li"]
__email__     = ["amanerik@purdue.edu", "li1208@purdue.edu"]
__status__    = "Prototype"
# -----------------------------------------------------------------------------

import warnings
warnings.filterwarnings('ignore')
import argparse
import importlib.util as config_loader

import warnings

warnings.filterwarnings('ignore')

from src.debisim_pipeline import *
from lib.decomposer.cdm_decomposer import *
from lib.misc.benchmark import *
import torchvision.transforms as tvt

img_suffixes = dict(
    cdm=['c', 'pe', 'z'],
    sirz=['ze', 'rho']
)

parser = argparse.ArgumentParser(
                description='Dataset Generator for DEBISim: \n'
                            '-----------\n'
                            'The script generates a simulated Two-view X-ray'
                            'dataset of randomized baggage configurations. '
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
                                         'config_default_two_view_conebeam_dect.py'),
                    help='config file location',
                    dest='config'
                    )

parser.add_argument('--sim_dir',
                    default=os.path.join(RESULTS_DIR,
                                         'example_default_two_view_conebeam_dect/'),
                    help='simulation directory for saving output'
                    )

parser.add_argument('--num_bags',
                    default=10,
                    help='number of bags to simulate',
                    type=int
                    )

args = parser.parse_args()


def run_two_view_xray_dataset_generator(num_bags,
                                        sim_dir,
                                        scanner,
                                        xray_src_mdl,
                                        bag_creator_args,
                                        decomposer='cdm',
                                        save_sino=False,
                                        basis_fn=None,
                                        decomposer_args=None,
                                        images_to_save=None,
                                        slicewise=True,
                                        compress_data=False,
                                        fwd_mdl_args=None,
                                        rotate_bag=False
):
    """
    ---------------------------------------------------------------------------
    Function to generate a CT baggage Dataset using the DebiSim pipeline.

    :param num_bags:            number of baggage instances to generate
    :param sim_dir:             simulation directory
    :param scanner:             scanner model defined from ScannerTemplate()
    :param xray_src_mdl:        Xray source specifications - these are defined
                                as a dictionary with the following
                                specifications:
                                - 'num_spectra' - No of X-ray sources/spectra
                                - 'kVp' - peak kV voltage for the X-ray source(s)
                                - 'spectra' - file paths for the each of the
                                              X-ray spectra. The spectrum files
                                              must contain a N x 2 array with the
                                              keV values in the first column
                                              and normalized photon distribution
                                              in the 2nd column. See
                                              /include/spectra/ for reference.
                                - 'dosage' - dosage count for each of the sources
    :param bag_creator_args:    The baggage creation arguments - this dictionary
                                contains the input arguments to the method
                                self.create_random_object_list() for
                                BaggageCreator3D() or BaggageCreator2d()
    :param decomposer:          The object(s) corresponding the DE Decomposer,
                                for e.g., CDMDecomposer().
    :param decomposer_args:     Input arguments for the specified decomposer(s)
    :param save_sino:           If simulated sinograms are to be saved
    :param basis_fn:            Energy basis functions for DECT processing
    :param images_to_save:      ground truth images to be saved:
                                {gt | lac_1 | lac_2 | compton | pe | zeff}
    :param slicewise:           whether to produce H x W X D 3D volumetric bag
                                or H x W 2D baggage cross-sections in D batches
    :param compress_data:       whether to compress saved FITS data
    :param fws_model_args       arguments for X-ray forward model

    :return:
    ---------------------------------------------------------------------------
    """

    if images_to_save is None:
        images_to_save = ['lac_1', 'pe', 'c', 'gt']

    if fwd_mdl_args is None:
        fwd_mdl_args = dict(
            add_poisson_noise=True,
            add_system_noise=True,
            system_gain=0.0025
        )

    # Organize processing with Benchmark
    bench = Benchmark(save_log=False,
                      save_remark=False)
    bench.set_remark(
        'Creating Randomized DEBISIM Dataset %s with %i bags ....'%(sim_dir,
                                                            len(num_bags)))
    sim_bag_dir = os.path.join(sim_dir, 'simulation_%03d/')
    bench.set_test_cases([sim_bag_dir % i for i in num_bags])
    bench.set_output_dir([sim_bag_dir % i for i in num_bags])

    # Process slice by slice
    def preprocess(test_case):
        # ---------------------------------------------------------------------
        # Setup the simulator
        # ---------------------------------------------------------------------

        pre = dict(test_case=test_case)
        return pre
        # ---------------------------------------------------------------------

    def run(pre):
        # ---------------------------------------------------------------------
        # Run the simulator
        # ---------------------------------------------------------------------
        bag_dir = pre['test_case']

        # Initialized the simulator
        simulator = DEBISimPipeline(
            sim_path=bag_dir,
            scanner_model=scanner,
            xray_source_model=xray_src_mdl,
            compress_data=compress_data
        )

        simulator.logger.info("="*80)
        simulator.logger.info(" "*20+"DEBISIM PIPELINE STARTS"+" "*20)
        simulator.logger.info("="*80)

        mu = simulator.mu.material('water')

        simulator.logger.info('\n'+"-" * 50+"BAG_GENERATOR"+"-" * 50+'\n')
        simulator.create_random_simulation_instance(bag_creator_args,
                                                    save_images=images_to_save,
                                                    slicewise=slicewise)

        # TODO random_rotation of bag
        table_height = 448
        bh, bw, bd = simulator.gt_image_3d.shape
        center_of_rotation = (np.random.randint(bh//2-50, bh//2+50),
                              np.random.randint(bd//2-50, bd//2+50))

        if rotate_bag:
            bag_chunk = simulator.gt_image_3d[:table_height, :, :].clone()

            bag_chunk = tvt.RandomRotation((-30,30),
                                           center=center_of_rotation
                                           )(bag_chunk.permute(1, 0, 2))
            bag_chunk = bag_chunk.permute(1,0,2)
            simulator.gt_image_3d[:table_height,:, :] = bag_chunk.clone()

        # simulator.keV_range = range(50, 75)

        simulator.logger.info('\n'+"-" * 50+"FORWARD_MODEL"+"-" * 50+'\n')
        simulator.run_fwd_model(**fwd_mdl_args)

        if simulator.xray_source_model['num_spectra'] == 1:
            two_view_image = simulator.data.copy()
            top_view_image, side_view_image = two_view_image[:,:,1], \
                                              two_view_image[:,:,2]

            save_fits_data(os.path.join(simulator.f_loc['image_dir'],
                                        'bag_top_view.fits.gz'),
                           top_view_image)
            save_fits_data(os.path.join(simulator.f_loc['image_dir'],
                                        'bag_side_view.fits.gz'),
                           side_view_image)

            quick_imshow(2, 1,
                         [top_view_image, side_view_image],
                         colorbar=False,
                         colormap='gist_yarg',
                         titles=['Top', 'Side'])
            plt.savefig(os.path.join(simulator.f_loc['image_dir'],
                                     'baggage_scans.png'))
            plt.close()

        elif simulator.xray_source_model['num_spectra'] == 2:
            two_view_image_1 = simulator.data1.copy()
            top_view_image_1, side_view_image_1 = two_view_image_1[:,:,1], \
                                                  two_view_image_1[:,:,2]

            save_fits_data(os.path.join(simulator.f_loc['image_dir'],
                                        'bag_top_view_spec1.fits.gz'),
                           top_view_image_1)
            save_fits_data(os.path.join(simulator.f_loc['image_dir'],
                                        'bag_side_view_spec1.fits.gz'),
                           side_view_image_1)

            quick_imshow(2, 1,
                         [top_view_image_1, side_view_image_1],
                         colorbar=False,
                         colormap='gist_yarg',
                         titles=['Top (Spectrum 1)', 'Side (Spectrum 1)'])
            plt.savefig(os.path.join(simulator.f_loc['image_dir'],
                                     'baggage_scans_spec1.png'))
            plt.close()

            two_view_image_2 = simulator.data2.copy()
            top_view_image_2, side_view_image_2 = two_view_image_2[:,:,1], \
                                                  two_view_image_2[:,:,2]

            save_fits_data(os.path.join(simulator.f_loc['image_dir'],
                                        'bag_top_view_spec2.fits.gz'),
                           top_view_image_2)
            save_fits_data(os.path.join(simulator.f_loc['image_dir'],
                                        'bag_side_view_spec2.fits.gz'),
                           side_view_image_2)

            quick_imshow(2, 1,
                         [top_view_image_2, side_view_image_2],
                         colorbar=False,
                         colormap='gist_yarg',
                         titles=['Top (Spectrum 2)', 'Side (Spectrum 2)'])
            plt.savefig(os.path.join(simulator.f_loc['image_dir'],
                                     'baggage_scans_spec2.png'))
            plt.close()

        f_loc = simulator.f_loc.copy()
        torch.cuda.empty_cache()

        if decomposer=='none':
            pass
        else:

            simulator.logger.info('\n'+'-'*50+"DECOMPOSER"+'-'*50+'\n')

            simulator.run_decomposer(type=decomposer,
                                     decomposer_args=decomposer_args,
                                     basis_fn=basis_fn,
                                     save_sino=save_sino)

        res = dict()

        if (simulator.logger.hasHandlers()):
            simulator.logger.handlers.clear()

        simulator.logger.propagate = False
        simulator.scanner.logger.propagate = False

        del simulator.logger
        del simulator

        return res
    # -------------------------------------------------------------------------

    def postprocess(res, outdir):

        if res is not None:
            for k, v in res.items():
                del v

        del res
        return None

    bench.set_handles(preprocess, run, postprocess, None)
    bench.start()
# ----------------------------------------------------------------------------


spec = config_loader.spec_from_file_location("config.params",
                                             args.config)
config = config_loader.module_from_spec(spec)
spec.loader.exec_module(config)

config.params['sim_dir'] = args.sim_dir

config.params['num_bags'] = range(1, args.num_bags+1)

run_two_view_xray_dataset_generator(**config.params)

