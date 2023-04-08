#!/usr/bin/env python

# ------------------------------------------------------------------------------
"""DEBISimPipeline: Class for generating X-ray projection data for single /
                      dual energy CT scanner models in randomized or user-
                      interactive modes.
"""

__author__    = "Ankit Manerikar"
__copyright__ = "Copyright (C) 2020, Robot Vision Lab"
__date__      = "12th January, 2021"
__credits__   = ["Ankit Manerikar", "Fangda Li", "Dr. Avinash Kak"]
__license__   = "Public Domain"
__version__   = "2.0.0"
__maintainer__= ["Ankit Manerikar", "Fangda Li"]
__email__     = ["amanerik@purdue.edu", "li1208@purdue.edu"]
__status__    = "Prototype"
# ------------------------------------------------------------------------------

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from numpy import *

from torch.distributions import Poisson, Normal

from lib.bag_generator.baggage_creator_3d import *
from lib.bag_generator.baggage_creator_2d import *
from lib.bag_generator.image_voxelizer_3d import *

from lib.forward_model.scanner_template import *
from lib.forward_model.scatter_simulator import *

from lib.decomposer.cdm_decomposer import *
from skimage.measure import regionprops
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
torch.set_default_tensor_type(torch.cuda.FloatTensor)

"""-----------------------------------------------------------------------------
* Module Description:

This module is the central module for the operation of the DEBISim pipeline - 
it runs the four different blocks of the pipeline to produce simulation data. 
The class DEBISimPipeline() can be used to generate projection data for 
a randomized data generation, for user-interactive data generation in the 
DEBISim GUI and from previously saved shape list data such as for phantoms & saved 
simulation directories. The class is explicitly used in the dataset generation 
in the DEBISim script for the randomized/mode gui. 

Usage:

When initialized, the class DEBISimPipeline() creates a simulation directory 
as specified by the  input argument and creates the following item within the 
directory:

- ground_truth/ - directory containing the ground truth images - these include 
                  gt_label_image, gt_compton, gt_pe_image, gt_zeff_image, 
                  gt_lac_1_image, gt_lac_2_image, gt_hu_1_image, 
                  gt_hu_2_image. The options for which images need to be saved 
                  can be specified while running the program. All images are 
                  saved as compressed .fits.gz files and can be read using 
                  util.read_fits_data files.
- sinograms/ -    This directory is reserved for saving the polychromatic 
                  projection data generating the module. The preferred 
                  nomenclature for the sinograms is sino_%i.fits.gz, where 
                  i = 1, 2, ... corresponding to the respective spectra of 
                  the projection.
- images/ -       This subdirectory is reserved for saving the reconstructed 
                  single-energy/multi-energy CT images from the projection 
                  data. The reconstructed images are saved as DICOM/DICOS 
                  files. Any post-processing output such as for MAR or 
                  Segmentation can be saved in this subdirectory. The preferred 
                  nomenclature for the image files is recon_image_%s.fits.gz 
                  (where %s - 1,2,...n - for the CT images, c - for compton, 
                  p - for PE and z - for effective atomic number.)
- sl_metadata.pyc This pickle file contains the Shape List for the simulated 
                  ground truth. See ShapeListHandle() for more details.

The ground truth data for the simulator instance can be created either in a 
randomized mode or a user specified mode. In the randomized mode, the 
simulator calls in the BaggageCreator3D module to spawn a randomized baggage 
simulation with objects randomly selected, placed and oriented as specified 
by the baggage creator arguments. In the user specified mode, the objects to 
be spawned in the baggage simulation are specified by a shape list provided 
as input. The parameters within the Shape List can be fed in using a 
Simulator GUI or manually by creating an SL dictionary for each object. 

An example for the use of the module in a randomized mode is given below. To 
create the instance, one must first specify (i) a scanner model using the
ScannerTemplate module, (ii) a Python dictionary specifying the X-ray source 
model, i.e., the spectrum pdfs, the dosage and peak tube voltages (see below) 
and (iii) the argument for creating a randomized bag using BaggageCreator3D
module - this involves selecting of materials, object dimensions and the number 
of objects. The ground truth image can then be generated using the 
self.create_random_simulation_instance() method. The projection data is 
created using the self.run_fwd_model(). Projection data generation includes 
options for adding Poisson noise and Gaussian shot noise to create noisy 
projections. The code for the example is as follows:


Methods:

__init__                                - Constructor 
create_random_simulation_instance       - Creates a random simulation instance 
                                          using BaggageCreator3D 
create_simulation_from_sl_file          - Creates simulaiton instance by reading 
                                          in a shape list
generate_polychromatic_ct_projection    - Generate polychromatic projection for 
                                          a single spectrum
run_fwd_model   - Generate projection scanner for the 
                                          entire scanner/source model
save_dect_ground_truth_images           - Save the ground truth label/coefficient 
                                          images

Attributes:

f_loc                                   - dictionary containing default file 
                                          locations
gt_image_3d                             - GT label image
image_shape_3d                          - dimensions of the volumetric ground 
                                          truth image
keV_range                               - the keV range for the Xray spectrum
material_curve                          - 2D array of attenuation curves for 
                                          all the materials in the image
maxkV                                   - maximum keV value encountered
mu                                      - instance of the MuDatabaseHandler 
                                          used in the image. 
reconstruction_geometry                 - reconstruction geometry read from 
                                          scanner model
scale                                   - image scale
scanner_geometry                        - scanner geometry read from scanner 
                                          model
scanner                                 - scanner model as an instance of 
                                          ScannerTemplate()                        
sf_obj_list                             - Shape List for the simulation 
                                          instance
slh                                     - ShapeListHandle() object for the 
                                          class
xray_source_model                       - dictionary of Xray source 
                                          specifications: {'num_spectra': number
                                          of spectra, 'dosage': list of dosage 
                                          counts for each spectra, 'spectrum':
                                          list of path for each spectrum file 
                                          (For format, see ./include/spectra/),
                                          'kVp': list of tube voltages for 
                                          each spectrum}
zwidth                                  - z-axis width of the image in mm.
                                          (Number of slices in unit mm)
-------------------------------------------------------------------------------
"""

default_file_locations = dict(
    simulation_dir=DEFAULT_SIM_DIR,
    scanner_dir=SCANNER_DIR,
    mass_attn_dir=MU_DIR,
    spectra_dir=SPECTRA_DIR,
    image_dir='images/',
    sino_dir='projections/',
    gt_dir='ground_truth/',
    gif_dir='gifs/',
    sino_file='sino_%i.fits.gz',
    img_file='recon_image_%i.fits.gz',
    gt_image='gt_label_image.fits.gz',
    sl_metadata='sl_metadata.pyc'
)


DEFAULT_FWD_MODEL_ARGS = dict(add_poisson_noise=True,
                              add_system_noise=True,
                              system_gain=1)


class DEBISimPipeline(object):

    # Methods -----------------------------------------------------------------

    def __init__(
            self,
            sim_path,
            scanner_model,
            xray_source_model,
            mu_handler=None,
            debug=False,
            logfile=None,
            compress_data=False
    ):
        """
        ------------------------------------------------------------------------
        Constructor for the DEBISimPipeline Class.

        :param sim_path: The path to simulation directory - when the simulation 
                         is complete this directory will be populated with 
                         the subdirectories images/, sinograms/ and 
                         ground_truth/ containing the res[ectively data.  
                         
        :param scanner_model: An instance of the ScannerTemplate where the 
                              scanner specifications are given.
        
        :param xray_source_model: dictionary specifying the xray source 
                             specifications. See Module Description for details

        :param mu_handler:    A MuDatabaseHandler object (optional)
        :param logfile:       log file for simulation
        :param debug:         set if in debug mode
        :param compress_data: set to compress saved FITS data
        ------------------------------------------------------------------------
        """

        # Select default values for the f_loc and scanner_specs
        self.f_loc = default_file_locations.copy()
        self.DECOMPOSER_FLAG = False

        self.scanner                 = scanner_model
        self.scanner_geometry        = scanner_model.machine_geometry.copy()
        self.reconstruction_geometry = scanner_model.recon_geometry.copy()
        self.xray_source_model       = xray_source_model.copy()
        self.zwidth                  = self.scanner.recon_params['image_dims'][2]
        self.debug                   = debug
        self.logfile                 = logfile
        self.compress_data           = compress_data

        self.maxkV = self.xray_source_model['kVp']  \
                     if   isscalar(self.xray_source_model['kVp']) \
                     else max(self.xray_source_model['kVp'])

        # Assign and create a simulation directory for the CT simulation
        self.f_loc['simulation_dir'] = sim_path

        os.makedirs(self.f_loc['simulation_dir'], exist_ok=True)

        # Assign absolute paths to all simulation files
        for sim_file in ['image_dir', 'sino_dir', 'gt_dir', 'gif_dir']:
            self.f_loc[sim_file] = os.path.join(self.f_loc['simulation_dir'],
                                                self.f_loc[sim_file])

            os.makedirs(self.f_loc[sim_file], exist_ok=True)

        self.f_loc['gt_image'] = os.path.join(self.f_loc['gt_dir'],
                                            self.f_loc['gt_image'])
        self.f_loc['sl_metadata'] = os.path.join(self.f_loc['simulation_dir'],
                                            self.f_loc['sl_metadata'])

        self.f_loc['sino_file'] = os.path.join(self.f_loc['sino_dir'],
                                            self.f_loc['sino_file'])

        # ---------------------------------------------------------------------

        # self.image_shape_3d = self.scanner.recon_params['image_dims']

        self.image_shape_3d = (int(self.scanner_geometry['gantry_diameter']),
                               int(self.scanner_geometry['gantry_diameter']),
                               int(self.scanner.recon_params['image_dims'][2]))

        self.slh = ShapeListHandle()
        self.mu = MuDatabaseHandler(self.debug, self.logfile) \
                  if mu_handler is None else mu_handler

        self.keV_range = arange(10, self.maxkV+1)

        spectra = [loadtxt(spec)[:self.maxkV-10,1]
                   for spec in self.xray_source_model['spectra']]

        self.mu.calculate_lac_hu_values('water', spectra)

        if self.logfile is None:
            self.logfile = os.path.join(self.f_loc['simulation_dir'],
                                        'debisim_bag.log')
            if not os.path.exists(self.logfile):
                open(self.logfile, 'w+').close()

        self.logger = get_logger('DEBISIM', self.logfile)

        header = ['CT Specifications', '']

        self.scale = 0.1 # self.scanner.recon_params['img_scale']

        print_table = []

        print_table.append(['Initialization Time',
                            time.strftime('%m-%d-%Y %H:%M:%S', 
                                          time.localtime())])
        print_table.append(['Simulation Directory',
                            self.f_loc['simulation_dir']])
        print_table.append(['Image Dimensions', self.image_shape_3d])
        print_table.append(['CT Scanner', self.scanner_geometry['scanner_name']])
        print_table.append(['Projection Dims. (views, rows, cols)',
                            [self.reconstruction_geometry['n_views'],
                             self.scanner_geometry['det_row_count'],
                             self.scanner_geometry['det_col_count']]
                            ])

        self.logger.info('\n'+tabulate(print_table, header, tablefmt='psql'))
        self.logger.info('\n')
    # --------------------------------------------------------------------------

    def create_random_simulation_instance(self,
                                          baggage_creator_args,
                                          prior_image=None,
                                          prior_list=[],
                                          save_images=['gt'],
                                          slicewise=False,
                                          template=2
                                          ):
        """
        ------------------------------------------------------------------------
        Create a random simulation phantom from the randomized BaggageCreator3D. 
        The functions also allows spawning randmized objects over a prior image.

        :param baggage_creator_args: arguments to run the 
                              BaggageCreator3D.create_random_object_list()
                              function. 
        :param prior_image:   Prior image if needs to be included
        :param prior_list:    Shape List of objects in the prior image
        :param save_images:   ground truth images to save (Options: {'gt', 
                              'compton', 'pe', 'zeff', 'lac_1', 'lac_2'})
        :return:
        ------------------------------------------------------------------------
        """

        # run BaggageCreator3D with the input creator args ---------------------
        bag_vol_shape = list(self.image_shape_3d)
        # bag_vol_shape[2] = max(bag_vol_shape[2], 350)

        if not slicewise:
            virtual_bag_creator = BaggageImage3D(img_vol=tuple(bag_vol_shape),
                                                 sim_dir=self.f_loc['gt_dir'],
                                                 logfile=self.logfile,
                                                 gantry_dia=int(
                                                     self.scanner_geometry['gantry_diameter']),
                                                 prior_image=prior_image,
                                                 debug=self.debug
                                         )
        else:
            virtual_bag_creator = BaggageImage2D(img_vol=tuple(bag_vol_shape),
                                                 sim_dir=self.f_loc['gt_dir'],
                                                 logfile=self.logfile,
                                                 gantry_dia=int(
                                                     self.scanner_geometry['gantry_diameter']),
                                                 prior_image=prior_image,
                                                 template=template,
                                                 debug=self.debug
                                         )

        # obtain randomized Object3D list
        obj_list = virtual_bag_creator.create_random_object_list(
                                            **baggage_creator_args
                                            )

        # run placement logic
        virtual_bag_creator.create_baggage_image(obj_list, 
                                                 save_data=False)
        sf_obj_list = virtual_bag_creator.param_file + prior_list

        # This is your final virtual bag
        self.gt_image_3d = virtual_bag_creator.virtual_bag

        virtual_bag_creator.logger.propagate = False
        # virtual_bag_creator.mu_handler.logger.propagate = False

        # adjust baggage volume to zwidth
        if self.zwidth < self.gt_image_3d.shape[2]:
            zd = (self.gt_image_3d.shape[2] - self.zwidth)//2
            self.gt_image_3d = self.gt_image_3d[:,:,zd:-zd]

        self.gt_image_3d = torch.as_tensor(self.gt_image_3d)

        if virtual_bag_creator.logger.hasHandlers():
            virtual_bag_creator.logger.handlers.clear()

        del virtual_bag_creator.logger
        del virtual_bag_creator
        # ---------------------------------------------------------------------

        self.sf_obj_list = sf_obj_list

        # create compton_image - this is used for forward modelling
        self.compton_image_3d = torch.zeros_like(self.gt_image_3d,
                                                 dtype=torch.float)

        # replace gt label with compton value ---------------------------------
        for sf_obj in sf_obj_list:
            self.compton_image_3d = torch.where(
                self.gt_image_3d==sf_obj['label'],
                torch.Tensor([self.mu.material(sf_obj['material'], 
                                               'compton')]),
                self.compton_image_3d)

            if sf_obj['lqd_flag']:
                self.compton_image_3d = torch.where(
                    self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                    torch.Tensor([self.mu.material(sf_obj['lqd_param']['lqd_material'],
                                                   'compton')]),
                    self.compton_image_3d)
        # ---------------------------------------------------------------------

        # get attenuation curves for each material in the bag -----------------
        spectra = [loadtxt(spec)[:self.maxkV,1]
                   for spec in self.xray_source_model['spectra']]

        self.mu.calculate_lac_hu_values('water', spectra)

        sim_obj_dict = dict()
        material_curve = dict()

        # iterating through each object
        for k, sf_obj in enumerate(sf_obj_list):

            mu_curve =  zeros(len(self.keV_range))
            self.mu.calculate_lac_hu_values(sf_obj['material'], spectra)
            curr_material = self.mu.material(sf_obj['material'])

            atten_curve =  curr_material['mu']  # original unscaled atten. curve

            # Note: The original attenuation curve has mac values in units of
            #       ((1/cm) / (g/cc) = cm^2/g) - the volume though is in units
            #       of mm therefore the coeffs are scaled down by 0.1 (self.scale)

            # the array limits for mu_curve and atten_curve are to ensure
            # they are of the same length, mac value is multiplied by density
            # to convert to lac
            mu_curve[:atten_curve.size] = \
                atten_curve[:len(self.keV_range)]*curr_material['density']*self.scale

            # save the mu_curve in obj metadata
            sf_obj['mu_curve'] = mu_curve
            sf_obj['mu_dict']  = curr_material.copy()

            # add the mu curve to material curves dictionary
            material_curve[sf_obj['material']] = mu_curve

            # repeat for liquids
            if sf_obj['lqd_flag']:
                mu_curve = zeros(len(self.keV_range))

                self.mu.calculate_lac_hu_values(sf_obj['lqd_param']['lqd_material'],
                                                spectra)
                curr_material = self.mu.material(sf_obj['lqd_param']['lqd_material'])
                atten_curve = curr_material['mu']

                mu_curve[:atten_curve.size] = \
                    atten_curve[:len(self.keV_range)] * curr_material['density'] * self.scale

                sf_obj['lqd_param']['mu_curve'] = mu_curve
                sf_obj['lqd_param']['mu_dict'] = curr_material.copy()
                material_curve[sf_obj['lqd_param']['lqd_material']] = mu_curve

            sim_obj_dict['%i' % sf_obj['label']] = sf_obj.copy()
        # ---------------------------------------------------------------------

        # save the dictionary of objects as metadata
        with open(self.f_loc['sl_metadata'], 'wb') as f:
            pickle.dump(sim_obj_dict, f)
            f.close()

        self.logger.info(f"Metadata generated and saved "
                         f"at {self.f_loc['sl_metadata']}")
        # ----------------------------------------------------------------------

        self.logger.info(f"Number of Objects: {len(sf_obj_list)}")
        self.logger.info("Details:")

        print_table = []
        header = ['Label', 'Shape', 'Material', 'Lqd.']

        if not slicewise:
            header = header + ['Center/Base', 'Dim/Apex', 'Rot/Radius']

        for sf_obj in sf_obj_list:

            print_list = [str(sf_obj['label']), sf_obj['shape'],
                          sf_obj['material'], sf_obj['lqd_flag']]

            if sf_obj['shape'] in ['B', 'S', 'T', 'M']:
                ind1, ind2, ind3 = 'center', 'dim', 'rot'
            elif sf_obj['shape'] in ['E']:
                ind1, ind2, ind3 = 'center', 'axes', 'rot'
            elif sf_obj['shape'] in ['Y']:
                ind1, ind2, ind3 = 'base', 'apex', 'radius'
            elif sf_obj['shape'] in ['C']:
                ind1, ind2, ind3 = 'base', 'apex', 'radius1'

            if not slicewise:
                print_list.append('(%i,%i,%i)' % (sf_obj['geom'][ind1][0],
                                                  sf_obj['geom'][ind1][1],
                                                  sf_obj['geom'][ind1][2]))
                print_list.append('(%i,%i,%i)' % (sf_obj['geom'][ind2][0],
                                                  sf_obj['geom'][ind2][1],
                                                  sf_obj['geom'][ind2][2]))

            if sf_obj['shape'] in ['B', 'E', 'S', 'T', 'M']:
                if not slicewise:
                    print_list.append('(%i,%i,%i)' % (sf_obj['geom'][ind3][0],
                                                      sf_obj['geom'][ind3][1],
                                                      sf_obj['geom'][ind3][2]))

            else:
                print_list.append('%i' % (sf_obj['geom'][ind3]))

            print_table.append(print_list)

        self.logger.info("\n"+tabulate(print_table,
                                  headers=header,
                                  tablefmt='grid'))
        # self.logger.info()

        self.logger.info('=' * 40)
        self.material_curve = material_curve

        del obj_list
        self.save_dect_ground_truth_images(images=save_images)

        torch.cuda.empty_cache()
    # --------------------------------------------------------------------------

    def create_simulation_from_sl_file(self,
                                       shape_list_file,
                                       gt_image=None,
                                       save_images=['gt']
                                       ):
        """
        ------------------------------------------------------------------------
        Function to initialize simulation by reading a shape list from a
        previously saved shape list. The function can voxelize a ground truth
        image from the shape list as long as it does not contain liquids or
        sheet objects, otherwise, gt_image needs to be provided.

        :param shape_list_file: SL instance or file-path of the SL file.
        :param gt_image:        GT Label image corresponding to the shape file
        :param save_images:     ground truth images to save (Options: {'gt',
                                'compton', 'pe', 'zeff', 'lac_1', 'lac_2'})
        :return:
        ------------------------------------------------------------------------
        """

        if isinstance(shape_list_file, str):
            with open(shape_list_file, 'rb') as f:
                sf_obj_dict = pickle.load(f, encoding='latin1')
                sf_obj_list = [sf_obj_dict['%i'%x]
                               for x in range(1, len(sf_obj_dict.keys())+1)
                               ]
                f.close()
        else:
            sf_obj_list = shape_list_file

        # ----------------------------------------------------------------------

        spectra = [loadtxt(spec)[:self.maxkV,1]
                   for spec in self.xray_source_model['spectra']]

        sim_obj_dict = dict()
        material_curve = dict()

        # TODO: check load sim with liquid objects
        for k, sf_obj in enumerate(sf_obj_list):
            mu_curve = zeros(len(self.keV_range))

            self.mu.calculate_lac_hu_values(sf_obj['material'], spectra)
            curr_material = self.mu.material(sf_obj['material'])
            atten_curve = curr_material['mu']
            mu_curve[:atten_curve.size] = atten_curve[:len(self.keV_range)] * curr_material['density'] * self.scale
            sf_obj['mu_curve'] = mu_curve
            sf_obj['mu_dict'] = curr_material.copy()
            material_curve[sf_obj['material']] = mu_curve.copy()

            if sf_obj['lqd_flag']:
                mu_curve = zeros(len(self.keV_range))

                self.mu.calculate_lac_hu_values(sf_obj['lqd_param']['lqd_material'],
                                                spectra)
                curr_material = self.mu.material(sf_obj['lqd_param']['lqd_material'])
                atten_curve = curr_material['mu']
                mu_curve[:atten_curve.size] = atten_curve[:len(self.keV_range)] * curr_material['density'] * self.scale
                sf_obj['lqd_param']['mu_curve'] = mu_curve
                sf_obj['lqd_param']['mu_dict'] = curr_material.copy()
                material_curve[sf_obj['lqd_param']['lqd_material']] = mu_curve.copy()

            sim_obj_dict['%i' % sf_obj['label']] = sf_obj.copy()

        with open(self.f_loc['sl_metadata'], 'wb') as f:
            pickle.dump(sim_obj_dict, f)
            f.close()

        self.logger.info("\nMetadata generated")
        # ----------------------------------------------------------------------

        self.logger.info("\nNumber of Objects: %i"%len(sf_obj_list))
        self.logger.info("Details:")

        print_table = []
        header = ['Label', 'Shape', 'Material', 'Lqd.',
                  'Center/Base', 'Dim/Apex', 'Rot/Radius']

        for sf_obj in sf_obj_list:

            print_list = [str(sf_obj['label']), sf_obj['shape'],
                          sf_obj['material'], sf_obj['lqd_flag']]

            if sf_obj['shape'] in ['B', 'S', 'T', 'M']:
                ind1, ind2, ind3 = 'center', 'dim', 'rot'
            elif sf_obj['shape'] in ['E']:
                ind1, ind2, ind3 = 'center', 'axes', 'rot'
            elif sf_obj['shape'] in ['Y']:
                ind1, ind2, ind3 = 'base', 'apex', 'radius'
            elif sf_obj['shape'] in ['C']:
                ind1, ind2, ind3 = 'base', 'apex', 'radius1'

            print_list.append('(%i,%i,%i)' % (sf_obj['geom'][ind1][0],
                                              sf_obj['geom'][ind1][1],
                                              sf_obj['geom'][ind1][2]))
            print_list.append('(%i,%i,%i)' % (sf_obj['geom'][ind2][0],
                                              sf_obj['geom'][ind2][1],
                                              sf_obj['geom'][ind2][2]))

            if sf_obj['shape'] in ['B', 'E', 'S', 'T', 'M']:
                print_list.append('(%i,%i,%i)' % (sf_obj['geom'][ind3][0],
                                                  sf_obj['geom'][ind3][1],
                                                  sf_obj['geom'][ind3][2]))
            else:
                print_list.append('%i' % (sf_obj['geom'][ind3]))

            print_table.append(print_list)

        self.logger.info("\n"+tabulate(print_table,
                                       headers=header,
                                       tablefmt='grid'))

        self.logger.info('-' * 40+'\n')

        self.sf_obj_list = sf_obj_list
        self.material_curve = material_curve

        torch.cuda.empty_cache()

        if gt_image is None:
            voxelizer = ImageVoxelizer3D(sf_list=self.sf_obj_list,
                                         imgshape=self.image_shape_3d)

            self.compton_image_3d, self.gt_image_3d = \
                voxelizer.voxelize_3d_image()

            del voxelizer
        else:
            self.gt_image_3d = torch.as_tensor(gt_image)
            self.compton_image_3d = torch.zeros_like(self.gt_image_3d,
                                                     dtype=torch.float)

            for sf_obj in sf_obj_list:
                self.compton_image_3d = torch.where(
                    self.gt_image_3d == sf_obj['label'],
                    torch.Tensor([self.mu.material(sf_obj['material'], 
                                                   'compton')]),
                    self.compton_image_3d)

        torch.cuda.empty_cache()

        self.save_dect_ground_truth_images(images=save_images)
        torch.cuda.empty_cache()
    # --------------------------------------------------------------------------

    def generate_polychromatic_ct_projection(self,
                                            add_poisson_noise=True,
                                            add_system_noise=True,
                                            system_gain=2.5e-3,
                                            shot_gain=5e-5,
                                            spectrum=1):
        """
        ------------------------------------------------------------------------
        Generate polyenergetic sinogram for the ground truth label image. The
        ground truth image must be generated prior to calling this function. The
        function generates the noisy polychromatic sinogram for the specified
        spectrum in the Xray source model dictionary, self.xray_source_model and
        saves it in the sino/ folder in the simulation directory.

        :param add_poisson_noise:   Set to True if Poisson noise is to be added.
        :param add_system_noise:    Set to True if Gaussian shot noise is to be
                                    added.
        :param system_gain:         Gain for Gaussian shot noise
        :param spectrum:            index of the spectrum as specified in self.
                                    xray_source_model.

        :return
        ------------------------------------------------------------------------
        """

        t0 = time.time()
        torch.cuda.empty_cache()

        projection_buffer = torch.zeros(
            self.scanner_geometry['det_row_count'],
            self.reconstruction_geometry['n_views'],
            self.scanner_geometry['det_col_count'],
            dtype=torch.float
        )

        i = spectrum
        self.logger.info("Generating Polyenergetic Sinograms "
                         "for Spectrum %i ..."%(i))

        curr_spectrum =  loadtxt(self.xray_source_model['spectra'][i-1])[:, 1]
        curr_pc = self.xray_source_model['dosage'][i-1]

        torch.cuda.empty_cache()

        material_list = unique(list(self.material_curve.keys()))
        pc_sum = 0

        # self.keV_range = range(50,53)

        kev_iter = tqdm(self.keV_range[:curr_spectrum.size])

        for e in kev_iter:
            k = e - self.keV_range[0]
            t1 = time.time()
            ref_image = torch.zeros_like(self.compton_image_3d)
            kev_iter.set_description(f"Processing Energy Level, {e} keV:\t", refresh=True)

            for mat in material_list:

                ref_image = torch.where(self.compton_image_3d==self.mu.material(mat,
                                                                                'compton'),
                                        torch.Tensor([self.material_curve[mat][k]]),
                                        ref_image)
                torch.cuda.empty_cache()

            curr_projection = self.scanner.projn_generator(ref_image)
            curr_projection = torch.mul(curr_projection, curr_pc)
            curr_projection = torch.mul(curr_projection, curr_spectrum[e-10]*system_gain*e)

            if add_poisson_noise:
                poisson_sampler = Poisson(curr_projection.type(torch.float))
                curr_projection = poisson_sampler.sample()
                del poisson_sampler
                torch.cuda.empty_cache()

            curr_projection = torch.as_tensor(curr_projection,
                                              dtype=torch.float)

            if add_system_noise:

                mask = torch.ones_like(curr_projection)
                g_noise = torch.normal(0.0, mask)
                g_noise = torch.mul(g_noise, shot_gain)
                curr_projection = torch.add(curr_projection, g_noise)
                del g_noise, mask
                torch.cuda.empty_cache()

            projection_buffer = torch.add(curr_projection, projection_buffer)

            del curr_projection
            torch.cuda.empty_cache()
            pc_sum += curr_pc*curr_spectrum[e-10]*system_gain*e

        projection_buffer = torch.where(projection_buffer<1,
                                        torch.Tensor([1.0]), projection_buffer)
        projection_buffer = torch.log(projection_buffer)
        projection_buffer = torch.neg(projection_buffer)
        projection_buffer = torch.add(projection_buffer,  log(pc_sum))
        projection_buffer = projection_buffer.cpu().numpy()
        torch.cuda.empty_cache()

        self.logger.info("Sinogram Created ...")

        save_fits_data(self.f_loc['sino_file']%spectrum,
                       projection_buffer,
                       self.compress_data)
        
        self.logger.info("Sinogram saved as %s"%(self.f_loc['sino_file']%spectrum))
        self.logger.info(f"Time Taken: {time.time() - t0}")
    # --------------------------------------------------------------------------

    def add_scatter_to_ct_projection_slice(self,
                                      add_poisson_noise=True,
                                      add_system_noise=True,
                                      system_gain=5e-3,
                                      spectrum=1,
                                      slice_no=150
                                      ):
        """
        ------------------------------------------------------------------------
        Add deterministic first-order scattering artifacts to the slice. The
        algorithm is adopted from:

        Freud, N., et al. "Deterministic simulation of first-order scattering
        in virtual X-ray imaging." Nuclear Instruments and Methods in Physics
        Research Section B: Beam Interactions with Materials and Atoms 222.1-2
        (2004): 285-300.

        :param add_poisson_noise:   Set to True if Poisson noise is to be added.
        :param add_system_noise:    Set to True if Gaussian shot noise is to be
                                    added.
        :param system_gain:         Gain for Gaussian shot noise
        :param spectrum:            index of the spectrum as specified in self.
                                    xray_source_model.
        :param slice_no:            Slice to which scatter is to be added.

        :return
        ------------------------------------------------------------------------
        """

        if not os.path.exists(self.f_loc['sino_file']%spectrum):
            self.generate_polychromatic_ct_projection(
                add_poisson_noise=add_poisson_noise,
                add_system_noise=add_system_noise,
                system_gain=system_gain,
                spectrum=spectrum
            )

        self.data = read_fits_data(self.f_loc['sino_file']%spectrum, 0)
        self.sino = self.data[slice_no, :, :].copy()
        del self.data

        i = spectrum
        self.logger.info("Generating Polyenergetic Sinograms "
                         "for Spectrum %i ..." % (i))

        curr_spectrum = loadtxt(self.xray_source_model['spectra'][i - 1])[:, 1]
        material_list = unique(list(self.material_curve.keys()))

        # set up the scatter simulation framework
        scatter_sim   = ScatterSimulator(self.scanner,
                                         self.sf_obj_list)
        scatter_sim.set_scatter_calculator(
            self.gt_image_3d[:, :, slice_no].cpu().numpy())

        kev_iter = tqdm(self.keV_range[:curr_spectrum.size])

        # initialize the scattering projection
        scatter_projn = zeros_like(self.sino.shape)

        # Calculate scatter projections for energy level
        for e in kev_iter:
            k = e - self.keV_range[0]
            ref_image = torch.zeros_like(self.compton_image_3d)
            kev_iter.set_description(f"Calculating Scatter at Energy Level, {e} keV:\t",
                                     refresh=True)

            for mat in material_list:
                ref_image = torch.where(
                    self.compton_image_3d == self.mu.material(mat, 'compton'),
                    torch.Tensor([self.material_curve[mat][k]]),
                    ref_image)

            scatter_projn = scatter_sim.get_scatter_projections(
                atten_image=ref_image[:,:,slice_no].cpu().numpy(),
                e=e,
                xray_specs=self.xray_source_model,
                spectrum=curr_spectrum
            )

        pc_sum = self.xray_source_model['dosage'][spectrum-1]

        projn = (self.sino - log(pc_sum))
        projn = np.exp(-projn)

        projn = projn + scatter_projn.T

        s_projn = torch.as_tensor(projn, dtype=torch.float, device='cuda')

        s_projn = torch.where(s_projn<1,
                              torch.Tensor([1.0]), s_projn)
        s_projn = torch.log(s_projn)
        s_projn = torch.neg(s_projn)
        s_projn = torch.add(s_projn,  log(pc_sum))
        self.scatter_sino = s_projn.cpu().numpy()

        self.scatter_recon = self.scanner.reconstruct_data(self.scatter_sino)

        spectra = [loadtxt(spec)[:self.maxkV,1]
                   for spec in self.xray_source_model['spectra']]

        self.mu.calculate_lac_hu_values('water', spectra)
        mu_w = self.mu.material('water')
        cmin, cmax, offset = -1000, 3.2e4, 0
        scale = 10 #self.scanner.recon_params['img_scale']

        self.scatter_recon *= scale
        self.scatter_recon = \
            (self.scatter_recon - mu_w['lac_1']) / mu_w['lac_1'] * 1000 + offset
        self.scatter_recon = clip(self.scatter_recon, cmin, cmax).astype(np.int16)

        self.recon = self.scanner.reconstruct_data(self.sino)
        self.recon *= scale
        self.recon = (self.recon - mu_w['lac_1']) / mu_w['lac_1'] * 1000 + offset
        self.recon = clip(self.recon, cmin, cmax).astype(np.int16)

        return self.scatter_sino, self.scatter_recon, self.sino, self.recon

    # --------------------------------------------------------------------------

    def run_bag_generator(self,
                          mode='randomized', 
                          bag_creator_dict=None, 
                          sf_file=None, 
                          sim_args={}):
        """
        ------------------------------------------------------------------------
        Run the Virtual Bag generator block - it can be run either randomized or
        manual mode. In the randomized mode, running the block creates a
        randomized virtual bag with randomly placed objects and randomly assign-
        ed material properties. In the manual mode, running the block reads in
        a shape list (location to a sl_metadata.pyc file) to create the 3D virtual
        bag.

        :param mode:              set the mode to 'manual' or 'randomized'
        :param bag_creator_dict:  dictionary of arguments for BaggageCreator3D
                                  to create a randomized bag
        :param sf_file:           a shape list or path to shape if running in
                                  'manual' mode
        :param sim_args:          additional optional arguments - see
                                  self.create_random_simulation_instance or
                                  self.create_simulation_from_sl_file
        :return: 
        -----------------------------------------------------------------------
        """
        if mode=='randomized':
            self.create_random_simulation_instance(bag_creator_dict, **sim_args)
        
        elif mode=='manual':
            self.create_simulation_from_sl_file(sf_file, **sim_args)
    # --------------------------------------------------------------------------

    def run_fwd_model(self,
                      add_poisson_noise=True,
                      add_system_noise=True,
                      system_gain=5e-4):
        """
        ------------------------------------------------------------------------
        Run the forward X-ray modelling block - generates polychromatic proj-
        ections for the entire system by iterating the function
        self.generate_polychromatic_ct_projection() over all the spectra in the
        model.

        :param add_poisson_noise:   Set to True if Poisson noise is to be added.
        :param add_system_noise:    Set to True if Gaussian shot noise is to be
                                    added.
        :param system_gain:         Gain for Gaussian shot noise
        :return:
        ------------------------------------------------------------------------
        """

        for spec_no in range(self.xray_source_model['num_spectra']):
            self.generate_polychromatic_ct_projection(
                add_poisson_noise=add_poisson_noise,
                add_system_noise=add_system_noise,
                system_gain=system_gain,
                spectrum=spec_no + 1
            )
            torch.cuda.empty_cache()

        self.logger.info("Xray data generated")
        self.logger.info("=" * 80)

        # Load the generated projection data
        if self.xray_source_model['num_spectra']==2:
            data1 = read_fits_data(self.f_loc['sino_file'] % 1, 0)
            data2 = read_fits_data(self.f_loc['sino_file'] % 2, 0)

            data1 = moveaxis(data1, -1, 0)
            data1 = data1[:, :, ::-1]
            data2 = moveaxis(data2, -1, 0)
            data2 = data2[:, :, ::-1]

            self.data1 = data1
            self.data2 = data2

        elif self.xray_source_model['num_spectra']==1:
            data = read_fits_data(self.f_loc['sino_file'] % 1, 0)

            data = moveaxis(data, -1, 0)
            data = data[:, :, ::-1]

            self.data = data
    # --------------------------------------------------------------------------

    def run_fwd_model_with_motion_artifacts(self,
                                            n_steps=3,
                                            blur_res=6,
                                            mode='bag',
                                            bag_params=None,
                                            lqd_params=None,
                                            obj_params=None,
                                            fwd_model_args=None
                                            ):
        """
        -----------------------------------------------------------------------

        :param n_steps:
        :param blur_res:
        :param mode:
        :param bag_params:
        :param lqd_params:
        :param obj_params:
        :return:
        -----------------------------------------------------------------------
        """

        if fwd_model_args is None:
            fwd_model_args = dict(add_poisson_noise=True,
                                  add_system_noise=True,
                                  system_gain=5e-4)

        assert mode in ['bag', 'objects']

        self.logger.info("Creating Data for Original Virtual Bag ...")
        self.f_loc['sino_file'] = os.path.join(self.f_loc['sino_dir'],
                                               'sino_%i_seq_00.fits.gz')
        self.run_fwd_model(**fwd_model_args)

        if mode=='bag':

            if bag_params is None:
                bag_params = dict(n_seqs=4,
                                  rotate=True,
                                  x_tol=4,
                                  t_tol=3,
                                  fixed_x=True)

            n_seqs = bag_params['objects']

            for s in range(1, bag_params['n_seqs']):

                self.logger.info("="*80)
                self.logger.info(f"Simulating Sequence {s}")
                # the motion translation, rotation parameters
                x_tol = np.random.choice(range(-bag_params['x_tol'],
                                                bag_params['x_tol']),
                                         size=3
                                         )
                t_tol = np.random.choice(range(-bag_params['t_tol'],
                                                bag_params['t_tol']),
                                         size=3)

                # if no vertical movement is allowed
                if bag_params['fixed_x']:
                    x_tol[0] = 0
                    t_tol[0], t_tol[2] = 0, 0

                self.gt_image_3d = self.gt_image_3d.cpu().numpy()
                self.gt_image_3d = sptx.rotate(self.gt_image_3d, t_tol[0],
                                               axes=(0,1)).astype(int)
                self.gt_image_3d = sptx.rotate(self.gt_image_3d, t_tol[1],
                                               axes=(1,2)).astype(int)
                self.gt_image_3d = sptx.rotate(self.gt_image_3d, t_tol[2],
                                               axes=(2, 0)).astype(int)

                self.gt_image_3d = sptx.shift(self.gt_image_3d,
                                              x_tol).astype(int)
                self.gt_image_3d = torch.from_numpy(self.gt_image_3d).to('cuda')

                self.compton_image_3d = torch.zeros_like(self.gt_image_3d,
                                                         dtype=torch.float)

                # create compton image for shifted virtual bag
                for sf_obj in self.sf_obj_list:
                    self.compton_image_3d = torch.where(
                        self.gt_image_3d == sf_obj['label'],
                        torch.Tensor([self.mu.material(sf_obj['material'],
                                                       'compton')]),
                        self.compton_image_3d)

                    if sf_obj['lqd_flag']:
                        self.compton_image_3d = torch.where(
                            self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                            torch.Tensor([
                                self.mu.material(sf_obj['lqd_param']['lqd_material'],
                                'compton')]),
                            self.compton_image_3d)

                self.f_loc['sino_file'] =  os.path.join(self.f_loc['sino_dir'],
                                                        f"sino_%i_seq_{s:02d}.fits.gz")
                self.run_fwd_model(**fwd_model_args)
                self.logger.info("="*80)
                self.logger.info("="*80)

                del self.compton_image_3d
                torch.cuda.empty_cache()

        elif mode=='objects':

            if obj_params is None:
                obj_params = dict(n_seqs=6,
                                  rotate=True,
                                  x_tol=6,
                                  t_tol=5,
                                  objects=10,
                                  fixed_x=True)

            obj_labels = [x['label'] for x in self.sf_obj_list
                          if x['label'] not in [1,2,3]]

            if isinstance(obj_params['objects'], int):
                obj_params['objects'] = np.random.choice(obj_labels,
                                                         size=obj_params['objects'])
            elif isinstance(obj_params['objects'], list):
                pass
            else:
                raise TypeError("Datatype for obj_params['objects'] not recognized!")

            self.gt_image_3d = self.gt_image_3d.cpu().numpy()
            orig_gt_image = self.gt_image_3d.copy()

            n_seqs = obj_params['n_seqs']

            for s in range(1, obj_params['n_seqs']):

                self.logger.info("="*80)
                self.logger.info(f"Simulating Sequence {s}")
                # the motion translation, rotation parameters
                x_tol = np.random.choice(range(-obj_params['x_tol'],
                                                obj_params['x_tol']),
                                         size=3)
                t_tol = np.random.choice(range(-obj_params['t_tol'],
                                                obj_params['t_tol']),
                                         size=3)

                # if no vertical movement is allowed
                if obj_params['fixed_x']:
                    x_tol[0] = 0
                    t_tol[0], t_tol[2] = 0, 0

                masked_gt_image  = orig_gt_image.copy()
                moving_obj_vol = zeros_like(masked_gt_image)

                for i in obj_params['objects']:
                    moving_obj_vol[orig_gt_image==i] = i
                    masked_gt_image[orig_gt_image==i] = 0

                # nz = moving_obj_vol.nonzero()

                moving_obj_vol = sptx.rotate(moving_obj_vol,
                                             t_tol[0],
                                             axes=(0,1),
                                             reshape=False).astype(int)
                moving_obj_vol = sptx.rotate(moving_obj_vol,
                                             t_tol[1],
                                             axes=(1,2),
                                             reshape=False).astype(int)
                moving_obj_vol = sptx.rotate(moving_obj_vol,
                                             t_tol[2],
                                             axes=(2, 0),
                                             reshape=False).astype(int)

                moving_obj_vol = sptx.shift(moving_obj_vol,
                                            x_tol).astype(int)

                masked_gt_image[moving_obj_vol>0] = moving_obj_vol[moving_obj_vol>0]
                self.gt_image_3d = torch.from_numpy(masked_gt_image).to('cuda')

                self.compton_image_3d = torch.zeros_like(self.gt_image_3d,
                                                         dtype=torch.float)

                # create compton image for shifted virtual bag
                for sf_obj in self.sf_obj_list:
                    self.compton_image_3d = torch.where(
                        self.gt_image_3d == sf_obj['label'],
                        torch.Tensor([self.mu.material(sf_obj['material'],
                                                       'compton')]),
                        self.compton_image_3d)

                    if sf_obj['lqd_flag']:
                        self.compton_image_3d = torch.where(
                            self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                            torch.Tensor([
                                self.mu.material(sf_obj['lqd_param']['lqd_material'],
                                'compton')]),
                            self.compton_image_3d)

                self.f_loc['sino_file'] =  os.path.join(self.f_loc['sino_dir'],
                                                        f"sino_%i_seq_{s:02d}.fits.gz")
                self.run_fwd_model(**fwd_model_args)
                self.logger.info("="*80)
                self.logger.info("="*80)

                del self.compton_image_3d
                torch.cuda.empty_cache()

        # Creating sinograms with motion artifacts
        if self.xray_source_model['num_spectra']==1:

            self.data = zeros((self.scanner.machine_geometry['det_row_count'],
                               self.scanner.recon_geometry['n_views'],
                               self.scanner.machine_geometry['det_col_count']))

            for s in range(n_seqs):
                self.f_loc['sino_file'] =  os.path.join(self.f_loc['sino_dir'],
                                                        f"sino_%i_seq_{s:02d}.fits.gz")

                data = read_fits_data(self.f_loc['sino_file'] % 1, 0)
                self.data[:, s::n_seqs, :] = \
                                    data[:, s::n_seqs,:]

            self.f_loc['sino_file'] =  os.path.join(self.f_loc['sino_dir'],
                                                    'sino_%i.fits.gz')
            save_fits_data(self.f_loc['sino_file'] % 1,
                           self.data,
                           self.compress_data)

            self.data = moveaxis(self.data, -1, 0)
            self.data = self.data[:, :, ::-1]

        elif self.xray_source_model['num_spectra']==2:

            self.data1 = zeros((self.scanner.machine_geometry['det_row_count'],
                                self.scanner.machine_geometry['n_views'],
                                self.scanner.machine_geometry['det_col_count']))
            self.data2 = zeros((self.scanner.machine_geometry['det_row_count'],
                                self.scanner.machine_geometry['n_views'],
                                self.scanner.machine_geometry['det_col_count']))

            for s in range(bag_params['n_seqs']):
                self.f_loc['sino_file'] = f"sino_%i_seq_{s:02d}.fits.gz"
                data1 = read_fits_data(self.f_loc['sino_file'] % 1, 0)
                data2 = read_fits_data(self.f_loc['sino_file'] % 2, 0)

                self.data1[:, s::bag_params['n_seqs'], :] = \
                                    data1[:, s::bag_params['n_seqs'], :]
                self.data2[:, s::bag_params['n_seqs'], :] = \
                                    data2[:, s::bag_params['n_seqs'], :]

            self.f_loc['sino_file'] = 'sino_%i.fits.gz'
            save_fits_data(self.f_loc['sino_file'] % 1, self.data1, self.compress_data)

            self.data1 = moveaxis(self.data1, -1, 0)
            self.data1 = self.data1[:, :, ::-1]

            save_fits_data(self.f_loc['sino_file'] % 2, self.data2, self.compress_data)

            self.data2 = moveaxis(self.data2, -1, 0)
            self.data2 = self.data2[:, :, ::-1]
    # -------------------------------------------------------------------------

    def run_decomposer(self,
                       type='cdm',
                       decomposer_args=None,
                       basis_fn=None,
                       save_sino=False):

        """
        ------------------------------------------------------------------------
        Run the Dual Energy Decomposition block -  uses either the CDM or SIRZ
        or LUTD to process Dual Energy data

        :param type:            select from {'cdm' | 'sirz' | 'lutd' }
        :param decomposer_args: additional arguments for DE decomposition
        :return:
        ------------------------------------------------------------------------
        """
        self.DECOMPOSER_FLAG = True
        self.decomposer_type = type
        self.basis_fn = basis_fn
        self.save_de_sino = save_sino

        assert type=='cdm', "Open-source version only support CDM method " \
                            "for dual energy decomposition"

        if type in ['cdm', 'sirz']:

            # Initialize CDM reconstructor
            sim_specs = dict(
                spctr_h_fname=self.xray_source_model['spectra'][0],
                spctr_l_fname=self.xray_source_model['spectra'][1],
                photon_count_high=self.xray_source_model['dosage'][0],
                photon_count_low=self.xray_source_model['dosage'][1],
                nangs=self.data1.shape[2],
                nbins=self.data1.shape[0],
                projector='cpu'
            )

            cdm_sim = CDMDecomposer(**sim_specs)
            # -----------------------------------------------------------------------------

            # Set basis function if not using Compton-PE basis
            if basis_fn is not None:
                cdm_sim.set_basis_functions(**basis_fn)

            sino_pe = zeros_like(self.data1)
            sino_c  = zeros_like(self.data2)
            nrows   = self.data1.shape[1]

            for i in range(nrows):
                self.logger.info("Row %d:" % i)
                cdm_sim.init_val = decomposer_args['init_val']
                # try:
                sino_pe[:, i, :], sino_c[:, i, :] = \
                    cdm_sim.decompose_dect_sinograms(
                        self.data1[:, i, :], self.data2[:, i, :],
                        solver=decomposer_args['cdm_solver'],
                        type=decomposer_args['cdm_type']
                    )
                # except:
                #     self.logger.info("GPufit error encountered")
                torch.cuda.empty_cache()

            self.sino_c = sino_c.copy()
            self.sino_pe = sino_pe.copy()

    # --------------------------------------------------------------------------

    def run_reconstructor(self,
                          img_type='HU',
                          recon='fbp',
                          plot_stats=True,
                          fname=None):
        """
        ------------------------------------------------------------------------
        Run the reconstructor block - uses the methods from ScannerTemplate to
        run the reconstructor for the scanner geometry that was used to
        initialize the ScannerTemplate class.

        :param img_type:    select the image unit: {'HU' | 'MHU' | 'LAC'}
        :param recon:       select reconstruction algo: {'sirt' | 'fbp'}
        :return:
        ------------------------------------------------------------------------
        """

        # reconstruct images from the sinograms
        if recon is not 'fbp':
            self.scanner.update_recon_algo(recon)

        if self.xray_source_model['num_spectra']==2:
            image_1 = self.scanner.reconstruct_data(self.data1,
                                                    full_range=True,
                                                    append_air_turns=True)

            self.logger.info("Reconstructed LAC Image 1 ...")

            image_2 = self.scanner.reconstruct_data(self.data2,
                                                    full_range=True,
                                                    append_air_turns=True)
            self.logger.info("Reconstructed LAC Image 2 ...")

            del self.data1, self.data2
        elif self.xray_source_model['num_spectra']==1:
            image_1 = self.scanner.reconstruct_data(self.data,
                                                    full_range=True,
                                                    append_air_turns=True)
            self.logger.info("Reconstructed LAC Image ...")
            del self.data

        # reconstruct any decomposed line integrals

        if self.DECOMPOSER_FLAG:

            image_c = self.scanner.reconstruct_data(self.sino_c, full_range=True,
                                               append_air_turns=True)
            self.logger.info("Reconstructed Compton Image ...")

            image_pe = self.scanner.reconstruct_data(self.sino_pe, full_range=True,
                                                append_air_turns=True)
            self.logger.info("Reconstructed PE Image ...")

            if self.decomposer_type in ['cdm', 'lutd']:
                image_z = effective_atomic_number(image_pe, image_c)
                img_suffixes = ['c', 'pe']
                self.logger.info("Created Zeff Image ...")

            elif self.decomposer_type=='sirz':
                image_z, image_rho =  self.sirz_decomp.run_sirz2_decomp(self.sino_c,
                                                                        self.sino_pe)
                img_suffixes = ['b1', 'b2']
                self.logger.info("Created Ze-Rhoe Images ...")

            if self.save_de_sino:
                if fname is None:
                    out_fname = os.path.join(self.f_loc['sino_dir'],
                                             'sino_%s.npz' % (img_suffixes[0]))
                else:
                    out_fname = os.path.join(self.f_loc['sino_dir'],
                                             fname % (img_suffixes[0]))

                self.logger.info(f"Saving results to: {out_fname}")
                self.sino_c = moveaxis(self.sino_c, -1, 0)
                savez_compressed(out_fname, self.sino_c)
            del self.sino_c

            if self.save_de_sino:
                if fname is None:
                    out_fname = os.path.join(self.f_loc['sino_dir'],
                                             'sino_%s.npz' % (img_suffixes[1]))
                else:
                    out_fname = os.path.join(self.f_loc['sino_dir'],
                                             fname % (img_suffixes[1]))

                self.logger.info(f"Saving results to: {out_fname}")
                self.sino_pe = moveaxis(self.sino_pe, -1, 0)
                savez_compressed(out_fname, self.sino_pe)
            del self.sino_pe

        scale = self.scanner.recon_params['img_scale']

        spectra = [loadtxt(spec)[:self.maxkV,1]
                   for spec in self.xray_source_model['spectra']]

        self.mu.calculate_lac_hu_values('water', spectra)
        mu_w = self.scanner.recon_params['mu_w'] if self.scanner.recon_params['mu_w'] \
                                                 is not None else self.mu.material('water')

        # convert to Hounsfield or Modified Hounsfield units
        if img_type=='HU':  cmin, cmax, offset = -1000, 3.2e4, 0
        if img_type=='MHU': cmin, cmax, offset = 0, 3.2e4, 1000

        if img_type in ['HU', 'MHU']:

            if self.xray_source_model['num_spectra'] == 2:

                if self.scanner.machine_geometry['scanner_name'] == 'default_parallelbeam':
                    image_1 = moveaxis(image_1.copy(), 0, -1)
                    image_2 = moveaxis(image_2.copy(), 0, -1)
                    image_1 = image_1[::-1,:,:].copy()
                    image_2 = image_2[::-1,:,:].copy()

                image_1 *= scale
                image_1 = (image_1 - mu_w['lac_1']) / mu_w['lac_1'] * 1000 + offset
                image_1 = clip(image_1, cmin, cmax).astype(np.int16)

                out_fname = os.path.join(self.f_loc['image_dir'],
                                         self.f_loc['img_file'] % 1)
                self.logger.info("Saving results to: "+out_fname)

                save_fits_data(out_fname, image_1, self.compress_data)

                image_2 *= scale
                image_2 = (image_2 - mu_w['lac_2']) / mu_w['lac_2'] * 1000 + offset
                image_2  = clip(image_2,  cmin, cmax).astype(np.int16)

                # Save reconstructed images
                out_fname = os.path.join(self.f_loc['image_dir'],
                                         self.f_loc['img_file'] % 2)
                self.logger.info("Saving results to: "+out_fname)
                save_fits_data(out_fname, image_2, self.compress_data)

                create_gif(os.path.join(self.f_loc['gif_dir'], 'gt_label_image.gif'),
                           self.gt_image_3d.cpu().numpy(),
                           stride=5)

                create_gif(os.path.join(self.f_loc['gif_dir'],
                                        'recon_image_%i.gif'%1),
                           np.clip(image_1+1000, 0, 3500), stride=2)
                create_gif(os.path.join(self.f_loc['gif_dir'],
                                        'recon_image_%i.gif'%2),
                           np.clip(image_2+1000, 0, 3500), stride=2)
                self.logger.info("Created GIFs for simulated volumes")

                if plot_stats:
                    self.plot_baggage_statistics()
                    self.logger.info("Plotted Baggage Statistics")

            elif self.xray_source_model['num_spectra'] == 1:
                image_1 *= scale
                image_1 = (image_1 - mu_w['lac_1']) / mu_w['lac_1'] * 1000 + offset
                image_1 = clip(image_1, cmin, cmax).astype(np.int16)

                if self.scanner.machine_geometry['scanner_name'] == 'default_parallelbeam':
                    image_1 = moveaxis(image_1.copy(), 0, -1)
                    image_1 = image_1[::-1,:,:].copy()

                out_fname = os.path.join(self.f_loc['image_dir'],
                                         self.f_loc['img_file']%1)
                self.logger.info("Saving results to: "+out_fname)
                save_fits_data(out_fname, image_1, self.compress_data)
                create_gif(os.path.join(self.f_loc['gif_dir'], 'gt_label_image.gif'),
                           self.gt_image_3d.cpu().numpy(),
                           stride=5)

                create_gif(os.path.join(self.f_loc['gif_dir'],
                                        'recon_image_%i.gif'%1),
                           np.clip(image_1+1000, 0, 3500), stride=2)

                self.logger.info("Created GIFs for simulated volumes")

                if plot_stats:
                    self.plot_baggage_statistics()
                    self.logger.info("Plotted Baggage Statistics")

            if self.DECOMPOSER_FLAG:
                if self.scanner.machine_geometry['scanner_name'] == 'default_parallelbeam':
                    image_c = moveaxis(image_c.copy(), 0, -1)
                    image_pe = moveaxis(image_pe.copy(), 0, -1)
                    image_z = moveaxis(image_z.copy(), 0, -1)
                    image_c = image_c[::-1,:,:].copy()
                    image_pe = image_pe[::-1,:,:].copy()
                    image_z = image_z[::-1,:,:].copy()

                image_c *= scale
                image_pe *= scale
                image_c = (image_c - mu_w['compton'])/mu_w['compton']*1000 + offset
                image_pe = (image_pe - mu_w['pe'])/mu_w['pe']*1000 + offset
                image_z = (image_z - mu_w['z'])/mu_w['z']*1000 + offset
                image_z[image_1 < cmin+200] = cmin
                image_z[image_2 < cmin+200] = cmin
                image_c  = clip(image_c,  cmin, cmax).astype(np.int16)
                image_pe = clip(image_pe, cmin, cmax).astype(np.int16)
                image_z  = clip(image_z,  cmin, cmax).astype(np.int16)

                out_fname = os.path.join(self.f_loc['image_dir'],
                                         self.f_loc['img_file'].replace('%i', '%s') % img_suffixes[0])
                self.logger.info("Saving results to: "+out_fname)
                save_fits_data(out_fname, image_c, self.compress_data)

                out_fname = os.path.join(self.f_loc['image_dir'],
                                         self.f_loc['img_file'].replace('%i', '%s') % img_suffixes[1])
                self.logger.info("Saving results to: "+out_fname)
                save_fits_data(out_fname, image_pe, self.compress_data)

                out_fname = os.path.join(self.f_loc['image_dir'],
                                         self.f_loc['img_file'].replace('%i', '%s') % 'z')
                self.logger.info("Saving results to: "+out_fname)
                save_fits_data(out_fname, image_z, self.compress_data)

                del image_c, image_pe, image_z, image_1, image_2

                if self.decomposer_type=='sirz':

                    image_rho *= scale
                    image_rho = (image_rho - mu_w['density']) / mu_w['density'] * 1000 + offset
                    image_rho = clip(image_rho, cmin, cmax).astype(np.int16)

                    out_fname = os.path.join(self.f_loc['image_dir'],
                                             'recon_image_%s.fits.gz' % 'rho')
                    self.logger.info("Saving results to: "+out_fname)
                    save_fits_data(out_fname, image_rho, self.compress_data)
    # --------------------------------------------------------------------------

    def save_dect_ground_truth_images(self, images=['gt']):
        """
        ------------------------------------------------------------------------
        Save 3D image files generated by the phantom voxelizer as .fits.gz files.

        :param images:  The ground truth images to be saved. Options = {'gt',
                        'compton', 'pe', 'zeff', 'lac_1', 'lac_2'}
        :return
        ------------------------------------------------------------------------
        """

        if 'gt' in images:
            save_fits_data(self.f_loc['gt_image'],
                           self.gt_image_3d.cpu().numpy().astype(int16), self.compress_data)

            self.logger.info("GT image saved as %s"%self.f_loc['gt_image'])

        if 'compton' in images:
            compton_image_file = os.path.join(self.f_loc['gt_dir'],
                                              'gt_compton_image.fits.gz')

            save_fits_data(compton_image_file,
                           self.compton_image_3d.cpu().numpy(), self.compress_data)

            self.logger.info("Compton image saved as %s"%compton_image_file)

        if 'pe' in images:
            pe_image_file = os.path.join(self.f_loc['gt_dir'],
                                         'gt_pe_image.fits.gz')

            pe_image_3d = torch.zeros_like(self.gt_image_3d).to(torch.float32)

            for sf_obj in self.sf_obj_list:
                pe_image_3d = torch.where(self.gt_image_3d == sf_obj['label'],
                                          torch.Tensor([self.mu.material(sf_obj['material'],
                                                                         'pe')]),
                                          pe_image_3d)

                if sf_obj['lqd_flag']:
                    lac_1_image_3d = torch.where(self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                                                 torch.Tensor([self.mu.material(sf_obj['lqd_param'][
                                                                                    'lqd_material'],
                                                                                'pe')]),
                                                 pe_image_3d)

            save_fits_data(pe_image_file, pe_image_3d.cpu().numpy(), self.compress_data)

            del pe_image_3d
            torch.cuda.empty_cache()

            self.logger.info("PE image saved as %s"%pe_image_file)

        if 'zeff' in images:
            zeff_image_file = os.path.join(self.f_loc['gt_dir'],
                                           'gt_zeff_image.fits.gz')

            zeff_image_3d = torch.zeros_like(self.gt_image_3d).to(torch.float32)

            for sf_obj in self.sf_obj_list:
                zeff_image_3d = torch.where(self.gt_image_3d == sf_obj['label'],
                                            torch.Tensor([self.mu.material(sf_obj['material'],
                                                                           'z')]),
                                            zeff_image_3d)

                if sf_obj['lqd_flag']:
                    lac_1_image_3d = torch.where(self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                                                 torch.Tensor([self.mu.material(sf_obj['lqd_param'][
                                                                                    'lqd_material'],
                                                                                'z')]),
                                                 zeff_image_3d)

            save_fits_data(zeff_image_file, zeff_image_3d.cpu().numpy(), self.compress_data)

            del zeff_image_3d

            torch.cuda.empty_cache()
            self.logger.info("Zeff image saved as %s"%zeff_image_file)

        if 'lac' in images:
            lac_image_file = os.path.join(self.f_loc['gt_dir'],
                                            'gt_lac_image.fits.gz')

            lac_image_3d = torch.zeros_like(self.gt_image_3d).to(torch.float32)

            for sf_obj in self.sf_obj_list:
                lac_image_3d = torch.where(self.gt_image_3d == sf_obj['label'],
                                             torch.Tensor([self.mu.material(sf_obj['material'],
                                                                            'lac')]),
                                             lac_image_3d)

                if sf_obj['lqd_flag']:
                    lac_1_image_3d = torch.where(self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                                                 torch.Tensor([self.mu.material(sf_obj['lqd_param']['lqd_material'],
                                                                                'lac')]),
                                                 lac_1_image_3d)

            save_fits_data(lac_image_file, lac_image_3d.cpu().numpy(), self.compress_data)

            del lac_image_3d

            torch.cuda.empty_cache()
            self.logger.info("LAC image saved as %s"%lac_image_file)

        if 'lac_1' in images:
            lac_1_image_file = os.path.join(self.f_loc['gt_dir'],
                                            'gt_lac_1_image.fits.gz')

            lac_1_image_3d = torch.zeros_like(self.gt_image_3d).to(torch.float32)

            for sf_obj in self.sf_obj_list:
                lac_1_image_3d = torch.where(self.gt_image_3d == sf_obj['label'],
                                             torch.Tensor([self.mu.material(sf_obj['material'],
                                                                            'lac_1')]),
                                             lac_1_image_3d)

                if sf_obj['lqd_flag']:
                    lac_1_image_3d = torch.where(self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                                                 torch.Tensor([self.mu.material(sf_obj['lqd_param'][
                                                                                    'lqd_material'],
                                                                                'lac_1')]),
                                                 lac_1_image_3d)

            save_fits_data(lac_1_image_file, lac_1_image_3d.cpu().numpy(), self.compress_data)

            del lac_1_image_3d

            torch.cuda.empty_cache()
            self.logger.info("LAC 1 image saved as %s"%lac_1_image_file)

        if 'lac_2' in images:
            lac_2_image_file = os.path.join(self.f_loc['gt_dir'],
                                            'gt_lac_2_image.fits.gz')

            lac_2_image_3d = torch.zeros_like(self.gt_image_3d).to(torch.float32)

            for sf_obj in self.sf_obj_list:
                lac_2_image_3d = torch.where(self.gt_image_3d == sf_obj['label'],
                                             torch.Tensor([self.mu.material(sf_obj['material'],
                                                                            'lac_2')]),
                                             lac_2_image_3d)

                if sf_obj['lqd_flag']:
                    lac_1_image_3d = torch.where(self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                                                 torch.Tensor([self.mu.material(sf_obj['lqd_param'][
                                                                                    'lqd_material'],
                                                                                'lac_2')]),
                                                 lac_2_image_3d)

            save_fits_data(lac_2_image_file, lac_2_image_3d.cpu().numpy(), self.compress_data)

            del lac_2_image_3d

            torch.cuda.empty_cache()
            self.logger.info("LAC 2 image saved as %s"%lac_2_image_file)
    # --------------------------------------------------------------------------

    def plot_baggage_statistics(self):
        """
        ------------------------------------------------------------------------


        :return:
        ------------------------------------------------------------------------
        """

        spec_num = self.xray_source_model['num_spectra']
        lqd_labels = [x['lqd_param']['lqd_label']
                      for x in self.sf_obj_list if x['lqd_flag']]

        lbl_materials = {
            **{x['label']: x['material'] for x in self.sf_obj_list},
            **{x['lqd_param']['lqd_label']: x['lqd_param']['lqd_material']
               for x in self.sf_obj_list if x['lqd_flag']}
        }

        max_label = max(list(lbl_materials.keys()))

        if spec_num==1:

            gt_img = read_fits_data(os.path.join(self.f_loc['gt_image']), 0)
            recon_img = read_fits_data(os.path.join(self.f_loc['image_dir'],
                                                    self.f_loc['img_file']%1), 0)

            r_shape, g_shape = recon_img.shape, gt_img.shape

            if r_shape != g_shape:
                r_buffer = np.ones_like(gt_img)*(-1000)

                init_pt = [g//2-r//2 for g, r in zip(g_shape, r_shape)]

                r_buffer[init_pt[0]:init_pt[0]+r_shape[0],
                         init_pt[1]:init_pt[1]+r_shape[1],
                         init_pt[2]:init_pt[2]+r_shape[2]] = recon_img.copy()
                recon_img = r_buffer.copy()

            rprops = regionprops(gt_img, recon_img)
            rprop_dict = {r['label']: r for r in rprops}

            headers = ['Label', 'Material',
                       'Mean',
                       'Min-Max', 'Vol', 'Bbox', 'Is Liquid']

            r_labels = [r['label'] for r in rprops]
            r_labels.sort()

            print_table = []

            for i in r_labels:
                print_table.append(
                    [i,
                     lbl_materials[i],
                     f"{rprop_dict[i]['mean_intensity']}",
                     f"({rprop_dict[i]['min_intensity']}, "
                     f"{rprop_dict[i]['max_intensity']})",
                     f"{rprop_dict[i]['area']}",
                     f"{rprop_dict[i]['bbox']}",
                     i in lqd_labels
                     ]
                )

            self.logger.info("\n" + tabulate(print_table,
                                             headers=headers,
                                             tablefmt='grid'))

            plt.figure()
            vectors = [recon_img[gt_img==i] for i in r_labels]
            plt.boxplot(vectors, showfliers=False)
            plt.xticks(r_labels,
                       [f"{k}, {lbl_materials[k]}"
                        for k in r_labels],
                        rotation='vertical')
            plt.xlabel("Material Labels")
            plt.ylabel("Hounsfield Units (HU)")
            plt.title("Baggage Statistics")
            plt.savefig(os.path.join(self.f_loc['simulation_dir'],
                                     'baggage_stats.png'))
            plt.tight_layout()
            plt.close()

        if spec_num==2:

            gt_img = read_fits_data(os.path.join(self.f_loc['gt_image']), 0)
            recon_img_1 = read_fits_data(os.path.join(self.f_loc['image_dir'],
                                                      self.f_loc['img_file']%1), 0)
            recon_img_2 = read_fits_data(os.path.join(self.f_loc['image_dir'],
                                                      self.f_loc['img_file']%2), 0)

            r_shape, g_shape = recon_img_1.shape, gt_img.shape

            if r_shape != g_shape:
                r_buffer = np.ones_like(gt_img)*(-1000)

                init_pt = [g//2-r//2 for g, r in zip(g_shape, r_shape)]

                r_buffer[init_pt[0]:init_pt[0]+r_shape[0],
                         init_pt[1]:init_pt[1]+r_shape[1],
                         init_pt[2]:init_pt[2]+r_shape[2]] = recon_img_1.copy()
                recon_img_1 = r_buffer.copy()

                r_buffer = np.ones_like(gt_img)*(-1000)

                init_pt = [g//2-r//2 for g, r in zip(g_shape, r_shape)]

                r_buffer[init_pt[0]:init_pt[0]+r_shape[0],
                         init_pt[1]:init_pt[1]+r_shape[1],
                         init_pt[2]:init_pt[2]+r_shape[2]] = recon_img_2.copy()
                recon_img_2 = r_buffer.copy()


            rprops1 = regionprops(gt_img, recon_img_1)
            rprops2 = regionprops(gt_img, recon_img_2)

            rprop1_dict = {r['label']: r for r in rprops1}
            rprop2_dict = {r['label']: r for r in rprops2}

            headers = ['Label', 'Material',
                       'Mean',
                       'Min-Max', 'Vol', 'Bbox', 'Is Liquid']

            r_labels = [r['label'] for r in rprops1]
            r_labels.sort()

            print_table = []

            lqd_labels = [x['lqd_param']['lqd_label']
                          for x in self.sf_obj_list if x['lqd_flag']]

            for i in r_labels:
                print_table.append(
                    [i,
                     lbl_materials[i],
                     f"{rprop1_dict[i]['mean_intensity']}, "
                     f"{rprop2_dict[i]['mean_intensity']}",
                     f"({rprop1_dict[i]['min_intensity']},{rprop1_dict[i]['max_intensity']}),"
                     f"({rprop2_dict[i]['min_intensity']},{rprop2_dict[i]['max_intensity']})",
                     f"{rprop1_dict[i]['area']}",
                     f"{rprop1_dict[i]['bbox']}",
                     i in lqd_labels
                     ]
                )

            self.logger.info("\n" + tabulate(print_table,
                                             headers=headers,
                                             tablefmt='grid'))

            plt.figure()
            vectors = [recon_img_1[gt_img==i] for i in r_labels]
            plt.boxplot(vectors, showfliers=False)
            plt.xticks(r_labels,
                       [f"{k}, {lbl_materials[k]}"
                        for k in r_labels],
                       rotation='vertical')
            plt.xlabel("Material Labels")
            plt.ylabel("Hounsfield Units (HU)")
            plt.title("Baggage Statistics (S1)")
            plt.savefig(os.path.join(self.f_loc['simulation_dir'],
                                     'baggage_stats_1.png'))
            plt.close()

            plt.figure()
            vectors = [recon_img_2[gt_img==i] for i in range(1, max_label+1)]
            plt.boxplot(vectors, showfliers=False)
            plt.xticks(r_labels,
                       [f"{k}, {lbl_materials[k]}"
                        for k in r_labels],
                       rotation='vertical')
            plt.xlabel("Material Labels")
            plt.ylabel("Hounsfield Units (HU)")
            plt.title("Baggage Statistics (S2)")
            plt.savefig(os.path.join(self.f_loc['simulation_dir'],
                                     'baggage_stats_2.png'))
            plt.close()
    # --------------------------------------------------------------------------


# ==============================================================================
# Class Ends
# ==============================================================================