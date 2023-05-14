#!/usr/bin/env python

# ------------------------------------------------------------------------------
"""scanner_template.py: Module for generating geometric forward model templates
                       for different CT scanner geometries with user specified
                       geometric specifications. The template sets up functions
                       for forward projection to create X-ray sinogram data as
                       well as functions for reconstructing from data collected
                       by the specified scanner geometries.
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


"""
--------------------------------------------------------------------------------
Module description:

This module contains support for generating forward models for different scanner 
geometries. The class ScannerTemplate() creates forward model objects for 
ConeBeam/FanBeam/ParallelBeam CT scanners with Spiral/Circular scanner paths.
The geometric and reconstruction parameters are user specified to cater to the 
setups and structures of physical scanners. The class creates an ASTRA object [1] 
corresponding to the scanner geometry and sets up methods:  

(i) to forward project baggage volumes to create scanner projections 
(ii) to reconstruct volumetric data from CT projections 

1. W. van Aarle, W. J. Palenstijn, J. Cant, E. Janssens, F. Bleichrodt, A. 
Dabravolski, J. De Beenhouwer, K. J. Batenburg, and J. Sijbers, 'Fast and 
Flexible X-ray Tomography Using the ASTRA Toolbox', Optics Express, 24(22), 
25129-25147, (2016), http://dx.doi.org/10.1364/OE.24.025129

2. W. van Aarle, W. J. Palenstijn, J. De Beenhouwer, T. Altantzis, S. Bals, 
K. J. Batenburg, and J. Sijbers, 'The ASTRA Toolbox: A platform for advanced 
algorithm development in electron tomography', Ultramicroscopy, 157, 35-47, 
(2015), http://dx.doi.org/10.1016/j.ultramic.2015.05.002


Usage:

ScannerTemplate() is initialized with the following args:

- geom - geometric setup of the scanner: 'cone', 'parallel', 'fan'
- scan - the scanner motion: 'helical', 'circular'
- recon - reconstruction algorithm: 'fbp', 'sirt'
- machine_dict - dictionary of parameters for machine geometry
- recon_dict   - dictionary of parameters for reconstruction setup

(The parameters for both these dictionaries and how to calculate them for a 
given scanner are explained using the example of default dictionaries below.)

If not specified, the parameters for the scanner are determined by the default 
values in the dictionaries default_machine_dict and default_recon_params which 
adopt these values from the Spiral Cone Beam CT scanner Siemens Sensation 64 [3].

3. Amin, AT Mohd, and AA Abd Rahni. "Modelling the Siemens SOMATOM Sensation 64 
   Multi-Slice CT (MSCT) Scanner." Journal of Physics: Conference Series. Vol. 
   851. No. 1. IOP Publishing, 2017.

Example of usage of the ScannerTemplate() class is given below:

> ---------------------------------------------------------------------------->
import numpy as np
from lib.misc.util import *
from lib.forward_model.scanner_template import *

siemens_sensation_32 = ScannerTemplate(
            geom='cone',
            scan='spiral',
            recon='fbp')

vol_img = np.zeros(664,664,350)             # volumetric image as example
vol_img[200:250, 250:280, 100:300] = 0.1    # spawning objects in the volume
vol_img[100:120, 400:600, 50:300] = 0.05
vol_img[400:500, 50:100, 200:250] = 0.4

siemens_sensation_32.set_recon_geometry(7, 1)
ct_projn = siemens_sensation_32.run_fwd_projector(vol_img)
recon_img = siemens_sensation_32.reconstruct_data(ct_projn)

slideshow(vol_img, vmin=0.0, vmax=1.0)
slideshow(recon_img, vmin=0.0, vmax=1.0)
> ---------------------------------------------------------------------------->

Once initialized, the class object for the specified scanner can be used to 
create X-ray projections and reconstructions. Volumetric data can be projected 
using the method self.run_fwd_projector() to create X-ray projections while 3D 
images can be reconstructed from the scanner projection data using the method 
self.reconstruct_data(). (For both forward projection and reconstruction, the 
dimensions of the input images/projections must match the geometric 
specifications used for initializing the scanner.)

This template is suited for use directly in code as well as can be accessed 
through the DebiSim GUI. 
--------------------------------------------------------------------------------
"""

import scipy.io as io

from lib.reconstructor.freect import *
import torch, torch.nn as nn
from numpy import *
from tabulate import *

import lib.forward_model.template_siemens_definition_as as siemens_definition_as
import lib.forward_model.template_siemens_force as siemens_force
import lib.forward_model.template_siemens_sensation_32 as siemens_sensation_32
import lib.forward_model.template_default_parallelbeam_scanner as default_scanner_parallel
import lib.forward_model.template_default_two_view_conebeam as default_two_view_conebeam
import lib.forward_model.template_default_two_view_parallelbeam as default_two_view_parallel

# =============================================================================
# Dictionary for Default Machine Geometry - the default machine is a Spiral
# CBCT scanner

default_machine_geometry = siemens_sensation_32.machine_geometry
default_recon_params     = siemens_sensation_32.recon_params
# =============================================================================

scanner_list = os.listdir(SCANNER_DIR)
scanner_files = [x.replace('.scanner','') for x in scanner_list if x[-8:]=='.scanner']

predefined_scanners = ['siemens_sensation_32', 'siemens_force', 'siemens_definition_as']
saved_scanners = [x.replace('.pyc','') for x in scanner_list if x[-4:]=='.pyc']

scanner_list = dict()

scanner_list['siemens_sensation_32']  = {'machine_geometry': siemens_sensation_32.machine_geometry,
                                         'recon_params': siemens_sensation_32.recon_params}
scanner_list['siemens_definition_as'] = {'machine_geometry': siemens_definition_as.machine_geometry,
                                         'recon_params': siemens_definition_as.recon_params}
scanner_list['siemens_force']         = {'machine_geometry': siemens_force.machine_geometry,
                                         'recon_params': siemens_force.recon_params}

for curr_scanner in saved_scanners:
    s_file = os.path.join(SCANNER_DIR, curr_scanner+'.pyc')
    with open(s_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        scanner_list[curr_scanner] = data.copy()


class ScannerTemplate(object):

    """
    ---------------------------------------------------------------------------
    Class for generating geometric forward model templates for different CT 
    scanner geometries with user specified geometric specifications. The 
    template sets up functions for forward projection to create X-ray sinogram 
    data as well as functions for reconstructing from data collected by the 
    specified scanner geometries.
    
    Methods:
    __init__()              - Constructor
    set_recon_geometry      - Creates geometric model based on specifications 
                              provided during initialization
    run_fwd_projector       - Perform forward projection on input 3D data to
                              generate X-ray projection data  
    reconstruct_data        - Reconstructs 3D volumetric image from input 
                              projection data
    
    Attributes:
    recon_geometry          - dictionary for reconstruction geometry
    machine_geometry        - dictionary for machine geometry
    recon_params            - dictionary for user-specified reconst. parameters
    proj_geom               - ASTRA object created by the class instance
    vecs                    - array of vectors specifying position of each CT 
                              detector in the frame of reference of the 
                              isocenter.
    ---------------------------------------------------------------------------
    """

    machine_geometry = None # Machine geometry
    recon_geometry   = None # Reconstruction geometry

    def __init__(self,
                 geometry='cone',
                 scan='spiral',
                 machine_dict=None,
                 recon='fbp',
                 recon_dict=None,
                 pscale=1.0,
                 logfile=None):
        """
        -----------------------------------------------------------------------
        Constructor for ScannerTemplate(). Note that the dictionary 
        specifications in machine_dict, recon_dict must match the options 
        provided during initialization - see above example for description of 
        these dictionary parameters. The options for scanner include 'cone'- 
        Cone Beam CT, 'parallel'- Parallel Beam CT and 'fan' - Fan Beam CT. 
        (Note that the current version only supports cone Beam CT geometries). 
        Models can be constructed for both circular and helical motion paths for
        data acquisition (through argument scan). Image Reconstruction using 
        either wFBP (through use of FreeCT_wFBP) or SIRT3D (ASTRA).
        
        :param geometry:        type of scanner geometry
                                options: {'cone', 'parallel', 'fan'}
                                (No support for 'fan' or 'parallel' in this 
                                version)
        :param scan:            Scanner data acquisition path
                                option: {'spiral', 'circular'}
        :param machine_dict:    dictionary for machine geometry
        :param recon:           Type of reconstruction algorithm used.
                                option: {'fbp', 'sirt'}
        :param recon_dict:      dictionary for reconstruction parameters.
        -----------------------------------------------------------------------
        """
        
        logfile = os.path.join(EXAMPLE_DIR, 'scanner.log') if logfile is None \
                  else logfile
        
        self.logger = get_logger('SCANNER', logfile)

        self.geom = geometry
        self.recon = recon
        self.scan = scan

        self.pscale = pscale
        setup_attr = ['_set_cone_vec_equi_space_geometry',
                      '_set_cone_geometry']
        recon_attr = ['_run_cone_vec_equi_space_fp',
                      '_run_sirt',
                      '_run_sirt_equi_space',
                      '_run_recon_fct_wfbp']

        # Set up the specifications for the geometric model of the scanner ----
        if machine_dict is None:
            self.machine_geometry = default_machine_geometry.copy()
        else:
            self.machine_geometry = machine_dict.copy()
        # ---------------------------------------------------------------------

        # Set reconstruction attributes depending on scanner geometry ---------
        if geometry=='cone':

            # Specify Scanner Motion ------------------------------------------
            if scan=='spiral':
                # Creates an ASTRA geometric model for spiral CBCT setup
                self.set_recon_geometry = self._set_cone_vec_equi_space_geometry
                self.run_fwd_projector = self._run_cone_vec_equi_space_fp

            elif scan=='circular':
                # Creates an ASTRA geometric model for Circular Cone Beam setup
                self.set_recon_geometry = self._set_cone_geometry
                self.run_fwd_projector = self._run_cone_equi_space_fp
                
            else:
                self.logger.info("Invalid input for argument 'scan'!")
                self.logger.info("Select from the following two options: "
                                 "{'circular', 'scan'}")
                raise IOError
            # ----------------------------------------------------------------

            # Specify Reconstruction Algorithm -------------------------------
            if recon == 'fbp':
                # reconstruction done using the weighted FBP algorithm
                self.reconstruct_data = self._run_recon_fct_wfbp

            elif recon == 'sirt':
                # reconstruction done using 3D SIRT algorithm
                self.reconstruct_data = self._run_sirt_equi_space

            else:
                self.logger.info("Invalid input for argument 'recon'!")
                self.logger.info("Select from the following two options:"
                                 " {'fbp', 'sirt'}")
                raise IOError
            # ----------------------------------------------------------------

        # For future versions -------------------------------------------------
        elif geometry=='fan':

            # TODO attributes for fan beam CT geometry
            raise NotImplementedError("No Support for Fan Beam Geometry implemented yet ...")

        elif geometry=='parallel':
            self.set_recon_geometry = self._set_parallel_beam_geometry
            self.run_fwd_projector  = self._run_parallel_beam_fp

            if recon=='fbp':
                self.reconstruct_data = self._run_fbp_parallel_beam
            elif recon=='sirt':
                self.reconstruct_data = self._run_sirt_parallel_beam
            elif recon=='mbir':
                self.reconstruct_data = self._run_mbir_parallel_beam
            else:
                self.logger.info("Invalid input for argument 'recon'!")
                self.logger.info("Select from the following two options: {'fbp', 'sirt', 'mbir'}")
                raise IOError

    # ---------------------------------------------------------------------

        if recon_dict is not None:
            self.recon_params = recon_dict.copy()
        else:
            self.recon_params = default_recon_params.copy()

        if self.machine_geometry['scanner_name'] not in scanner_files \
                and geometry=='spiral':
            self.create_new_scanner_file()
    # -------------------------------------------------------------------------

    def update_recon_algo(self, recon):
        """
        -----------------------------------------------------------------------
        changes the reconstruction algorithms to be used ot specified algorithm.
        The update method can be called while running the DEBISim Pipeline.

        :param recon:   reconstruction algo: {'fbp' | 'sirt'}
        :return:
        -----------------------------------------------------------------------
        """

        # change Reconstruction Algorithm -------------------------------
        if recon == 'fbp':
            # reconstruction done using the weighted FBP algorithm
            self.reconstruct_data = self._run_recon_fct_wfbp

        elif recon == 'sirt':
            # reconstruction done using 3D SIRT algorithm
            self.reconstruct_data = self._run_sirt_equi_space

        else:
            self.logger.info("Invalid input for argument 'recon'!")
            self.logger.info("Select from the following two options: {'fbp', 'sirt'}")
            raise IOError
        # ----------------------------------------------------------------
    # -------------------------------------------------------------------------

    def create_new_scanner_file(self, scanner_file_args=None):
        """
        -----------------------------------------------------------------------
        Creates a new .scanner file for a new scanner geometry described for
        running a FreeCT reconstruction - the scanner file arguments can be
        provided manually or can be calculated from the scanner geometry

        :param scanner_file_args    dictionary containing field-value pairs of
                                    the .scanner file
        :return:
        -----------------------------------------------------------------------
        """
        scanner_file_path = SCANNER_DIR

        if scanner_file_args is None:

            self.logger.info("Calculating scanner file arguments ...")

            scanner_file_args = dict()

            src_det = self.machine_geometry['source_origin'] + self.machine_geometry['origin_det']

            scanner_file_args['RSrcToIso']      = self.machine_geometry['source_origin']
            scanner_file_args['RSrcToDet']      = self.machine_geometry['origin_det']
            scanner_file_args['AnodeAngle']     = self.machine_geometry['anode_angle']
            scanner_file_args['FanAngleInc']    = 0.98*self.machine_geometry['sens_spacing_x']/\
                                                  src_det

            theta = 2*arcsin(
                (self.machine_geometry['det_row_count']*
                    self.machine_geometry['sens_spacing_y']/(2*src_det)
                 )
            )
            scanner_file_args['ThetaCone']      = theta

            scanner_file_args['CentralChannel'] = self.machine_geometry['det_col_count']/2.- 0.50
            scanner_file_args['NProjTurn']      = self.recon_params['n_views']
            scanner_file_args['NChannels']      = self.machine_geometry['det_col_count']
            scanner_file_args['ReverseRowInterleave'] = 0
            scanner_file_args['ReverseChanInterleave'] = 0

        scanner_fname = os.path.join(scanner_file_path,
                                     '%s.scanner'%self.machine_geometry['scanner_name'])

        f = open(scanner_fname, 'w')
        for k, v in scanner_file_args.iteritems():
            f.write("{}:\t{}\n".format(k, v))
        f.close()
        self.logger.info("Generated %s" % scanner_fname)
    # -------------------------------------------------------------------------

    def _set_cone_vec_equi_space_geometry(self):
        """
        -----------------------------------------------------------------------
        Set up cone vector geometry that models all 4 sections as one detector.
        The detector is modelled as one flat panel by using equispace weights.

        :param n_rot:       Number of turns made by CBCT scanner
                            (Must cover the z-length of the input volumetric 
                            image - else the image will be cropped - a good 
                            estimate is vol_image_z_length/pitch)
        :param view_stride: number of views to skip (can be used to decrease 
                            projection resolution)
        :return:
        -----------------------------------------------------------------------
        """

        n_views         = self.recon_params['n_views']
        pitch           = self.recon_params['pitch']
        image_dims      = self.recon_params['image_dims']

        n_rot = ceil(image_dims[2]/pitch)
        n_rot = int(self.pscale*n_rot)

        g = {}
        g['views_per_rot']  = n_views
        g['n_views']        = int(g['views_per_rot'] * n_rot)

        theta               = linspace(-360 * n_rot/2.,
                                       360 * n_rot/2.,
                                       g['n_views'],
                                       endpoint=False)
        g['angles'] = deg2rad(mod(theta, 360))

        adj_ratio = 1.

        g['pitch'] = pitch/adj_ratio
        g['src_angles'] = g['angles']
        g['det_angles'] = g['angles'] + pi
        g['translation'] = linspace(-n_rot * pitch / 2.,
                                     n_rot * pitch / 2.,
                                    g['n_views'],
                                    endpoint=False)
        g['n_turns'] = n_rot
        self.recon_geometry = g.copy()

        # Modify the projection geometry to accommodate helical cone-beam
        vecs = zeros((self.recon_geometry['n_views'], 12), dtype=float32)

        # Source center
        vecs[:, 0] =  sin(self.recon_geometry['src_angles']) * self.machine_geometry['source_origin']
        vecs[:, 1] = -cos(self.recon_geometry['src_angles']) * self.machine_geometry['source_origin']
        vecs[:, 2] = self.recon_geometry['translation']

        # Detector center
        vecs[:, 3] =  sin(self.recon_geometry['det_angles']) * self.machine_geometry['origin_det']
        vecs[:, 4] = -cos(self.recon_geometry['det_angles']) * self.machine_geometry['origin_det']
        vecs[:, 5] = self.recon_geometry['translation']

        # From detector pixel (r0,c0) to (0,1), col
        vecs[:, 6] = -cos(self.recon_geometry['det_angles']) * self.machine_geometry['sens_spacing_x']
        vecs[:, 7] = -sin(self.recon_geometry['det_angles']) * self.machine_geometry['sens_spacing_x']
        vecs[:, 8] = 0

        # From detector pixel (r0,c0) to (1,0), row
        vecs[:, 9] = 0
        vecs[:, 10] = 0
        vecs[:, 11] = -self.machine_geometry['sens_spacing_y']*self.pscale

        self.proj_geom = astra.create_proj_geom('cone_vec',
                                                self.machine_geometry['det_row_count'],
                                                self.machine_geometry['n_sens_x'],
                                                vecs)

        self.vecs = vecs
    # -------------------------------------------------------------------------

    def _set_cone_geometry(self, n_rot=1, view_stride=1):
        """
        -----------------------------------------------------------------------
        Setup geometry necessary for circular Cone Beam CT geometry.

        :param n_rot:       Number of turns made by CBCT scanner
                            (Must cover the z-length of the input volumetric
                            image - else the image will be cropped - a good
                            estimate is vol_image_z_length/pitch)
        :param view_stride: number of views to skip (can be used to decrease
                            projection resolution)
        :return
        -----------------------------------------------------------------------
        """

        n_views = self.recon_params['n_views']
        pitch = self.recon_params['pitch']
        # fan_angle_0   = self.recon_params['fan_angle_0']
        # fan_angle_inc = self.recon_params['fan_angle_inc']

        # Reconstruction-specific geometry parameter
        self.machine_geometry['pitch'] = pitch

        g = {}

        # gives the angle for each view of each projection
        g['views_per_rot'] = n_views / view_stride
        g['n_views'] = int(g['views_per_rot'] * n_rot)
        theta = linspace(0.0, 360.0 * n_rot,
                         g['n_views'],
                         endpoint=False)
        g['angles'] = deg2rad(mod(theta, 360))

        # no. of slices = pitch x no of rotations / slice thickness
        g['n_slices'] = int(
            self.machine_geometry['pitch'] * n_rot
            / self.recon_params['slice_thickness'])
        # g['src_angles'] = repeat(g['angles'],
        #                          self.machine_geometry['n_gums'])

        # Offsets for section centers
        # det_angles = repeat(g['angles'],
        #                     self.machine_geometry['n_gums'])
        #
        # det_angles = det_angles.reshape(g['n_views'],
        #                                 self.machine_geometry['n_gums']) \
        #                                 + self.machine_geometry['angle_offsets'].reshape(1,
        #                                                                                  self.machine_geometry['n_gums'])
        # g['det_angles'] = det_angles.flatten()
        g['src_angles'] = g['angles']
        g['det_angles'] = g['angles']
        # translation = linspace(0.0, self.machine_geometry['pitch'] * n_rot, g['n_views'],
        #                        endpoint=False)
        # g['translation'] = translation

        # from flat panel detector to curved panel detector

        # fan_angle = fan_angle_0 + fan_angle_inc * np.arange(1024)
        # equi_space_weights = (np.cos(np.deg2rad(self.machine_geometry['columnAngle'])) /
        #                       np.cos(np.deg2rad(fan_angle)))
        # g['equi_space_weights'] = equi_space_weights
        # w_1 = (np.arange(
        #     self.machine_geometry['gums_row_count'] * (g['views_per_rot'] + 1))) * 1.0 / g[
        #           'views_per_rot']
        # w_1 = np.hstack((w_1, w_1[::-1][self.machine_geometry['gums_row_count']:]))
        # w_1 = w_1.reshape(2 * g['views_per_rot'] + 1,
        #                   self.machine_geometry['gums_row_count']).T
        # g['w_1'] = w_1
        # w_f = (np.arange(g['views_per_rot'] + 1)) * 1.0 / g['views_per_rot']
        # w_b = w_f[::-1]
        # g['w_b'] = w_b

        g['recon_slice_no'] = 63
        g['view_stride'] = view_stride

        # create attribute fro mrecon geometry
        self.recon_geometry = g

        # Astra GPU projector with circular cone beam
        self.proj_geom = astra.create_proj_geom(
            'cone',  # type of geometry
            self.machine_geometry['sens_spacing_x'],  # horizontal spacing between detector cells
            self.machine_geometry['sens_spacing_y'],  # vertical spacing between detector cells
            self.machine_geometry['det_row_count'],   # number of detector rows
            self.machine_geometry['n_sens_x'],        # number of detector columns - no. of channels in sino
            g['angles'],                              # total number of angles for entire images
            self.machine_geometry['source_origin'],   # distance between source and origin
            self.machine_geometry['origin_det'])      # distance between detector and origin

        self.logger.info("Done processing reconstruction geometry...")

    # -------------------------------------------------------------------------

    def _set_parallel_beam_geometry(self):
        """
        -----------------------------------------------------------------------
        Set up parallel beam geometry for generating set of 2D projections for
        a 3D volume

        :return:
        -----------------------------------------------------------------------
        """

        g = {}
        g['n_views']        = self.recon_params['n_views']
        g['image_dims']     = self.recon_params['image_dims']
        g['view_range']     = self.recon_params['view_range']
        g['fov']            = self.machine_geometry['gantry_diameter']
        view_range          = self.recon_params['view_range']

        theta               = linspace(- view_range/2,
                                         view_range/2,
                                       g['n_views'],
                                       endpoint=False)[::-1]
        g['angles'] = deg2rad(mod(theta, 360))

        self.recon_geometry = g.copy()
        self.proj_geom = astra.create_proj_geom('parallel3d',
                                                self.machine_geometry['det_spacing_y'],
                                                self.machine_geometry['det_spacing_x'],
                                                self.machine_geometry['det_row_count'],
                                                self.machine_geometry['det_col_count'],
                                                self.recon_geometry['angles']
                                                )

        ramlak = np.squeeze(io.loadmat(self.recon_params['ramlak'])['myFilter'])
        self.ramlak = ramlak.astype(complex)
    # -------------------------------------------------------------------------

    def _run_cone_vec_equi_space_fp(self, vol_data, verbose=False):
        """
        -----------------------------------------------------------------------
        Forward the input 3D volume to create X-ray projection data for the
        scanner. (Spiral CBCT)

        :param vol_data:        3D volumetric image to be projected
                                (dim.: gantry_diameter x gantry_diametry x
                                 no_of_slices)
        :param verbose:         Set to True to print results on the terminal
        :return:                projection data
                                (dim.: n_rows x n_cols x (n_views*n_rot))
        -----------------------------------------------------------------------
        """
        # Run forward projection on a 3D volume

        # Check the cone_vec geometry
        assert self.proj_geom['type'] == 'cone_vec'

        # Create volume geometry, astra requires a bit exchange of axes
        vol_geom = astra.create_vol_geom(*vol_data.shape)
        if verbose:  self.logger.info(f"Volume data has shape {vol_data.shape}")

        # Do forward projection based on cone vec geometry
        if verbose:  self.logger.info("Starting cone_vec forward projection...")
        t0 = time.time()
        proj_id, proj_data = astra.create_sino3d_gpu(
            moveaxis(vol_data, -1, 0),
            self.proj_geom,
            vol_geom
        )
        if verbose:  self.logger.info("Done, took %.3fs..." % (time.time() - t0))
        if verbose:  self.logger.info(f"Projection has shape {proj_data.shape}")

        # Clean up
        astra.data3d.delete(proj_id)

        return proj_data
    # -------------------------------------------------------------------------

    def _run_parallel_beam_fp(self, vol_data, verbose=False):
        """
        -----------------------------------------------------------------------
        Forward the input 3D volume to create X-ray projection data for the
        scanner. (3D Parallel Beam CT)

        :param vol_data:        3D volumetric image to be projected
                                (dim.: gantry_diameter x gantry_diametry x
                                 no_of_slices)
        :param verbose:         Set to True to print results on the terminal
        :return:                projection data
                                (dim.: n_rows x n_cols x (n_views*n_rot))
        -----------------------------------------------------------------------
        """
        # Run forward projection on a 3D volume

        # Check the cone_vec geometry
        assert self.proj_geom['type'] == 'parallel3d'

        vol_geom = astra.create_vol_geom(*vol_data.shape)
        vol_data = transpose(vol_data, (2,0,1))

        # Create volume geometry, astra requires a bit exchange of axes

        if verbose:  self.logger.info(f"Volume data has shape {vol_data.shape}")

        dims = self.recon_geometry['image_dims'][2], \
               self.recon_geometry['image_dims'][0], \
               self.recon_geometry['image_dims'][1]
        proj_id, proj_data = astra.create_sino3d_gpu(
            vol_data, self.proj_geom, vol_geom)

        # Do forward projection based on cone vec geometry
        if verbose:  self.logger.info("Starting forward projection...")
        t0 = time.time()

        if verbose:  self.logger.info("Done, took %.3fs..." % (time.time() - t0))
        if verbose:  self.logger.info(f"Projection has shape {proj_data.shape}")

        astra.data3d.delete(proj_id)

        return proj_data
    # -------------------------------------------------------------------------

    def _run_cone_equi_space_fp(self, vol_data, verbose=False):
        """
        -----------------------------------------------------------------------
        Forward the input 3D volume to create X-ray projection data for the
        scanner. (Circular CBCT)

        :param vol_data:        3D volumetric image to be projected
                                (dim.: gantry_diameter x gantry_diametry x
                                 no_of_slices)
        :param verbose:         Set to True to print results on the terminal
        :return:                projection data
                                (dim.: n_rows x n_cols x (n_views*n_rot))
        -----------------------------------------------------------------------
        """

        # Run forward projection on a 3D volume

        # Check the cone_vec geometry
        assert self.proj_geom['type'] == 'cone'

        # Create volume geometry, astra requires a bit exchange of axes
        vol_geom = astra.create_vol_geom(*vol_data.shape)
        if verbose:  self.logger.info(f"Volume data has shape {vol_data.shape}")

        # Do forward projection based on cone vec geometry
        if verbose:  self.logger.info("Starting cone_vec forward projection...")
        t0 = time.time()
        proj_id, proj_data = astra.create_sino3d_gpu(
            moveaxis(vol_data, -1, 0), self.proj_geom, vol_geom)
        if verbose:  self.logger.info("Done, took %.3fs..." % (time.time() - t0))
        if verbose:  self.logger.info(f"Projection has shape {proj_data.shape}")

        # Clean up
        astra.data3d.delete(proj_id)

        return proj_data
    # -------------------------------------------------------------------------

    def _run_sirt(self, sino, n_iter=100, sino_mask=None, verbose=False):
        """
        -----------------------------------------------------------------------
        Run SIRT3D algorithm to reconstruct volume from  projection data for
        CBCT. For details of the algorithm see ASTRA's SIRT3D_CUDA algorithm.

        :param sino:        input projection data
        :param n_iter:      number of iterations for SIRT
        :param sino_mask:   portion of the sinogram to skip
        :return: reconstructed 3d image
        -----------------------------------------------------------------------
        """

        # Voxel size is 1x1x1.78mm
        vol_geom = astra.create_vol_geom(512, 512, self.recon_geometry['n_slices'])
        if verbose: self.logger.info(f"Volume geometry is {vol_geom}")

        # print self.proj_geom
        # print sino.shape

        sino_id = astra.data3d.create('-sino', self.proj_geom_2, sino)
        rec_id = astra.data3d.create('-vol', vol_geom)

        # create configuration
        cfg = astra.astra_dict('SIRT3D_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['option'] = {'MinConstraint': 0}

        # Create sinogram mask if needed
        if sino_mask is not None:
            sino_mask_id = astra.data3d.create('-sino', self.proj_geom,
                                               sino_mask)
            cfg['option']['SinogramMaskId'] = sino_mask_id
        if verbose: self.logger.info("SIRT configuration:", cfg)
        if verbose: self.logger.info("Running SIRT with %d iterations" % n_iter)

        # Create and run the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        if verbose: self.logger.info("Starting reconstruction...")
        t0 = time.time()
        astra.algorithm.run(alg_id, n_iter)
        if verbose: self.logger.info("Reconstruction took %.2fs" % (time.time() - t0))

        # Get the result
        rec = astra.data3d.get(rec_id)
        if verbose: self.logger.info("Result info:", rec.max(), rec.min(), rec.shape)

        # Clean up. Note that GPU memory is tied up in the algorithm object,
        # and main RAM in the data objects.
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(sino_id)

        return rec
    # -------------------------------------------------------------------------

    def _run_sirt_equi_space(self,
                             sino,
                             n_iter=100,
                             sino_mask=None,
                             full_range=False, flags='',
                             append_air_turns=False,
                             clip_for_annotation=True):
        """
        -----------------------------------------------------------------------
        Run SIRT3D algorithm to reconstruct volume from  projection data for
        CBCT. For details of the algorithm see ASTRA's SIRT3D_CUDA algorithm.
        (Spiral CBCT)

        :param sino:        Projection data. Shape: (n_views, n_rows, n_cols)
        :param n_iter:      Number of iterations for SIRT
        :param sino_mask:   portion of the sinogram to skip
        :return:
        -----------------------------------------------------------------------
        """
        self.logger.info("\n\nRunning flat panel SIRT...")
        # Convert to equispace

        # Voxel size is 1x1x1.78mm

        vol_geom = astra.create_vol_geom(self.recon_params['image_dims'][0],
                                         self.recon_params['image_dims'][1],
                                         self.recon_params['image_dims'][2])
        self.logger.info(f"Volume geometry is {vol_geom}")

        # print self.proj_geom
        sino = moveaxis(sino,0,-1)
        self.logger.info(f"Input conogram shape is {sino.shape}")

        sino_id = astra.data3d.create('-sino', self.proj_geom, sino)
        rec_id = astra.data3d.create('-vol', vol_geom)

        # create configuration
        cfg = astra.astra_dict('SIRT3D_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['option'] = {'MinConstraint': 0}

        # Create sinogram mask if needed
        if sino_mask is not None:
            sino_mask_id = astra.data3d.create('-sino', self.proj_geom,
                                               sino_mask)
            cfg['option']['SinogramMaskId'] = sino_mask_id
        self.logger.info("SIRT configuration:", cfg)
        self.logger.info("Running SIRT with %d iterations" % n_iter)

        # Create and run the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        self.logger.info("Starting reconstruction...")
        t0 = time.time()
        astra.algorithm.run(alg_id, n_iter)
        self.logger.info("Reconstruction took %.2fs" % (time.time() - t0))

        # Get the result
        rec = astra.data3d.get(rec_id)
        self.logger.info("Result info:", rec.max(), rec.min(), rec.shape)

        # Clean up. Note that GPU memory is tied up in the algorithm object,
        # and main RAM in the data objects.
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(sino_id)

        return rec
    # -------------------------------------------------------------------------

    def _run_fbp_parallel_beam(self,
                               sino,
                               verbose=False,
                               full_range=True,
                               append_air_turns=True
                               ):
        """
        -----------------------------------------------------------------------
        Run FBP algorithm to reconstruct volume from  projection data for parallel
        beam geometry.

        :param sino:        Projection data. Shape: (n_views, n_rows, n_cols)
        :param n_iter:      Number of iterations for SIRT
        :param sino_mask:   portion of the sinogram to skip
        :return:
        -----------------------------------------------------------------------
        """
        self.logger.info("\n\nRunning parallel beam FBP...")

        self.logger.info(f"Input sinogram shape is {sino.shape}")
        t0 = time.time()

        if len(sino.shape)==2:
            if verbose: self.logger.info("Starting reconstruction...")
            self.proj_geom2d = astra.create_proj_geom('parallel',
                                                    self.machine_geometry['det_spacing_y'],
                                                    self.machine_geometry['det_col_count'],
                                                    self.recon_geometry['angles']
                                                    )

            im_geom = astra.create_vol_geom(self.recon_geometry['image_dims'][0],
                                            self.recon_geometry['image_dims'][1])
            self.proj2d_id = astra.create_projector('cuda', self.proj_geom2d, im_geom)
            self.w_2d = astra.OpTomo(self.proj2d_id)

            rec = self.w_2d.reconstruct('FBP_CUDA', sino)
            if verbose: self.logger.info("Reconstruction took %.2fs" % (time.time() - t0))

            del self.proj2d_id, self.w_2d
            torch.cuda.empty_cache()

        elif len(sino.shape)==3:

            sino = transpose(sino, (1, 2, 0))
            assert sino.shape == (self.machine_geometry['det_row_count'],
                                  self.recon_params['n_views'],
                                  self.machine_geometry['det_col_count'])

            vol_geom = astra.create_vol_geom(*self.recon_geometry['image_dims'])

            if verbose: self.logger.info("Starting reconstruction...")

            sino = moveaxis(sino, -1, 0)
            sino_freq = np.fft.fft(sino, axis=0)  # FFT
            filtered_freq = self.ramlak[:,newaxis,:] * sino_freq  # use ram-lak filter
            filtered_sino = np.fft.ifft(filtered_freq, axis=0)  # IFFT
            filtered_sino = np.real(filtered_sino[0:1024, :])

            sino = moveaxis(filtered_sino, 0, -1)

            sino_id = astra.data3d.create('-sino', self.proj_geom, sino)
            rec_id = astra.data3d.create('-vol', vol_geom)

            # create configuration
            cfg = astra.astra_dict('BP3D_CUDA')
            # cfg = astra.astra_dict('CGLS3D_CUDA')
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sino_id

            # Create and run the algorithm object from the configuration structure
            alg_id = astra.algorithm.create(cfg)
            self.logger.info("Starting reconstruction...")
            t0 = time.time()
            astra.algorithm.run(alg_id)

            # Get the result
            rec = astra.data3d.get(rec_id)

            # Clean up. Note that GPU memory is tied up in the algorithm object,
            # and main RAM in the data objects.
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(rec_id)
            astra.data3d.delete(sino_id)

            rec = rec*1e-2

            if verbose: self.logger.info("Reconstruction took %.2fs" % (time.time() - t0))
            torch.cuda.empty_cache()

        else:
            self.logger.info("No Idea what you have fed as input! Sino Shape: ", sino.shape)
            raise IOError

        self.logger.info(f"Result info:, {rec.max()}, {rec.min()}, {rec.shape}")
        self.logger.info("Reconstruction took %.2fs" % (time.time() - t0))

        return rec
    # -------------------------------------------------------------------------

    def _run_recon_fct_wfbp(self,
                            data,
                            full_range=False,
                            flags='',
                            append_air_turns=False,
                            clip_for_annotation=True):
        """
        -----------------------------------------------------------------------
        Run FreeCT wFBP algorithm on data. Takes in three turns of data, returns
        only the reconstruction for the central turn if the full_range option is
        not turned on.

        :param data:         sinogram data of the dimensions
                             n_channels x n_rows x n_views
        :param flags:       flags for running fct_wfbp
        :param full_range:  whether cover full range skip the padding turns
        :param append_air_turns: add additional turns to avoid cropped
                            reconstructions
        :param show_plots   display reconstructed image after operation
        :return:  reconstructed data, (row, column, z)
        -----------------------------------------------------------------------
        """
        self.logger.info("\n\nRunning freeCT...")

        n_channels, n_rows, n_views = data.shape

        if append_air_turns:
            self.logger.info("Appending one air turn before and one after...")
            air_turn = zeros((n_channels, n_rows,
                              self.machine_geometry['n_views_per_rot']))
            data = dstack([air_turn, data, air_turn])
            n_channels, n_rows, n_views = data.shape

        gantry_pitch = self.recon_geometry['pitch']

        n_proj_ffs = self.machine_geometry['n_views_per_rot']
        array_direction = -1  # Scanner constant
        add_projections_ffs = 143  # Scanner constant

        projection_padding = gantry_pitch * (n_proj_ffs / 2 +
                                             add_projections_ffs + 256) / n_proj_ffs

        data_begin_pos     = n_views / float(n_proj_ffs) * gantry_pitch \
                             + array_direction * projection_padding
        data_end_pos       = 1 / float( n_proj_ffs) * gantry_pitch \
                             - array_direction * projection_padding

        if full_range:
            # To avoid minor conflicts
            recon_begin_pos = floor(data_begin_pos)
            recon_end_pos = ceil(data_end_pos)
        else:
            # Determine the range of the central turn
            data_center_pos = (data_begin_pos + data_end_pos) / 2
            recon_begin_pos = data_center_pos +  gantry_pitch / 2
            recon_end_pos = data_center_pos - gantry_pitch / 2

        self.logger.info("Recon range: %.2f - %.2f" % (recon_begin_pos, recon_end_pos))

        scn_fname = os.path.join(SCANNER_FILE_NAME%self.machine_geometry['scanner_name'])
        temp_dir = FCT_TMP_DIR
        os.makedirs(temp_dir, exist_ok=True)
        prj_fname = 'prj.bin'
        prm_fname = os.path.join(temp_dir, 'tmp.prm')
        img_fname = os.path.join(temp_dir, prj_fname + '.img')
        w, h = self.recon_params['image_dims'][0], self.recon_params['image_dims'][1]

        os.makedirs(temp_dir, exist_ok=True)

        paramdict = dict(
            RawDataDir=temp_dir,
            RawDataFile=prj_fname,
            OutputDir=temp_dir,
            Nrows=self.machine_geometry['det_row_count'],
            CollSlicewidth=self.machine_geometry['sens_spacing_y']*self.pscale,
            StartPos=recon_begin_pos,
            EndPos=recon_end_pos,
            TubeStartAngle=180,
            SliceThickness= self.recon_params['slice_thickness']*self.pscale,
            PitchValue= gantry_pitch,
            AcqFOV=700,
            ReconFOV=503,
            ReconKernel=3,
            Readings=n_views,
            Xorigin=0.0,
            Yorigin=0.0,
            Zffs=0,
            Phiffs=0,
            Scanner=scn_fname,
            FileType=0,
            FileSubType=0,
            RawOffset=0,
            Nx=w,
            Ny=h,
        )

        header = ['Parameter', 'Value']

        prm_table = []
        for k, v in paramdict.items():
            prm_table.append([k, v])

        self.logger.info("\n"+tabulate(prm_table, headers=header, tablefmt='psql'))

        # Prepare data into the binary file that FreeCT understands
        fct_write_prj_file(os.path.join(temp_dir, prj_fname), data)

        # Generate the prm file
        fct_write_prm_file(prm_fname, paramdict)

        t_wait = time.time()
        while not os.path.exists(prm_fname):
            if (time.time()-t_wait)>10:
                raise FileNotFoundError("Trouble writing to %s"%prm_fname)
            pass
        # Run
        fct_run(prm_fname, flags=flags)

        # Read the reconstruction results
        img = fct_read_img_file(img_fname, w, h)
        img = moveaxis(img, 0, -1)  # Third axis is z

        # Cleaning up
        shutil.rmtree(temp_dir)

        if clip_for_annotation:
            z_mid = img.shape[2]//2
            z_clip = ceil(self.recon_params['image_dims'][2]/self.recon_params['slice_thickness'])
            z_clip = int(ceil(z_clip/2))

            img = img[ :, :, max(0, z_mid-z_clip):z_mid+z_clip]

        self.logger.info(f"Clipped Image Dimensions: {img.shape}")

        return img
    # -------------------------------------------------------------------------

    def projn_generator(self, vol_data):
        """
        -----------------------------------------------------------------------
        Function that runs that forward model projector to create CT
        projections of input volumetric data.

        :param vol_data:    3D volumetric image torch.Tensor
        :return: projn:     torch Tensor represent representing the X-ray
                            projection (this is raw projection - not log
                            attenuation)
        -----------------------------------------------------------------------
        """
        if self.pscale !=1.0:
            up_sampler = nn.Upsample(size=(vol_data.size(0),
                                      vol_data.size(1),
                                      int(vol_data.size(2)*self.recon_geometry[
                                          'projn_scale'])),
                                     mode='nearest')
            vol_data = up_sampler(vol_data.unsqueeze(0).unsqueeze(0))

        projn = self.run_fwd_projector(vol_data.squeeze().cpu().numpy())
        projn = torch.as_tensor(projn, dtype=torch.float)
        projn = torch.neg(projn)
        projn = torch.exp(projn)
        return projn
    # -------------------------------------------------------------------------

    def set_ring_artifact_params(self,
                                 mode='random',
                                 severity=0.1,
                                 gains=None):
        """
        -----------------------------------------------------------------------
        Set up parameters for generating ring artifacts in the CT
        reconstructions

        :param mode:        randomly generate detector gains
        :param severity:    for random mode, set the number of faulty detectors
        :param gains:       for fixed mode, feed the detector gains
        :return:
        -----------------------------------------------------------------------
        """

        assert mode in ['fixed', 'random']
        assert 0<severity<1

        self.ra_mode = mode
        self.severity = severity
        self.gains = gains

        if self.geom == 'cone':
            detector_panel = ones_like(self.machine_geometry['det_col_count']*
                                       self.machine_geometry['det_row_count'])

            if mode=='fixed' and gains is None:

                num_faulty_detectors = int(severity*detector_panel.size)

                faulty_detectors = \
                    np.random.normal(1.0, 0.2,
                                     num_faulty_detectors)

                detector_panel[:num_faulty_detectors] = faulty_detectors
                np.random.shuffle(detector_panel)
                detector_panel = detector_panel.reshape(
                    self.machine_geometry['det_col_count'],
                    self.machine_geometry['det_row_count']
                )

                self.detector_gain = detector_panel

            elif mode=='fixed' and isinstance(gains, np.ndarray):

                self.detector_gain = gains

            elif mode=='random':
                self.num_faulty_detectors = int(severity*detector_panel.size)
                self.det_panel_shape = (
                    self.machine_geometry['det_col_count'],
                    self.machine_geometry['det_row_count']
                )
                self.detector_gain = None

        elif self.geom in ['parallel', 'fan']:
            detector_panel = ones_like(self.machine_geometry['det_col_count'])

            if mode == 'fixed' and gains is None:

                num_faulty_detectors = int(severity * detector_panel.size)

                faulty_detectors = \
                    np.random.normal(1.0, 0.2,
                                     num_faulty_detectors)

                detector_panel[:num_faulty_detectors] = faulty_detectors
                np.random.shuffle(detector_panel)
                self.detector_gain = detector_panel

            elif mode == 'fixed' and isinstance(gains, np.ndarray):

                self.detector_gain = gains

            elif mode == 'random':
                self.num_faulty_detectors = int(severity * detector_panel.size)
                self.det_panel_shape = (self.machine_geometry['det_col_count'])
                self.detector_gain = None
    # -------------------------------------------------------------------------

    def add_ring_artifacts(self, projn):
        """
        -----------------------------------------------------------------------
        Adds ring artifacts to input projection (not sinogram).
        if mode is random and gains are none, then a new gain is calculated
                with each function call
        if mode is fixed and gains are none, then the gain is fixed with the
                randomly initialized values.
        if mode is fixed and gains are not none, then the gains are set as the
                input ones.

        :param projn:
        -----------------------------------------------------------------------
        """

        if self.geom == 'cone':
            detector_panel = ones(self.machine_geometry['det_col_count'] *
                                       self.machine_geometry['det_row_count'])

            if self.ra_mode == 'random' and self.gains is None:

                num_faulty_detectors = int(self.severity *detector_panel.size)

                faulty_detectors = \
                    np.random.normal(1, 0.2,
                                     num_faulty_detectors)
                faulty_detectors[faulty_detectors>1] = 1

                detector_panel[:num_faulty_detectors] = faulty_detectors
                np.random.shuffle(detector_panel)
                detector_panel = detector_panel.reshape(
                    self.machine_geometry['det_col_count'],
                    self.machine_geometry['det_row_count']
                )

                self.detector_gain = detector_panel

            noisy_projn = projn*self.detector_gain[:, newaxis, :]
            return noisy_projn

        elif self.geom in ['parallel', 'fan']:

            detector_panel = ones(self.machine_geometry['det_col_count'])

            if self.ra_mode == 'random' and self.gains is None:
                num_faulty_detectors = int(self.severity * detector_panel.size)

                faulty_detectors = \
                    np.random.normal(1, 0.2,
                                     num_faulty_detectors)
                faulty_detectors[faulty_detectors>1] = 1

                detector_panel[:num_faulty_detectors] = faulty_detectors
                np.random.shuffle(detector_panel)
                self.detector_gain = detector_panel

            noisy_projn = projn*self.detector_gain[newaxis, newaxis, :]
            return noisy_projn
    # -------------------------------------------------------------------------


if __name__=="__main__":
    default_scanner = ScannerTemplate(
        geometry='parallel',
        scan='circular',
        machine_dict=default_scanner_parallel.machine_geometry,
        recon='fbp',
        recon_dict=default_scanner_parallel.recon_params,
        pscale=1.0
    )

    default_scanner.set_recon_geometry()

    vol_img = np.zeros((512, 512, 350))  # volumetric image as example
    vol_img[200:250, 250:280, 100:300] = 0.1  # spawning objects in the volume
    vol_img[100:120, 400:600, 50:300] = 0.05
    vol_img[400:500, 50:100, 200:250] = 0.4

    ct_projn = default_scanner.run_fwd_projector(vol_img, verbose=True)
    ct_projn = moveaxis(ct_projn, 2, 0)
    recon_img = default_scanner.reconstruct_data(ct_projn)
    recon_img = moveaxis(recon_img, 0, 2)
    quick_imshow(1,2, [recon_img[:,:,200], vol_img[:,:,200]], titles=['Recon', 'GT'])

