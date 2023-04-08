#!/usr/bin/env python

# ------------------------------------------------------------------------------
"""de_decomposer.py:  Base Class for generating a 3D Dual energy CT projection
                     decomposition module for a pair of low/high energy CT
                     images"""
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
Module Description:

This module contains the base class DEDecomposer() for creating a Python module 
for performing Dual Energy Decomposition using different methodologies. All the 
modules for DE Decompostion in this library are inherited from this class. The 
class contains methods and attributes for calculating many useful quantities in 
the downstream decomposition process.

* Dual Energy Decomposition: 
Dual Energy Decomposition is the primary step in the processing of Dual Energy 
CT projections to generate X-ray CT / Atomic Number / Density Images from 
projection/sinogram data received Dual Energy Xray Scanners. It involves 
decomposing a pair of X-ray projections obtained for the same image by a scanner 
using two different X-ray spectra. The decomposition model as shown in 
Alvarez and Macovski:

Alvarez, Robert E., and Albert Macovski. "Energy-selective reconstructions in 
x-ray computerised tomography." Physics in Medicine & Biology 21.5 (1976): 733.

can be used to extract Compton and Photoelectric line-integrals from the pair 
of dual energy projections. These line integrals when projected back into the 
image space form Photoelectric and Compton images that are used for voxel-wise 
material characterization of the scanned image using Compton, PE coefficients 
and the effective atomic number Zeff. Depending upon the methodology used, 
there are subtle variations in the implementation of the decomposition 
algorithm - this module is the base class for implementing any such 
decomposition algorithm within the DebiSim framework.  
-------------------------------------------------------------------------------
"""

from tabulate import tabulate
import torch

import warnings
warnings.filterwarnings('ignore')

from lib.misc.fdlib import *
from lib.misc.ctlib import *
from lib.misc.util  import *


class DEDecomposer(object):
    """
    ---------------------------------------------------------------------------
    * Description:

    Base class for Dual Energy Decomposition Module in DebiSim.

    The base class defines attributes and methods that are necessary for
    performing any type of decomposition for DECT data. The initialization of
    the class requires the specification of the following input arguments:

    - Scanner DE spectral model          -  spctrm_h_fname, spctrm_l_fname,
                                            photon_count_high,
                                            photon_count_low

    - projection data specifications     -  (nangs, nbins)

    - optimization parameters            -  projector, out_dir, opt_specs

    Other class modules for Decomposition can be implemented by inheriting from
    DEDecomposer() as shown below:

    > ------------------------------------------------------------------>
    >   class CDMDecomposer(DEDecomposer):                              >
    >                                                                   >
    >   def __init__(self, ...,):                                       >
    >       print "Initializing Class ..."                              >
    >       DEDecomposer.__init__(self, spctr_h_fname, spctr_l_fname,   >
    >                             photon_count_low, photon_count_high,  >
    >                             nangs, nbins, circle, R, out_dir)     >
    >       print "DEDecomposer inherited"                              >
    >                                                                   >
    > ------------------------------------------------------------------>

    DEDecomposer() contains a set of attributes and methods that are useful to
    build functions for performing DE decomposition and analyzing it.

    * Methods:

    __init__                        - Constructor
    _load_spectra                   - Load the Dual Energy Spectra
    _compute_auxiliary_spectra      - Calculate differential spectra for Jacobian
    _effective_atomic_number        - get effective atomic number from Compton/PE
                                      values
    view_dect_specs                 - View DE Specifications
    view_opt_specs                  - View Optimization Specs

    set_photon_counts               - Set value for high/low energy photon count
    set_basis_functions             - Set DE basis functions in decomposition model
                                      (by default these are klein-nishina and photoelectric,
                                       can be changed to custom basis functions such as
                                       for Material Basis Decomposition)

    radon                           - Function for Radon transform
    iradon                          - Function for Inverse Radon transform

    pc_images_to_hl_sinograms       - Convert from Compton/PE to High/Low CT sinograms
    pc_images_to_hl_images          - Convert from Compton/PE to High/Low CT images
    pc_images_to_pc_sinograms       - Convert from Compton/PE images to High/Low CT sinograms
    pc_sinograms_to_hl_sinograms    - Convert from High/Low CT to Compton/PE  sinograms
    pc_sinograms_to_pc_images       - Convert from High/Low CT to Compton/PE  images

    (The above 6 methods only apply to 2D images slices for Parallel Beam CT. For more
     practical scanner geometries, see /src/simulator/)

    dummy_phantom                   - Dummy phantom for analysis
    add_poisson_noise               - Add poisson noise to a simulated sinogram
    cramer_rao_lower_bound          - Calculate Cramer-Rao Lower Variance bounds for Compton
                                      and PE estimates (see Alvarez and Macovski for derivation)

    photon_count_to_log_projection  - Convert X-ray attenuation to X-ray projection images
    log_projection_to_photon_count  - Convert X-ray projection  to X-ray attenuation images


    * Attributes:

    circle              - circle perimeter argument for radon transform
    dect_specs          - dictionary of DE specifications

    img_shape           - dimensions of the output CT image
    sino_shape          - dimensions of the input sinogram

    klein_nishina       - function handle for the Klein-Nishina Cross Section equation
    photoelectric       - function handle for the Photoelectric Basis equation

    m1_cp               - Compton/PE values for material basis function 1
                          (For Material Basis Decomposition)
    m2_cp               - Compton/PE values for material basis function 2
                          (For Material Basis Decomposition)

    n_img_pxls          - Number of image pixels
    n_sino_pxls         - Number of Sinogram pixels
    nangs               - Number of Views in the sinogram
    nbins               - Number of channels in the sinogram

    opt_specs           - Dictionary for Optimization specifications
    out_dir             - Output directory for storing decomposition results and logs
    projector           - Mode for optimization:
                          'cpu' - for CPU machines
                          'gpu' - for GPU machines (requires CUDA toolkit and Gpufit
                                                    installed)

    photon_count_high   - High Energy Photon Count
    photon_count_low    - Low  Energy Photon Count

    spctr_h_fname       - File name for high energy Xray spectrum
    spctr_l_fname       - File name for low energy Xray spectrum

    (these point to .txt file that contain the energy spectral distribution given by keV range
    and the corresponding energy spectral pdf. See /include/spectra/ for examples.)

    spctrm_h            - Loaded high energy spectra
    spctrm_l            - Loaded low  energy spectra

    spctrm_h_kn, spctrm_h_ph, spctrm_l_kn, spctrm_l_ph
                        - auxiliary derivative spectra for calculating the Jacobian

    theta               - Angle values for the sinogram views
    -------------------------------------------------------------------------"""

    def __init__(self,
                 spctr_h_fname,
                 spctr_l_fname,
                 photon_count_low,
                 photon_count_high,
                 nangs,
                 nbins,
                 projector,
                 out_dir=None):
        """
        -----------------------------------------------------------------------
        Constructor for the DEDecomposer class

        :param spctr_h_fname:       file location for low energy source spectrum
        :param spctr_l_fname:       file location for high energy source spectrum

        (These point to .txt files representing the spectral distribution of the
        Xray sources, i.e., a 2D array of keV values and corresponding normalized
        spectra pdf. See /include/spectra/ for examples)

        :param photon_count_low:    photon count for the high energy source
        :param photon_count_high:   photon count for the low energy source

        (Number of photons emitted by the Xray source)

        :param nangs:               number of views for sinogram
        :param nbins:               number of channels for sinogram

        (these arguments define a sinogram of dimensions nangs x nbins)

        :param projector:           'gpu' or 'cpu' - use gpu for faster operation
        :param out_dir:             directory for saving the output images
                                    and logs
        -----------------------------------------------------------------------
        """

        self.t_str = time.strftime("-%Y%b%d-%H%M")

        if out_dir is not None:
            self.out_dir = out_dir
            log_name = os.path.join(out_dir, "DECT" + self.t_str + ".log")
            sys.stdout = Logger(log_name)

        self.spctr_h_fname  = spctr_h_fname
        self.spctr_l_fname  = spctr_l_fname
        self.spctrm_h       = None
        self.spctrm_l       = None
        self.photon_count_high = photon_count_high
        self.photon_count_low = photon_count_low

        self.nangs = nangs
        self.nbins = nbins
        self.theta = linspace(0., 180.0, self.nangs, endpoint=False)
        self.sino_shape = (nbins, nangs)

        self.img_shape = (int(nbins / sqrt(2)), int(nbins / sqrt(2)))
        self.width = self.img_shape[0]

        self.n_img_pxls = self.img_shape[0] * self.img_shape[1]
        self.n_sino_pxls = self.sino_shape[0] * self.sino_shape[1]

        self.klein_nishina = klein_nishina
        self.photoelectric = photoelectric

        self._load_spectra()
        self.opt_specs = dict()

        # Prepare Radon transpose, sparse row for efficient dot product
        if projector == 'cpu':
            # Astra CPU projector
            proj_geom = astra.create_proj_geom('parallel', 1.0, self.nbins,
                                               self.theta / 180 * pi)
            vol_geom = astra.create_vol_geom(self.width, self.width)
            R = wrapped_astra_projector('line', proj_geom, vol_geom)

        elif projector == 'gpu':
            # Astra GPU projector
            proj_geom = astra.create_proj_geom('parallel', 1.0, self.nbins,
                                               self.theta / 180 * pi)
            vol_geom = astra.create_vol_geom(self.width, self.width)
            R = wrapped_astra_projector('cuda', proj_geom, vol_geom)

        elif sp.issparse(projector):
            # Pre-computed sparse matrix
            assert sp.issparse(projector)
            Rsp = projector.tocsc()

            def _matvec(v):     return Rsp.dot(v)

            def _rmatvec(v):    return Rsp.transpose().dot(v)

            R = sp.linalg.LinearOperator(
                shape=(self.n_sino_pxls, self.n_img_pxls), matvec=_matvec,
                rmatvec=_rmatvec)
        else:
            # skimage projector
            def _matvec(v):
                return tr.radon(v.reshape(self.img_shape), theta=self.theta,
                                circle=self.circle)

            def _rmatvec(v):
                return radon_transpose(v, self.img_shape, self.sino_shape,
                                       self.circle)

            R = sp.linalg.LinearOperator(
                shape=(self.n_sino_pxls, self.n_img_pxls), matvec=_matvec,
                rmatvec=_rmatvec)

        self.R = R
        self.projector = projector

        self.dect_specs = dict(
            spctr_h_fname=spctr_h_fname,
            spctr_l_fname=spctr_l_fname,
            nangs=self.nangs,
            nbins=self.nbins,
            photon_count_low=self.photon_count_low,
            photon_count_high=self.photon_count_high,
            projector=projector,
        )
        self.view_dect_specs()

        self.m1_cp = None
        self.m2_cp = None

    # -------------------------------------------------------------------------

    def _load_spectra(self):
        """
        -----------------------------------------------------------------------
        Read the low/high energy spectra from provided file locations
        -----------------------------------------------------------------------
        """

        self.spctrm_l = loadtxt(self.spctr_l_fname)
        self.spctrm_h = loadtxt(self.spctr_h_fname)
        self._compute_auxiliary_spectra()
    # -------------------------------------------------------------------------

    def _compute_auxiliary_spectra(self):
        """
        -----------------------------------------------------------------------
        Update auxiliary spectra useful in Jacobian calculations. Called every
        time bases are changed.

        -----------------------------------------------------------------------
        """

        self.spctrm_h_ph = array(self.spctrm_h)
        self.spctrm_h_ph[:, 1] = self.spctrm_h_ph[:, 1] * self.photoelectric(
            self.spctrm_h[:, 0])
        self.spctrm_h_kn = array(self.spctrm_h)
        self.spctrm_h_kn[:, 1] = self.spctrm_h_kn[:, 1] * self.klein_nishina(
            self.spctrm_h[:, 0])
        self.spctrm_l_ph = array(self.spctrm_l)
        self.spctrm_l_ph[:, 1] = self.spctrm_l_ph[:, 1] * self.photoelectric(
            self.spctrm_l[:, 0])
        self.spctrm_l_kn = array(self.spctrm_l)
        self.spctrm_l_kn[:, 1] = self.spctrm_l_kn[:, 1] * self.klein_nishina(
            self.spctrm_l[:, 0])
    # -------------------------------------------------------------------------

    def _effective_atomic_number(self, img_p, img_c):
        """
        ------------------------------------------------------------------------
        Generate Z_eff image

        :param  img_p - photoelectric image
        :param  img_c - compton image
        :return img_z - Z_eff image
        ------------------------------------------------------------------------
        """

        Kp = 1 / 2.501
        n = 3.5
        img_p = maximum(img_p, 0)
        img_c = maximum(img_c, 0)
        img_z = zeros(img_c.shape)
        nz = img_c.nonzero()
        img_z[nz] = Kp * (img_p[nz] / img_c[nz]) ** (1 / n)
        return img_z
    # -------------------------------------------------------------------------

    def view_dect_specs(self):
        """
        ------------------------------------------------------------------------
        View the Dual Energy CT specifications

        ---------------------------------------------------------------------
        """
        print('\n\n')
        header = ['Decomposer Specifications', '']
        print_table = []

        for key in self.dect_specs:
            print_table.append([key, self.dect_specs[key]])

        print(tabulate(print_table, header, tablefmt='psql'))

        print('\n')
    # -------------------------------------------------------------------------

    def view_opt_specs(self):
        """
        ------------------------------------------------------------------------
        View the Dual Energy CT optimization specifications

        -----------------------------------------------------------------------
        """

        print("-" * 80)
        print("\nOptimization Specifications:")

        for key in self.opt_specs:
            print(key, ":\t", self.opt_specs[key])
        print("-"*80)
    # -------------------------------------------------------------------------

    def set_photon_counts(self, pc_h, pc_l):
        """
        -----------------------------------------------------------------------
        Set the high/low energy dosage.

        -----------------------------------------------------------------------
        """

        self.photon_count_high = pc_h
        self.photon_count_low = pc_l
    # -------------------------------------------------------------------------

    def dummy_phantom(self, cp=None):
        """
        -----------------------------------------------------------------------
        Generate a dummy phantom.

        :param cp    - A list of tuples, each describes the cp of an object.
        :return img_c, img_p - Compton, PE images
        ---------------------------------------------------------------------
        """

        if cp == None:
            cp = [(0.3, 10000), (0.6, 20000), (0.9, 30000)]

        # Draw a circle
        img_c = ones((128, 128)) * 0.0
        img_p = ones((128, 128)) * 0.0

        ri, ci = dr.circle(31, 63, radius=16, shape=(128, 128))
        img_c[ri, ci] = cp[0][0]
        img_p[ri, ci] = cp[0][1]

        # Draw a box
        img_c[88 - 16:88 + 16, 40 - 16:40 + 16] = cp[1][0]
        img_p[88 - 16:88 + 16, 40 - 16:40 + 16] = cp[1][1]

        # Draw a thin rectangle
        img_c[88 - 16:88 + 16, 92:96] = cp[2][0]
        img_p[88 - 16:88 + 16, 92:96] = cp[2][1]

        return img_c, img_p
    # -------------------------------------------------------------------------

    def add_poisson_noise(self, sino_h, sino_l, input='projection'):
        """
        ------------------------------------------------------------------------
        Add Poisson noise to a pair of sinograms.

        :param sino_h, sino_l - input sinograms
        :param input          - whether input is 'projection' - a log projection
                                or 'count' - line integral

        :return  out_h, out_l    - noisy output sinogram
        -----------------------------------------------------------------------
        """

        if input == 'projection':
            # Add Poisson noise to the photon count
            # First convert energy projections to photon count
            y_h = exp(-sino_h) * self.photon_count_high
            y_l = exp(-sino_l) * self.photon_count_low
        elif input == 'count':
            y_h, y_l = sino_h, sino_l
        else:
            raise NameError("Input can only be projection or count!")
        # Then add Poisson noise to photon counts
        pssn_y_h = poisson(y_h)
        pssn_y_l = poisson(y_l)
        if pssn_y_h.min() < 1 or pssn_y_l.min() < 1:
            print("Warning: photon starvation...")
            # Assume no photon starvation
            pssn_y_h = maximum(pssn_y_h, 1)
            pssn_y_l = maximum(pssn_y_l, 1)
        if input == 'projection':
            # Finally convert back to energy projection
            pssn_sino_h = -log(pssn_y_h) + log(self.photon_count_high)
            pssn_sino_l = -log(pssn_y_l) + log(self.photon_count_low)
            return pssn_sino_h, pssn_sino_l
        else:
            return pssn_y_h, pssn_y_l
    # -------------------------------------------------------------------------

    def radon(self, img):
        """
        ------------------------------------------------------------------------
        Convenient function for scipy's radon transform.

        :param img     - input image array, can be either flattened or 2D
        :return sino   - output radon image, same # of dimensions as input
        ------------------------------------------------------------------------
        """

        if img.ndim == 1:
            img = img.reshape(self.img_shape)
            return self.R.matvec(img)
        else:
            return self.R.matvec(img.flatten()).reshape(self.sino_shape)
    # -------------------------------------------------------------------------

    def iradon(self, sino, filter='ramp'):
        """
        ------------------------------------------------------------------------
        Convenient function for scipy's iradon transform

        :param  sino    - input radon image array, can be flattened or 2D
        :param  filter  - FBP filter (Options are the same as for scipy.iradon)
        :return img     - output image, same # of dimensions as input
        ------------------------------------------------------------------------
        """

        if sino.ndim == 1:
            sino = sino.reshape(self.sino_shape)
            flat = True
        if self.projector == 'gpu':
            vol_geom = astra.create_vol_geom(self.width, self.width)
            proj_geom = astra.create_proj_geom('parallel', 1.0, self.nbins,
                                               deg2rad(self.theta))
            # Create a data object for the reconstruction
            rec_id = astra.data2d.create('-vol', vol_geom)
            # Set up the parameters for a reconstruction algorithm using the GPU
            cfg = astra.astra_dict('FBP_CUDA')
            # sinogram_id, sinogram_gpu = astra.create_sino(sino, proj_id)
            sino_id = astra.data2d.create('-sino', proj_geom, sino.transpose())
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sino_id
            # Create the algorithm object from the configuration structure
            alg_id = astra.algorithm.create(cfg)
            # Run
            astra.algorithm.run(alg_id)
            # Get the result
            img = astra.data2d.get(rec_id)
        else:
            img = tr.iradon(sino, theta=self.theta,
                            circle=self.circle, filter=filter)
    # -------------------------------------------------------------------------

    def set_basis_functions(self, fc, fp, m1_cp, m2_cp):
        """
        ------------------------------------------------------------------------
        Overwrite the default Compton and photoelectric basis functions.

        :param fc, fp  - must take in a scalar energy keV and return the
                         corresponding coefficient. The functions must also be
                         broadcastable to numpy array of energy level inputs.
                         They could be material bases.

        Return:
        ------------------------------------------------------------------------
        """

        self.klein_nishina = fc
        self.photoelectric = fp
        self.m1_cp = m1_cp
        self.m2_cp = m2_cp
        self._compute_auxiliary_spectra()

        print("==================== INFO ====================")
        print("Basis functions were changed!")
        print("Material CP values are", m1_cp, m2_cp)
        print("==============================================")
    # -------------------------------------------------------------------------

    def pc_images_to_hl_sinograms(self,
                                  img_p, img_c,
                                  spctrm_h=None,
                                  spctrm_l=None,
                                  neglog=True):
        """
        ------------------------------------------------------------------------
        Given the pc images, use radon transform to construct the H/L
        sinograms.

        :param img_p   - Photoelectric image
        :param img_c   - Compton image
        :param spctrm_h, spctrm_l  - high/low energy spectra to be used
        :param neglog  - output is log projection if set to true, photon
                         count otherwise

        :return sino_h  - high energy sinogram
        :return sino_l  - low energy sinogram
        ------------------------------------------------------------------------
        """

        # Perform Radon transform
        # When circle is set to False, radon will pad the image with 0s to a
        # square image with the size of diagonal by diagonal. So if our phantom
        # has objects outside the inscribed circle, circle should be set to False.

        if spctrm_h is None or spctrm_l is None:
            spctrm_h = self.spctrm_h
            spctrm_l = self.spctrm_l

        A_p, A_c = self.pc_images_to_pc_sinograms(img_p, img_c)

        # Calculate projection values
        sino_h, sino_l = self.pc_sinograms_to_hl_sinograms(A_p,
                                                           A_c,
                                                           spctrm_h,
                                                           spctrm_l,
                                                           neglog=neglog)
        return sino_h, sino_l
    # -------------------------------------------------------------------------

    def pc_images_to_hl_images(self, img_p, img_c):
        """
        ------------------------------------------------------------------------
        Given PE-Compton Image convert them to High-Low Energy images (this only
        works for a 2D image slice for a parallel beam scanner.)

        :param img_p -      PE image
        :param img_c -      Compton Image

        :return: high-low energy CT images
        ------------------------------------------------------------------------
        """

        sino_h, sino_l = self.pc_images_to_hl_sinograms(
            img_p, img_c,
            spctrm_h=None,
            spctrm_l=None,
            neglog=True)

        return self.iradon(sino_h), self.iradon(sino_l)
    # -------------------------------------------------------------------------

    def pc_images_to_pc_sinograms(self, img_p, img_c):
        """
        ------------------------------------------------------------------------
        Given the pc images, use radon transform to construct the pc sinograms.

        :param  img_p   - Photoelectric image
        :param  img_c   - Compton image
        :return sino_p  -  sinogram
        :return sino_c  - low energy sinogram
        -----------------------------------------------------------------------
        """

        A_p = self.radon(img_p)
        A_c = self.radon(img_c)
        return A_p, A_c
    # -------------------------------------------------------------------------

    def pc_sinograms_to_hl_sinograms(self,
                                     A_p, A_c,
                                     spctrm_h=None,
                                     spctrm_l=None,
                                     neglog=True,
                                     useGPU=True):
        """
        ------------------------------------------------------------------------
        Given the pc sinograms, return the hl sinograms. Mostly used as part of
        the forward model.

        :param A_p          - Photoelectric line integrals
        :param A_c          - Compton line integrals
        :param neglog       - True for logarithmic projection and false for
                              photon count
        :return sino_h      - high energy projection/count sinogram
        :return sino_l      - low energy projection/count sinogram
        -----------------------------------------------------------------------
        """

        if spctrm_h is None or spctrm_l is None:
            spctrm_h = self.spctrm_h
            spctrm_l = self.spctrm_l
        flat = True

        if A_p.ndim > 1 and A_c.ndim > 1:
            # All numerical calculations are done with flat arrays
            flat = False
            A_p = A_p.flatten()
            A_c = A_c.flatten()

        # Calculate projection values, using GPU whenever possible
        if self.projector == 'gpu' and useGPU:
            t_spctrm_h_kn = torch.from_numpy(
                self.klein_nishina(spctrm_h[:, 0])).type(torch.cuda.FloatTensor)
            t_spctrm_l_kn = torch.from_numpy(
                self.klein_nishina(spctrm_l[:, 0])).type(torch.cuda.FloatTensor)
            t_spctrm_h_p = torch.from_numpy(
                self.photoelectric(spctrm_h[:, 0])).type(torch.cuda.FloatTensor)
            t_spctrm_l_p = torch.from_numpy(
                self.photoelectric(spctrm_l[:, 0])).type(torch.cuda.FloatTensor)
            t_spctrm_h = torch.from_numpy(spctrm_h[:, 1]).type(
                torch.cuda.FloatTensor)
            t_spctrm_l = torch.from_numpy(spctrm_l[:, 1]).type(
                torch.cuda.FloatTensor)
            t_A_c = torch.from_numpy(A_c).type(torch.cuda.FloatTensor)
            t_A_p = torch.from_numpy(A_p).type(torch.cuda.FloatTensor)
            t_A_h = -torch.ger(t_A_c, t_spctrm_h_kn) - torch.ger(t_A_p,
                                                                 t_spctrm_h_p)
            t_A_l = -torch.ger(t_A_c, t_spctrm_l_kn) - torch.ger(t_A_p,
                                                                 t_spctrm_l_p)
            t_A_h = torch.matmul(torch.exp(t_A_h), t_spctrm_h)
            t_A_l = torch.matmul(torch.exp(t_A_l), t_spctrm_l)
            A_h = t_A_h.to('cpu').numpy()
            A_l = t_A_l.to('cpu').numpy()

        else:
            A_l = -outer(A_c, self.klein_nishina(spctrm_l[:, 0])) - \
                  outer(A_p, self.photoelectric(spctrm_l[:, 0]))
            A_l = dot(exp(A_l), spctrm_l[:, 1])
            A_h = -outer(A_c, self.klein_nishina(spctrm_h[:, 0])) - \
                  outer(A_p, self.photoelectric(spctrm_h[:, 0]))
            A_h = dot(exp(A_h), spctrm_h[:, 1])
        # Account for the actual photon count
        A_l *= self.photon_count_low
        A_h *= self.photon_count_high
        if neglog:

            # Convert photon count to attenuation (line integrals)
            A_h = maximum(A_h, 1)
            A_l = maximum(A_l, 1)
            A_l = -log(A_l) + log(self.photon_count_low)
            A_h = -log(A_h) + log(self.photon_count_high)
        # Maintain the input format
        if not flat:
            A_h = A_h.reshape(self.sino_shape)
            A_l = A_l.reshape(self.sino_shape)
        return A_h, A_l
    # -------------------------------------------------------------------------

    def pc_sinograms_to_pc_images(self, A_p, A_c):
        """
        ------------------------------------------------------------------------
        Given the pc sinograms, use iradon transform to construct the pc images.

        :param sino_p   - photoelectric sinogram
        :param sino_c   - Compton sinogram
        :return img_p   - Photoelectric image
        :return img_c   - Compton image
        -----------------------------------------------------------------------
        """

        return self.iradon(A_p), self.iradon(A_c)
    # -------------------------------------------------------------------------

    def cramer_rao_lower_bound(self, A_c, A_p):
        """
        ------------------------------------------------------------------------
        Return the Cramer Rao lower bounds in flattened arrays.

        :param Ac, Ap  - ground truth pair of line integrals
        :return sig_Ac, sig_Ap  - respective CR bounds
        ------------------------------------------------------------------------
        """
        lmbd_h, lmbd_l = self.pc_sinograms_to_hl_sinograms(A_p, A_c,
                                                           neglog=False)
        dlmbdh_dAp, dlmbdl_dAp = self.pc_sinograms_to_hl_sinograms(
            A_p, A_c, self.spctrm_h_ph, self.spctrm_l_ph, neglog=False)
        dlmbdh_dAc, dlmbdl_dAc = self.pc_sinograms_to_hl_sinograms(
            A_p, A_c, self.spctrm_h_kn, self.spctrm_l_kn, neglog=False)

        hp = dlmbdh_dAp ** 2 / lmbd_h + dlmbdl_dAp ** 2 / lmbd_l
        lc = dlmbdh_dAc ** 2 / lmbd_h + dlmbdl_dAc ** 2 / lmbd_l
        hc = lp = dlmbdh_dAp * dlmbdh_dAc / lmbd_h + dlmbdl_dAp * dlmbdl_dAc / lmbd_l
        H_det = abs(hp * lc - hc * lp)
        sig_Ap = sqrt(lc / H_det)
        sig_Ac = sqrt(hp / H_det)
        return sig_Ac, sig_Ap
    # -------------------------------------------------------------------------

    def photon_count_to_log_projection(self, pc_h, pc_l):
        """
        ------------------------------------------------------------------------
        Photon count to logarithmic projection
        :param pc_h, pc_l  - H/L photon count
        :param A_h, A_l    - H/L logarithmic projection
        ------------------------------------------------------------------------
        """

        # assert pc_h.min() > 0 and pc_l.min() > 0
        if pc_h.min() < 1 or pc_l.min() < 1:
            print("Photon starvation, clipping to 1...")
            pc_h = maximum(pc_h, 1)
            pc_l = maximum(pc_l, 1)
        return log(self.photon_count_high / pc_h), \
               log(self.photon_count_low / pc_l)
    # -------------------------------------------------------------------------

    def log_projection_to_photo_count(self, A_h, A_l):
        """
        ------------------------------------------------------------------------
        Logarithmic projection to photon count

        :param  A_h, A_l    - H/L logarithmic projection
        :return pc_h, pc_l  - H/L photon count
        ------------------------------------------------------------------------
        """

        return exp(-A_h) * self.photon_count_high, \
               exp(-A_l) * self.photon_count_low
    # -------------------------------------------------------------------------

    # =========================================================================
    # Class Ends
    # =========================================================================
