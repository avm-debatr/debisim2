#!/usr/bin/env python

# -----------------------------------------------------------------------------
"""
cdm_decomposer.py:   Class for carrying out Dual Energy Decomposition pair using
                    Constrained Decomposition Method. (C. Crawford et al.)
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

"""
-------------------------------------------------------------------------------
Module Description:

This module contains the class CDMDecomposer() for carrying out Dual Energy 
Decomposition for a pair of Dual Energy CT images using the Constrained 
Decomposition Method (CDM). The DE Decomposition model for CDM is described in:

Ying, Zhengrong, Ram Naidu, and Carl R. Crawford. "Dual Energy Computed 
Tomography for Explosive Detection." Journal of X-ray Science and Technology 
14.4 (2006): 235-256.

It establishes a contrained NLS optimization model for extracting PE and 
Compton Line Integrals from Dual Energy Projection pairs. The module contains 
the implementation of this optimization model that performs decomposition on 
an elementwise level for an input pair of X-ray projections. The module 
contains support for single-CPU/ multi-CPU / GPU operation as well as 
definition of custom material basis functions. 

The module is built upon the parent class DEDecomposer() and inherits its 
attributes and methods from DEDecomposer(). Like DEDecomposer(), the 
operation of the module requires input knowledge of the DE spectral model, 
projection data specifications and the optimization specifications. The class 
uses the DogBox method (scipy.optimize.least_squares) to perform optimization
and extract the Compton/PE values. The module also contains support to change 
the basis functions for decomposition for methods like Material Basis 
Decomposition.
-------------------------------------------------------------------------------
"""

from lib.decomposer.de_decomposer import *
import pygpufit.gpufit as gf


class CDMDecomposer(DEDecomposer):
    """
    ---------------------------------------------------------------------------
    Module for performing Dual Energy Decomposition using Constrained
    Decomposition Method.

    The module is initialized like DEDecomposer() with the decomposition,
    optimization and projection specifications as shown below. The decomposition
    is then carried out using the method
    CDMDecomposer.decompose_dect_sinograms() which takes in the pair of dual
    energy projections and returns the Compton and PE line integrals.

    > --------------------------------------------------------------------- >
    > import cdm_ecomposer as cdm                                           >
    >                                                                       >
    > cdm_decomposer = cdm.CDMDecomposer(                                   >
    >       spctr_l_fname='/include/spectra/example_spectrum_95kV.txt',     >
    >       spctr_h_fname='/include/spectra/example_spectrum_130kV.txt',    >
    >       photon_count_low=1.7e5,                                         >
    >       photon_count_high=1.8e5,                                        >
    >       nangs=360,                                                      >
    >       nbins=512,                                                      >
    >       R='gpu'                                                         >
    >       )                                                               >
    >                                                                       >
    > sino_compton, sino_pe = cdm_decomposer.decompose_dect_sinograms(      >
    >                               sino_high,                              >
    >                               sino_low,                               >
    >                               solver='gpu',                           >
    >                               type='cpd'                              >
    >                              )                                        >
    >                                                                       >
    > --------------------------------------------------------------------- >

    Note that the high/low energy sinograms fed in as sino_high / sino_low
    must correspond to the respective Xray spectra spctr_l_fname and
    spctr_h_fname. The choice of the solver determines whether the operation is
    carried on a CPU ('cpu') / GPU ('gpu') machines (It is recommended to use
    solver='gpu' if CUDA support is available for fast operation.). The
    decomposition can be performed for Compton-PE basis (type='cpd') r for
    Material-Basis Decomposition (type='mbd')

    * Methods:

    __init__                    - Constructor
    decompose_dect_sinograms    - function to decompose the input projection pair
    cdm_worker                  - CDM decomposer worker function for multi-threading
    _cost_calc                  - function to calculate CDM quadratic cost

    * Attributes:

    init_val                    - value for initialization of the DE optimization
                                  operation, default value is array([0.1, 0.1])
    pix_no                      - Current pixel being processed
    ---------------------------------------------------------------------------
    """

    # Methods ------------------------------------------------------------------

    def __init__(self,
                 spctr_h_fname, spctr_l_fname,
                 photon_count_low, photon_count_high,
                 nangs, nbins, projector,
                 out_dir=None
                 ):
        """---------------------------------------------------------------------
        Constructor for CDMDecomposer

        :param  spctrm_h, spctrm_l      - high/low energy spectra
        :param  photon_count_high       - high/low energy dosage
        :param  photon_count_low        - high/low energy dosage
        :param  out_dir                 - location for output files
        :param  dect_specs              - dictionary of DECT Specifications
        :param  nangs                   - angle resolution for sinogram
        :param  nbins                   - bin resolution for sinogram
        :param  theta                   - angles where projections took place
        :param  projector               - projector to use
        ---------------------------------------------------------------------"""

        DEDecomposer.__init__(self, spctr_h_fname, spctr_l_fname,
                              photon_count_low, photon_count_high,
                              nangs, nbins, projector, out_dir)
        self.init_val = array([0.1, 0.1])

    # --------------------------------------------------------------------------

    def decompose_dect_sinograms(self, sino_h, sino_l, solver='gpu',
                                 method='lsq', type='cpd'):
        """
        -----------------------------------------------------------------------
        Carry out CDM Dual Decomposition on the sinogram pair.

        :param  sino_h  - high energy sinogram (log projection)
        :param  sino_l  - low energy sinogram (log projection)

        (sino_high, sino_low must have the same dimensions)

        :param  solver  - one of multi-threaded 'cpu', LM on 'gpu' or 'vec'
        :param  method  - 'lsq' for using scipy's bounded least squares
                            optimizer; otherwise exactly follows the paper
        :param  type    - 'cpd' - Compton-PE basis decomposition;
                          'mbd' - Material Basis Decomposition (requires
                                  calling set_basis_functions to change the
                                  bases from Compton-PE to the material bases)

        :return  sino_p  - sinogram for photoelectric image
        :return  sino_c  - sinogram for compton image
        -----------------------------------------------------------------------
        """

        # Compute the energy dependent coefficients
        self.pix_no = 0
        t = time.time()
        scc, scp = 1e0, 1e0

        # Decompose every energy pair in the sinograms
        def _decomp(p_h, p_l):
            """
            -----------------------------------------------------------------
            Decompose by optimization

            :param  p_h - high energy sinogram pixel
            :param  p_l - low energy sinogram pixel
            Returns:    decomposed output pixels p_c, p_pe
            -----------------------------------------------------------------
            """

            if (self.pix_no % 10000) == 0:
                print("Pixel No", self.pix_no, '\tTime Elapsed:\t', \
                    time.time() - t, 'seconds')
            self.pix_no += 1

            # If either projection is non-positive, return no attenuation
            if p_h <= 0 or p_l <= 0:
                return (0.0, 0.0)

            # Residuals -------------------------------------------------------
            def _residual(a, mul):
                """
                ---------------------------------------------------------------
                Calculate Residual

                :param  a - input a_c, a_p array
                :return:    residual vector
                ---------------------------------------------------------------
                """

                a_p = a[0] * mul[0] / scp
                a_c = a[1] * mul[1] / scc
                tp_h, tp_l = self.pc_sinograms_to_hl_sinograms(a_p, a_c)
                tp_h, tp_l = tp_h[0], tp_l[0]

                return array([tp_h - p_h, tp_l - p_l])
                # -------------------------------------------------------------

            def _jacobian(a, mul):
                """
                ---------------------------------------------------------------
                Calculate Jacobian

                :param  a - input a_c, a_p array
                :return:    jacobian matrix
                ---------------------------------------------------------------
                """

                a_p = a[0] * mul[0] / scp
                a_c = a[1] * mul[1] / scc
                tp_h, tp_l = self.pc_sinograms_to_hl_sinograms(
                    a_p, a_c, neglog=False)
                dph_dap, dpl_dap = self.pc_sinograms_to_hl_sinograms(
                    a_p, a_c, self.spctrm_h_ph, self.spctrm_l_ph, False)
                dph_dac, dpl_dac = self.pc_sinograms_to_hl_sinograms(
                    a_p, a_c, self.spctrm_h_kn, self.spctrm_l_kn, False)
                dph_da = array([[dph_dap[0], dph_dac[0]]]) / tp_h
                dpl_da = array([[dpl_dap[0], dpl_dac[0]]]) / tp_l
                J = vstack((dph_da, dpl_da))
                J[:, 0] *= mul[0]
                J[:, 1] *= mul[1]

                return J
                # -------------------------------------------------------------

            # Solve Ac and Ap for a single pair of projections
            if method == 'lsq':
                if type == 'cpd':
                    res = op.least_squares(_residual,
                                           self.init_val,
                                           jac=_jacobian,
                                           bounds=(0, inf),
                                           method='dogbox',
                                           args=([1.0, 1.0],))
                elif type == 'mbd':
                    res = op.least_squares(_residual,
                                           self.init_val,
                                           jac=_jacobian,
                                           method='dogbox',
                                           args=([1.0, 1.0],))
                else:
                    print("Please specify type: 'mbd' or 'cpd' ")
            else:
                # Find the roots first
                res = op.root(_residual, self.init_val, jac=_jacobian,
                              args=([1.0, 1.0],))

                # If either Ac or Ap is negative, do additional Gauss-Newton
                # with additional constraint that either Ac or Ap is zero.
                if res.x[0] < 0 or res.x[1] < 0:
                    res_p0 = op.least_squares(_residual,
                                              array([0.0, res.x[1]]),
                                              jac=_jacobian,
                                              method='lm',
                                              args=([0.0, 1.0],))
                    assert abs(res_p0.x[0]) < 1e-4
                    res_c0 = op.least_squares(_residual,
                                              array([res.x[0], 0.0]),
                                              jac=_jacobian,
                                              method='lm',
                                              args=([1.0, 0.0],))
                    assert abs(res_c0.x[1]) < 1e-4
                    if res_p0.cost > res_c0.cost:
                        res = res_c0
                    else:
                        res = res_p0

            self.init_val = res.x

            return tuple(res.x)

        if solver == 'cpu':
            t0 = time.time()
            mp = MultiProcessor(self.cdm_worker)
            print("CDM is using %d processes..." % mp.num_processes)
            # Assign each worker an angle
            for angle in arange(self.nangs):
                mp.add_job((self, sino_h[:, angle], sino_l[:, angle], angle))
            # Close out by stitching the angles back together
            res = mp.close_out()
            t1 = time.time()
            print("Done, multiprocessed CDM took %.5fs..." % (t1 - t0))
            res = concatenate(res, 1)

            # Reorder the results that were out of sync
            idx = res[-1, :, 0].astype(int)
            res[:, idx, :] = array(res)
            sino_p = res[:-1, :, 0] / scp
            sino_c = res[:-1, :, 1] / scc
            # quick_imshow(1,2,[sino_c, sino_p])

        elif solver == 'vec':
            print("CDM is using vectorization...")
            print("Total number of projections:", self.n_sino_pxls)
            # decompose entire sinogram
            t0 = time.time()
            vdecomp = np.vectorize(_decomp)
            t1 = time.time()
            res = vdecomp(sino_h, sino_l)
            t2 = time.time()
            print("Vectorize took %.5fs, Execution took %.5fs..." \
                  % (t1 - t0, t2 - t1))
            res = np.asarray(res)
            sino_p = res[0, :] / scp
            sino_c = res[1, :] / scc

        elif solver == 'gpu' and type == 'cpd':
            print("CDM+CPD is using GPU...")
            t0 = time.time()
            GFtol = 1e-4
            GFmaxIter = 20
            # User info
            GFui = concatenate((
                [scc, scp],
                [self.photon_count_high, self.photon_count_low],
                [self.spctrm_h.shape[0], self.spctrm_l.shape[0]],
                self.spctrm_h[:, 0].copy().flatten(),
                self.spctrm_l[:, 0].copy().flatten(),
                self.spctrm_h[:, 1].copy().flatten(),
                self.spctrm_l[:, 1].copy().flatten(),
                self.spctrm_h_ph[:, 1].copy().flatten(),
                self.spctrm_l_ph[:, 1].copy().flatten(),
                self.spctrm_h_kn[:, 1].copy().flatten(),
                self.spctrm_l_kn[:, 1].copy().flatten(),
            ))

            # Arrays
            mh, ml = sino_h.flatten(), sino_l.flatten()
            pch, pcl = self.log_projection_to_photo_count(mh, ml)
            GFdata = vstack((
                mh, ml
            )).transpose()
            GFw = vstack((
                pch, pcl
            )).transpose()
            GFinit = vstack((
                ones(self.n_sino_pxls, dtype=np.float32) * self.init_val[1],
                ones(self.n_sino_pxls, dtype=np.float32) * self.init_val[0]
            )).transpose()
            # GFinit = vstack((
            #     ones(mh.size, dtype=float32) * self.init_val[1],
            #     ones(ml.size, dtype=float32) * self.init_val[0]
            # )).transpose()

            # GPU uses float32
            GFui_ac = ascontiguousarray(GFui, dtype=np.float32)
            GFdata_ac = ascontiguousarray(GFdata, dtype=np.float32)
            GFw_ac    = ascontiguousarray(GFw, dtype=np.float32)
            GFinit_ac = ascontiguousarray(GFinit, dtype=np.float32)

            GFres = gf.fit(
                GFdata_ac, GFw_ac, gf.ModelID.COMPTON_PE, GFinit_ac, GFtol,
                GFmaxIter, None, gf.EstimatorID.LSE, GFui_ac)

            GF_out = zeros_like(GFres[0])
            GF_out.data = GF_out.data

            sino_c = GF_out[:, 0].flatten().reshape(self.sino_shape)
            sino_p = GF_out[:, 1].flatten().reshape(self.sino_shape)
            t1 = time.time()

            print("Done, GPU CDM took %.5fs..." % (t1 - t0))
            parameters, states, chi_squares, number_iterations, execution_time = GFres
            number_fits = self.n_sino_pxls
            converged = states == 0
            print('\ngpufit stats:')
            print(
                'iterations:      {:.2f}'.format(
                    np.mean(number_iterations[converged])))
            print('time:            {:.2f} s'.format(execution_time))

            # get fit states
            number_converged = np.sum(converged)
            print('ratio converged         {:6.2f} %'.format(
                number_converged * 1.0 / number_fits * 100))
            print('ratio max it. exceeded  {:6.2f} %'.format(
                np.sum(states == 1) * 1.0 / number_fits * 100))
            print('ratio singular hessian  {:6.2f} %'.format(
                np.sum(states == 2) * 1.0 / number_fits * 100))
            print('ratio neg curvature MLE {:6.2f} %'.format(
                np.sum(states == 3) * 1.0 / number_fits * 100))

            del GFdata, GFmaxIter, GFtol, GFui, GFw, GFinit
            del GFdata_ac, GFui_ac, GFw_ac, GFinit_ac

        elif solver == 'gpu' and type == 'mbd':
            print("CDM+MBD is using GPU...")
            t0 = time.time()
            GFtol = 1e-4
            GFmaxIter = 20
            # User info
            GFui = concatenate((
                [scc, scp],
                [self.photon_count_high, self.photon_count_low],
                [self.spctrm_h.shape[0], self.spctrm_l.shape[0]],
                list(self.m1_cp),
                list(self.m2_cp),
                self.spctrm_h[:, 0].flatten(),
                self.spctrm_l[:, 0].flatten(),
                self.spctrm_h[:, 1].flatten(),
                self.spctrm_l[:, 1].flatten(),
                self.spctrm_h_ph[:, 1].flatten(),
                self.spctrm_l_ph[:, 1].flatten(),
                self.spctrm_h_kn[:, 1].flatten(),
                self.spctrm_l_kn[:, 1].flatten(),
            ))
            GFui = ascontiguousarray(GFui, dtype=float32)
            # Arrays
            mh, ml = sino_h.flatten(), sino_l.flatten()
            pch, pcl = self.log_projection_to_photo_count(mh, ml)
            GFdata = vstack((
                mh, ml
            )).transpose()
            GFw = vstack((
                pch, pcl
            )).transpose()
            GFinit = vstack((
                ones(self.n_sino_pxls, dtype=float32) * self.init_val[1],
                ones(self.n_sino_pxls, dtype=float32) * self.init_val[0]
            )).transpose()
            # GPU uses float32
            GFdata = ascontiguousarray(GFdata, dtype=float32)
            GFw = ascontiguousarray(GFw, dtype=float32)
            GFinit = ascontiguousarray(GFinit, dtype=float32)
            GFres = gf.fit(
                GFdata, GFw, gf.ModelID.MATERIAL_BASIS, GFinit, GFtol,
                GFmaxIter, None, gf.EstimatorID.LSE, GFui)
            sino_c = GFres[0][:, 0].flatten().reshape(self.sino_shape)
            sino_p = GFres[0][:, 1].flatten().reshape(self.sino_shape)
            t1 = time.time()
            print("Done, GPU CDM took %.5fs..." % (t1 - t0))
            parameters, states, chi_squares, number_iterations, execution_time = GFres
            number_fits = self.n_sino_pxls
            converged = states == 0
            print('\ngpufit stats:')
            print(
                'iterations:      {:.2f}'.format(
                    np.mean(number_iterations[converged])))
            print('time:            {:.2f} s'.format(execution_time))

            # get fit states
            number_converged = np.sum(converged)
            print('ratio converged         {:6.2f} %'.format(
                number_converged * 1.0 / number_fits * 100))
            print('ratio max it. exceeded  {:6.2f} %'.format(
                np.sum(states == 1) * 1.0 / number_fits * 100))
            print('ratio singular hessian  {:6.2f} %'.format(
                np.sum(states == 2) * 1.0 / number_fits * 100))
            print('ratio neg curvature MLE {:6.2f} %'.format(
                np.sum(states == 3) * 1.0 / number_fits * 100))

        else:
            raise NotImplementedError("%s is not implemented!" % solver)
        print("Range of A_p:", sino_p.max(), sino_p.min())
        print("Range of A_c:", sino_c.max(), sino_c.min())

        return sino_p, sino_c
        # --------------------------------------------------------------------------

    @staticmethod
    @worker
    def cdm_worker(args):
        # Every worker process a given slice of sinogram
        self, sino_h, sino_l, angle = args
        sino_p, sino_c = zeros(self.nbins + 1), zeros(self.nbins + 1)
        sino_p[-1], sino_c[-1] = angle, angle

        # Residuals --------------------------------------------------------
        def _residual(a, extra_args):
            """
            ----------------------------------------------------------------
            Calculate Residual

            :param  a - input a_c, a_p array
            :return residual vector
            ----------------------------------------------------------------
            """

            a_p = a[0] * extra_args[0]
            a_c = a[1] * extra_args[1]
            p_h = extra_args[2]
            p_l = extra_args[3]
            tp_h, tp_l = self.pc_sinograms_to_hl_sinograms(a_p, a_c,
                                                           useGPU=False)
            tp_h, tp_l = tp_h[0], tp_l[0]
            return array([tp_h - p_h, tp_l - p_l])

        def _jacobian(a, extra_args):
            """
            ----------------------------------------------------------------
            Calculate Jacobian

            :param  a - input a_c, a_p array
            :return jacobian matrix
            ----------------------------------------------------------------
            """

            a_p = a[0] * extra_args[0]
            a_c = a[1] * extra_args[1]
            tp_h, tp_l = self.pc_sinograms_to_hl_sinograms(
                a_p, a_c, neglog=False, useGPU=False)
            dph_dap, dpl_dap = self.pc_sinograms_to_hl_sinograms(
                a_p, a_c, self.spctrm_h_ph, self.spctrm_l_ph, False,
                useGPU=False)
            dph_dac, dpl_dac = self.pc_sinograms_to_hl_sinograms(
                a_p, a_c, self.spctrm_h_kn, self.spctrm_l_kn, False,
                useGPU=False)
            dph_da = array([[dph_dap[0], dph_dac[0]]]) / tp_h
            dpl_da = array([[dpl_dap[0], dpl_dac[0]]]) / tp_l
            J = vstack((dph_da, dpl_da))
            J[:, 0] *= extra_args[0]
            J[:, 1] *= extra_args[1]
            return J

        x0 = array([0, 0])
        # Decompose every energy pair in the sinograms
        for i, (p_h, p_l) in enumerate(zip(sino_h, sino_l)):
            # If either projection is non-positive, return no attenuation
            if p_h <= 0 or p_l <= 0:
                sino_p[i], sino_c[i] = 0.0, 0.0
                continue

            # Solve Ac and Ap for a single pair of projections
            res = op.least_squares(_residual,
                                   x0,
                                   jac=_jacobian,
                                   # bounds=(0, inf),
                                   method='dogbox',
                                   args=([1.0, 1.0, p_h, p_l],))
            x0 = res.x
            sino_p[i], sino_c[i] = res.x[0], res.x[1]

        return stack((sino_p.reshape(-1, 1), sino_c.reshape(-1, 1)), 2)
    # -------------------------------------------------------------------------

    def _cost_calc(self, plrow, phrow, w_0, p0):
        """
        -----------------------------------------------------------------------
        Function to calculate the residual cost for a given dual energy CT
        array pair.

        :param plrow:   row of p_l values
        :param phrow:   row of p_h values
        :param w_0:     initial w
        :param p0:
        :return:        calculated cost
        -----------------------------------------------------------------------
        """

        s_max = len(plrow)

        def _cost_func(w_c):
            cost = []
            for s in range(s_max):
                c_curr = 0.0

                for sc in range(max([s, 0]),
                                min([int(s_max - 1), int(s + w_c)])):
                    if phrow[sc - int(np.floor(0.5 * w_c))] - p0 > 0:
                        c_curr += phrow[sc - int(np.floor(0.5 * w_c))] - p0
                cost.append(c_curr)
            return max(cost)

        w = op.least_squares(_cost_func,
                             np.array([w_0]),
                             bounds=(0, s_max),
                             method='dogbox')

        return int(np.floor(w['x']))
    # -------------------------------------------------------------------------

    # =========================================================================
    # End Class
    # =========================================================================