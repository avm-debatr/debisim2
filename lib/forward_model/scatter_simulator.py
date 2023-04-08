#!/usr/bin/env python

# ------------------------------------------------------------------------------
"""scatter_simulator.py: Module for add second-order scatter to simulated scan
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

from lib.forward_model.mu_database_handler import *
from lib.forward_model.scanner_template import *
from tqdm import tqdm


class ScatterSimulator(object):

    def __init__(self,
                 scanner,
                 sf_list):
        """
        -----------------------------------------------------------------------


        :param scanner:
        :param gt_image:
        :param sf_list:
        -----------------------------------------------------------------------
        """

        self.mu = MuDatabaseHandler()

        self.scanner = scanner
        self.sf_list = sf_list.copy()
        self.metal_dict = {x['label']: x['material']
                            for x in self.sf_list
                            if x['material'] in self.mu.metals
                            }
        self.metal_labels = list(self.metal_dict.keys())

        self.scanner_proj_geom = astra.create_proj_geom(
            'parallel',
            self.scanner.machine_geometry['det_spacing_y'],
            self.scanner.machine_geometry['det_col_count'],
            self.scanner.recon_geometry['angles']
        )

        self.scanner_proj_geom_vec = astra.geom_2vec(self.scanner_proj_geom)
        self.dcs_vec = vectorize(self.dcs)
    # -------------------------------------------------------------------------

    def set_scatter_calculator(self,
                               gt_image):
        """
        -----------------------------------------------------------------------

        :param gt_image:
        :return:
        -----------------------------------------------------------------------
        """

        metal_mask = zeros_like(gt_image)

        for i in self.metal_labels:
            metal_mask[gt_image==i] = 1

        self.metal_voxels = array(metal_mask.nonzero()).T
        self.metal_mask   = metal_mask

        m_labels = gt_image[metal_mask.nonzero()]
        self.metal_voxels = column_stack((self.metal_voxels, m_labels))

        self.img_ctr = array([gt_image.shape[0] / 2,
                              gt_image.shape[1] / 2])

        self.metal_voxels[:, :2] = self.metal_voxels[:, :2] - self.img_ctr

        self.det_coords = array([
            [self.scanner.machine_geometry['gantry_diameter'] / 2,
             (i - self.scanner.machine_geometry['det_col_count'] / 2) * \
             self.scanner.machine_geometry['det_spacing_y']]
            for i in range(self.scanner.machine_geometry['det_col_count'])]
        )

        self.det_angles = array([
            arctan2(self.det_coords[:,1]-x[1], self.det_coords[:,0]-x[0])
            for x in self.metal_voxels]
        )

    # -------------------------------------------------------------------------

    def get_scatter_projections(self,
                                atten_image,
                                e,
                                xray_specs,
                                spectrum,
                                chl=1):
        """
        -----------------------------------------------------------------------

        :param atten_image:
        :param e:
        :param x_ray_specs:
        :param spectrum:
        :param chl:
        :return:
        -----------------------------------------------------------------------
        """

        scatter_sino = zeros(shape=(self.scanner.machine_geometry['det_col_count'],
                                    self.scanner.recon_params['n_views'])
                             )
        n_a = 6.02214e23
        vol_cc = self.scanner.machine_geometry['det_spacing_y'] * \
                 self.scanner.machine_geometry['det_spacing_x']

        for i, mcoord in enumerate(self.metal_voxels):

            metal_mm = self.mu.molar_masses[self.metal_dict[mcoord[2]]]  # 26.1
            d = self.mu.material(self.metal_dict[mcoord[2]], 'density')
            v_factor = (d * vol_cc * 1e4) / metal_mm * n_a

            # create projn_geometry that calculate line integrals
            # upto voxel for all views

            self.src_vox_geom = SourceToVoxelGeometry(self.scanner,
                                                      mcoord)

            vox_log_attenuations = self.src_vox_geom.get_log_attenuations(atten_image)
            vox_photon_vals = \
                xray_specs['dosage'][chl-1] * spectrum[e - 10] * np.exp(-vox_log_attenuations)
            vox_det_geom = VoxelToDetectorGeometry(self.scanner,
                                                   mcoord)
            scatter_line_integrals = vox_det_geom.get_line_integrals(atten_image)
            scatter_projn_val = vox_photon_vals[:, newaxis]*scatter_line_integrals

            self.rot_det_coords = array([
                [self.det_coords[:, 1]*cos(t) - self.det_coords[:, 0]*sin(t),
                 self.det_coords[:, 1]*cos(t) + self.det_coords[:, 0]*cos(t)]
                for t in self.scanner.recon_geometry['angles']
            ])
            self.rot_det_angles = array([ arctan2(
                self.det_coords[:, 1]*cos(t) - self.det_coords[:, 0]*sin(t) - mcoord[0],
                self.det_coords[:, 1]*cos(t) + self.det_coords[:, 0]*cos(t) - mcoord[1]
            )
                for t in self.scanner.recon_geometry['angles']
            ])
            scatter_projn_val = self.dcs(
                self.rot_det_angles,
                e*ones_like(self.rot_det_angles)
            )*scatter_projn_val*v_factor

            scatter_sino = scatter_sino + scatter_projn_val.T

        return scatter_sino
    # -------------------------------------------------------------------------

    def dcs(self, thetas, e_in):
        """
        -----------------------------------------------------------------------

        :param theta:
        :param e:
        :return:
        -----------------------------------------------------------------------
        """

        c   = 2.98e8
        m_e = 511e3/(c**2)#9.109e-31
        r_e = 0.386e-12

        e_in = e_in*1e3
        kappa = e_in/(m_e*(c**2))
        e_lambda = 1/(1 + kappa*(1-cos(thetas)))

        dcs_out =0.5*(r_e**2)
        dcs_out *= e_lambda**2
        dcs_out *= (e_lambda + 1/e_lambda - sin(thetas)**2)

        return dcs_out
    # -------------------------------------------------------------------------


class SourceToVoxelGeometry(object):

    def __init__(self, scanner, metal_voxel):


        g = dict()
        self.scanner = scanner
        self.metal_voxel = metal_voxel
        g['n_views'] = self.scanner.recon_params['n_views']

        view_range = self.scanner.recon_params['view_range']

        theta = linspace(- view_range/2,
                           view_range/2,
                          self.scanner.recon_params['n_views'],
                          endpoint=False)[::-1]
        g['angles'] = deg2rad(mod(theta, 360))
        g['src_angles'] = g['angles']
        g['det_angles'] = g['angles'] + pi

        g_dia = self.scanner.machine_geometry['gantry_diameter']
        vecs = zeros((g['n_views'], 6), dtype=float32)

        # Source Center
        vecs[:,0] = sin(g['src_angles']) * g_dia/2
        vecs[:,1] = -cos(g['src_angles']) * g_dia/2

        # Detector Values
        vecs[:,2] = self.metal_voxel[0]
        vecs[:,3] = self.metal_voxel[1]

        # Ortho vector from detector 0 to detector 1
        vecs[:,4] = \
            - cos(g['det_angles'])* self.scanner.machine_geometry['det_spacing_y']
        vecs[:,5] = \
            - cos(g['det_angles'])* self.scanner.machine_geometry['det_spacing_y']

        self.proj_geom = astra.create_proj_geom('fanflat_vec',
                                                1,
                                                vecs)
    # -------------------------------------------------------------------------

    def get_log_attenuations(self, atten_image):

        im_geom = astra.create_vol_geom(*atten_image.shape)
        proj_id = astra.create_projector('cuda', self.proj_geom, im_geom)
        # self.w_2d = astra.OpTomo(self.proj2d_id)

        # rec = self.w_2d.reconstruct('FBP_CUDA', sino)

        vol_geom = astra.create_vol_geom(*atten_image.shape)
        # projn_id = astra.create_projector()
        out_id, projn_data = astra.create_sino(atten_image,
                                       proj_id,
                                       vol_geom)
        astra.data2d.delete(proj_id)
        astra.data2d.delete(out_id)

        return projn_data.squeeze()
    # -------------------------------------------------------------------------


class VoxelToDetectorGeometry(object):

    def __init__(self, scanner, metal_voxel):
        """
        -----------------------------------------------------------------------

        :param scanner:
        :param metal_voxel:
        -----------------------------------------------------------------------
        """
        g = dict()
        self.scanner = scanner
        self.metal_voxel = metal_voxel
        g['n_views'] = self.scanner.recon_params['n_views']

        view_range = self.scanner.recon_params['view_range']

        theta = linspace(- view_range/2,
                           view_range/2,
                          self.scanner.recon_params['n_views'],
                          endpoint=False)[::-1]
        g['angles'] = deg2rad(mod(theta, 360))
        g['src_angles'] = g['angles']
        g['det_angles'] = g['angles'] + pi

        g_dia = self.scanner.machine_geometry['gantry_diameter']
        vecs = zeros((g['n_views'], 6), dtype=float32)

        # Source Center
        vecs[:,0] = self.metal_voxel[0]
        vecs[:,1] = self.metal_voxel[1]

        # Detector Values
        vecs[:,2] = sin(g['det_angles']) * g_dia/2
        vecs[:,3] = -cos(g['det_angles']) * g_dia/2

        # Ortho vector from detector 0 to detector 1
        vecs[:,4] = \
            - cos(g['det_angles'])* self.scanner.machine_geometry['det_spacing_y']
        vecs[:,5] = \
            - cos(g['det_angles'])* self.scanner.machine_geometry['det_spacing_y']

        self.proj_geom = astra.create_proj_geom(
            'fanflat_vec',
             self.scanner.machine_geometry['det_col_count'],
             vecs)
    # -------------------------------------------------------------------------

    def get_line_integrals(self, atten_image):

        im_geom = astra.create_vol_geom(*atten_image.shape)
        proj_id = astra.create_projector('cuda', self.proj_geom, im_geom)
        # self.w_2d = astra.OpTomo(self.proj2d_id)

        # rec = self.w_2d.reconstruct('FBP_CUDA', sino)

        vol_geom = astra.create_vol_geom(*atten_image.shape)
        # projn_id = astra.create_projector()
        out_id, projn_data = astra.create_sino(atten_image,
                                       proj_id,
                                       vol_geom)
        astra.data2d.delete(proj_id)
        astra.data2d.delete(out_id)

        return np.exp(-projn_data)
    # -------------------------------------------------------------------------


if __name__=="__main__":

    s = ScannerTemplate(
        geometry='parallel',
        scan='circular',
        machine_dict=default_scanner_parallel.machine_geometry,
        recon='fbp',
        recon_dict=default_scanner_parallel.recon_params,
        pscale=1.0
    )

    s.set_recon_geometry()

    d = ScatterSimulator(s, [])

    t = deg2rad(linspace(0,360, 360))

    dout = d.dcs(t, 30)

