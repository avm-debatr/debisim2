#!/usr/bin/env python

# ------------------------------------------------------------------------------
"""
image_voxelizer_3d: Module for voxelizing 3D images from primitive shape
                  descriptions.
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
Module Description:

This module contains the set of functions to facilitate the voxelization of 
ground truth images in DEBISim as described by a shape list. The shape 
list is a data structure used in DEBISim to store the ground truth meta data 
for the different objects spawned within a ground truth image. The D.S. is a 
list of dictionaries each describing a single object within the image. For 
detailed information about shape lists, see the module ShapeListHandle.
The module ImageVoxelizer3D reads in a shape list for a given ground truth 
image and generates a corresponding 3D label image. The module is instrumental 
for loading and handling saved ground truth data from the shape list files. 

Usage:

The class for ImageVoxelizer3D is initialized with an input shape list and 
optionally, with the 3D image dimensions (default dimensions are (664,664,350)).
Running the method self.voxelize_3d_image() then generates a 3D label image 
wherein each label corresponds to the shape list label of te object.

> ---------------------------------------------------------------------------- >
from image_voxelizer_3D import *
from mu_database_handler import *
from util import *
import pickle, torch

file_name = './examples/phantom_shape_lists/battelle_phantom.pyc'
with open(file_name, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    f.close()

Mu = MuDatabaseHandler()
shape_list = [data[int(x)] for x in shape_list.keys()]
voxelizer  = ImageVoxelizer3D(shape_list, imgshape=(664,664,886),
                              mu_dbase=Mu)

# The ground turth label image and corresponding compton image are outputted as 
# torch.Tensor type
gt_image, compton_image = voxelizer.voxelize_3d_image()
slide_show(gt_image.cpu().numpy())
> ---------------------------------------------------------------------------- >

Note:
The simulator program saves the ground truth image as a pickle .pyc file - when 
read using the pickle.load(), the file is loaded as a dictionary with the object 
labels as the keys. The read data can be converted to a shape list with the 
following code:

> ---------------------------------------------------------------------------- >
file_name = './include/examples/phantom_shape_lists/acr_phantom.pyc'
with open(file_name, 'rb') as f:
    data = pickle.load(f, enoding='latin1')
    f.close()

shape_list = [data[int(x)] for x in shape_list.keys()]
> ---------------------------------------------------------------------------- >

* Methods:

__init__ ()                 - Constructor
voxelize_3d_image()         - function to create the 3D voxelized image
_draw_ellipsoid()           - draw an ellipsoid from its geometric specs
_draw_cylinder()            - draw a  cylinder  from its geometric specs
_draw_truncated_cone()      - draw a  cone      from its geometric specs
_draw_box()                 - draw a  box       from its geometric specs

* Attributes:

imgshape                    - dimensions of the image    
sf_list                     - the input shape list 

--------------------------------------------------------------------------------
"""


from skimage.draw import circle
import scipy.ndimage as sptx

from lib.forward_model.mu_database_handler import *
import torch


class ImageVoxelizer3D(object):

    def __init__(self,
                 sf_list,
                 imgshape=(664, 664, 450),
                 mu_dbase=None
                 ):
        """
        ------------------------------------------------------------------------
        Constructor for a 3D Image Voxelizer. To initialize, one must provide
        a shape list, i.e., a list of shape objects wherein each shape is
        defined as a dictionary with keys describing geometry and material.

        :param sf_list:         sf file data
        :param imgshape:        dimensions of ct image (height, width, slices)
        :param mu_dbase:        current instance of MuDatabaseHandler
        ------------------------------------------------------------------------
        """

        print("\nInitializing Phantom Voxelizer ...")

        self.sf_list = sf_list
        self.imgshape = tuple(  array(imgshape).astype(  uint16))

        if mu_dbase is None:
            self.Mu = MuDatabaseHandler()
        else:
            self.Mu = mu_dbase

        print("SF list Obtained - run voxelize_phantom() to generate the 3D " 
              "phantom image\n")
    # --------------------------------------------------------------------------

    def voxelize_3d_image(self):
        """
        ------------------------------------------------------------------------
        Function to voxelize the 3D volumetric image using the input shape
        list, self.sf_list.

        :return:
        ------------------------------------------------------------------------
        """

        t0 = time.time()
        print("="*80)
        print("Voxelization started at ", time.strftime('%m-%d-%Y %H:%M:%S',
                                                    time.localtime()))

        ref_image  = torch.zeros(self.imgshape)

        for k, sf_obj in enumerate(self.sf_list):

            print("Current Object:\t", sf_obj['material'], sf_obj['shape'])

            if sf_obj['shape']=='E':
                center = tuple(sf_obj['geom']['center'])
                axes   = tuple(sf_obj['geom']['axes']*2)
                theta  = tuple(sf_obj['geom']['rot'])
                obj_mask = self._draw_ellipsoid(center, axes, theta)

            elif sf_obj['shape']=='B':
                center = tuple(sf_obj['geom']['center'])
                dims   = tuple(sf_obj['geom']['dim'])
                theta  = tuple(sf_obj['geom']['rot'])
                obj_mask = self._draw_box(center, dims, theta)

            elif sf_obj['shape']=='Y':
                center1 = tuple(sf_obj['geom']['base'])
                center2 = tuple(sf_obj['geom']['apex'])
                radius  = int(sf_obj['geom']['radius'])
                obj_mask = self._draw_cylinder(center1, center2, radius)

            elif sf_obj['shape']=='C':
                center1 = tuple(sf_obj['geom']['base'])
                center2 = tuple(sf_obj['geom']['apex'])
                radius1, radius2   = int(sf_obj['geom']['radius1']), \
                                     int(sf_obj['geom']['radius2'])
                obj_mask = self._draw_truncated_cone(center1,
                                                     center2,
                                                     radius1,
                                                     radius2)

            obj_mask  = torch.as_tensor(obj_mask,
                                        dtype=torch.float)

            if sf_obj['material']=='air':
                ref_image = torch.where(obj_mask == 1,
                                        torch.Tensor([0.]),
                                        ref_image)

            else:
                ref_image = torch.where(obj_mask == 1,
                                        torch.Tensor([k+1]),
                                        ref_image)
            torch.cuda.empty_cache()
            print("Object Voxelized ...\n")

        compton_image =  torch.zeros_like(ref_image)

        for k,sf_obj in enumerate(self.sf_list):

            if sf_obj['material']=='air':
                compton_image = torch.where(ref_image==(k+1),
                                            torch.Tensor([0.0]),
                                            compton_image)
            else:
                compton_image = torch.where(ref_image==(k+1),
                                            torch.Tensor([self.Mu.material(sf_obj['material'], 'compton')]),
                                            compton_image)

        print("Voxelization completed at ", time.strftime('%m-%d-%Y %H:%M:%S',
                                            time.localtime()))

        print("Time Taken:\t", time.time()-t0)
        print("="*80, '\n')

        return compton_image, ref_image
    # --------------------------------------------------------------------------

    def _draw_ellipsoid(self, center, axes, theta):
        """
        ------------------------------------------------------------------------
        Generate a 3D ellipsoid.

        :param center:  center of ellipsoid
        :param axes:    length of principal axes.
        :param theta:   orientations
        :return: bin_image: 3D binary image output
        ------------------------------------------------------------------------
        """

        scale       = 100
        rshape      = (scale, scale, scale)
        mgrid       =   np.lib.index_tricks.nd_grid()
        cshape      =   asarray(1j) * rshape
        coords      = mgrid[-1:1:cshape[0], -1:1:cshape[1], -1:1:cshape[2]]

        val         =   square(  asarray(coords))
        mask        = val.sum(axis=0)<=1
        mask.resize(rshape)

        zoom_scale = (axes[0]*1.0/scale, axes[1]*1.0/scale,axes[2]*1.0/scale)
        mask = sptx.zoom(mask, zoom=zoom_scale, order=0)

        mask = sptx.rotate(mask, theta[0], axes=(1,0), order=0)
        mask = sptx.rotate(mask, theta[1], axes=(2,1), order=0)
        mask = sptx.rotate(mask, theta[2], axes=(0,2), order=0)

        bx, by, bz = self.imgshape
        mx, my, mz = mask.shape
        buf_image =   zeros((max(mx,bx), max(my,by), max(mz,bz)))
        buf_image[:mx, :my, :mz] = mask

        buf_image = sptx.shift(buf_image,
                             (center[0]-mx/2,
                              center[1]-my/2,
                              center[2]-mz/2),
                              order=0)
        bin_image = buf_image[:bx, :by, :bz]
        return bin_image
    # --------------------------------------------------------------------------

    def _draw_cylinder(self, center1, center2, radius):
        """
        ------------------------------------------------------------------------
        Generate a 3D Cylinder.

        :param center1:  center of first disc
        :param center2:  center of last disc
        :param radius:   radius of cylinder
        :return: bin_image: 3D binary image output
        ------------------------------------------------------------------------
        """

        ctr1 =   asarray(center1)
        ctr2 =   asarray(center2)
        dist =   linalg.norm(ctr2-ctr1)
        ref_image =   zeros((2*radius+1, 2*radius+1, 2*int(dist)))

        cyl_circle =   zeros((2*radius+1, 2*radius+1))
        rr, cc = circle(radius + 1, radius + 1, radius)
        cyl_circle[rr, cc] = 1

        for i in range(int(dist),2*int(dist)):
            ref_image[:, :, i] = cyl_circle

        t   =   arccos((ctr2[2]-ctr1[2])/dist)
        mask = sptx.rotate(ref_image,   rad2deg(t), axes=(2, 0), order=0)

        diff = abs(ctr2-ctr1)
        if diff[0]<0.01 and diff[1]<0.01:
            pass
        elif diff[0]<0.01 and diff[1]>0.01:
            mask = sptx.rotate(mask, 90, axes=(1, 0), order=0)
        else:
            rho =   arccos((ctr2[1] - ctr1[1]) / (ctr2[0] - ctr1[0]))
            mask = sptx.rotate(mask,   deg2rad(rho), axes=(1, 0), order=0)

        mx, my, mz = mask.shape
        bx, by, bz = self.imgshape
        buf_image =   zeros((max(mx,bx), max(my,by), max(mz,bz)))
        buf_image[:mx, :my, :mz] = mask

        buf_image = sptx.shift(buf_image,
                             (center1[0]-mx/2,
                              center1[1]-my/2,
                              center1[2]-mz/2),
                             order=0)

        bin_image = buf_image[:bx, :by, :bz]
        return  bin_image
    # --------------------------------------------------------------------------

    def _draw_truncated_cone(self, center1, center2, radius1, radius2):
        """
        ------------------------------------------------------------------------
        Generate a 3D Truncated Cone.

        :param center1:  center of first disc
        :param center2:  center of last disc
        :param radius1:   radius of 1st disc
        :param radius2:   radius of last disc
        :return: bin_image: 3D binary image output
        ------------------------------------------------------------------------
        """

        ctr1 =   asarray(center1)
        ctr2 =   asarray(center2)
        dist =   linalg.norm(ctr2-ctr1)
        rd = abs(radius1-radius2)

        ang =   arcsin(rd/dist)

        if radius1<=radius2:
            ref_image =   zeros((2*radius2+1, 2*radius2+1, 2*int(dist)))

            for i in range(int(dist),2*int(dist)):
                if   isnan(  tan(ang)) or   isnan(dist) or   isnan(ang):
                    circ_rad = radius2
                else:
                    circ_rad = int(radius1+(i-int(dist))*  tan(ang))

                if circ_rad >radius2:
                    circ_rad = radius2

                cyl_circle =   zeros((2 * radius2 + 1, 2 * radius2 + 1))
                rr, cc = circle(radius2 + 1, radius2 + 1, circ_rad)
                cyl_circle[rr, cc] = 1
                ref_image[:, :, i] = cyl_circle
        else:
            ref_image =   zeros((2 * radius1 + 1,
                                  2 * radius1 + 1,
                                  2 * int(dist)))

            for i in range(int(dist), 2 * int(dist)):
                if   isnan(  tan(ang)) or   isnan(dist) or   isnan(ang):
                    circ_rad = radius2
                else:
                    circ_rad = int(radius1 -   floor(i-int(dist))*  tan(ang))

                if circ_rad >radius1:
                    circ_rad = radius1
                cyl_circle =   zeros((2 * radius1 + 1, 2 * radius1 + 1))
                rr, cc = circle(radius1 + 1, radius1 + 1, circ_rad)
                cyl_circle[rr, cc] = 1
                ref_image[:, :, i] = cyl_circle

        t   =   arccos((ctr2[2]-ctr1[2])/dist)

        mask = sptx.rotate(ref_image,   rad2deg(t), axes=(2, 0), order=0)

        diff = abs(ctr2-ctr1)
        if diff[0]<0.01 and diff[1]<0.01:
            pass
        elif diff[0]<0.01 and diff[1]>0.01:
            mask = sptx.rotate(mask, 90, axes=(1, 0), order=0)
        else:
            rho =   arccos((ctr2[1] - ctr1[1]) / (ctr2[0] - ctr1[0]))
            mask = sptx.rotate(mask,   rad2deg(rho), axes=(1, 0), order=0)


        bx, by, bz = self.imgshape
        mx, my, mz = mask.shape
        buf_image =   zeros((max(mx,bx), max(my,by), max(mz,bz)))
        buf_image[:mx, :my, :mz] = mask

        buf_image = sptx.shift(buf_image,
                             (center1[0]-mx/2,
                              center1[1]-my/2,
                              center1[2]-mz/2),
                             order=0)

        bin_image = buf_image[:bx, :by, :bz]
        # print("Cone Generated")
        return bin_image
    # --------------------------------------------------------------------------

    def _draw_box(self, center, dims, theta):
        """
        ------------------------------------------------------------------------
        Generate a 3D box.

        :param center:  center of box
        :param dims:    dimensions.
        :param theta:   orientations
        :return: bin_image: 3D binary image output
        ------------------------------------------------------------------------
        """

        scale       = 100
        rshape      = (scale, scale, scale)
        mgrid       =   np.lib.index_tricks.nd_grid()
        cshape      =   asarray(1j) * rshape
        coords      = mgrid[-1:1:cshape[0], -1:1:cshape[1], -1:1:cshape[2]]

        mask        =   ones(shape=coords[0].shape)
        mask.resize(rshape)

        zoom_scale = (dims[0]*1.0/scale, dims[1]*1.0/scale, dims[2]*1.0/scale)
        mask = sptx.zoom(mask, zoom=zoom_scale, order=0)

        mask = sptx.rotate(mask, theta[0], axes=(1,0), order=0)
        mask = sptx.rotate(mask, theta[1], axes=(2,1), order=0)
        mask = sptx.rotate(mask, theta[2], axes=(0,2), order=0)

        bx, by, bz = self.imgshape
        mx, my, mz = mask.shape
        buf_image =   zeros((max(mx,bx), max(my,by), max(mz,bz)))
        buf_image[:mx, :my, :mz] = mask
        buf_image = sptx.shift(buf_image,
                             (center[0]-mx/2,
                              center[1]-my/2,
                              center[2]-mz/2),
                             order=0)
        bin_image = buf_image[:bx, :by, :bz]
        # print("Box Generated")
        return bin_image
    # --------------------------------------------------------------------------

# ==============================================================================
# Class Ends
# ==============================================================================
