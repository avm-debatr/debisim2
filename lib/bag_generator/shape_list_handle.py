#!usr/bin/env python

# -----------------------------------------------------------------------------
"""shape_list_handle.py: Module containing functions to handle shape list for
                         DEBISim simulation"""
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

from lib.misc.util import *
from lib.misc.ctlib import *
from skimage.draw import *

"""
-------------------------------------------------------------------------------
Module Description:

This module contains support for creating, handling and managing Shape Lists 
used in DEBISim simulations. The shape list is a data structure used 
to store the meta data for the different objects spawned within the 
ground truth image. The D.S. is a list of dictionaries each describing a single 
object within the image - the metadata within the SL dictionary describes 
(i) geometric specs of the objects, and (ii) material composition of the object. 
The detailed description of a shape list is given below:

* Shape List:
Each object SL dictionary contains the following key parameters:

'label'     - integer label of the object within the ground truth image.
'shape'     - the shape object - current support includes the primitive shapes:
                - 'E' - Ellipsoid
                - 'Y' - Cylinder
                - 'C' - Truncated Cone
                - 'B' - Box
                - 'S' - Deformable Sheet
                - 'M' - a custom shape (a 3D binary mask - read from .stl/fits.gz files) 
                - 'T' - fixed objects(table, tray and bag boundary)
'geom'      - dictionary containing Geometric specifications for the object 
              shape - varies for object 
              shape:
              
              - 'E' - 'center'  - Center co-ordinates [x,y,z] 
                    - 'axes'    - principal axes [a1, a2, a3] 
                    - 'rot'     - rotation (Euler) angles in degrees [t1, t2, t3]
              - 'Y' - 'base'    - Center co-ordinates of the cylinder base
                    - 'apex'    - Center co-ordinates of the cylinder apex
                    - 'radius'  - radius 
              - 'C' - 'base'    - Center co-ordinates of the cone base
                    - 'apex'    - Center co-ordinates of the cone apex
                    - 'radius1' - base radius 
                    - 'radius2' - apex radius
              - 'B' - 'center'  - Center co-ordinates [x,y,z]
                    - 'dim'     - dimensions of the box [a1, a2, a3] 
                    - 'rot'     - rotation (Euler) angles in degrees [t1, t2, t3]
              - 'S' - 'center'  - Center co-ordinates [x,y,z] of the sheet
                    - 'dim'     - dimensions of the undeformed sheet [a1, a2, a3] 
                    - 'rot'     - rotation (Euler) angles in degrees [t1, t2, t3]
              - 'M' - 'center'  - Center co-ordinates [x,y,z] of the shape
                    - 'dim'     - bounding box dimensions of the shape [a1, a2, a3]
                    - 'rot'     - rotation (Euler) angles in degrees [t1, t2, t3]
                    - 'scale'   - scale w.r.t. the original custom shape mask
                    - 'mask'    - 3d object mask
                    - 'src'     - source .stl/.fits.gz file

'material'  - material composing the object
'coeffs'    - dictionary the material coefficient values of the assigned material.
              The dictionary is directly copied from the XCOM Mu Database.
              
'lqd_flag'  - set to True if the object is a liquid-filled container
'lqd_param' - set of liquid parameters - specified only if lqd_flag is set 
              to True:
              'cntr_thickness' - thickness of the liquid container
              'lqd_level'      - level to which the liquid is filled (specified as 
                                 a fraction from 0 to 1)
              'lqd_label'      - liquid label 
              'lqd_material'   - liquid material
              
The shape list thus spawned provides the description for the objects' geometry 
and material composition which can then be used to assign training labels for 
the baggage image samples  when creating a dataset. The simulator in both 
the randomized mode or the user-interactive mode creates and saves a shape list 
for any baggage data it creates. Note the shape list only contains the 
metadata for the objects spawned in the ground truth image - it would be 
difficult to construct ground truth label simply from the shape list. An 
exception to this is through the use of ImageVoxelizer3D(). The module can be 
used to construct ground truth images of phantoms described whose contents are 
described by primitive shapes (ImageVoxelizer3D cannot construct for sheet 
type objects or liquid-filled containers).

Thus, the shape list can be created in one of the three ways:
(i)   randomly generated through use of BaggageShapeGrammar module
(ii)  manually generated by use of the Simulator GUI
(iii) manually generated by feeding numerical values to the SL dictionary (this
      is not recommended but can be useful for generating simulations for 
      phantoms.)

Usage:

The module is mainly used internally in bag_generator scripts to create shape 
lists from the shape grammar/GUI outputs.

> --------------------------------------------------------------------------- >
from shape_list_handle import *

slh = ShapeListHandle()

# Creates a sample shape list for an acrylic ellipsoid pierced with two 
# metal rods

sample_shape_list = [
    slh.create_sim_object('B',
                      [
                        ('center', array([100, 100, 100 ])),
                        ('dim', array([10, 10, 80])),
                        ('rot', array([0, 90, 0]))
                      ],
                      'Al'),
    
        slh.create_sim_object('B',
                      [
                        ('center', array([100, 100, 100 ])),
                        ('dim', array([10, 10, 80])),
                        ('rot', array([90, 0, 0]))
                      ],
                      'Cu'),
        slh.create_sim_object('B',
                      [
                        ('center', array([100, 100, 100 ])),
                        ('dim', array([50, 50, 50])),
                        ('rot', array([0, 0, 45]))
                      ],
                      'acrylic')
]

> --------------------------------------------------------------------------- >

Methods:

__init__()              - constructor
create_sim_object()     - function to create an SL dictionary to add to a 
                          shape list
get_table_img()         - generates an SL dictionary for the scanner table
get_tray_img()          - generates an SL dictionary for the scanner tray
get_bag_boundary()      - generates an SL dictionary for the bag boundary
 
-------------------------------------------------------------------------------
"""


class ShapeListHandle(object):

    default_sim_object = dict(
        shape='Y',
        geom=dict(trans=array([0.0, 0.0, 0.0]),
                  rot=array([0.0, 0.0, 0.0]),
                  dim=array([0.0, 0.0, 0.0])),
        material='air',
        # coeffs={'compton': 0.0, 'pe': 0.0, 'zeff': 0.0, 'mu': 0.0,
        #         'density': 0.0},
        lqd_flag=False,
        lqd_param=None
    )

    ellipsoid_geom = {'center': [], 'axes': [], 'rot': []}
    cone_geom = {'base': [], 'apex': [], 'radius1': [], 'radius2': []}
    cylinder_geom = {'base': [], 'apex': [], 'radius': []}
    box_geom = {'center': [], 'dim': [], 'rot': []}
    sheet_geom = {'center': [], 'dim': [], 'rot': []}
    custom_geom = {'center': [], 'dim': [], 'rot': [],
                   'mask': None, 'src': '', 'scale': 1.0}

    def __init__(self):
        """
        -----------------------------------------------------------------------
        Constructor for the Shape List Handler
        
        :return
        -----------------------------------------------------------------------
        """

        print("Shape List Handler initialized ...")
    # -------------------------------------------------------------------------

    def create_sim_object(self, 
                          geom,
                          shape='Y',
                          obj_material='air',
                          label=0,
                          lqd_flag=False,
                          lqd_param=None
                          ):
        """
        -----------------------------------------------------------------------
        Function to create an object dictionary to add to the shape list. To 
        use this function to create a shape list, check the input arguments 
        carefully as they in a different form then required in the object 
        dictionary.  
        
        :param geom:            geometric specs for object shape
        :param shape:           type of object shape
        :param obj_material:    material assigned to object
        :param label:           the label for the obje in the ground truth 
                                label image
        :param lqd_flag:        set to True if the object is a liquid filled 
                                container
        :param lqd_param:       dictionary containing specs for the liquid 
                                filled container. (See shape List description 
                                for detailed description of the dictionary)
        :return: SL dictionary
        -----------------------------------------------------------------------
        """
        geom      = dict(geom)
        geom_keys = list(geom.keys()).sort()
        
        if shape=='E':
            keys = list(self.ellipsoid_geom.keys()).sort()
            assert  keys == geom_keys
        if shape=='Y':
            keys = list(self.cylinder_geom.keys()).sort()
            assert  keys == geom_keys
        if shape=='C':
            keys = list(self.cone_geom.keys()).sort()
            assert  keys == geom_keys
        if shape=='B':
            keys = list(self.box_geom.keys()).sort()
            assert  keys == geom_keys
        if shape=='S':
            keys = list(self.sheet_geom.keys()).sort()
            assert  keys == geom_keys
        if shape=='M':
            keys = list(self.custom_geom.keys()).sort()
            assert  keys == geom_keys

        return {
            'shape': shape, 'geom': geom, 'label':label,
            'material': obj_material, 'lqd_flag': lqd_flag, 
            'lqd_param': lqd_param
        }
    # -------------------------------------------------------------------------

    def get_bag_background(self,
                           vol_dim,
                           gantry_dia,
                           has_tray=True,
                           materials=None,
                           template=2):
        """
        -----------------------------------------------------------------------
        Produces the cross section image with the bag boundary, the table and
        the tray.

        :param vol_dim:         dimensions of the scanned volume
        :param gantry_dia:      gantry diameter - preferably the the size of
                                the cross section vol_dim[0] or vol_dim[1]
        :param has_tray:        if table+tray are to be added
        :param materials:       materials for the bag objects
        :return:
        -----------------------------------------------------------------------
        """

        # table_dims = (250, 15)

        if materials is None:
            materials = {1:'polyethylene',
                         2:'polystyrene',
                         3:'neoprene'}

        bag_bg_img = zeros((gantry_dia, gantry_dia))
        gantry_x, gantry_y = circle(gantry_dia//2,
                                    gantry_dia//2,
                                    gantry_dia//2*0.9,
                                    (gantry_dia, gantry_dia))
        gantry_cavity = bag_bg_img.copy()
        gantry_cavity[gantry_x, gantry_y] = 1

        sf_list = []

        # Bag Boundary --------------------------------------------------------

        if template is None:

            bb_h, bb_t = gantry_dia//4-1, 3

            img_ctr = [int(vol_dim[0] * 0.8) - bb_h - 42,
                       int(vol_dim[1] / 2),
                       int(vol_dim[2] / 2)]

            bag_in_pts = array([
                [img_ctr[0] - bb_h, img_ctr[1] - bb_h],
                [img_ctr[0] + bb_h, img_ctr[1] - bb_h],
                [img_ctr[0] + bb_h, img_ctr[1] + bb_h],
                [img_ctr[0] - bb_h, img_ctr[1] + bb_h],
                [img_ctr[0] - bb_h - 40, img_ctr[1] + bb_h - 60],
                [img_ctr[0] - bb_h - 40, img_ctr[1] - bb_h + 60]
            ])

            bag_out_pts = array([
                [img_ctr[0] - bb_h - bb_t, img_ctr[1] - bb_h - bb_t],
                [img_ctr[0] + bb_h + bb_t, img_ctr[1] - bb_h - bb_t],
                [img_ctr[0] + bb_h + bb_t, img_ctr[1] + bb_h + bb_t],
                [img_ctr[0] - bb_h - bb_t, img_ctr[1] + bb_h + bb_t],
                [img_ctr[0] - bb_h - bb_t - 40, img_ctr[1] + bb_h + bb_t - 60],
                [img_ctr[0] - bb_h - bb_t - 40, img_ctr[1] - bb_h - bb_t + 60]
            ])

            rr, cc = polygon(bag_out_pts[:, 0], bag_out_pts[:, 1])
            bag_bg_img[rr, cc] = 3

            rr, cc = polygon(bag_in_pts[:, 0], bag_in_pts[:, 1])
            bag_bg_img[rr, cc] = 0

            # ---------------------------------------------------------------------

            # Tray ----------------------------------------------------------------

            # if has_tray:
            #
            bag_base = img_ctr[0] + bb_h + bb_t + 2
            tray_h = 5

            tray_pts = array([
                [bag_base - 120,      img_ctr[1] - (bb_h+20) - 70 - tray_h],
                [bag_base + tray_h,   img_ctr[1] - (bb_h+20)],
                [bag_base + tray_h,   img_ctr[1] + (bb_h+20)],
                [bag_base - 120,      img_ctr[1] + (bb_h+20) + 70 + tray_h],
                [bag_base - 120,      img_ctr[1] + (bb_h+20) + 70 - tray_h],
                [bag_base,            img_ctr[1] + (bb_h+20)],
                [bag_base,            img_ctr[1] - (bb_h+20)],
                [bag_base - 120,      img_ctr[1] - (bb_h+20) - 70 + tray_h]
            ])
            rr, cc = polygon(tray_pts[:, 0], tray_pts[:, 1])
            bag_bg_img[rr, cc] = 2
            # -----------------------------------------------------------------

            # Table -----------------------------------------------------------
            tray_base = bag_base + 7
            table_t, table_w = 30, 150

            bag_bg_img[tray_base:tray_base+table_t,
                       img_ctr[1]-table_w:img_ctr[1]+table_w] = 1
            # -----------------------------------------------------------------

            bag_bound, bag_mask = None, None
        # ---------------------------------------------------------------------

        else:
            # load the saved bag templates ------------------------------------
            template_npz = load(os.path.join(INC_DIR,
                                             'bags',
                                             'bag_%i_template.npz'%template))
            bag_bg_img = template_npz['bag']
            bag_bound = template_npz['bag_bound']
            bag_mask = template_npz['bag_mask']

            # find the dims and extents of bag_mask for adjusting to the
            # scanner isocenter

            nz = bag_mask.nonzero()
            bbox = nz[0].min(), nz[0].max(), nz[1].min(), nz[1].max()
            extents = bbox[1]-bbox[0], bbox[3]-bbox[2]+5

            bb_h = max(extents)//2
            bb_t = 10
            img_ctr = [bbox[0]+extents[0]//2,
                       bbox[2]+extents[1]//2,
                       int(vol_dim[2] / 2)]

            table_t, table_w = 30, 150

        bb_dict = self.create_sim_object([
            ('center', array(img_ctr)),
            ('dim', array([2 * bb_h, 2 * bb_h, vol_dim[2]])),
            ('rot', array([0, 0, 0]))],
            'T',
            materials[3],
            3
        )

        sf_list.append(bb_dict)

        tray_dict = self.create_sim_object([
            ('center', array(img_ctr)),
            ('dim', array([5, 2 * bb_h, vol_dim[2]])),
            ('rot', array([0, 0, 0]))],
            'T',
            materials[2],
            2
        )

        table_dict = self.create_sim_object([
            ('center', array(img_ctr)),
            ('dim', array([table_t, table_w, vol_dim[2]])),
            ('rot', array([0, 0, 0]))],
            'T',
            materials[1],
            1
        )

        sf_list.append(tray_dict)
        sf_list.append(table_dict)

        return img_ctr, (bb_h, bb_t), sf_list, bag_bg_img, \
               gantry_cavity, bag_bound, bag_mask
    # -------------------------------------------------------------------------

    def get_table_img(self,
                      orig_dim,
                      table_thickness=15, 
                      material='polyethylene'):
        """
        -----------------------------------------------------------------------
        Function to create a default scanner table object in the image.
        
        :param image_dim:           CT image dimension
        :param table_thickness:     thickness of the scanner table
        :param material:            material assigned to the table
        :return tray_img:           a 2D cross section image of the table
        :return table_dict          an SL dictionary for the table
        -----------------------------------------------------------------------
        """

        image_dim = (664,664,orig_dim[2])

        table_width = 250
        table_img = zeros((image_dim[0], image_dim[1]))
        
        img_ctr  = int(image_dim[0]*0.8), image_dim[1]//2
        table_img[
        img_ctr[0] - table_thickness:img_ctr[0] + table_thickness,
        img_ctr[1] - table_width:img_ctr[1] + table_width
        ] = 1

        table_img = table_img[
                   image_dim[0]//2-orig_dim[0]//2:image_dim[0]//2+orig_dim[0]//2,
                   image_dim[1]//2-orig_dim[1]//2:image_dim[1]//2+orig_dim[1]//2]

        table_dict = \
        self.create_sim_object(
                          [('center', array([int(image_dim[0] * 0.8),
                                             int(image_dim[1] / 2),
                                             int(image_dim[2] / 2)])),
                           ('dim', array([2*table_thickness,
                                          2*table_width,
                                          int(image_dim[2])])),
                           ('rot', array([0, 0, 0]))],
                          'T',
                          material,
                          label=1)
    
        return table_img, table_dict
    # -------------------------------------------------------------------------

    def get_tray_img(self, orig_dim, material='polystyrene'):
        """
        -----------------------------------------------------------------------
        Function to create a default scanner tray object in the image.
        
        :param orig_dim:           CT image dimension
        :param material:            material assigned to the tray
        :return tray_img:           a 2D cross section image of the tray
        :return tray_dict          an SL dictionary for the tray
        -----------------------------------------------------------------------
        """
    
        label = 2

        image_dim = (664,664,orig_dim[2])

        tray_pts = array([
            [int(image_dim[0] * 0.8) - 33 - 125, int(image_dim[1] / 2) - 185 - 70 - 5],
            [int(image_dim[0] * 0.8) - 33 + 5, int(image_dim[1] / 2) - 185],
            [int(image_dim[0] * 0.8) - 33 + 5, int(image_dim[1] / 2) + 185],
            [int(image_dim[0] * 0.8) - 33 - 125, int(image_dim[1] / 2) + 185 + 70 + 5],
            [int(image_dim[0] * 0.8) - 33 - 125, int(image_dim[1] / 2) + 185 + 70 - 5],
            [int(image_dim[0] * 0.8) - 33 - 5, int(image_dim[1] / 2) + 185],
            [int(image_dim[0] * 0.8) - 33 - 5, int(image_dim[1] / 2) - 185],
            [int(image_dim[0] * 0.8) - 33 - 125, int(image_dim[1] / 2) - 185 - 70 + 5]
        ])
    
        rr, cc = polygon(tray_pts[:,0], tray_pts[:,1])
    
        tray_img =   zeros((image_dim[0], image_dim[1]))
        tray_img[rr, cc] = label

        tray_img = tray_img[
                   image_dim[0]//2-orig_dim[0]//2:image_dim[0]//2+orig_dim[0]//2,
                   image_dim[1]//2-orig_dim[1]//2:image_dim[1]//2+orig_dim[1]//2]
    
        table_dict = \
            self.create_sim_object(
                              [('center', array([int(orig_dim[0] * 0.8)-97,
                                                 int(orig_dim[1] // 2),
                                                 int(orig_dim[2] // 2)])),
                               ('dim', array([10, 370, int(orig_dim[2])])),
                               ('rot', array([0, 0, 0]))],
                              'T',
                              material,
                              label)
    
        return tray_img, table_dict
    # -------------------------------------------------------------------------

    def get_bag_boundary(self, orig_dim, bag_dim, material = 'neoprene'):
        """
        -----------------------------------------------------------------------
        Function to create a default scanner tray object in the image.

        :param image_dim:           CT image dimension
        :param material:            material assigned to the tray
        :param bag_dim:             dimensions of the bag:
                                    (bb_h, bag_thickness)
        :return bb_img:             2D cross section of the bag boundary
        :return bag_dict:           SL dictionary for the bag
        :return img_ctr:            center of the bag
        -----------------------------------------------------------------------
        """

        label = 3

        image_dim = (664,664,orig_dim[2])
        bb_size, bb_thickness = bag_dim
    
        img_ctr = [int(image_dim[0] * 0.8) - bb_size-42, \
                   int(image_dim[1] / 2), \
                   int(image_dim[2] / 2)]
    
        bag_in_pts = array([
            [img_ctr[0] - bb_size, img_ctr[1] - bb_size],
            [img_ctr[0] + bb_size, img_ctr[1] - bb_size],
            [img_ctr[0] + bb_size, img_ctr[1] + bb_size],
            [img_ctr[0] - bb_size, img_ctr[1] + bb_size],
            [img_ctr[0] - bb_size-40, img_ctr[1] + bb_size-60],
            [img_ctr[0] - bb_size-40, img_ctr[1] - bb_size+60]
        ])
    
        bag_out_pts = array([
            [img_ctr[0] - bb_size - bb_thickness, img_ctr[1] - bb_size - bb_thickness],
            [img_ctr[0] + bb_size + bb_thickness, img_ctr[1] - bb_size - bb_thickness],
            [img_ctr[0] + bb_size + bb_thickness, img_ctr[1] + bb_size + bb_thickness],
            [img_ctr[0] - bb_size - bb_thickness, img_ctr[1] + bb_size + bb_thickness],
            [img_ctr[0] - bb_size - bb_thickness - 40, img_ctr[1] + bb_size + bb_thickness - 60],
            [img_ctr[0] - bb_size - bb_thickness - 40, img_ctr[1] - bb_size - bb_thickness + 60]
        ])
    
        bb_img =   zeros((image_dim[0], image_dim[1]))
    
        rr, cc = polygon(bag_out_pts[:,0], bag_out_pts[:,1])
        bb_img[rr, cc] = label
    
        rr, cc = polygon(bag_in_pts[:,0], bag_in_pts[:,1])
        bb_img[rr, cc] = 0

        bb_img = bb_img[
                   image_dim[0]//2-orig_dim[0]//2:image_dim[0]//2+orig_dim[0]//2,
                   image_dim[1]//2-orig_dim[1]//2:image_dim[1]//2+orig_dim[1]//2]

        img_ctr[0] = img_ctr[0] - (image_dim[0]//2 - orig_dim[0]//2)
        img_ctr[1] = img_ctr[1] - (image_dim[1]//2 - orig_dim[1]//2)

        table_dict = \
            self.create_sim_object(
                              [('center', array([int(image_dim[0] * 0.8)-97,
                                                 int(image_dim[1] / 2),
                                                 int(image_dim[2] / 2)])),
                               ('dim', array([10, 370, int(image_dim[2])])),
                               ('rot', array([0, 0, 0]))],
                              'T',
                              material,
                              label)
    
        return bb_img, table_dict, img_ctr
    # -------------------------------------------------------------------------

    def _insert(self, slh, obj_dict, label):
        """
        -----------------------------------------------------------------------
        Insert an object dictionary into the shape list at given label

        :param slh:         Shape List
        :param obj_dict:    Object Dictionary
        :param label:       Label at which the dictionary is to be inserted
        :return: modified shape list
        -----------------------------------------------------------------------
        """
        obj_dict['label'] = label
        num_obj = len(slh)

        if label>num_obj:
            slh.append(obj_dict)
        else:
            for l in range(num_obj, label, -1):
                for k in range(len(slh)):
                    if slh[k]['label']==l: slh[k]['label'] += 1
                    if slh[k]['lqd_flag']: slh[k]['lqd_param']['lqd_label'] += 1

            slh.append(obj_dict)

        return slh
    # -------------------------------------------------------------------------

    def _delete(self, slh, label):
        """
        -----------------------------------------------------------------------
        Delete an object dictionary from shape list.

        :param slh:         Shape List
        :param label:       Label at which the dictionary is to be deleted
        :return: modified shape list
        -----------------------------------------------------------------------
        """

        num_obj = len(slh)

        if label > num_obj:
            print("Label does not exist!")

        else:
            for k in range(len(slh)):
                if slh[k]['label'] == label: slh.remove(slh[k])

            for l in range(num_obj, label, -1):
                for k in range(len(slh)):
                    if slh[k]['label'] == l: slh[k]['label'] -= 1
                    if slh[k]['lqd_flag']: slh[k]['lqd_param']['lqd_label'] -= 1

        return slh
    # -------------------------------------------------------------------------

    def _search(self, slh, label):
        """
        -----------------------------------------------------------------------
        Search for a given label

        :param slh:     Shape List
        :param label:   Label to be searched
        :return:
        -----------------------------------------------------------------------
        """
        for k in range(len(slh)):
            if slh[k]['label'] == label: return slh[k]

        print("Label does not exist!")

        return None
    # -------------------------------------------------------------------------
