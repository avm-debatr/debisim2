#!/usr/bin/env python

# ------------------------------------------------------------------------------
"""
mu_database_handler.py: Module to handle the NIST XCOM database for generating and
                      managing attenuation response data for elements, compounds
                      and target materials.
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

This module is responsible for handling the NIST XCOM database for generating 
and managing attenuation response data for elements, compounds and target 
materials. The attenuation curve in the CT simulation for DEBISim is derived 
from the NIST XCOM Photon Cross Sections Database [1] which is based on a 
FORTRAN script that handles the attenuation data. The modules performs two 
functions: (i) it acts as a Python wrapper for creating and managing the 
attenuation data obtained from the XCOM database, and (ii) single-/dual-energy 
CT coefficients from the response curves to be put to use in the simulation. 
Any material, with the knowledge of chemical formula can be conveniently 
created and used in the dataset generation script or in the Simulator GUI.

The Database Handler contains supports for:
(i)   Obtaining attenuation / CT coefficient data for atomic elements
(ii)  Creating  attenuation / CT coefficient data for compounds
(iii) Creating  attenuation / CT coefficient data for target materials
(iv)  Obtaining attenuation / CT coefficient data for compounds
(v)   Obtaining attenuation / CT coefficient data for target materials 

The CT coefficients used here are Compton coefficients, PE coefficients, 
Effective Atomic Number and Density. Upon selection of the forward projection 
model, the linear attenuation coefficient (LAC) and Hounsfeld values are also 
added to the list of coefficient values.

* Usage:

The main function of the MuDatabaseHandler() is to serve as a handle for 
accessing materials and their propoerties during simulation. The handleer 
can also be used to generate attenuation specified materials:

> --------------------------------------------------------------------------- >  
from mu_database_handler import *

mu_handle = MuDatabaseHandler()
print('Copper:', mu_handle.element['Cu'])
print('-'*20')
print('Water:',  mu_handle.compound['water'])

Output:

Copper:
{'compton': 910.4351012828125,
 'density': 8.95,
 'mu': array([2.279e+04, 1.797e+04, 1.432e+04, 1.155e+04, 9.436e+03, 7.814e+03,
              .
              .
              .
              2.253e+01, 2.233e+01, 2.213e+01, 2.194e+01, 2.175e+01, 2.156e+01,
              2.138e+01]),
 'pe': 625.6699690326266,
 'z': 29}
 ------------------
 
 Water:
 {'compton': 0.1636710077950889,
 'density': 1.0,
 'mu': array([5.33  , 4.026 , 3.126 , 2.487 , 2.021 , 1.672 , 1.408 , 1.203,
                .
                .
                .
              0.1518, 0.1515, 0.1512, 0.1508, 0.1505, 0.1502, 0.1499, 0.1496,
              0.1493, 0.149 , 0.1487, 0.1484, 0.1481, 0.1478, 0.1475]),
 'pe': 5058.233649345908,
 'z': 7.420007543374948}
> --------------------------------------------------------------------------- >  

Attenuation data can also generated for a given compound by specifying its 
molecular formula and density.

> --------------------------------------------------------------------------- >  
mu_handle.create_compound_mu(compound_formula='NaCl', compound_name='salt',
                             density=2.16)
print('Salt:', mu_handle.compound['salt'])

Output:
Attenuation Data generated for salt, NaCl.

Salt:
{'compton': 0.2831475714237009,
 'density': 2.16,
 'mu': array([40.23  , 30.52  , 23.68  , 18.74  , 15.07  , 12.3   , 10.17  ,
              .
              .
              .
              0.1348,  0.1344,  0.134 ,  0.1336,  0.1333,  0.1329,  0.1325,
              0.1321,  0.1318,  0.1314,  0.131 ]),
 'pe': 87277.26936840561,
 'z': 14.315062719492778}
> --------------------------------------------------------------------------- >  

Target materials or Materials of interests for CT segmentation such as H2O2 
(liquid threat detection) or rust, FeO2 (in metal assembly inspection) can be 
separately created and accessed in a similar manner as normal compounds, using 
the method self.create_target_mu() and attribute like self.target['H2O2']. 

Once initialized, the data for a material (element, compound or target) is 
stored as a dictionary with the following keys:

- 'density' - material density in g/cc
- 'compton' - Compton scattering coefficient in 1/cm  
- 'pe'      - Pohotelectric absorption coefficient in 1/cm
- 'z'       - Atomic Number for an element or Effective atomic number for a 
              compound/target
- 'mu'      - X-ray attenuation curve - 1D array of attenuation values (in 
              cm^2/g) for the keV range specified by self.kev_range (Default: 
              10 -161 keV)
- 'lac', 'lac_i' - LAC for the i^th spectrum 
- 'HU', 'HU_i'   - Hounsfeld Units for the i^th spectrum 
('lac' and 'HU' can be accessed only after the spectral model is specified.)

The materials and their properties can be accessed using the method 
self.material;

> --------------------------------------------------------------------------- >  
print('Acrylic:', mu_handle.material('acrylic'))
print('-'*20)
print('Acrylic Z:', mu_handle.material('acrylic', 'z'))

Output:

Acrylic:
{'compton': 0.16904512642438155,
 'density': 1.18,
 'mu': array([3.398 , 2.616 , 2.063 , 1.662 , 1.364 , 1.139 , 0.9664, 0.8314,
              .
              .
              .
              0.1352, 0.1349, 0.1346, 0.1344, 0.1341, 0.1339, 0.1336, 0.1334,
              0.1331, 0.1329, 0.1326, 0.1324, 0.1321]),
 'pe': 3804.98491468153,
 'z': 6.777477825789594}
-------------------

Acrylic Z: 6.777477825789594
> --------------------------------------------------------------------------- >  


* Methods:

__init__()                  - Constructor
create_compound_mu()        - Create attenuation data for a compound 
                              specified by its molecular formula and density 
                              in g/cc
create_targets_mu()         - Create attenuation data for a target material 
                              specified by its molecular formula and density 
                              in g/cc
material()                  - Get attenuation data for a specified material
calculate_lac_hu_values()   - Calculate the LAC / Hounsfeld for a given 
                              material and spectral model

* Attributes:

element                     - dictionary of periodic leement from Z = 1-100
compound                    - dictionary of current compounds retained by 
                              the mu_handler
target                      - dictionary of current targets retained by 
                              the mu_handler
                              
curr_compounds_list         - list of compounds currently saved in DEBISim
curr_targets_list           - list of compounds currently saved in DEBISim
elements_list               - list of the atomic symbols of all elementa
f_loc                       - dictionary of dbase file locations in DEBiSim
kev_range                   - Current Energy range for attenuation curves 
                              (default: 10 - 161 keV with 1 keV)
materials_list              - list of all saved materials including elements, 
                              compounds and targets 
--------------------------------------------------------------------------------
"""

import sys, os

from lib.misc.fdlib import *
from lib.misc.util import *
from lib.misc.ctlib import *
from numpy import *
import subprocess as sub


class MuDatabaseHandler(object):
    """
    ----------------------------------------------------------------------------

    This module is responsible for handling the NIST XCOM database for generating
    and managing attenuation response data for elements, compounds and target
    materials. The attenuation curve in the CT simulation for DEBISim is derived
    from the NIST XCOM Photon Cross Sections Database [1] which is based on a
    FORTRAN script that handles the attenuation data. The modules performs two
    functions: (i) it acts as a Python wrapper for creating and managing the
    attenuation data obtained from the XCOM database, and (ii) single-/dual-energy
    CT coefficients from the response curves to be put to use in the simulation.
    Any material, with the knowledge of chemical formula can be conveniently
    created and used in the dataset generation script or in the Simulator GUI.

    The Database Handler contains supports for:
    (i)   Obtaining attenuation / CT coefficient data for atomic elements
    (ii)  Creating  attenuation / CT coefficient data for compounds
    (iii) Creating  attenuation / CT coefficient data for target materials
    (iv)  Obtaining attenuation / CT coefficient data for compounds
    (v)   Obtaining attenuation / CT coefficient data for target materials

    The CT coefficients used here are Compton coefficients, PE coefficients,
    Effective Atomic Number and Density. Upon selection of the forward projection
    model, the linear attenuation coefficient (LAC) and Hounsfeld values are also
    added to the list of coefficient values.
    ----------------------------------------------------------------------------
    """

    # Attributes

    # file locations
    f_loc = {
        'mu_dir': os.path.join(ROOT_DIR, 'include/mu/'),
        'xcom_dir': os.path.join(ROOT_DIR,'include/mu/xcom/XCOM/'),
        'elements_dbase': os.path.join(ROOT_DIR,'include/mu/elements/'),
        'compounds_dbase': os.path.join(ROOT_DIR,'include/mu/compounds/'),
        'targets_dbase': os.path.join(ROOT_DIR, 'include/mu/targets/'),
        'elements_density': os.path.join(ROOT_DIR,'include/mu/elements_density.txt'),
        'compounds_density': os.path.join(ROOT_DIR,'include/mu/compounds_density.txt'),
        'targets_density': os.path.join(ROOT_DIR,'include/mu/targets_density.txt'),
        'xcom_setup': 'gfortran XCOM.f',
        'xcom_run': './a.out'
    }

    # List of symbols for periodic elements
    elements_list = ['H', 'He',  'Li', 'Be', 'B',   'C',  'N',  'O',  'F',  'Ne',
                     'Na', 'Mg', 'Al', 'Si', 'P',   'S',  'Cl', 'Ar', 'K', 'Ca',
                     'Sc', 'Ti', 'V',  'Cr', 'Mn',  'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                     'Ga', 'Ge', 'As', 'Se', 'Br',  'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                     'Nb', 'Mo', 'Tc', 'Ru', 'Rh',  'Pd', 'Ag', 'Cd', 'In', 'Sn',
                     'Sb', 'Te', 'I',  'Xe', 'Cs',  'Ba', 'La', 'Ce', 'Pr', 'Nd',
                     'Pm', 'Sm', 'Eu', 'Gd', 'Tb',  'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                     'Lu', 'Hf', 'Ta', 'W',  'Re',  'Os', 'Ir', 'Pt', 'Au', 'Hg',
                     'Tl', 'Pb', 'Bi', 'Po', 'At',  'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                     'Pa', 'U',  'Np', 'Pu', 'Am',  'Cm', 'Bk', 'Cf', 'Es', 'Fm']

    curr_compounds_list = None
    curr_targets_list = None
    kev_range = range(10,161)

    element = dict()
    compound = dict()
    target = dict()

    materials_list = elements_list

    def __init__(self, debug=False, logfile=None):
        """
        -----------------------------------------------------------------------
        Constructor for MuDatabaseHandler().

        -----------------------------------------------------------------------
        """

        self.debug = debug
        
        if self.debug: 
            if logfile is None: 
                raise IOError("Must specify logger for debuging")
            else:
                self.logger = get_logger('MU_HANDLER', logfile)

                self.logger.info("Initializing MuDatabaseHandler ...")
                self.logger.info("="*80)

        # List of compounds and targets currently saved in DEBISim database
        self.curr_compounds_list = [x.replace('mass_atten_',
                                              '').replace('.txt', '')
                                    for x in os.listdir(
                                                self.f_loc['compounds_dbase'])]
        self.curr_targets_list   = [x.replace('mass_atten_',
                                               '').replace('.txt', '')
                                    for x in os.listdir(
                                                self.f_loc['targets_dbase'])]

        self.materials_list = self.materials_list + self.curr_compounds_list \
                              + self.curr_targets_list

        self.elements_density  =   loadtxt(self.f_loc['elements_density']
                                               )[:, 1]
        self.compounds_density =   loadtxt(self.f_loc['compounds_density'],
                                            dtype=dtype(str, float))[:, 1]
        self.targets_density   =   loadtxt(self.f_loc['targets_density'],
                                            dtype=dtype(str, float))[:, 1]
        self.compounds_name =   loadtxt(self.f_loc['compounds_density'],
                                            dtype=dtype(str, float))[:, 0]
        self.targets_name   =   loadtxt(self.f_loc['targets_density'],
                                            dtype=dtype(str, float))[:, 0]

        self.metals = self.elements_list[2:4] + self.elements_list[10:13] + \
                      self.elements_list[18:31] + self.elements_list[36:50] + \
                      self.elements_list[54:83] + self.elements_list[87:]

        self.periodic_table = recfromcsv(os.path.join(MU_DIR,
                                                      'periodic_table.csv'))

        self.molar_masses = {x[2]: x[3] for x in self.periodic_table}

        # Load Attenuation data from the database -----------------------------

        # Data for elements
        for k, elem in enumerate(self.elements_list):
            curr_elem = dict()
            curr_elem['mu'] = loadtxt(os.path.join(self.f_loc['elements_dbase'],
                                                   'mass_atten_%s.txt'%elem))

            curr_elem['density'] = float(self.elements_density[k])
            curr_elem['compton'], curr_elem['pe'] = \
                calculate_pe_compton_coeffs(self.kev_range,
                                            curr_elem['mu'],
                                            density=curr_elem['density'])
            curr_elem['z'] = k+1
            self.element[elem] = curr_elem

        # Data for compounds
        for k, mat in enumerate(self.curr_compounds_list):
            curr_mat = dict()
            curr_mat['mu'] = loadtxt(os.path.join(self.f_loc['compounds_dbase'],
                                                  'mass_atten_%s.txt'%mat))

            ind = list(self.compounds_name).index(mat)
            curr_mat['density'] = float(self.compounds_density[ind])
            curr_mat['compton'], curr_mat['pe'] = \
                calculate_pe_compton_coeffs(range(10, 10+len(curr_mat['mu'])),
                                            curr_mat['mu'],
                                            density=curr_mat['density'])
            curr_mat['z'] = effective_atomic_number(curr_mat['pe'],
                                                    curr_mat['compton'])
            self.compound[mat] = curr_mat

        # Data for target materials
        for k, mat in enumerate(self.curr_targets_list):
            curr_mat = dict()
            curr_mat['mu'] = loadtxt(os.path.join(self.f_loc['targets_dbase'],
                                                  'mass_atten_%s.txt'%mat))
            ind = list(self.targets_name).index(mat)
            curr_mat['density'] = float(self.targets_density[ind])
            curr_mat['compton'], curr_mat['pe'] = \
                calculate_pe_compton_coeffs(range(10, 10+len(curr_mat['mu'])),
                                            curr_mat['mu'],
                                            density=curr_mat['density'])
            curr_mat['z'] = effective_atomic_number(curr_mat['pe'],
                                                    curr_mat['compton'])
            self.target[mat] = curr_mat

        if self.debug: self.logger.info("Database Contents:")
        if self.debug: self.logger.info('-'*40)
        if self.debug: self.logger.info(f"Number of Elements: {len(self.element)}")
        if self.debug: self.logger.info(f"Number of Compounds: {len(self.compound)}")
        if self.debug: self.logger.info(f"Number of Targets: {len(self.target)}")
        if self.debug: self.logger.info("="*80)
        # ---------------------------------------------------------------------

    def create_compound_mu(self, compound_formula, compound_name, density):
        """
        -----------------------------------------------------------------------
        Generate Attenuation data for a given compound specified by its chemical
        formula and density

        :param compound_formula:       Molecular formula for a compound
        :param compound_name:          Name to be assigned to the material
        :param density:                Density of the compound in g/cm^2

        :return: -
        -----------------------------------------------------------------------
        """

        # run the XCOM fortran script to load the attenuation curve data
        os.chdir(self.f_loc['xcom_dir'])
        proc = sub.Popen([self.f_loc['xcom_run']],
                         stdin=sub.PIPE, stdout=sub.PIPE)
        proc.stdin.write("%s\n" % compound_name)
        proc.stdin.write("3\n")
        proc.stdin.write("%s\n" % compound_formula)
        proc.stdin.write("3\n")
        proc.stdin.write("1\n")
        proc.stdin.write("%i\n" % len(self.kev_range))

        for kev in self.kev_range:
            proc.stdin.write("%.3f\n" % (kev*0.001))
        proc.stdin.write("N\n")
        proc.stdin.write("%s\n"%(os.path.join(self.f_loc['mu_dir'], 'tmp')))

        proc.stdin.write("1\n")
        time.sleep(5)

        # Save the generated attenuation data into the database
        with open(os.path.join(self.f_loc['mu_dir'], 'tmp'), 'r+') as cdataf:

            indata = cdataf.readlines()
            outdata = indata[13:57] + indata[71:115] + indata[129:173] + \
                      indata[187:]

            with open(os.path.join(self.f_loc['compounds_dbase'],
                                   'mass_atten_%s.txt'%compound_name), 'w') as f:
                f.writelines(outdata)
                f.close()
            cdataf.close()

        mu_data = loadtxt(os.path.join(self.f_loc['compounds_dbase'],
                                       'mass_atten_%s.txt' % compound_name))[:,-1]
        mu_line = ''

        for m in mu_data:   mu_line = mu_line + '%.5f\t'%m
        mu_line =  mu_line[:-1]

        with open(os.path.join(self.f_loc['compounds_dbase'],
                               'mass_atten_%s.txt' % compound_name), 'w') as f:
            f.write(mu_line)
            f.close()

        curr_mat = dict()

        # Calculate density, compton, pe, z values for the material
        curr_mat['mu'] = mu_data

        curr_mat['density'] = density
        curr_mat['compton'], curr_mat['pe'] = \
            calculate_pe_compton_coeffs(range(10, 10 + len(curr_mat['mu'])),
                                        curr_mat['mu'],
                                        density=curr_mat['density'])
        curr_mat['z'] = effective_atomic_number(curr_mat['pe'],
                                                curr_mat['compton'])
        self.compound[compound_name] = curr_mat

        with open(self.f_loc['compounds_density'], 'a+') as f:
            f.write("\n%s\t%.3f"%(compound_name, density))
            f.close()

        self.materials_list = self.materials_list + [compound_name]
        self.curr_compounds_list = self.curr_compounds_list + [compound_name]
        if self.debug: self.logger.info(f"Attenuation Data generated for "
                         f"{compound_name} , {compound_formula}")

    # -------------------------------------------------------------------------

    def create_mixture_mu(self, mixture_name, component_list, is_target=True):
        """
        -----------------------------------------------------------------------
        Generate Attenuation data for a given compound specified by its chemical
        formula and density

        :param mixture_name:       Name of the compound
        :param component_list:     List of tuples of the components of the
                                   mixture in the form:
                                   (formula, fraction, density in g/cc)
        :param is_target:          Set to True if the material is a target

        :return: -
        -----------------------------------------------------------------------
        """

        # run the XCOM fortran script to load the attenuation curve data
        os.chdir(self.f_loc['xcom_dir'])
        proc = sub.Popen([self.f_loc['xcom_run']],
                         stdin=sub.PIPE, stdout=sub.PIPE)
        proc.stdin.write("%s\n" % mixture_name)
        proc.stdin.write("4\n")
        proc.stdin.write("%i\n"%(len(component_list)))

        for component in component_list:
            proc.stdin.write("%s\n" % component[0])
            proc.stdin.write("%.3f\n" % component[1])

        proc.stdin.write("1\n")
        proc.stdin.write("3\n")
        proc.stdin.write("1\n")
        proc.stdin.write("%i\n" % len(self.kev_range))

        for kev in self.kev_range:
            proc.stdin.write("%.3f\n" % (kev*0.001))
        proc.stdin.write("N\n")
        proc.stdin.write("%s\n"%(os.path.join(self.f_loc['mu_dir'], 'tmp')))

        proc.stdin.write("1\n")

        time.sleep(5)

        if is_target:
            save_file = os.path.join(self.f_loc['targets_dbase'],
                                     'mass_atten_%s.txt' % mixture_name)
        else:
            save_file = os.path.join(self.f_loc['compounds_dbase'],
                                     'mass_atten_%s.txt' % mixture_name)

        # Save the generated attenuation data into the database
        with open(os.path.join(self.f_loc['mu_dir'], 'tmp'), 'r+') as cdataf:

            indata = cdataf.readlines()
            outdata = indata[13:57] + indata[71:115] + indata[129:173] + \
                      indata[187:]

            with open(save_file, 'w') as f:
                f.writelines(outdata)
                f.close()
            cdataf.close()

        mu_data = loadtxt(save_file)[:,-1]
        mu_line = ''

        for m in mu_data:   mu_line = mu_line + '%.5f\t'%m
        mu_line =  mu_line[:-1]

        with open(save_file, 'w') as f:
            f.write(mu_line)
            f.close()

        curr_mat = dict()

        # Calculate density, compton, pe, z values for the material
        curr_mat['mu'] = mu_data

        curr_mat['density'] = sum(array([x[1]*x[2] for x in component_list]))
        curr_mat['compton'], curr_mat['pe'] = \
            calculate_pe_compton_coeffs(range(10, 10 + len(curr_mat['mu'])),
                                        curr_mat['mu'],
                                        density=curr_mat['density'])
        curr_mat['z'] = effective_atomic_number(curr_mat['pe'],
                                                curr_mat['compton'])
        self.compound[mixture_name] = curr_mat

        if is_target:
            with open(self.f_loc['targets_density'], 'a+') as f:
                f.write("\n%s\t%.3f"%(mixture_name, curr_mat['density']))
                f.close()
            self.curr_compounds_list = self.curr_compounds_list + [mixture_name]
        else:
            with open(self.f_loc['compounds_density'], 'a+') as f:
                f.write("\n%s\t%.3f"%(mixture_name, curr_mat['density']))
                f.close()
            self.curr_compounds_list = self.curr_compounds_list + [mixture_name]

        self.materials_list = self.materials_list + [mixture_name]
        if self.debug: self.logger.info(f"Attenuation Data generated for "
                         f"{mixture_name}, {component_list}")

    # -------------------------------------------------------------------------

    def create_targets_mu(self, target_formula, target_name, density):

        """
        -----------------------------------------------------------------------
        Generate Attenuation data for a given target specified by its chemical
        formula and density

        :param target_formula:       Molecular formula for a compound
        :param target_name:          Name to be assigned to the material
        :param density:              Density of the compound in g/cm^2

        :return: -
        -----------------------------------------------------------------------
        """

        # run the XCOM fortran script to load the attenuation curve data
        os.chdir(self.f_loc['xcom_dir'])
        proc = sub.Popen([self.f_loc['xcom_run']],
                         stdin=sub.PIPE, stdout=sub.PIPE)
        proc.stdin.write("%s\n" % target_name)
        proc.stdin.write("3\n")
        proc.stdin.write("%s\n" % target_formula)
        proc.stdin.write("3\n")
        proc.stdin.write("1\n")
        proc.stdin.write("%i\n" % len(self.kev_range))


        for kev in self.kev_range:
            proc.stdin.write("%.3f\n" % (kev*0.001))
        proc.stdin.write("N\n")
        proc.stdin.write("%s\n"%(os.path.join(self.f_loc['mu_dir'], 'tmp')))

        proc.stdin.write("1\n")
        time.sleep(5)

        # Save the generated attenuation data into the database
        with open(os.path.join(self.f_loc['mu_dir'], 'tmp'), 'r+') as cdataf:

            indata = cdataf.readlines()
            outdata = indata[13:57] + indata[71:115] + indata[129:173] + \
                      indata[187:]

            with open(os.path.join(self.f_loc['targets_dbase'],
                                   'mass_atten_%s.txt'%target_name), 'w') as f:
                f.writelines(outdata)
                f.close()
            cdataf.close()

        mu_data = loadtxt(os.path.join(self.f_loc['targets_dbase'],
                                       'mass_atten_%s.txt' % target_name))[:,-1]
        mu_line = ''

        for m in mu_data:   mu_line = mu_line + '%.5f\t'%m
        mu_line =  mu_line[:-1]

        with open(os.path.join(self.f_loc['targets_dbase'],
                             'mass_atten_%s.txt' % target_name), 'w') as f:
            f.write(mu_line)
            f.close()

        curr_mat = dict()

        # calculate density, compton, pe and z value for target
        curr_mat['mu'] = mu_data

        curr_mat['density'] = density
        curr_mat['compton'], curr_mat['pe'] = \
            calculate_pe_compton_coeffs(range(10, 10 + len(curr_mat['mu'])),
                                        curr_mat['mu'],
                                        density=curr_mat['density'])
        curr_mat['z'] = effective_atomic_number(curr_mat['pe'],
                                                curr_mat['compton'])
        self.target[target_name] = curr_mat

        with open(self.f_loc['targets_density'], 'a+') as f:
            f.write("\n%s\t%.3f"%(target_name, density))
            f.close()

        self.materials_list = self.materials_list + [target_name]
        self.curr_targets_list = self.curr_targets_list + [target_name]

        if self.debug: self.logger.info(f"Attenuation Data generated "
                         f"for {target_name}, {target_formula}")

    # -------------------------------------------------------------------------

    def material(self, mat, prop=None):
        """
        -----------------------------------------------------------------------
        Function to read material / material properties.

        :param mat:     material name
        :param prop:    material property: {'compton', 'pe', 'z', 'mu', 'density'}
        :return:   material dictionary or property value
        -----------------------------------------------------------------------
        """

        if mat in list(self.element.keys()):
            if prop is None: return self.element[mat]
            else:            return self.element[mat][prop]

        elif mat in list(self.compound.keys()):
            if prop is None: return self.compound[mat]
            else:            return self.compound[mat][prop]

        elif mat in list(self.target.keys()):
            if prop is None: return self.target[mat]
            else:            return self.target[mat][prop]
        else:
            print("Unknown Material!")
            print("Select from the current material list or add the new "
                  "material to the list!")

            raise IOError

    # -------------------------------------------------------------------------

    def calculate_lac_hu_values(self, mat, spectrum_list):
        """
        -----------------------------------------------------------------------
        Calculate the LAC and HU values for the given material and spectrum

        :param mat:             material name
        :param spectrum_list:   spectrum array or list of 1D spectra
        :return:
        -----------------------------------------------------------------------
        """

        def set_atten_coeffs(cmat, spec, k=None):
            """
            -----------------------------------------------------
            Sets LAC / HU for given material and spectrum.

            :param cmat:    material
            :param spec:    1D spectrum
            :param k:       coefficient suffix
            :return:
            -----------------------------------------------------
            """

            if cmat in self.element.keys():

                s_len = min(spec.size, self.element[cmat]['mu'].size)

                self.element[cmat]['lac%s'%k] =  -log(sum(spec[:s_len]*
                                                    exp(-self.element[cmat]['mu'][:s_len]
                                                         *self.element[cmat]['density'])))

                self.element[cmat]['HU%s'%k] = (self.element[cmat]['lac%s'%k]
                - self.compound['water']['lac%s'%k])/self.compound['water']['lac%s'%k]*1000

            elif cmat in self.compound.keys():

                s_len = min(spec.size, self.compound[cmat]['mu'].size)

                self.compound[cmat]['lac%s'%k] = -log(sum(spec[:s_len]*
                                                    exp(-self.compound[cmat]['mu'][:s_len]
                                                         *self.compound[cmat]['density'])))

                self.compound[cmat]['HU%s'%k] = (self.compound[cmat]['lac%s'%k]
                - self.compound['water']['lac%s'%k])/self.compound['water']['lac%s'%k]*1000

            elif cmat in self.target.keys():
                s_len = min(spec.size, self.target[cmat]['mu'].size)

                self.target[cmat]['lac%s'%k] = -log(sum(spec[:s_len]*
                                                    exp(-self.target[cmat]['mu'][:s_len]
                                                         *self.target[cmat]['density'])))

                self.target[cmat]['HU%s'%k] = (self.target[cmat]['lac%s'%k]
                - self.compound['water']['lac%s'%k])/self.compound['water']['lac%s'%k]*1000

            else:
                self.logger.info("Unknown Material!")
                self.logger.info("Select from the current material list or add the new "
                      "material to the list!")

                raise IOError
            # -----------------------------------------------

        if isinstance(spectrum_list, list):
            for ind, curr_spec in enumerate(spectrum_list):
                set_atten_coeffs(mat, curr_spec, k='_%i'%(ind+1))
        else:
            set_atten_coeffs(mat, spectrum_list, k='')
    # -------------------------------------------------------------------------

# =============================================================================
# Class Ends
# =============================================================================


if __name__=="__main__":
    mu_handler = MuDatabaseHandler()
