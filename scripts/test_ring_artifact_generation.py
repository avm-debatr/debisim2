#!/usr/bin/env python

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

from src.debisim_pipeline import *
from lib.forward_model.mu_database_handler import *
import argparse

parser = argparse.ArgumentParser(
            description="Test ring artifact simulation on an example sinogram")

parser.add_argument('--in_dir',
                    default='results/example_parallelbeam_2d/simulation_001',
                    help="Input simulation directory")

parser.add_argument('--out_dir',
                    default='results/ring_artifact_sim',
                    help="Output simulation directory")

parser.add_argument('--scanner',
                    default='default',
                    choices=['sensation_32', 'definition_as', 'force', 'default'],
                    help="Scanner for simulation")

parser.add_argument('--zslice',
                    default=59,
                    type=int,
                    help="CT Z slice to process")

args = parser.parse_args()

scanner = {'sensation_32': siemens_sensation_32,
           'definition_as': siemens_definition_as,
           'force': siemens_force,
           'default': default_scanner_parallel}[args.scanner]

t0 = time.time()

# Define scanner model
id1 = ScannerTemplate(geometry='parallel' if args.scanner=='default' else 'cone',
                      scan='circular'  if args.scanner=='default' else 'spiral',
                      machine_dict=scanner.machine_geometry,
                      recon='fbp',
                      recon_dict=scanner.recon_params,
                      pscale=1.0
                      )

id1.set_recon_geometry()

# X-ray source specifications
xray_source_specs = dict(
    num_spectra=1,
    kVp=130,
    spectra=[os.path.join(SPECTRA_DIR, 'example_spectrum_130kV.txt')],
    dosage=[1.8e5]
)

simulator = DEBISimPipeline(
    sim_path=DEFAULT_SIM_DIR,
    scanner_model=id1,
    xray_source_model=xray_source_specs
)

fname = os.path.join(args.in_dir, 'projections/sino_1.fits.gz')

mu_w = simulator.mu.material('water')
sino_1 = read_fits_data(fname, 0)
orig_recon = id1.reconstruct_data(moveaxis(sino_1, 2, 0))
orig_recon *= id1.recon_params['img_scale']
orig_recon = (orig_recon - mu_w['lac_1']) / mu_w['lac_1'] * 1000
orig_recon = clip(orig_recon, -1000, 3.2e4).astype(np.int16)

s_no = args.zslice
n_recons = [orig_recon[s_no,:,:]]

for s in [0.05 ,0.1, 0.5]:

    id1.set_ring_artifact_params(severity=s)

    # convert to projn
    projn = sino_1 - log(xray_source_specs['dosage'][0])
    projn = np.exp(-projn)

    n_projn = id1.add_ring_artifacts(projn)

    n_projn[n_projn < 1] = 1
    n_sino = -np.log(n_projn) + log(xray_source_specs['dosage'][0])

    noisy_recon = id1.reconstruct_data(moveaxis(n_sino, 2, 0))

    noisy_recon *= id1.recon_params['img_scale']
    noisy_recon  = (noisy_recon - mu_w['lac_1'])/ mu_w['lac_1'] * 1000
    noisy_recon  = clip(noisy_recon, -1000, 3.2e4).astype(np.int16)

    n_recons.append(noisy_recon[s_no,:,:])

quick_imshow(2, 2, n_recons, titles=['Original',
                                     'Severity=0.05',
                                     'Severity=0.1',
                                     'Severity=0.5'],
             vmin=-1000,
             vmax=500,
             colorbar=False,
             colormap='gray'
             )
plt.show()

