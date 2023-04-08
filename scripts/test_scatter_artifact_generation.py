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
import argparse

parser = argparse.ArgumentParser(
            description="Test ring artifact simulation on an example sinogram")

parser.add_argument('--in_dir',
                    default='results/example_parallelbeam_2d/simulation_001',
                    help="Input simulation directory")

parser.add_argument('--out_dir',
                    default='results/scatter_artifact_sim',
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


# Define scanner model
id1 = ScannerTemplate(geometry='parallel' if args.scanner=='default' else 'cone',
                      scan='circular'  if args.scanner=='default' else 'spiral',
                      machine_dict=scanner.machine_geometry,
                      recon='fbp',
                      recon_dict=scanner.recon_params,
                      pscale=1.0
                      )

t0 = time.time()
id1.set_recon_geometry()

# X-ray source specifications
xray_source_specs = dict(
    num_spectra=1,
    kVp=130,
    spectra=[os.path.join(SPECTRA_DIR, 'example_spectrum_130kV.txt')],
    dosage=[1.8e5]
)


# Initialize simulator
simulator = DEBISimPipeline(
    sim_path=args.out_dir,
    scanner_model=id1,
    xray_source_model=xray_source_specs
)

gt_image_file = os.path.join(args.in_dir, 'ground_truth/gt_label_image.fits.gz')
metadata_file = os.path.join(args.in_dir, 'sl_metadata.pyc')

sim_args = dict(
    gt_image=read_fits_data(gt_image_file, 0).astype(np.float32),
    save_images=[]
)

simulator.run_bag_generator(mode='manual',
                            sim_args=sim_args,
                            sf_file=metadata_file)

print("Total Time:", time.time() - t0)

s_sino, s_recon, o_sino, o_recon = simulator.add_scatter_to_ct_projection_slice(
                                        add_poisson_noise=True,
                                        add_system_noise=True,
                                        system_gain=1.0,
                                        slice_no=args.zslice
                                    )

quick_imshow(1, 2, [s_sino, o_sino], colormap='gray')

quick_imshow(1, 3,
             [s_recon[::-1,:], o_recon[::-1,:], s_recon[::-1,:] - o_recon[::-1,:]],
             colormap='gray',
             colorbar=False,
             vmin=-1000, vmax=500
             )
plt.show()


