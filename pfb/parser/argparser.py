import argparse
from pfb.utils import str2bool


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str, nargs='+',
                   help="List of measurement sets to image")
    p.add_argument("--data_column", default="CORRECTED_DATA", type=str,
                   help="The column to image.")
    p.add_argument("--weight_column", default='WEIGHT_SPECTRUM', type=str,
                   help="Weight column to use.")
    p.add_argument("--imaging_weight_column", default=None, type=str,
                   help="Weight column to use.")
    p.add_argument("--model_column", default='MODEL_DATA', type=str,
                   help="Column to write model data to")
    p.add_argument("--flag_column", default='FLAG', type=str)
    p.add_argument("--row_chunks", default=100000, type=int,
                   help="Rows per chunk")
    p.add_argument("--chan_chunks", default=8, type=int,
                   help="Channels per chunk (only used for writing component"
                   " model).")
    p.add_argument("--write_model", type=str2bool, nargs='?', const=True,
                   default=True, help="Whether to write model visibilities to"
                   " model_column")
    p.add_argument("--interp_model", type=str2bool, nargs='?', const=True,
                   default=True, help="Interpolate final model with "
                   "integrated polynomial")
    p.add_argument("--spectral_poly_order", type=int, default=4,
                   help="Order of interpolating polynomial")
    p.add_argument("--mop_flux", type=str2bool, nargs='?', const=True,
                   default=True, help="If True then positivity and sparsity "
                   "will be relaxed at the end and a flux mop will be applied"
                   " inside the mask.")
    p.add_argument("--make_restored", type=str2bool, nargs='?', const=True,
                   default=True, help="Whather to produce a restored image "
                   "or not.")
    p.add_argument("--deconv_mode", type=str, default='sara',
                   help="Select minor cycle to use. Current options are "
                   "'spotless' (default), 'sara' or 'clean'")
    p.add_argument("--weighting", type=str, default=None,
                   help="Imaging weights to apply. None means natural, "
                   "anything else is either Briggs or Uniform depending of "
                   "the value of robust.")
    p.add_argument("--robust", type=float, default=None,
                   help="Robustness value for Briggs weighting. None means "
                   "uniform.")
    p.add_argument("--dirty", type=str,
                   help="Fits file with dirty cube")
    p.add_argument("--psf", type=str,
                   help="Fits file with psf cube")
    p.add_argument("--psf_oversize", default=2.0, type=float,
                   help="Increase PSF size by this factor")
    p.add_argument("--outfile", type=str, default='pfb',
                   help='Base name of output file.')
    p.add_argument("--fov", type=float, default=None,
                   help="Field of view in degrees")
    p.add_argument("--super_resolution_factor", type=float, default=1.2,
                   help="Pixel sizes will be set to Nyquist divided by this "
                   "factor unless specified by cell_size.")
    p.add_argument("--nx", type=int, default=None,
                   help="Number of x pixels. Computed automatically from fov "
                   "if None.")
    p.add_argument("--ny", type=int, default=None,
                   help="Number of y pixels. Computed automatically from fov "
                   "if None.")
    p.add_argument('--cell_size', type=float, default=None,
                   help="Cell size in arcseconds. Computed automatically from"
                   " super_resolution_factor if None")
    p.add_argument("--nband", default=None, type=int,
                   help="Number of imaging bands in output cube")
    p.add_argument("--mask", type=str, default=None,
                   help="A fits mask (True where unmasked)")
    p.add_argument("--beam_model", type=str, default=None,
                   help="Power beam pattern for Stokes I imaging. Pass in a "
                   "fits file or set to JimBeam to use katbeam.")
    p.add_argument("--band", type=str, default='l',
                   help="Band to use with JimBeam. L or UHF")
    p.add_argument("--do_wstacking", type=str2bool, nargs='?', const=True,
                   default=True, help='Whether to use wstacking or not.')
    p.add_argument("--epsilon", type=float, default=1e-5,
                   help="Accuracy of the gridder")
    p.add_argument("--nthreads", type=int, default=0)
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Step size of 'primal' update.")
    p.add_argument("--peak_factor", type=float, default=0.025,
                   help="Clean peak factor.")
    p.add_argument("--maxit", type=int, default=5,
                   help="Number of pfb iterations")
    p.add_argument("--minormaxit", type=int, default=5,
                   help="Number of minor cycle iterations per major "
                   "iteration")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Tolerance")
    p.add_argument("--minortol", type=float, default=1e-3,
                   help="Tolerance")
    p.add_argument("--report_freq", type=int, default=1,
                   help="How often to save output images during "
                   "deconvolution")
    p.add_argument("--beta", type=float, default=None,
                   help="Lipschitz constant of F")
    p.add_argument("--sig_21", type=float, default=1e-3,
                   help="Initial strength of l21 regulariser."
                   "Initialise to nband x expected rms in MFS dirty if "
                   "uncertain.")
    p.add_argument("--sigma_frac", type=float, default=0.5,
                   help="Fraction of peak MFS residual to use in "
                   "preconditioner at each iteration.")
    p.add_argument("--positivity", type=str2bool, nargs='?', const=True,
                   default=True, help="Whether to impose a positivity "
                   "constraint or not.")
    p.add_argument("--psi_levels", type=int, default=3,
                   help="Wavelet decomposition level")
    p.add_argument("--psi_basis", type=str, default=None, nargs='+',
                   help="Explicitly set which bases to use for psi out of:"
                   "[self, db1, db2, db3, db4, db5, db6, db7, db8]")
    p.add_argument("--x0", type=str, default=None,
                   help="Initial guess in form of fits file")
    p.add_argument("--first_residual", default=None, type=str,
                   help="Residual corresponding to x0")
    p.add_argument("--reweight_iters", type=int, default=None, nargs='+',
                   help="Set reweighting iters explicitly. "
                   "Default is to reweight at 4th, 5th, 6th, 7th, 8th and 9th"
                   " iterations.")
    p.add_argument("--reweight_alpha_percent", type=float, default=10,
                   help="Set alpha as using this percentile of non zero "
                   "coefficients")
    p.add_argument("--reweight_alpha_ff", type=float, default=0.5,
                   help="reweight_alpha_percent will be scaled by this factor"
                   " after each reweighting step.")
    p.add_argument("--cgtol", type=float, default=1e-4,
                   help="Tolerance for cg updates")
    p.add_argument("--cgmaxit", type=int, default=150,
                   help="Maximum number of iterations for the cg updates")
    p.add_argument("--cgminit", type=int, default=25,
                   help="Minimum number of iterations for the cg updates")
    p.add_argument("--cgverbose", type=int, default=0,
                   help="Verbosity of cg method used to invert Hess. Set to "
                   "2 for debugging.")
    p.add_argument("--pmtol", type=float, default=1e-5,
                   help="Tolerance for power method used to compute spectral "
                   "norms")
    p.add_argument("--pmmaxit", type=int, default=50,
                   help="Maximum number of iterations for power method")
    p.add_argument("--pmverbose", type=int, default=0,
                   help="Verbosity of power method used to get spectral norm "
                   "of approx Hessian. Set to 2 for debugging.")
    p.add_argument("--pdtol", type=float, default=1e-4,
                   help="Tolerance for primal dual")
    p.add_argument("--pdmaxit", type=int, default=250,
                   help="Maximum number of iterations for primal dual")
    p.add_argument("--pdverbose", type=int, default=0,
                   help="Verbosity of primal dual used to solve backward "
                   "step. Set to 2 for debugging.")
    p.add_argument("--hbgamma", type=float, default=0.1,
                   help="Minor loop gain of Hogbom")
    p.add_argument("--hbpf", type=float, default=0.1,
                   help="Peak factor of Hogbom")
    p.add_argument("--hbmaxit", type=int, default=5000,
                   help="Maximum number of iterations for Hogbom")
    p.add_argument("--hbverbose", type=int, default=0,
                   help="Verbosity of Hogbom. Set to 2 for debugging or "
                   "zero for silence.")
    p.add_argument("--tidy", type=str2bool, nargs='?', const=True,
                   default=True, help="Switch off if you prefer it dirty.")
    p.add_argument("--real_type", type=str, default='f4',
                   help="Dtype of real valued images. f4/f8 for single or "
                   "double precision respectively.")
    return p
