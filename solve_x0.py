import numpy as np
import dask
from scipy.linalg import norm, svd
from scipy.fftpack import next_fast_len
from scipy.stats import laplace
from pfb.opt import pcg, hpd
from pfb.sara import set_Psi
from pfb.utils import load_fits, save_fits, data_from_header, prox_21, str2bool
from pyrap.tables import table
from pfb.operators import Gridder, Prior, PSF
from africanus.constants import c as lightspeed
from astropy.io import fits
import argparse

ms_name = "testing/test_data/point_gauss_nb.MS_p0"
result_name = 'testing/ref_images/hpd_minor.npz'

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dirty", type=str,
                   help="Fits file with dirty cube")
    p.add_argument("--psf", type=str,
                   help="Fits file with psf cube")
    p.add_argument("--outfile", default='image', type=str,
                   help='base name of output.')
    p.add_argument("--ncpu", default=0, type=int,
                   help='Number of threads to use.')
    p.add_argument("--gamma0", type=float, default=0.9,
                   help="Initial primal step size.")
    p.add_argument("--maxit", type=int, default=20,
                   help="Number of hpd iterations")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Tolerance")
    p.add_argument("--report_freq", type=int, default=2,
                   help="How often to save output images during deconvolution")
    p.add_argument("--beta", type=float, default=None,
                   help="Lipschitz constant of F")
    p.add_argument("--sig_21", type=float, default=1e-3,
                   help="Tolerance for cg updates")
    p.add_argument("--use_psi", type=str2bool, nargs='?', const=True, default=False,
                   help="Use SARA basis")
    p.add_argument("--psi_levels", type=int, default=2,
                   help="Wavelet decomposition level")
    p.add_argument("--x0", type=str, default=None,
                   help="Initial guess in form of fits file")
    p.add_argument("--reweight_start", type=int, default=20,
                   help="When to start l1 reweighting scheme")
    p.add_argument("--reweight_freq", type=int, default=2,
                   help="How often to do l1 reweighting")
    p.add_argument("--reweight_alpha", type=float, default=0.5,
                   help="Determines how aggressively the reweighting is applied."
                   "1.0 = very mild whereas close to zero = aggressive.")
    p.add_argument("--reweight_alpha_ff", type=float, default=0.25,
                   help="Determines how quickly the reweighting progresses."
                   "alpha will grow like alpha/(1+i)**alpha_ff.")
    p.add_argument("--cgtol", type=float, default=1e-2,
                   help="Tolerance for cg updates")
    p.add_argument("--cgmaxit", type=int, default=10,
                   help="Maximum number of iterations for the cg updates")
    p.add_argument("--cgverbose", type=int, default=0,
                   help="Verbosity of cg method used to invert Hess. Set to 1 or 2 for debugging.")
    p.add_argument("--pmtol", type=float, default=1e-14,
                   help="Tolerance for power method used to compute spectral norms")
    p.add_argument("--pmmaxit", type=int, default=25,
                   help="Maximum number of iterations for power method")
    return p

def main(args):
    # load dirty and psf
    dirty = load_fits(args.dirty)
    real_type = dirty.dtype
    hdr = fits.getheader(args.dirty)
    freq = data_from_header(hdr, axis=3)
    
    nchan, nx, ny = dirty.shape
    psf_array = load_fits(args.psf)
    hdr_psf = fits.getheader(args.psf)
    try:
        assert np.array_equal(freq, data_from_header(hdr_psf, axis=3))
    except:
        raise ValueError("Fits frequency axes dont match")
    
    psf_max = np.amax(psf_array.reshape(nchan, 4*nx*ny), axis=1)
    
    # set operators
    psf = PSF(psf_array, args.ncpu)
    K = Prior(freq/np.mean(freq), 1.0, 0.25, nx, ny, nthreads=args.ncpu)
    def hess(x):
        return psf.convolve(x) + K.idot(x)

    if args.beta is None:
        from pfb.opt import power_method
        beta = power_method(hess, dirty.shape, tol=args.pmtol, maxit=args.pmmaxit)
    else:
        beta = args.beta   # 212346753.18840605
    print("beta = ", beta)

    # fidelity and gradient term
    def fprime(x):
        diff = psf.convolve(x) - dirty
        tmp = K.idot(x)
        return 0.5*np.vdot(x, diff) - 0.5*np.vdot(x, dirty) + 0.5*np.vdot(x, tmp), diff + tmp

    
    if args.use_psi:
        # wavelet basis and regulariser
        nchan, nx, ny = dirty.shape
        PSI, PSIT = set_Psi(nx, ny)
        psi = {}
        psi['PSI'] = PSI
        psi['PSIT'] = PSIT
        prox = lambda p, sig21, w21: prox_21(p, sig21, w21, psi=psi)

        def reg(x):
            nchan, nx, ny = x.shape
            nbasis = len(PSIT)
            norm21 = 0.0
            for k in range(nbasis):
                v = np.zeros((nchan, nx*ny), x.dtype)
                for l in range(nchan):
                    v[l] = PSIT[k](x[l])
                l2norm = norm(v, axis=0)
                norm21 += np.sum(l2norm)
            return norm21
    else:
        prox = prox_21
    
        def reg(x):
            nchan, nx, ny = x.shape
            # normx = norm(x.reshape(nchan, nx*ny), axis=0)
            normx = np.mean(x.reshape(nchan, nx*ny), axis=0)
            return np.sum(np.abs(normx))


    if args.x0 is None:
        x0 = pcg(hess, dirty, np.zeros((nchan, nx, ny)), M=K.dot, tol=args.cgtol, maxit=2*args.cgmaxit)
    else:
        x0 = load_fits(args.x0)
        
    model, objhist, fidhist, reghist = hpd(fprime, prox, reg, x0, args.gamma0, beta, args.sig_21, psi=psi,
                                           hess=hess, cgprecond=K.dot, cgtol=args.cgtol, cgmaxit=args.cgmaxit, cgverbose=args.cgverbose,
                                           alpha0=args.reweight_alpha, alpha_ff=args.reweight_alpha_ff, reweight_start=args.reweight_start, reweight_freq=args.reweight_freq,
                                           tol=args.tol, maxit=args.maxit, report_freq=1, verbosity=2)

    save_fits(args.outfile + '_model.fits', model, hdr, dtype=real_type)

    residual = dirty - psf.convolve(model)

    save_fits(args.outfile + '_residual.fits', residual/psf_max[:, None, None], hdr)

    import matplotlib.pyplot as plt
    plt.figure('obj')
    plt.plot(np.arange(args.maxit+1), objhist + 1.1*np.abs(objhist.min()), 'r', alpha=0.5)
    plt.plot(np.arange(args.maxit+1), fidhist + 1.1*np.abs(fidhist.min()), 'b', alpha=0.5)
    plt.yscale('log')

    plt.savefig(args.outfile + '_obj_hist.png', dpi=250)

    plt.figure('reg')
    plt.plot(np.arange(args.maxit+1), reghist, 'k')

    plt.savefig(args.outfile + '_reg_hist.png', dpi=250)


if __name__=="__main__":
    args = create_parser().parse_args()

    if args.ncpu:
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(args.ncpu))
    else:
        import multiprocessing
        args.ncpu = multiprocessing.cpu_count()

    GD = vars(args)
    print('Input Options:')
    for key in GD.keys():
        print(key, ' = ', GD[key])

    main(args)


