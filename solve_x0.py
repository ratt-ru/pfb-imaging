import numpy as np
import dask
from scipy.linalg import norm, svd
from scipy.fftpack import next_fast_len
from scipy.stats import laplace
from pfb.opt import pcg, hpd
from pyrap.tables import table
from pfb.operators import Gridder, Prior, PSF
from africanus.constants import c as lightspeed
from astropy.io import fits
import argparse

ms_name = "testing/test_data/point_gauss_nb.MS_p0"
result_name = 'testing/ref_images/hpd_minor.npz'

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ddict", type=str,
                   help=".npz dictionary holding psf, dirty and freq")
    p.add_argument("--outname", default='image', type=str,
                   help='base name of output.')
    p.add_argument("--ncpu", default=0, type=int,
                   help='Number of threads to use.')
    p.add_argument("--maxit", type=int, default=20,
                   help="Number of hpd iterations")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Tolerance")
    p.add_argument("--cgtol", type=float, default=1e-2,
                   help="Tolerance for cg updates")
    p.add_argument("--cgmaxit", type=int, default=10,
                   help="Maximum number of iterations for the cg updates")
    p.add_argument("--beta", type=float, default=None,
                   help="Lipschitz constant of F")
    p.add_argument("--sig_21", type=float, default=1e-3,
                   help="Tolerance for cg updates")
    p.add_argument("--ref_image", type=str, default=None,
                   help="Reference image to pinch header from")
    p.add_argument("--x0", type=str, default=None,
                   help="Initial guess in form of fits file")
    p.add_argument("--reweight_start", type=int, default=20,
                   help="When to start l1 reweighting scheme")
    p.add_argument("--reweight_freq", type=int, default=2,
                   help="How often to do l1 reweighting")
    
    return p

def main(args):
    # load dirty and psf
    ddict = np.load(args.ddict)
    dirty = ddict['dirty']
    nchan, nx, ny = dirty.shape
    psf_array = ddict['psf']
    psf_max = np.amax(psf_array.reshape(nchan, 4*nx*ny), axis=1)
    freq = ddict['freq']
    freq /= np.mean(freq)
    
    # set operators
    psf = PSF(psf_array, args.ncpu)
    K = Prior(freq, 1.0, 0.25, nx, ny, nthreads=args.ncpu)
    def hess(x):
        return psf.convolve(x) + K.idot(x)

    if args.beta is None:
        from pfb.opt import power_method
        beta = power_method(hess, dirty.shape, tol=1e-15, maxit=25)
    else:
        beta = args.beta   # 212346753.18840605
    print("beta = ", beta)

    # fidelity and gradient term
    def fprime(x):
        diff = psf.convolve(x) - dirty
        tmp = K.idot(x)
        return 0.5*np.vdot(x, diff) - 0.5*np.vdot(x, dirty) + 0.5*np.vdot(x, tmp), diff + tmp

    def prox(p, sig_21, weights_21):
        # l21 norm
        nchan, nx, ny = p.shape
        # meanp = norm(p.reshape(nchan, nx*ny), axis=0)
        meanp = np.mean(p.reshape(nchan, nx*ny), axis=0)
        l2_soft = np.maximum(meanp - sig_21 * weights_21, 0.0) 
        indices = np.nonzero(meanp)
        ratio = np.zeros(meanp.shape, dtype=np.float64)
        ratio[indices] = l2_soft[indices]/meanp[indices]
        x = (p.reshape(nchan, nx*ny) * ratio[None, :]).reshape(nchan, nx, ny)  
        x[x<0] = 0.0
        return x

    def reg(x):
        nchan, nx, ny = x.shape
        # normx = norm(x.reshape(nchan, nx*ny), axis=0)
        normx = np.mean(x.reshape(nchan, nx*ny), axis=0)
        return np.sum(np.abs(normx))


    if args.x0 is None:
        x0 = pcg(hess, dirty, np.zeros((nchan, nx, ny)), M=K.dot, tol=1e-3, maxit=50)
    else:
        x0 = fits.getdata(args.x0)
        x0 = np.transpose(x0[:,:, ::-1], axes=(0, 2, 1)).astype(np.float64)
        
    model, objhist, fidhist, reghist = hpd(fprime, prox, reg, x0, 1.0, beta, args.sig_21, 
                                           hess=hess, cgprecond=K.dot, cgtol=1e-2, cgmaxit=10, 
                                           alpha0=0.5, alpha_ff=0.25, reweight_start=args.reweight_start, reweight_freq=args.reweight_freq,
                                           tol=1e-5, maxit=args.maxit, report_freq=1)

    if args.ref_image is not None:
        hdr = fits.getheader(args.ref_image)
    else:
        hdu = fits.PrimaryHDU()
        hdr = hdu.header

    hdu = fits.PrimaryHDU(header=hdr)
    hdu.data = np.transpose(model, axes=(0, 2, 1))[:, ::-1].astype(np.float32)
    hdu.writeto(args.outname + '_model.fits', overwrite=True)

    residual = dirty - psf.convolve(model)

    hdu = fits.PrimaryHDU(header=hdr)
    hdu.data = np.transpose(residual/psf_max[:, None, None], axes=(0, 2, 1))[:, ::-1].astype(np.float32)
    hdu.writeto(args.outname + '_residual.fits', overwrite=True)

    
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.figure('obj')
    plt.plot(np.arange(args.maxit+1), objhist + 1.1*np.abs(objhist.min()), 'r', alpha=0.5)
    plt.plot(np.arange(args.maxit+1), fidhist + 1.1*np.abs(fidhist.min()), 'b', alpha=0.5)
    plt.yscale('log')

    plt.figure('reg')
    plt.plot(np.arange(args.maxit+1), reghist, 'k')
    plt.show()

    plt.show()


if __name__=="__main__":
    args = create_parser().parse_args()

    if args.ncpu:
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(args.ncpu))
    else:
        import multiprocessing
        args.ncpu = multiprocessing.cpu_count()

    print("Using %i threads"%args.ncpu)

    main(args)


