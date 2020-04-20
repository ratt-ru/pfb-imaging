import numpy as np
from numba import njit, prange
from pyrap.tables import table
from daskms import xds_from_table
import dask
import dask.array as da
from pfb.opt import power_method, hpd, fista, pcg
from scipy.fftpack import next_fast_len
from scipy.linalg import norm
from time import time
import argparse
from astropy.io import fits
from pfb.utils import str2bool, set_wcs, load_fits, save_fits, compare_headers, prox_21
from pfb.operators import OutMemGridder, PSF, Prior, PSI
import scipy.linalg as la
from scipy.stats import laplace

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("table_name", type=str,
                   help="Name of table to load concatenated Stokes I vis from")
    p.add_argument("--data_column", default="CORRECTED_DATA", type=str,
                   help="The column to image.")
    p.add_argument("--weight_column", default='WEIGHT_SPECTRUM', type=str,
                   help="Weight column to use. Will use WEIGHT if no WEIGHT_SPECTRUM is found")
    p.add_argument("--dirty", type=str,
                   help="Fits file with dirty cube")
    p.add_argument("--psf", type=str,
                   help="Fits file with psf cube")
    p.add_argument("--outfile", type=str, default='pfb',
                   help='Base name of output file.')
    p.add_argument("--fov", type=float, default=None,
                   help="Field of view in degrees")
    p.add_argument("--super_resolution_factor", type=float, default=1.2,
                   help="Pixel sizes will be set to Nyquist divided by this factor unless specified by cell_size.") 
    p.add_argument("--nx", type=int, default=None,
                   help="Number of x pixels. Computed automatically from fov if None.")
    p.add_argument("--ny", type=int, default=None,
                   help="Number of y pixels. Computed automatically from fov if None.")
    p.add_argument('--cell_size', type=float, default=None,
                   help="Cell size in arcseconds. Computed automatically from super_resolution_factor if None")
    p.add_argument('--precision', default=1e-7, type=float,
                   help='Precision of the gridder.')
    p.add_argument("--do_wstacking", type=str2bool, nargs='?', const=True, default=True,
                   help="Whether to do wide-field correction or not")
    p.add_argument("--channels_out", default=None, type=int,
                   help="Number of channels in output cube")
    p.add_argument("--field", type=int, default=0, nargs='+',
                   help="Which fields to image")
    p.add_argument("--ncpu", type=int, default=0)
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
    p.add_argument("--sig_l2", default=1.0, type=float,
                   help="The strength of the l2 norm regulariser")
    p.add_argument("--lfrac", default=0.2, type=float,
                   help="The length scale of the frequency prior will be lfrac * fractional bandwidth")
    p.add_argument("--sig_21_start", type=float, default=1e-3,
                   help="The strength of the l21 norm regulariser")
    p.add_argument("--sig_21_end", type=float, default=1e-3,
                   help="The strength of the l21 norm regulariser")
    p.add_argument("--use_psi", type=str2bool, nargs='?', const=True, default=False,
                   help="Use SARA basis")
    p.add_argument("--psi_levels", type=int, default=4,
                   help="Wavelet decomposition level")
    p.add_argument("--x0", type=str, default=None,
                   help="Initial guess in form of fits file")
    p.add_argument("--reweight_start", type=int, default=1,
                   help="When to start l1 reweighting scheme")
    p.add_argument("--reweight_freq", type=int, default=1,
                   help="How often to do l1 reweighting")
    p.add_argument("--reweight_end", type=int, default=20,
                   help="When to end the l1 reweighting scheme")
    p.add_argument("--reweight_alpha", type=float, default=1e-6,
                   help="Determines how aggressively the reweighting is applied."
                   " >= 1 is very mild whereas << 1 is aggressive.")
    p.add_argument("--reweight_alpha_ff", type=float, default=0.0,
                   help="Determines how quickly the reweighting progresses."
                   "alpha will grow like alpha/(1+i)**alpha_ff.")
    p.add_argument("--cgtol", type=float, default=5e-3,
                   help="Tolerance for cg updates")
    p.add_argument("--cgmaxit", type=int, default=35,
                   help="Maximum number of iterations for the cg updates")
    p.add_argument("--cgverbose", type=int, default=1,
                   help="Verbosity of cg method used to invert Hess. Set to 1 or 2 for debugging.")
    p.add_argument("--hpdtol", type=float, default=1e-5,
                   help="Tolerance for hpd sub-iters")
    p.add_argument("--hpdmaxit", type=int, default=20,
                   help="Maximum number of iterations for hpd sub-iters")
    p.add_argument("--pmtol", type=float, default=1e-14,
                   help="Tolerance for power method used to compute spectral norms")
    p.add_argument("--pmmaxit", type=int, default=25,
                   help="Maximum number of iterations for power method")   
    p.add_argument("--tidy",type=str2bool, nargs='?', const=True, default=False,
                   help="Tidy after clean")
    p.add_argument("--make_restored", type=str2bool, nargs='?', const=True, default=True,
                   help="Make 'restored' image")
    return p

def main(args):
    if args.precision > 1e-6:
        real_type = np.float32
        complex_type=np.complex64
    else:
        real_type = np.float64
        complex_type=np.complex128

    # get max uv coords over all fields
    uvw = []
    xds = xds_from_table(args.table_name, group_cols=('FIELD_ID'), columns=('UVW'), chunks={'row':-1})
    for ds in xds:
        uvw.append(ds.UVW.data.compute())
    uvw = np.concatenate(uvw)
    from africanus.constants import c as lightspeed
    u_max = np.abs(uvw[:, 0]).max()
    v_max = np.abs(uvw[:, 1]).max()
    del uvw

    # get Nyquist cell size
    freq = xds_from_table(args.table_name+"::FREQ")[0].FREQ.data.compute().squeeze()
    uv_max = np.maximum(u_max, v_max)
    cell_N = 1.0/(2*uv_max*freq.max()/lightspeed)

    if args.cell_size is not None:
        cell_rad = args.cell_size * np.pi/60/60/180
        print("Super resolution factor = ", cell_N/cell_rad)
    else:
        cell_rad = cell_N/args.super_resolution_factor
        args.cell_size = cell_rad*60*60*180/np.pi
        print("Cell size set to %5.5e arcseconds" % args.cell_size)
    
    if args.nx is None:
        fov = args.fov*3600
        nx = int(fov/args.cell_size)
        from scipy.fftpack import next_fast_len
        args.nx = next_fast_len(nx)

    if args.ny is None:
        fov = args.fov*3600
        ny = int(fov/args.cell_size)
        from scipy.fftpack import next_fast_len
        args.ny = next_fast_len(ny)

    if args.channels_out is None:
        args.channels_out = freq.size

    print("Image size set to (%i, %i, %i)"%(args.channels_out, args.nx, args.ny))


    # init gridder
    R = OutMemGridder(args.table_name, args.nx, args.ny, args.cell_size, freq, 
                      nband=args.channels_out, field=args.field, precision=args.precision, 
                      ncpu=args.ncpu, do_wstacking=args.do_wstacking)
    freq_out = R.freq_out

    # get headers
    radec = xds_from_table(args.table_name+"::RADEC")[0].RADEC.data.compute().squeeze()
    hdr = set_wcs(args.cell_size/3600, args.cell_size/3600, args.nx, args.ny, radec, freq_out)
    hdr_mfs = set_wcs(args.cell_size/3600, args.cell_size/3600, args.nx, args.ny, radec, np.mean(freq_out))
    hdr_psf = set_wcs(args.cell_size/3600, args.cell_size/3600, 2*args.nx, 2*args.ny, radec, freq_out)
    hdr_psf_mfs = set_wcs(args.cell_size/3600, args.cell_size/3600, 2*args.nx, 2*args.ny, radec, np.mean(freq_out))
    
    # psf
    if args.psf is not None:
        compare_headers(hdr_psf, fits.getheader(args.psf))
        psf_array = load_fits(args.psf)
    else:
        psf_array = R.make_psf()
        save_fits(args.outfile + '_psf.fits', psf_array, hdr_psf, dtype=real_type)
    nband = R.nband
    psf_max = np.amax(psf_array.reshape(nband, 4*args.nx*args.ny), axis=1)
    psf = PSF(psf_array, args.ncpu)
    psf_max[psf_max < 1e-15] = 1e-15

    # dirty
    if args.dirty is not None and args.x0 is None:  # no use for dirty if we are starting from input image
        compare_headers(hdr, fits.getheader(args.dirty))
        dirty = load_fits(args.dirty)
        model = np.zeros((nband, args.nx, args.ny), dtype=real_type)
    else:
        if args.x0 is None:
            model = np.zeros((nband, args.nx, args.ny), dtype=real_type)
            dirty = R.make_dirty()
            save_fits(args.outfile + '_dirty.fits', dirty, hdr, dtype=real_type)
        else:
            compare_headers(hdr, fits.getheader(args.x0))
            model = load_fits(args.x0, dtype=real_type)
            dirty = R.make_residual(model)
            save_fits(args.outfile + '_first_residual.fits', dirty, hdr, dtype=real_type)

    # mfs residual 
    wsum = np.sum(psf_max)
    dirty_mfs = np.sum(dirty, axis=0)/wsum 
    rmax = np.abs(dirty_mfs).max()
    rms = np.std(dirty_mfs)
    print("Peak of dirty is %f and rms is %f"%(rmax, rms))
    save_fits(args.outfile + '_dirty_mfs.fits', dirty_mfs, hdr_mfs)       
    
    #  preconditioning matrix
    l = args.lfrac * (freq_out.max() - freq_out.min())/np.mean(freq_out)
    print("GP prior over frequency sigma_f = %f l = %f"%(args.sig_l2, l))
    K = Prior(freq_out, args.sig_l2, l, args.nx, args.ny, nthreads=args.ncpu)
    def hess(x):  
        return psf.convolve(x) + K.idot(x)
    if args.beta is None and args.tidy:
        print("Getting spectral norm of update operator")
        beta = power_method(hess, dirty.shape, tol=args.pmtol, maxit=args.pmmaxit)
    else:
        beta = args.beta
    print(" beta = %f "%beta)

    # Reweighting
    reweight_iters = list(np.arange(args.reweight_start, args.reweight_end, args.reweight_freq))
    reweight_iters.append(args.reweight_end)

    # Reporting    
    print("At iteration 0 peak of residual is %f and rms is %f" % (rmax, rms))
    report_iters = list(np.arange(0, args.maxit, args.report_freq))
    if report_iters[-1] != args.maxit-1:
        report_iters.append(args.maxit-1)

    # fidelity and gradient term
    def fprime(x, y, A, mu=0.0):
        d = y - mu
        tmp = A(d-x)
        return 0.5*np.vdot(d-x, tmp), -tmp

    # mean function fitting
    w = (freq_out/np.mean(freq_out)).reshape(nband, 1)
    order = 3
    X = np.tile(w, order)**np.arange(0,order)
    WX = (psf_max[:, None] * X)
    XTX = X.T @ WX
    XTXinv = np.linalg.pinv(XTX)
    Xinv = XTXinv @ WX.T

    # regulariser
    sig_21 = np.linspace(args.sig_21_start, args.sig_21_end, args.maxit)
    if args.use_psi:
        # set up wavelet basis
        nchan, nx, ny = dirty.shape
        psi = PSI(nchan, nx, ny, nlevels=args.psi_levels)
        nbasis = psi.nbasis
        prox = lambda p, sig21, w21: prox_21(p, sig21, w21, psi=psi)

        def prox_func(p, sigma_21, weights_21, psi, mu):
            x = mu + p
            x = prox_21(x, sigma_21, weights_21, psi=psi)
            return x - mu

        weights_21 = np.empty(psi.nbasis, dtype=object)
        weights_21[0] = np.ones(nx*ny, dtype=real_type)
        for m in range(1, psi.nbasis):
            weights_21[m] = np.ones(psi.ntot, dtype=real_type)

        def reg(x):
            nchan, nx, ny = x.shape
            nbasis = len(psi.nbasis)
            norm21 = 0.0
            for k in range(psi.nbasis):
                v = psi.hdot(x, k)
                l2norm = norm(v, axis=0)
                norm21 += np.sum(l2norm)
            return norm21
    else:
        prox = prox_21
        psi = None
        weights_21 = np.ones(nx*ny, dtype=real_type)
        def reg(x):
            nchan, nx, ny = x.shape
            # normx = norm(x.reshape(nchan, nx*ny), axis=0)
            normx = np.mean(x.reshape(nchan, nx*ny), axis=0)
            return np.sum(np.abs(normx))

    # deconvolve
    eps = 1.0
    i = 0
    gamma = args.gamma0
    residual = dirty.copy()
    for i in range(args.maxit):
        x = pcg(hess, residual - K.idot(model), np.zeros(dirty.shape, dtype=real_type), M=K.dot, tol=args.cgtol, maxit=args.cgmaxit, verbosity=args.cgverbose)
        
        # update model
        modelp = model
        model = modelp + gamma * x

        # print("Before = ", model[2].max())

        # compute prox
        if args.tidy:
            theta = Xinv @ model.reshape(nband, args.nx * args.ny)
            mu = (X @ theta).reshape(nband, args.nx, args.ny)
            fp = lambda x: fprime(x, model.copy(), hess, mu)
            prox = lambda p : prox_func(p, sig_21[i], weights_21, psi, mu)
            upd, fid, fidu = fista(fp, prox, np.zeros(dirty.shape, dtype=real_type), beta, tol=0.01*args.cgtol, maxit=2*args.cgmaxit)
            model = mu + upd
        else:
            model = prox(model, sig_21[i], weights_21)

        # print("After = ", model[2].max())

        # reweighting
        if i >= args.reweight_start and not i%args.reweight_freq:
            alpha = args.reweight_alpha/(1+i)**args.reweight_alpha_ff
            if psi is None:
                l2norm = norm(model.reshape(nchan, npix), axis=0)
                weights_21 = 1.0/(l2norm + alpha)
            else:
                for m in range(psi.nbasis):
                    v = psi.hdot(model, m)
                    l2norm = norm(v, axis=0)
                    weights_21[m] = 1.0/(l2norm + alpha)

        # get residual
        residual = R.make_residual(model)
       
        # check stopping criteria
        residual_mfs = np.sum(residual, axis=0)/wsum 
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        if i in report_iters:
            # save current iteration
            save_fits(args.outfile + str(i+1) + '_model.fits', model, hdr, dtype=real_type)

            save_fits(args.outfile + str(i+1) + '_mu.fits', mu, hdr, dtype=real_type)
            
            model_mfs = np.mean(model, axis=0)
            save_fits(args.outfile + str(i+1) + '_model_mfs.fits', model_mfs, hdr_mfs)

            save_fits(args.outfile + str(i+1) + '_update.fits', x, hdr)

            save_fits(args.outfile + str(i+1) + '_residual.fits', residual, hdr, dtype=real_type)

            save_fits(args.outfile + str(i+1) + '_residual_mfs.fits', residual_mfs, hdr_mfs)

        print("At iteration %i peak of residual is %f, rms is %f, current eps is %f" % (i, rmax, rms, eps))

    if args.make_restored:
        # get the uninformative Wiener filter soln
        op = lambda x: psf.convolve(x) + 0.0001*x
        M = lambda x: x/0.0001
        x = pcg(op, residual, np.zeros(dirty.shape, dtype=real_type), M=M, tol=0.01*args.cgtol, maxit=2*args.cgmaxit)
        restored = model + x
        
        # get residual
        residual = R.make_residual(restored)
        residual_mfs = np.sum(residual, axis=0)/wsum 
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)

        print("After restoring peak of residual is %f and rms is %f" % (rmax, rms))

        # save current iteration
        save_fits(args.outfile + '_restored.fits', restored, hdr, dtype=real_type)

        restored_mfs = np.mean(restored, axis=0)
        save_fits(args.outfile + '_restored_mfs.fits', restored_mfs, hdr_mfs)

        save_fits(args.outfile + '_restored_residual.fits', residual/psf_max[:, None, None], hdr)

        save_fits(args.outfile + '_restored_residual_mfs.fits', residual_mfs, hdr_mfs)


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
