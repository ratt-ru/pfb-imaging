import numpy as np
from numba import njit, prange
from pyrap.tables import table
from daskms import xds_from_table
import dask
import dask.array as da
from pfb.opt import power_method, hpd, fista, pcg
from scipy.fftpack import next_fast_len
from time import time
import argparse
from astropy.io import fits
from pfb.utils import str2bool, set_wcs, load_fits, save_fits, compare_headers
from pfb.operators import OutMemGridder, PSF, Prior
import scipy.linalg as la
from scipy.stats import laplace

def create_parser():
    p.add_argument("table_name", type=str,
                   help="Name of table to load concatenated Stokes I vis from")
    p.add_argument("--data_column", default="CORRECTED_DATA", type=str,
                   help="The column to image.")
    p.add_argument("--weight_column", default='WEIGHT_SPECTRUM', type=str,
                   help="Weight column to use. Will use WEIGHT if no WEIGHT_SPECTRUM is found")
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
    p.add_argument("--sig_21", type=float, default=1e-3,
                   help="Tolerance for cg updates")
    p.add_argument("--x0", type=str, default=None,
                   help="Initial guess in form of fits file")
    p.add_argument("--reweight_start", type=int, default=20,
                   help="When to start l1 reweighting scheme")
    p.add_argument("--reweight_freq", type=int, default=2,
                   help="How often to do l1 reweighting")
    p.add_argument("--reweight_alpha", type=float, default=0.5,
                   help="Determines how aggressively the reweighting is applied."
                   "1.0 = very mild whereas close to zero = aggressive.")
    p.add_argument("--reweight_aplha_ff", type=float, default=0.25,
                   help="Determines how quickly the reweighting progresses."
                   "alpha will grow like alpha/(1+i)**alpha_ff.")
    p.add_argument("--cgtol", type=float, default=1e-2,
                   help="Tolerance for cg updates")
    p.add_argument("--cgmaxit", type=int, default=10,
                   help="Maximum number of iterations for the cg updates")
    p.add_argument("--cgverbose", type=int, default=0,
                   help="Verbosity of cg method used to invert Hess. Set to 1 or 2 for debugging.")
    p.add_argument("--hpdtol", type=float, default=1e-2,
                   help="Tolerance for hpd sub-iters")
    p.add_argument("--hpdmaxit", type=int, default=10,
                   help="Maximum number of iterations for hpd sub-iters")
    p.add_argument("--pmtol", type=float, default=1e-14,
                   help="Tolerance for power method used to compute spectral norms")
    p.add_argument("--pmmaxit", type=int, default=25,
                   help="Maximum number of iterations for power method")    
    return p

def main(args, table_name, freq, radec):
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
        print("Making PSF.")
        psf_array = R.make_psf()
        save_fits(args.outfile + '_psf.fits', psf_array, hdr_psf, dtype=real_type)
    nband = R.nband
    psf_max = np.amax(psf_array.reshape(nband, 4*args.nx*args.ny), axis=1)
    psf = PSF(psf_array, args.ncpu)

    # dirty
    if args.dirty is not None and args.x0 is None:  # no use for dirty if we are starting from input image
        compare_headers(hdr, fits.getheader(args.dirty))
        dirty = load_fits(args.dirty)
    else:
        if args.x0 is None:
            print("Making dirty.")
            model = np.zeros((nband, args.nx, args.ny), dtype=real_type)
            dirty = R.make_dirty()
            save_fits(args.outfile + '_dirty.fits', dirty, hdr, dtype=real_type)
        else:
            compare_headers(hdr, fits.getheader(args.x0))
            print("Making first residual")
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
    def Uop(x):  
        return psf.convolve(x) + K.idot(x)
    if args.beta is None:
        print("Getting spectral norm of update operator")
        beta = power_method(Uop, dirty.shape, tol=args.pmtol, maxit=args.pmmaxit)
    else:
        beta = args.beta
    print(" beta = %5.5e "%beta)

    # Reweighting
    reweight_iters = list(np.arange(args.reweight_start, args.maxit, args.reweight_freq))

    # Reporting    
    print("At iteration 0 peak of residual is %f and rms is %f" % (rmax, rms))
    report_iters = list(np.arange(0, args.nmiter, args.report_freq))
    if report_iters[-1] != args.nmiter-1:
        report_iters.append(args.nmiter-1)

    # fidelity and gradient term
    def fprime(x, y, A=None):
        if A is None:
            grad = y-x
        else:
            grad = A(y-x)
        return 0.5*np.vdot(y-x, grad), grad
    
    # regulariser
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
        # solve enet vanishing to TK
        # fct = 1e-3/2.0**i
        # print("Thresholding values below %f " % (fct*rmax))
        # x, y, L = fista(op, residual, x, y, L, sig_l2, fct*rmax*L, tol=args.cgtol, maxit=args.cgmaxit, positivity=False)
        x = pcg(Uop, residual, np.zeros(*dirty.shape, dtype=real_type), M=K.dot, tol=args.cgtol, maxit=args.cgmaxit)
        
        # update model
        modelp = model
        model = modelp + gamma * x

        if args.tidy:
            fp = lambda x: fprime(x, model, A=Uop)
        else:
            fp = lambda x: fprime(x, model)

        # compute prox
        model, objhist, fidhist, reghist = hpd(fp, prox_21, reg, modelp, args.gamma0, beta, args.sig_21, 
                                               alpha0=args.reweight_alpha, alpha_ff=args.reweight_alpha_ff, reweight_start=1, reweight_freq=1,
                                               tol=args.hpdtol, maxit=args.hpdmaxit, report_freq=1)

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
            

            mfs_model = np.mean(model, axis=0)
            save_fits(args.outfile + str(i+1) + '_model_mfs.fits', model_mfs, hdr_mfs)

            save_fits(args.outfile + str(i+1) + '_update.fits', x, hdr)

            save_fits(args.outfile + str(i+1) + '_residual.fits', model, hdr, dtype=real_type)
            hdu = fits.PrimaryHDU(header=hdr)
            hdu.data = np.transpose(residual/psf_max[:, None, None], axes=(0, 2, 1))[:, ::-1].astype(np.float32)
            hdu.writeto(args.outfile + args.outname + str(i+1) + '_residual.fits', overwrite=True)

            hdu = fits.PrimaryHDU(header=hdr_mfs)
            hdu.data = residual_mfs.T[::-1].astype(np.float32)
            hdu.writeto(args.outfile + args.outname + str(i) + '_residual_mfs.fits', overwrite=True)

        i += 1
        print("At iteration %i peak of residual is %f, rms is %f, current eps is %f" % (i, rmax, rms, eps))

    # cache results so we can resume if needs be
    np.savez(result_cache_name, model=model, L=L, residual=residual)

    if args.make_restored:
        # get the (flat) Wiener filter soln
        x, y, L = fista(op, residual, 
                        x, y, L, 1.0, 0.0,
                        positivity=False, tol=args.cgtol, maxit=args.cgmaxit)
        
        # x = pcg(A, residual, x, M=K.dot, tol=1e-10, maxit=args.cgmaxit)
        restored = model + x
        # get residual
        residual = R.make_residual(restored)

        residual_mfs = np.sum(residual, axis=0)/wsum 
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)

        print("After restoring peak of residual is %f and rms is %f" % (rmax, rms))

        # save current iteration
        hdu = fits.PrimaryHDU(header=hdr)
        hdu.data = np.transpose(restored, axes=(0, 2, 1))[:, ::-1].astype(np.float32)
        hdu.writeto(args.outfile + args.outname + '_restored.fits', overwrite=True)

        mfs_restored = np.mean(restored, axis=0)
        hdu = fits.PrimaryHDU(header=hdr_mfs)
        hdu.data = mfs_restored.T[::-1].astype(np.float32)
        hdu.writeto(args.outfile + args.outname + '_restored_mfs.fits', overwrite=True)

        hdu = fits.PrimaryHDU(header=hdr)
        hdu.data = np.transpose(residual/psf_max[:, None, None], axes=(0, 2, 1))[:, ::-1].astype(np.float32)
        hdu.writeto(args.outfile + args.outname + '_restored_residual.fits', overwrite=True)

        hdu = fits.PrimaryHDU(header=hdr_mfs)
        hdu.data = residual_mfs.T[::-1].astype(np.float32)
        hdu.writeto(args.outfile + args.outname + '_restored_residual_mfs.fits', overwrite=True)


if __name__=="__main__":
    args = create_parser().parse_args()

    if args.ncpu:
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(args.ncpu))
    else:
        import multiprocessing
        args.ncpu = multiprocessing.cpu_count()

    if args.numba_threads:
        import os
        os.environ.update(NUMBA_NUM_THREADS = str(args.numba_threads))
    else:
        os.environ.update(NUMBA_NUM_THREADS = str(args.ncpu))

    if args.outfile[-1] != '/':
        args.outfile += '/'

    print("Using %i threads"%args.ncpu)

    GD = vars(args)
    print('Input Options:')
    for key in GD.keys():
        print(key, ' = ', GD[key])
    
    # try to open concatenated Stokes I table if it exists otherwise create it
    if args.table_name is None:
        table_name = args.outfile + args.outname + ".table"
    else:
        table_name = args.table_name

    try:
        tbl = xds_from_table(table_name)
        freq = xds_from_table(args.ms[0] + '::SPECTRAL_WINDOW')[0].CHAN_FREQ.data.compute()[0]
        radec = xds_from_table(args.ms[0] + '::FIELD')[0].PHASE_DIR.data.compute().squeeze()
        print("Successfully loaded cached data at %s"%table_name)
    except:
        print("%s does not exist or is invalid. Computing Stokes I visibilities."%table_name)
        # import subprocess
        # subprocess.run("rm -r %s"%table_name)
        freq, radec = concat_ms_to_I_tbl(args.ms, table_name)
        
    
    main(args, table_name, freq, radec)
