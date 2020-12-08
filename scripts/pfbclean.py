#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import numpy as np
from daskms import xds_from_ms, xds_from_table
from scipy.linalg import norm
import dask
import dask.array as da
from pfb.opt import power_method, pcg, primal_dual
import argparse
from astropy.io import fits
from pfb.utils import str2bool, set_wcs, load_fits, save_fits, compare_headers, prox_21
from pfb.operators import Gridder, PSF, DaskPSI, PSI

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
    p.add_argument("--write_model", type=str2bool, nargs='?', const=True, default=True,
                   help="Whether to write model visibilities to model_column")
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
    p.add_argument("--nband", default=None, type=int,
                   help="Number of imaging bands in output cube")
    p.add_argument("--mask", type=str, default=None,
                   help="A fits mask (True where unmasked)")
    p.add_argument("--do_wstacking", type=str2bool, nargs='?', const=True, default=True,
                   help='Whether to use wstacking or not.')
    p.add_argument("--epsilon", type=float, default=1e-4,
                   help="Accuracy of the gridder")
    p.add_argument("--nthreads", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.95,
                   help="Step size of 'primal' update.")
    p.add_argument("--maxit", type=int, default=10,
                   help="Number of pfb iterations")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Tolerance")
    p.add_argument("--report_freq", type=int, default=2,
                   help="How often to save output images during deconvolution")
    p.add_argument("--beta", type=float, default=None,
                   help="Lipschitz constant of F")
    p.add_argument("--sig_l2", default=1.0, type=float,
                   help="The strength of the l2 norm regulariser")
    p.add_argument("--sig_21", type=float, default=1e-3,
                   help="Strength of l21 regulariser")
    p.add_argument("--positivity", type=str2bool, nargs='?', const=True, default=True,
                   help='Whether to impose a positivity constraint or not.')
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
                   help="Set reweighting iters explicitly")
    p.add_argument("--reweight_start", type=int, default=50,
                   help="When to start l21 reweighting scheme")
    p.add_argument("--reweight_freq", type=int, default=2,
                   help="How often to do l21 reweighting")
    p.add_argument("--reweight_end", type=int, default=90,
                   help="When to end the l21 reweighting scheme")
    p.add_argument("--reweight_alpha_min", type=float, default=1.0e-8,
                   help="MInimum alpha in l21 reweighting formula.")
    p.add_argument("--reweight_alpha_percent", type=float, default=10)
    p.add_argument("--reweight_alpha_ff", type=float, default=0.5,
                   help="Determines how quickly the reweighting progresses."
                   "reweight_alpha_percent will be scaled by this factor after each reweighting step.")
    p.add_argument("--cgtol", type=float, default=1e-5,
                   help="Tolerance for cg updates")
    p.add_argument("--cgmaxit", type=int, default=200,
                   help="Maximum number of iterations for the cg updates")
    p.add_argument("--cgminit", type=int, default=100,
                   help="Minimum number of iterations for the cg updates")
    p.add_argument("--cgverbose", type=int, default=1,
                   help="Verbosity of cg method used to invert Hess. Set to 1 or 2 for debugging.")
    p.add_argument("--pmtol", type=float, default=1e-4,
                   help="Tolerance for power method used to compute spectral norms")
    p.add_argument("--pmmaxit", type=int, default=50,
                   help="Maximum number of iterations for power method")
    p.add_argument("--pdtol", type=float, default=1e-4,
                   help="Tolerance for primal dual")
    p.add_argument("--pdmaxit", type=int, default=300,
                   help="Maximum number of iterations for primal dual")
    p.add_argument("--make_restored", type=str2bool, nargs='?', const=True, default=True,
                   help="Relax positivity and sparsity constraints at final iteration")
    return p

def main(args):
    # get max uv coords over all fields
    uvw = []
    u_max = 0.0
    v_max = 0.0
    all_freqs = []
    for ims in args.ms:
        xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'), columns=('UVW'), chunks={'row':args.row_chunks})

        spws = xds_from_table(ims + "::SPECTRAL_WINDOW", group_cols="__row__")
        spws = dask.compute(spws)[0]

        for ds in xds:
            uvw = ds.UVW.data
            u_max = da.maximum(u_max, abs(uvw[:, 0]).max())
            v_max = da.maximum(v_max, abs(uvw[:, 1]).max())
            uv_max = da.maximum(u_max, v_max)

            spw = spws[ds.DATA_DESC_ID]
            tmp_freq = spw.CHAN_FREQ.data.squeeze()
            all_freqs.append(list([tmp_freq]))

    uv_max = u_max.compute()
    del uvw

    # get Nyquist cell size
    from africanus.constants import c as lightspeed
    all_freqs = dask.compute(all_freqs)
    freq = np.unique(all_freqs)
    cell_N = 1.0/(2*uv_max*freq.max()/lightspeed)

    if args.cell_size is not None:
        cell_rad = args.cell_size * np.pi/60/60/180
        print("Super resolution factor = ", cell_N/cell_rad)
    else:
        cell_rad = cell_N/args.super_resolution_factor
        args.cell_size = cell_rad*60*60*180/np.pi
        print("Cell size set to %5.5e arcseconds" % args.cell_size)
    
    if args.nx is None or args.ny is None:
        fov = args.fov*3600
        npix = int(fov/args.cell_size)
        if npix % 2:
            npix += 1
        args.nx = npix
        args.ny = npix

    if args.nband is None:
        args.nband = freq.size

    print("Image size set to (%i, %i, %i)"%(args.nband, args.nx, args.ny))

    # init gridder
    R = Gridder(args.ms, args.nx, args.ny, args.cell_size, nband=args.nband, nthreads=args.nthreads,
                do_wstacking=args.do_wstacking, row_chunks=args.row_chunks,
                data_column=args.data_column, weight_column=args.weight_column,
                epsilon=args.epsilon, imaging_weight_column=args.imaging_weight_column,
                model_column=args.model_column, flag_column=args.flag_column)
    freq_out = R.freq_out
    radec = R.radec

    # get headers
    hdr = set_wcs(args.cell_size/3600, args.cell_size/3600, args.nx, args.ny, radec, freq_out)
    hdr_mfs = set_wcs(args.cell_size/3600, args.cell_size/3600, args.nx, args.ny, radec, np.mean(freq_out))
    hdr_psf = set_wcs(args.cell_size/3600, args.cell_size/3600, 2*args.nx, 2*args.ny, radec, freq_out)
    hdr_psf_mfs = set_wcs(args.cell_size/3600, args.cell_size/3600, 2*args.nx, 2*args.ny, radec, np.mean(freq_out))
    
    # psf
    if args.psf is not None:
        try:
            compare_headers(hdr_psf, fits.getheader(args.psf))
            psf_array = load_fits(args.psf)
        except:
            psf_array = R.make_psf()
            save_fits(args.outfile + '_psf.fits', psf_array, hdr_psf)
    else:
        psf_array = R.make_psf()
        save_fits(args.outfile + '_psf.fits', psf_array, hdr_psf)

    psf_max = np.amax(psf_array.reshape(args.nband, 4*args.nx*args.ny), axis=1)
    wsum = np.sum(psf_max)
    counts = np.sum(psf_max > 0)
    psf_max_mean = wsum/counts  # normalissation for more intuitive sig_21 values
    psf_array /= psf_max_mean
    psf = PSF(psf_array, args.nthreads)
    psf_max = np.amax(psf_array.reshape(args.nband, 4*args.nx*args.ny), axis=1)
    wsum = np.sum(psf_max)
    psf_max[psf_max < 1e-15] = 1e-15  # LB - is this the right thing to do?

    psf_mfs = np.sum(psf_array, axis=0)/wsum
    save_fits(args.outfile + '_psf_mfs.fits', psf_mfs[args.nx//2:3*args.nx//2, 
                                                      args.ny//2:3*args.ny//2], hdr_mfs)

    # dirty
    if args.dirty is not None:
        try:
            compare_headers(hdr, fits.getheader(args.dirty))
            dirty = load_fits(args.dirty)
        except:
            dirty = R.make_dirty()
            save_fits(args.outfile + '_dirty.fits', dirty, hdr)
    else:
        dirty = R.make_dirty()
        save_fits(args.outfile + '_dirty.fits', dirty, hdr)
    
    dirty_mfs = np.sum(dirty/psf_max_mean, axis=0)/wsum 
    save_fits(args.outfile + '_dirty_mfs.fits', dirty_mfs, hdr_mfs)
    
    if args.x0 is not None:
        try:
            compare_headers(hdr, fits.getheader(args.x0))
            model = load_fits(args.x0, dtype=np.float64)
            if args.first_residual is not None:
                try:
                    compare_headers(hdr, fits.getheader(args.first_residual))
                    residual = load_fits(args.first_residual, dtype=np.float64)
                except:
                    residual = R.make_residual(model)
                    save_fits(args.outfile + '_first_residual.fits', residual, hdr)
            else:
                residual = R.make_residual(model)
                save_fits(args.outfile + '_first_residual.fits', residual, hdr)
        except:
            model = np.zeros((args.nband, args.nx, args.ny))
            residual = dirty.copy()
    else:
        model = np.zeros((args.nband, args.nx, args.ny))
        residual = dirty.copy()

    # normalise for more intuitive hypers
    residual /= psf_max_mean
    residual_mfs = np.sum(residual, axis=0)/wsum 
    save_fits(args.outfile + '_first_residual_mfs.fits', residual_mfs, hdr_mfs)

    # mask
    if args.mask is not None:
        mask = load_fits(args.mask, dtype=np.int64)[None, :, :]
        if mask.shape != (1, args.nx, args.ny):
            raise ValueError("Mask has incorrect shape")
    else:
        mask = np.ones((1, args.nx, args.ny), dtype=np.int64)

    #  preconditioning matrix
    def hess(x):  
        return mask*psf.convolve(mask*x) + x / args.sig_l2**2 
    
    if args.beta is None:
        print("Getting spectral norm of update operator")
        beta = power_method(hess, dirty.shape, tol=args.pmtol, maxit=args.pmmaxit)
    else:
        beta = args.beta
    print(" beta = %f "%beta)

    # set up wavelet basis
    if args.psi_basis is None:
        print("Using Dirac + db1-4 dictionary")
        # psi = DaskPSI(args.nband, args.nx, args.ny, nlevels=args.psi_levels,
        #                 nthreads=args.nthreads)
        psi = PSI(args.nband, args.nx, args.ny, nlevels=args.psi_levels)
    else:
        if not isinstance(args.psi_basis, list):
            args.psi_basis = list(args.psi_basis)
        print("Using ", args.psi_basis, " dictionary")
        # psi = DaskPSI(args.nband, args.nx, args.ny, nlevels=args.psi_levels,
        #                 nthreads=args.nthreads, bases=args.psi_basis)
        psi = PSI(args.nband, args.nx, args.ny, nlevels=args.psi_levels, bases=args.psi_basis)
    nbasis = psi.nbasis
    weights_21 = np.ones((psi.nbasis, psi.nmax), dtype=np.float64)
    dual = np.zeros((psi.nbasis, args.nband, psi.nmax), dtype=np.float64)

    # Reweighting
    if args.reweight_iters is not None:
        if not isinstance(args.reweight_iters, list):
            reweight_iters = [args.reweight_iters]
        else:    
            reweight_iters = list(args.reweight_iters)
    else:
        reweight_iters = list(np.arange(args.reweight_start, args.reweight_end, args.reweight_freq))
        reweight_iters.append(args.reweight_end)

    # Reporting
    report_iters = list(np.arange(0, args.maxit, args.report_freq))
    if report_iters[-1] != args.maxit-1:
        report_iters.append(args.maxit-1)

    # deconvolve
    eps = 1.0
    i = 0
    rmax = np.abs(residual_mfs).max()
    rms = np.std(residual_mfs)
    M = lambda x: x * args.sig_l2**2  # preconditioner
    print("Peak of initial residual is %f and rms is %f" % (rmax, rms))
    for i in range(1, args.maxit):
        x = pcg(hess, mask*residual, np.zeros(dirty.shape, dtype=np.float64), M=M, tol=args.cgtol,
                maxit=args.cgmaxit, minit=args.cgminit, verbosity=args.cgverbose)

        if i in report_iters:
            save_fits(args.outfile + str(i) + '_update.fits', x, hdr)
        
        # update model
        modelp = model
        model = modelp + args.gamma * x
        model, dual = primal_dual(hess, model, modelp, dual, args.sig_21, psi, weights_21, beta,
                                  tol=args.pdtol, maxit=args.pdmaxit, report_freq=100, mask=mask,
                                  positivity=args.positivity)

        # reweighting
        if i in reweight_iters:
            v = psi.hdot(model)
            l2_norm = norm(v, axis=1)
            l2_norm = np.where(l2_norm < args.sig_21*weights_21, 0.0, l2_norm)
            for m in range(psi.nbasis):
                indnz = l2_norm[m].nonzero()
                alpha = np.percentile(l2_norm[m, indnz].flatten(), args.reweight_alpha_percent)
                alpha = np.maximum(alpha, args.reweight_alpha_min)
                print("Reweighting - ", m, alpha)
                weights_21[m] = alpha/(l2_norm[m] + alpha)
            args.reweight_alpha_percent *= args.reweight_alpha_ff
            # print(" reweight alpha percent = ", args.reweight_alpha_percent)

        # get residual
        residual = R.make_residual(model)/psf_max_mean
       
        # check stopping criteria
        residual_mfs = np.sum(residual, axis=0)/wsum 
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        if i in report_iters:
            # save current iteration
            save_fits(args.outfile + str(i) + '_model.fits', model, hdr)
            
            model_mfs = np.mean(model, axis=0)
            save_fits(args.outfile + str(i) + '_model_mfs.fits', model_mfs, hdr_mfs)

            save_fits(args.outfile + str(i) + '_residual.fits', residual, hdr)

            save_fits(args.outfile + str(i) + '_residual_mfs.fits', residual_mfs, hdr_mfs)

        print("At iteration %i peak of residual is %f, rms is %f, current eps is %f" % (i, rmax, rms, eps))

    if args.write_model:
        R.write_model(model)

    if args.make_restored:
        x = pcg(hess, residual, np.zeros(dirty.shape, dtype=np.float64), M=M, tol=args.cgtol, maxit=args.cgmaxit)
        restored = model + x
        
        # get residual
        residual = R.make_residual(restored)/psf_max_mean
        residual_mfs = np.sum(residual, axis=0)/wsum 
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)

        print("After restoring peak of residual is %f and rms is %f" % (rmax, rms))

        # save current iteration
        save_fits(args.outfile + '_restored.fits', restored, hdr)

        restored_mfs = np.mean(restored, axis=0)
        save_fits(args.outfile + '_restored_mfs.fits', restored_mfs, hdr_mfs)

        save_fits(args.outfile + '_restored_residual.fits', residual, hdr)

        save_fits(args.outfile + '_restored_residual_mfs.fits', residual_mfs, hdr_mfs)


if __name__=="__main__":
    args = create_parser().parse_args()

    if args.nthreads:
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(args.nthreads))
    else:
        import multiprocessing
        args.nthreads = multiprocessing.cpu_count()

    if not isinstance(args.ms, list):
        args.ms = [args.ms]

    GD = vars(args)
    print('Input Options:')
    for key in GD.keys():
        print(key, ' = ', GD[key])
    

    main(args)
