#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import numpy as np
from daskms import xds_from_ms, xds_from_table
from scipy.linalg import norm
import dask
import dask.array as da
from pfb.opt import power_method, pcg, fista, hogbom, primal_dual
import argparse
from astropy.io import fits
from pfb.utils import str2bool, set_wcs, load_fits, save_fits, compare_headers
from pfb.operators import Gridder, PSF, Prior, Dirac


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
    p.add_argument("--interp_model", type=str2bool, nargs='?', const=True, default=True,
                   help="Interpolate final model with integrated polynomial")
    p.add_argument("--spectral_poly_order", type=int, default=4,
                   help="Order of interpolating polynomial")
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
    p.add_argument("--point_mask", type=str, default=None,
                   help="A fits mask with a priori known point source locations (True where unmasked)")
    p.add_argument("--do_wstacking", type=str2bool, nargs='?', const=True, default=True,
                   help='Whether to use wstacking or not.')
    p.add_argument("--nthreads", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Step size of 'primal' update.")
    p.add_argument("--maxit", type=int, default=10,
                   help="Number of pfb iterations")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Tolerance")
    p.add_argument("--report_freq", type=int, default=2,
                   help="How often to save output images during deconvolution")
    p.add_argument("--sig_l2", default=1.0, type=float,
                   help="The strength of the l2 norm regulariser")
    p.add_argument("--sig_21", type=float, default=1e-3,
                   help="Strength of l21 regulariser")
    p.add_argument("--positivity", type=str2bool, nargs='?', const=True, default=True,
                   help='Whether to impose a positivity constraint or not.')
    p.add_argument("--peak_factor", type=float, default=0.1,
                   help="Clean peak factor")
    p.add_argument("--cgamma", type=float, default=0.1,
                   help="Clean step size")
    p.add_argument("--cgtol", type=float, default=1e-3,
                   help="Tolerance for cg updates")
    p.add_argument("--cgmaxit", type=int, default=50,
                   help="Maximum number of iterations for the cg updates")
    p.add_argument("--cgverbose", type=int, default=0,
                   help="Verbosity of cg method used to invert Hess. Set to 1 or 2 for debugging.")
    p.add_argument("--pmtol", type=float, default=1e-14,
                   help="Tolerance for power method used to compute spectral norms")
    p.add_argument("--pmmaxit", type=int, default=25,
                   help="Maximum number of iterations for power method")
    p.add_argument("--pdtol", type=float, default=1e-6,
                   help="Tolerance for fista")
    p.add_argument("--pdmaxit", type=int, default=50,
                   help="Maximum number of iterations for fista")
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
            all_freqs.append(list(tmp_freq))

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
                do_wstacking=args.do_wstacking, row_chunks=args.row_chunks, optimise_chunks=True,
                data_column=args.data_column, weight_column=args.weight_column, imaging_weight_column=imaging_weight_column,
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
        compare_headers(hdr_psf, fits.getheader(args.psf))
        psf_array = load_fits(args.psf)
    else:
        psf_array = R.make_psf()
        save_fits(args.outfile + '_psf.fits', psf_array, hdr_psf)

    
    psf_max = np.amax(psf_array.reshape(args.nband, 4*args.nx*args.ny), axis=1)
    wsum = np.sum(psf_max)
    counts = np.sum(psf_max > 0)
    psf_max_mean = wsum/counts
    psf_array /= psf_max_mean
    psf = PSF(psf_array, args.nthreads)
    psf_max = np.amax(psf_array.reshape(args.nband, 4*args.nx*args.ny), axis=1)
    psf_max[psf_max < 1e-15] = 1e-15


    if args.dirty is not None:
        compare_headers(hdr, fits.getheader(args.dirty))
        dirty = load_fits(args.dirty)
    else:
        dirty = R.make_dirty()
        save_fits(args.outfile + '_dirty.fits', dirty, hdr)

    dirty /= psf_max_mean
    
    # mfs residual 
    wsum = np.sum(psf_max)
    dirty_mfs = np.sum(dirty, axis=0)/wsum 
    rmax = np.abs(dirty_mfs).max()
    rms = np.std(dirty_mfs)
    save_fits(args.outfile + '_dirty_mfs.fits', dirty_mfs, hdr_mfs)       

    psf_mfs = np.sum(psf_array, axis=0)/wsum
    save_fits(args.outfile + '_psf_mfs.fits', psf_mfs[args.nx//2:3*args.nx//2, 
                                                      args.ny//2:3*args.ny//2], hdr_mfs)
    
    # mask
    if args.mask is not None:
        compare_headers(hdr_mfs, fits.getheader(args.mask))
        mask = load_fits(args.mask, dtype=np.bool)
    else:
        mask = np.ones((args.nx, args.ny), dtype=np.bool)

    if args.point_mask is not None:
        compare_headers(hdr_mfs, fits.getheader(args.point_mask))
        pmask = load_fits(args.point_mask, dtype=np.bool)
    else:
        pmask = None

    # Reporting    
    print("At iteration 0 peak of residual is %f and rms is %f" % (rmax, rms))
    report_iters = list(np.arange(0, args.maxit, args.report_freq))
    if report_iters[-1] != args.maxit-1:
        report_iters.append(args.maxit-1)

    # set up point sources
    psi = Dirac(args.nband, args.nx, args.ny, mask=pmask)
    dual = np.zeros((args.nband, args.nx, args.ny), dtype=np.float64)
    weights_21 = np.where(psi.mask, 1, np.inf)

    # preconditioning matrix
    def hess(beta):
        return psi.hdot(psf.convolve(psi.dot(beta))) + beta/args.sig_l2**2  # vague prior on beta

    # deconvolve
    eps = 1.0
    i = 0
    residual = dirty.copy()
    model = np.zeros(dirty.shape, dtype=dirty.dtype)
    for i in range(1, args.maxit):
        # find point source candidates
        model_tmp = hogbom(mask[None] * residual/psf_max[:, None, None], psf_array/psf_max[:, None, None], gamma=args.cgamma, pf=args.peak_factor)
        psi.update_locs(np.any(model_tmp, axis=0))

        # get new spectral norm
        L = power_method(hess, model.shape, tol=args.pmtol, maxit=args.pmmaxit)
        
        # solve for beta updates
        x = pcg(hess, psi.hdot(residual), psi.hdot(model_tmp), 
                M=lambda x: x * args.sig_l2**2, tol=args.cgtol,
                maxit=args.cgmaxit, verbosity=args.cgverbose)

        modelp = model.copy()
        model += args.gamma * x

        # impose sparsity and positivity in point sources
        weights_21 = np.where(psi.mask, 1, np.inf)
        model, dual = primal_dual(hess, model, modelp, dual, args.sig_21, psi, weights_21, L,
                                  tol=args.pdtol, maxit=args.pdmaxit, axis=0,
                                  positivity=args.positivity)

        # update Dirac dictionary (remove zero components)
        psi.trim_fat(model)

        # get residual
        residual = R.make_residual(model)/psf_max_mean
       
        # check stopping criteria
        residual_mfs = np.sum(residual, axis=0)/wsum 
        rmax = np.abs(mask * residual_mfs).max()
        rms = np.std(mask * residual_mfs)
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        if i in report_iters:
            # save current iteration
            save_fits(args.outfile + str(i) + '_model.fits', model, hdr)
            
            model_mfs = np.mean(model, axis=0)
            save_fits(args.outfile + str(i) + '_model_mfs.fits', model_mfs, hdr_mfs)

            save_fits(args.outfile + str(i) + '_residual.fits', residual/psf_max[:, None, None], hdr)

            save_fits(args.outfile + str(i) + '_residual_mfs.fits', residual_mfs, hdr_mfs)

        print("At iteration %i peak of residual is %f, rms is %f, current eps is %f" % (i, rmax, rms, eps))

        if eps < args.tol:
            break

    # final iteration with only a positivity constraint on pixel locs
    x = pcg(hess, psi.hdot(residual), psi.hdot(model), 
            M=lambda x: x * args.sig_l2**2, tol=args.cgtol,
            maxit=args.cgmaxit, verbosity=args.cgverbose)
    
    modelp = model.copy()
    model += x
    model, dual = primal_dual(hess, model, modelp, dual, 0.0,
                              psi, weights_21, L, tol=args.fistatol,
                              maxit=args.fistamaxit, axis=0)
    
    # get residual
    residual = R.make_residual(model)/psf_max_mean
    
    # check stopping criteria
    residual_mfs = np.sum(residual, axis=0)/wsum 
    rmax = np.abs(residual_mfs).max()
    rms = np.std(residual_mfs)
    print("At final iteration peak of residual is %f and rms is %f" % (rmax, rms))

    save_fits(args.outfile + '_model.fits', model, hdr)
            
    model_mfs = np.mean(model, axis=0)
    save_fits(args.outfile + '_model_mfs.fits', model_mfs, hdr_mfs)

    save_fits(args.outfile + '_residual.fits', residual/psf_max[:, None, None], hdr)

    save_fits(args.outfile + '_residual_mfs.fits', residual_mfs, hdr_mfs)
    

    if args.write_model:
        R.write_model(model)

    if args.interp_model:
        nband = args.nband
        order = args.spectral_poly_order
        psi.trim_fat(model)
        I = np.argwhere(psi.mask).squeeze()
        Ix = I[:, 0]
        Iy = I[:, 1]
        npix = I.shape[0]
        
        # get components
        beta = model[:, Ix, Iy]

        # fit integrated polynomial to model components
        # we are given frequencies at bin centers, convert to bin edges
        ref_freq = np.mean(freq_out)
        delta_freq = freq_out[1] - freq_out[0]
        wlow = (freq_out - delta_freq/2.0)/ref_freq
        whigh = (freq_out + delta_freq/2.0)/ref_freq
        wdiff = whigh - wlow

        # set design matrix for each component
        Xdesign = np.zeros([freq_out.size, args.spectral_poly_order])
        for i in range(1, args.spectral_poly_order+1):
            Xdesign[:, i-1] = (whigh**i - wlow**i)/(i*wdiff)

        # # get un-averaged model from monomial design matrix
        # w = (freq_out / ref_freq).reshape(freq_out.size, 1)
        # Xdesign = np.tile(w, order) ** np.arange(0, order)

        dirty_comps = Xdesign.T.dot(beta)
        
        hess_comps = Xdesign.T.dot(Xdesign)
        
        comps = np.linalg.solve(hess_comps, dirty_comps)

        beta_rec = Xdesign.dot(comps)
        model_interp = np.zeros(model.shape, dtype=model.dtype)
        for i, xy in enumerate(I):
            ix = xy[0]
            iy = xy[1]
            model_interp[:, ix, iy] = beta_rec[:, i]

        save_fits(args.outfile + '_model_interp.fits', model_interp, hdr)

        np.savez(args.outfile + "spectral_comps", comps=comps, ref_freq=ref_freq, mask=np.any(model, axis=0))

    # if args.make_restored:
    #     # # get the clean beam
    #     # L = power_method(hess, dirty.shape, tol=args.pmtol, maxit=args.pmmaxit)
    #     # psf_residual = psf_array[:, args.nx//2:3*args.nx//2, args.ny//2:3*args.ny//2]
    #     # eps = 1.0
    #     # psf_clean = np.zeros(dirty.shape, dtype=dirty.dtype)
    #     # M = lambda x: x * args.sig_l2**2
    #     # for k in range(args.maxit):
    #     #     x = pcg(hess, psf_residual, np.zeros(dirty.shape, dtype=np.float64), M=M, tol=args.cgtol, maxit=args.cgmaxit)

    #     #     psf_cleanp = psf_clean
    #     #     psf_clean += x

    #     #     # impose positivity
    #     #     psf_clean, _, _ = fista(hess, 
    #     #                             psf_clean, psf_cleanp, 0.0,  L,
    #     #                             tol=args.fistatol, maxit=args.fistamaxit, report_freq=10)

    #     #     psf_residual = psf_array[:, args.nx//2:3*args.nx//2, args.ny//2:3*args.ny//2] - psf.convolve(psf_clean)

    #     #     eps = np.linalg.norm(psf_clean - psf_cleanp)/np.linalg.norm(psf_clean)

    #     #     if eps < args.tol:
    #     #         break

    #     # # save clean beam
    #     # save_fits(args.outfile + '_psf_clean.fits', psf_clean, hdr)

    #     # quit()

    #     # augment data
    #     y = np.append(residual.reshape(args.nband, args.nx*args.ny), H.hdot(residual), axis=1)

    #     # augment dictionary
    #     def psi(x):
    #         alpha = x[:, 0:args.nx*args.ny].reshape(args.nband, args.nx, args.ny)
    #         beta = x[:, args.nx*args.ny::] 
    #         return alpha + H.dot(beta)
        
    #     def psih(x):
    #         y = x.reshape(args.nband, args.nx*args.ny)
    #         y = np.append(y, H.hdot(x), axis=1)
    #         return y

    #     # augment hess
    #     def hess(x):
    #         return psih(psf.convolve(psi(x))) + x/args.sig_l2**2
        
    #     # get Wiener filter soln
    #     M = lambda x: x * args.sig_l2**2
    #     x = pcg(hess, y, np.zeros(y.shape, dtype=np.float64), M=M, tol=args.cgtol, maxit=args.cgmaxit)
    #     restored = model + psi(x)
        
    #     # get residual
    #     residual = R.make_residual(restored)
    #     residual_mfs = np.sum(residual, axis=0)/wsum 
    #     rmax = np.abs(residual_mfs).max()
    #     rms = np.std(residual_mfs)

    #     print("After restoring peak of residual is %f and rms is %f" % (rmax, rms))

    #     # save current iteration
    #     save_fits(args.outfile + '_restored.fits', restored, hdr)

    #     restored_mfs = np.mean(restored, axis=0)
    #     save_fits(args.outfile + '_restored_mfs.fits', restored_mfs, hdr_mfs)

    #     save_fits(args.outfile + '_restored_residual.fits', residual/psf_max[:, None, None], hdr)

    #     save_fits(args.outfile + '_restored_residual_mfs.fits', residual_mfs, hdr_mfs)


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
