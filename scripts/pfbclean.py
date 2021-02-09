#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import numpy as np
from daskms import xds_from_ms, xds_from_table
import dask
import dask.array as da
import argparse
from astropy.io import fits
from numpy.lib.histograms import _ravel_and_check_weights
from pfb.utils import str2bool, set_wcs, load_fits, save_fits, compare_headers
from pfb.operators import Gridder
from pfb.deconv import sara

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
                   help="Channels per chunk (only used for writing component model")
    p.add_argument("--write_model", type=str2bool, nargs='?', const=True, default=True,
                   help="Whether to write model visibilities to model_column")
    p.add_argument("--interp_model", type=str2bool, nargs='?', const=True, default=True,
                   help="Interpolate final model with integrated polynomial")
    p.add_argument("--spectral_poly_order", type=int, default=4,
                   help="Order of interpolating polynomial")
    p.add_argument("--deconv_mode", type=str, default='sara',
                   help="Select minor cycle to use. Current options are 'sara' (default) or 'clean'")
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
    p.add_argument("--power_beam", type=str, default=None,
                   help="Power beam pattern for Stokes I imaging")
    p.add_argument("--point_mask", type=str, default=None,
                   help="A fits mask with a priori known point source locations (True where unmasked)")
    p.add_argument("--do_wstacking", type=str2bool, nargs='?', const=True, default=True,
                   help='Whether to use wstacking or not.')
    p.add_argument("--epsilon", type=float, default=1e-5,
                   help="Accuracy of the gridder")
    p.add_argument("--nthreads", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Step size of 'primal' update.")
    p.add_argument("--maxit", type=int, default=5,
                   help="Number of pfb iterations")
    p.add_argument("--minormaxit", type=int, default=15,
                   help="Number of pfb iterations")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Tolerance")
    p.add_argument("--minortol", type=float, default=1e-3,
                   help="Tolerance")
    p.add_argument("--report_freq", type=int, default=1,
                   help="How often to save output images during deconvolution")
    p.add_argument("--beta", type=float, default=None,
                   help="Lipschitz constant of F")
    p.add_argument("--sig_21", type=float, default=1e-3,
                   help="Initial strength of l21 regulariser."
                   "Initialise to nband x expected rms in MFS dirty if uncertain.")
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
                   help="Set reweighting iters explicitly. "
                   "Default is to reweight at 4th, 5th, 6th, 7th, 8th and 9th iterations.")
    p.add_argument("--reweight_alpha_percent", type=float, default=10,
                   help="Set alpha as using this percentile of non zero coefficients")
    p.add_argument("--reweight_alpha_ff", type=float, default=0.5,
                   help="reweight_alpha_percent will be scaled by this factor after each reweighting step.")
    p.add_argument("--cgtol", type=float, default=1e-6,
                   help="Tolerance for cg updates")
    p.add_argument("--cgmaxit", type=int, default=150,
                   help="Maximum number of iterations for the cg updates")
    p.add_argument("--cgminit", type=int, default=25,
                   help="Minimum number of iterations for the cg updates")
    p.add_argument("--cgverbose", type=int, default=1,
                   help="Verbosity of cg method used to invert Hess. Set to 2 for debugging.")
    p.add_argument("--pmtol", type=float, default=1e-5,
                   help="Tolerance for power method used to compute spectral norms")
    p.add_argument("--pmmaxit", type=int, default=50,
                   help="Maximum number of iterations for power method")
    p.add_argument("--pmverbose", type=int, default=1,
                   help="Verbosity of power method used to get spectral norm of approx Hessian. Set to 2 for debugging.")
    p.add_argument("--pdtol", type=float, default=1e-6,
                   help="Tolerance for primal dual")
    p.add_argument("--pdmaxit", type=int, default=250,
                   help="Maximum number of iterations for primal dual")
    p.add_argument("--tidy", type=str2bool, nargs='?', const=True, default=True,
                   help="Switch off if you prefer it dirty.")
    p.add_argument("--pdverbose", type=int, default=1,
                   help="Verbosity of primal dual used to solve backward step. Set to 2 for debugging.")
    p.add_argument("--make_restored", type=str2bool, nargs='?', const=True, default=True,
                   help="Relax positivity and sparsity constraints at final iteration")
    p.add_argument("--real_type", type=str, default='f4',
                   help="Dtype of real valued images. f4/f8 for single or double precision respectively.")
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
        if cell_N/cell_rad < 1:
            raise ValueError("Requested cell size too small. Super resolution factor = ", cell_N/cell_rad)
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

    # mask
    if args.mask is not None:
        mask_array = load_fits(args.mask, dtype=args.real_type)[None, :, :]
        if mask_array.shape != (1, args.nx, args.ny):
            raise ValueError("Mask has incorrect shape")
        mask = lambda x: mask_array * x
    else:
        mask_array = np.ones((1, args.nx, args.ny), dtype=args.real_type)
        mask = lambda x: x

    if args.power_beam is not None:
        power_beam = load_fits(args.power_beam, dtype=args.real_type)
        # TODO - check correct header
        if power_beam.shape != (args.nband, args.nx, args.ny):
            raise ValueError("Power beam has incorrect shape")
        power_beam *= mask_array
        mask = lambda x: power_beam * x

    # init gridder
    R = Gridder(args.ms, args.nx, args.ny, args.cell_size, nband=args.nband, nthreads=args.nthreads,
                do_wstacking=args.do_wstacking, row_chunks=args.row_chunks, psf_oversize=args.psf_oversize,
                data_column=args.data_column, weight_column=args.weight_column,
                epsilon=args.epsilon, imaging_weight_column=args.imaging_weight_column,
                model_column=args.model_column, flag_column=args.flag_column, mask=mask)
    freq_out = R.freq_out
    radec = R.radec

    # get headers
    hdr = set_wcs(args.cell_size/3600, args.cell_size/3600, args.nx, args.ny, radec, freq_out)
    hdr_mfs = set_wcs(args.cell_size/3600, args.cell_size/3600, args.nx, args.ny, radec, np.mean(freq_out))
    hdr_psf = set_wcs(args.cell_size/3600, args.cell_size/3600, R.nx_psf, R.ny_psf, radec, freq_out)
    
    # psf
    if args.psf is not None:
        try:
            compare_headers(hdr_psf, fits.getheader(args.psf))
            psf = load_fits(args.psf, dtype=args.real_type)
        except:
            psf = R.make_psf()
            save_fits(args.outfile + '_psf.fits', psf, hdr_psf)
    else:
        psf = R.make_psf()
        save_fits(args.outfile + '_psf.fits', psf, hdr_psf)

    # Normalising by wsum (so that the PSF always sums to 1) results in the
    # most intuitive sig_21 values and by far the least bookkeeping.
    # However, we won't save the cubes that way as it destroys information
    # about the noise in image space. Note only the MFS images will have the
    # usual units of Jy/beam.
    wsums = np.amax(psf.reshape(args.nband, R.nx_psf*R.ny_psf), axis=1)
    wsum = np.sum(wsums)
    psf /= wsum
    psf_mfs = np.sum(psf, axis=0)

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
    
    dirty /= wsum
    dirty_mfs = np.sum(dirty, axis=0) 
    save_fits(args.outfile + '_dirty_mfs.fits', dirty_mfs, hdr_mfs)

    # initial model and residual 
    if args.x0 is not None:
        try:
            compare_headers(hdr, fits.getheader(args.x0))
            model = load_fits(args.x0, dtype=args.real_type)
            if args.first_residual is not None:
                try:
                    compare_headers(hdr, fits.getheader(args.first_residual))
                    residual = load_fits(args.first_residual, dtype=args.real_type)
                except:
                    residual = R.make_residual(model)
                    save_fits(args.outfile + '_first_residual.fits', residual, hdr)
            else:
                residual = R.make_residual(model)
                save_fits(args.outfile + '_first_residual.fits', residual, hdr)
            residual /= wsum
        except:
            model = np.zeros((args.nband, args.nx, args.ny))
            residual = dirty.copy()
    else:
        model = np.zeros((args.nband, args.nx, args.ny))
        residual = dirty.copy()

    residual_mfs = np.sum(residual, axis=0) 
    save_fits(args.outfile + '_first_residual_mfs.fits', residual_mfs, hdr_mfs)
        

    # Reweighting
    if args.reweight_iters is not None:
        if not isinstance(args.reweight_iters, list):
            reweight_iters = [args.reweight_iters]
        else:
            reweight_iters = list(args.reweight_iters)
    else:
        reweight_iters = np.arange(5, args.minormaxit, dtype=np.int)

    # Reporting
    report_iters = list(np.arange(0, args.maxit, args.report_freq))
    if report_iters[-1] != args.maxit-1:
        report_iters.append(args.maxit-1)

    # deconvolve
    rmax = np.abs(residual_mfs).max()
    rms = np.std(residual_mfs)
    redo_dirty = False
    print("PFB - Peak of initial residual is %f and rms is %f" % (rmax, rms))
    for i in range(0, args.maxit):
        # run minor cycle of choice
        modelp = model.copy()
        if args.deconv_mode == 'sara':
            # we want to reuse these in subsequent iterations
            if i == 0:
                dual = None 
                weights21 = None 
            
            model, dual, residual_mfs_minor, weights21 = sara(
                psf, model, residual, mask, args.sig_21, dual=dual, weights21=weights21,
                nthreads=args.nthreads,  maxit=args.minormaxit, gamma=args.gamma,  tol=args.minortol,
                psi_levels=args.psi_levels, psi_basis=args.psi_basis,
                reweight_iters=reweight_iters, reweight_alpha_ff=args.reweight_alpha_ff, reweight_alpha_percent=args.reweight_alpha_percent,
                pdtol=args.pdtol, pdmaxit=args.pdmaxit, pdverbose=args.pdverbose, positivity=args.positivity, tidy=args.tidy,
                cgtol=args.cgtol, cgminit=args.cgminit, cgmaxit=args.cgmaxit, cgverbose=args.cgverbose, 
                pmtol=args.pmtol, pmmaxit=args.pmmaxit, pmverbose=args.pmverbose)
            
            # by default do l21 reweighting every iteration from the second major cycle onwards 
            if args.reweight_iters is None:
                reweight_iters = np.arange(args.minormaxit, dtype=np.int)

        else:
            raise ValueError("Unknown deconvolution mode ", args.deconv_mode)


        # get residual
        if redo_dirty:
            # Need to do this if weights or Jones has changed 
            # (i.e. if we change robustness factor, do l2 reweighting or calibration)
            psf = R.make_psf()
            wsums = np.amax(psf.reshape(args.nband, R.nx_psf*R.ny_psf), axis=1)
            wsum = np.sum(wsums)
            psf /= wsum
            dirty = R.make_dirty()/wsum

        
        # compute in image space
        residual = dirty - R.convolve(model)/wsum

        residual_mfs = np.sum(residual, axis=0)

        if i in report_iters:
            # save current iteration
            model_mfs = np.mean(model, axis=0)
            save_fits(args.outfile + str(i+1) + '_model_mfs.fits', model_mfs, hdr_mfs)

            save_fits(args.outfile + str(i+1) + '_residual_mfs.fits', residual_mfs, hdr_mfs)

            save_fits(args.outfile + str(i+1) + '_residual_mfs_minor.fits', residual_mfs_minor, hdr_mfs)

        # check stopping criteria
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        print("PFB - At iteration %i peak of residual is %f, rms is %f, current eps is %f" % (i+1, rmax, rms, eps))

        if eps < args.tol:
            break

    # save model cube and last residual cube
    save_fits(args.outfile + '_model.fits', model, hdr)
    save_fits(args.outfile + '_last_residual.fits', residual*wsum, hdr)
    
    if args.write_model:
        R.write_model(model)


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
