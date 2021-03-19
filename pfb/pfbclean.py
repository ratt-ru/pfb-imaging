#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import sys
import numpy as np
from daskms import xds_from_ms, xds_from_table
import dask
import dask.array as da
from astropy.io import fits
from pfb.utils import (set_wcs, load_fits, save_fits, compare_headers,
                       data_from_header, fitcleanbeam, Gaussian2D)
from pfb.operators import Gridder, PSF
from pfb.deconv import sara, clean, spotless
from pfb.opt import pcg
from pfb.parser import create_parser
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('PFB')


def _main(args, dest=sys.stdout):
    # get max uv coords over all fields
    uvw = []
    u_max = 0.0
    v_max = 0.0
    all_freqs = []
    for ims in args.ms:
        xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                          columns=('UVW'), chunks={'row': args.row_chunks})

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
    cell_N = 1.0 / (2 * uv_max * freq.max() / lightspeed)

    if args.cell_size is not None:
        cell_rad = args.cell_size * np.pi / 60 / 60 / 180
        if cell_N / cell_rad < 1:
            raise ValueError("Requested cell size too small. "
                             "Super resolution factor = ", cell_N / cell_rad)
        print("Super resolution factor = %f" % (cell_N / cell_rad), file=dest)
    else:
        cell_rad = cell_N / args.super_resolution_factor
        args.cell_size = cell_rad * 60 * 60 * 180 / np.pi
        print("Cell size set to %5.5e arcseconds" % args.cell_size, file=dest)

    if args.nx is None or args.ny is None:
        fov = args.fov * 3600
        npix = int(fov / args.cell_size)
        if npix % 2:
            npix += 1
        args.nx = npix
        args.ny = npix

    if args.nband is None:
        args.nband = freq.size

    print("Image size set to (%i, %i, %i)" % (args.nband, args.nx, args.ny),
          file=dest)

    # mask
    if args.mask is not None:
        mask_array = load_fits(args.mask, dtype=args.real_type).squeeze()
        if mask_array.shape != (args.nx, args.ny):
            raise ValueError("Mask has incorrect shape.")
        # add freq axis
        mask_array = mask_array[None]
        def mask(x): return mask_array * x
    else:
        mask_array = None
        def mask(x): return x

    # init gridder
    R = Gridder(args.ms, args.nx, args.ny, args.cell_size, nband=args.nband,
                nthreads=args.nthreads, do_wstacking=args.do_wstacking,
                row_chunks=args.row_chunks, psf_oversize=args.psf_oversize,
                data_column=args.data_column, epsilon=args.epsilon,
                weight_column=args.weight_column,
                imaging_weight_column=args.imaging_weight_column,
                model_column=args.model_column, flag_column=args.flag_column,
                weighting=args.weighting, robust=args.robust)
    freq_out = R.freq_out
    radec = R.radec

    # get headers
    hdr = set_wcs(args.cell_size / 3600, args.cell_size / 3600,
                  args.nx, args.ny, radec, freq_out)
    hdr_mfs = set_wcs(args.cell_size / 3600, args.cell_size / 3600, args.nx,
                      args.ny, radec, np.mean(freq_out))
    hdr_psf = set_wcs(args.cell_size / 3600, args.cell_size / 3600, R.nx_psf,
                      R.ny_psf, radec, freq_out)
    hdr_psf_mfs = set_wcs(args.cell_size / 3600, args.cell_size / 3600,
                          R.nx_psf, R.ny_psf, radec, np.mean(freq_out))

    # psf
    if args.psf is not None:
        try:
            compare_headers(hdr_psf, fits.getheader(args.psf))
            psf = load_fits(args.psf, dtype=args.real_type).squeeze()
        except BaseException:
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
    wsums = np.amax(psf.reshape(args.nband, R.nx_psf * R.ny_psf), axis=1)
    wsum = np.sum(wsums)
    psf /= wsum
    psf_mfs = np.sum(psf, axis=0)

    # fit restoring psf
    GaussPar = fitcleanbeam(psf_mfs[None], level=0.5, pixsize=1.0)
    GaussPars = fitcleanbeam(psf, level=0.5, pixsize=1.0)

    cpsf_mfs = np.zeros(psf_mfs.shape, dtype=args.real_type)
    cpsf = np.zeros(psf.shape, dtype=args.real_type)

    lpsf = np.arange(-R.nx_psf / 2, R.nx_psf / 2)
    mpsf = np.arange(-R.ny_psf / 2, R.ny_psf / 2)
    xx, yy = np.meshgrid(lpsf, mpsf, indexing='ij')

    cpsf_mfs = Gaussian2D(xx, yy, GaussPar[0], normalise=False)

    for v in range(args.nband):
        cpsf[v] = Gaussian2D(xx, yy, GaussPars[v], normalise=False)

    from pfb.utils import add_beampars
    GaussPar = list(GaussPar[0])
    GaussPar[0] *= args.cell_size / 3600
    GaussPar[1] *= args.cell_size / 3600
    GaussPar = tuple(GaussPar)
    hdr_psf_mfs = add_beampars(hdr_psf_mfs, GaussPar)

    save_fits(args.outfile + '_cpsf_mfs.fits', cpsf_mfs, hdr_psf_mfs)
    save_fits(args.outfile + '_psf_mfs.fits', psf_mfs, hdr_psf_mfs)

    GaussPars = list(GaussPars)
    for b in range(args.nband):
        GaussPars[b] = list(GaussPars[b])
        GaussPars[b][0] *= args.cell_size / 3600
        GaussPars[b][1] *= args.cell_size / 3600
        GaussPars[b] = tuple(GaussPars[b])
    GaussPars = tuple(GaussPars)
    hdr_psf = add_beampars(hdr_psf, GaussPar, GaussPars)

    save_fits(args.outfile + '_cpsf.fits', cpsf, hdr_psf)

    # dirty
    if args.dirty is not None:
        try:
            compare_headers(hdr, fits.getheader(args.dirty))
            dirty = load_fits(args.dirty).squeeze()
        except BaseException:
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
            model = load_fits(args.x0, dtype=args.real_type).squeeze()
            if args.first_residual is not None:
                try:
                    compare_headers(hdr, fits.getheader(args.first_residual))
                    residual = load_fits(args.first_residual,
                                         dtype=args.real_type).squeeze()
                except BaseException:
                    residual = R.make_residual(model)
                    save_fits(args.outfile + '_first_residual.fits',
                              residual, hdr)
            else:
                residual = R.make_residual(model)
                save_fits(args.outfile + '_first_residual.fits', residual,
                          hdr)
            residual /= wsum
        except BaseException:
            model = np.zeros((args.nband, args.nx, args.ny))
            residual = dirty.copy()
    else:
        model = np.zeros((args.nband, args.nx, args.ny))
        residual = dirty.copy()

    residual_mfs = np.sum(residual, axis=0)
    save_fits(args.outfile + '_first_residual_mfs.fits', residual_mfs,
              hdr_mfs)

    # smooth beam
    if args.beam_model is not None:
        if args.beam_model[-5:] == '.fits':
            beam_image = load_fits(args.beam_model,
                                   dtype=args.real_type).squeeze()
            if beam_image.shape != (args.nband, args.nx, args.ny):
                raise ValueError("Beam has incorrect shape")

        elif args.beam_model == "JimBeam":
            from katbeam import JimBeam
            if args.band.lower() == 'l':
                beam = JimBeam('MKAT-AA-L-JIM-2020')
            else:
                beam = JimBeam('MKAT-AA-UHF-JIM-2020')
            beam_image = np.zeros((args.nband, args.nx, args.ny),
                                  dtype=args.real_type)

            l_coord, ref_l = data_from_header(hdr, axis=1)
            l_coord -= ref_l
            m_coord, ref_m = data_from_header(hdr, axis=2)
            m_coord -= ref_m
            xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')

            for v in range(args.nband):
                beam_image[v] = beam.I(xx, yy, freq_out[v])

        def beam(x): return beam_image * x
    else:
        beam_image = None
        def beam(x): return x

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
    if report_iters[-1] != args.maxit - 1:
        report_iters.append(args.maxit - 1)

    # deconvolve
    rmax = np.abs(residual_mfs).max()
    rms = np.std(residual_mfs)
    redo_dirty = False
    print("Peak of initial residual is %f and rms is %f" % (rmax, rms),
          file=dest)
    for i in range(0, args.maxit):
        # run minor cycle of choice
        modelp = model.copy()
        if args.deconv_mode == 'sara':
            # we want to reuse these in subsequent iterations
            if i == 0:
                dual = None
                weights21 = None

            model, dual, residual_mfs_minor, weights21 = sara(
                psf, model, residual, sig_21=args.sig_21,
                sigma_frac=args.sigma_frac, mask=mask_array, beam=beam_image,
                dual=dual, weights21=weights21, nthreads=args.nthreads,
                maxit=args.minormaxit, gamma=args.gamma, tol=args.minortol,
                psi_levels=args.psi_levels, psi_basis=args.psi_basis,
                reweight_iters=reweight_iters,
                reweight_alpha_ff=args.reweight_alpha_ff,
                reweight_alpha_percent=args.reweight_alpha_percent,
                pdtol=args.pdtol, pdmaxit=args.pdmaxit,
                pdverbose=args.pdverbose, positivity=args.positivity,
                tidy=args.tidy, cgtol=args.cgtol, cgminit=args.cgminit,
                cgmaxit=args.cgmaxit, cgverbose=args.cgverbose,
                pmtol=args.pmtol, pmmaxit=args.pmmaxit,
                pmverbose=args.pmverbose)

            # by default do l21 reweighting every iteration from
            # the second major cycle onwards
            if args.reweight_iters is None:
                reweight_iters = np.arange(args.minormaxit, dtype=np.int)
        elif args.deconv_mode == 'clean':
            model, residual_mfs_minor = clean(
                psf, model, residual, mask=mask_array, beam=beam_image,
                nthreads=args.nthreads, maxit=args.minormaxit,
                gamma=args.gamma, peak_factor=args.peak_factor,
                threshold=args.threshold, hbgamma=args.hbgamma,
                hbpf=args.hbpf, hbmaxit=args.hbmaxit,
                hbverbose=args.hbverbose)
        elif args.deconv_mode == 'spotless':
            model, residual_mfs_minor = spotless(
                psf, model, residual, mask=mask_array, beam=beam_image,
                sig_21=args.sig_21, sigma_frac=args.sigma_frac,
                nthreads=args.nthreads, tidy=args.tidy, gamma=args.gamma,
                maxit=args.minormaxit, tol=args.minortol,
                threshold=args.threshold, positivity=args.positivity,
                hbgamma=args.hbgamma, hbpf=args.hbpf, hbmaxit=args.hbmaxit,
                hbverbose=args.hbverbose, pdtol=args.pdtol,
                pdmaxit=args.pdmaxit, pdverbose=args.pdverbose,
                cgtol=args.cgtol, cgminit=args.cgminit,
                cgmaxit=args.cgmaxit, cgverbose=args.cgverbose,
                pmtol=args.pmtol, pmmaxit=args.pmmaxit,
                pmverbose=args.pmverbose)
        else:
            raise ValueError("Unknown deconvolution mode ", args.deconv_mode)

        # get residual
        if redo_dirty:
            # Need to do this if weights or Jones has changed
            # (eg. if we change robustness factor, reweight or calibrate)
            psf = R.make_psf()
            wsums = np.amax(psf.reshape(args.nband, R.nx_psf * R.ny_psf),
                            axis=1)
            wsum = np.sum(wsums)
            psf /= wsum
            dirty = R.make_dirty() / wsum

        # compute in image space
        # residual = dirty - R.convolve(beam(mask(model))) / wsum
        residual = R.make_residual(beam(mask(model)))/wsum

        residual_mfs = np.sum(residual, axis=0)

        if i in report_iters:
            # save current iteration
            model_mfs = np.mean(model, axis=0)
            save_fits(args.outfile + str(i + 1) + '_model_mfs.fits',
                      model_mfs, hdr_mfs)

            save_fits(args.outfile + str(i + 1) + '_residual_mfs.fits',
                      residual_mfs, hdr_mfs)

            save_fits(args.outfile + str(i + 1) + '_residual_mfs_minor.fits',
                      residual_mfs_minor, hdr_mfs)

        # check stopping criteria
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)
        eps = np.linalg.norm(model - modelp) / np.linalg.norm(model)

        print("At iteration %i peak of residual is %f, rms is %f, current "
              "eps is %f" % (i + 1, rmax, rms, eps), file=dest)

        if eps < args.tol:
            break

    if args.mop_flux:
        print("Mopping flux", file=dest)
        # extacts flux where model is non-zero
        mask_array2 = np.any(model, axis=0)[None]
        def mask2(x): return mask_array2 * x
        psfo = PSF(psf, nthreads=args.nthreads, imsize=residual.shape)

        # vague Gaussian prior on x
        def hess(x):
            return mask2(beam(psfo.convolve(mask2(beam(x))))) + 1e-6 * x

        def M(x): return x / 1e-6  # preconditioner
        x = pcg(
            hess,
            mask2(
                beam(residual)),
            np.zeros(
                residual.shape,
                dtype=residual.dtype),
            M=M,
            tol=args.cgtol,
            maxit=args.cgmaxit,
            minit=args.cgminit,
            verbosity=args.cgverbose)

        model += x
        # residual = dirty - R.convolve(beam(mask(model))) / wsum
        residual = R.make_residual(beam(mask(model)))/wsum

        save_fits(args.outfile + '_mopped_model.fits', model, hdr)
        save_fits(args.outfile + '_mopped_residual.fits', residual, hdr)
        model_mfs = np.mean(model, axis=0)
        save_fits(args.outfile + '_mopped_model_mfs.fits', model_mfs, hdr_mfs)
        residual_mfs = np.sum(residual, axis=0)
        save_fits(args.outfile + '_mopped_residual_mfs.fits', residual_mfs,
                  hdr_mfs)

        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)

        print("After mopping flux peak of residual is %f, rms is %f" %
              (rmax, rms), file=dest)

    # save model cube and last residual cube
    save_fits(args.outfile + '_model.fits', model, hdr)
    save_fits(args.outfile + '_last_residual.fits', residual * wsum, hdr)

    if args.write_model:
        print("Writing model", file=dest)
        R.write_model(model)

    if args.make_restored:
        print("Making restored", file=dest)
        cpsfo = PSF(cpsf, nthreads=args.nthreads, imsize=residual.shape)
        restored = cpsfo.convolve(model)

        # residual needs to be in Jy/beam before adding to convolved model
        wsums = np.amax(psf.reshape(-1, R.nx_psf * R.ny_psf), axis=1)
        restored += residual / wsums[:, None, None]

        save_fits(args.outfile + '_restored.fits', restored, hdr)
        restored_mfs = np.mean(restored, axis=0)
        save_fits(args.outfile + '_restored_mfs.fits', restored_mfs, hdr_mfs)
        residual_mfs = np.sum(residual, axis=0)
        save_fits(args.outfile + '_mopped_residual_mfs.fits', residual_mfs,
                  hdr_mfs)


def main():
    args = create_parser().parse_args()

    if args.nthreads:
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(args.nthreads))
    else:
        import multiprocessing
        args.nthreads = multiprocessing.cpu_count()

    if not isinstance(args.ms, list):
        args.ms = [args.ms]

    pyscilog.log_to_file(args.outfile + '.log')
    pyscilog.enable_memory_logging(level=3)

    GD = vars(args)
    print('Input Options:', file=log)
    for key in GD.keys():
        print('     %25s = %s' % (key, GD[key]), file=log)

    _main(args, dest=log)
