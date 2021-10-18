# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('FORWARD')

@cli.command()
@click.option('-xds', '--xds', type=str, required=True,
              help="Path to xarray dataset containing")
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-nb', '--nband', type=int, required=True,
              help="Number of imaging bands")
@click.option('-rchunk', '--row-chunk', type=int, default=-1,
              help="Number of rows in a chunk.")
@click.option('-mask', '--mask',
              help="Path to mask.fits.")
@click.option('-pmask', '--point-mask',
              help="Path to point source mask.fits.")
@click.option('-band', '--band', default='L',
              help='L or UHF band when using JimBeam.')
@click.option('-bases', '--bases', default='self',
              help='Wavelet bases to use. Give as str separated by | eg.'
              '-bases self|db1|db2|db3|db4')
@click.option('-nlevels', '--nlevels', default=3,
              help='Number of wavelet decomposition levels')
@click.option('-otype', '--output-type', default='f4',
              help="Data type of output")
@click.option('-eps', '--epsilon', type=float, default=1e-5,
              help='Gridder accuracy')
@click.option('-sinv', '--sigmainv', type=float, default=1.0,
              help='Standard deviation of assumed GRF prior.')
@click.option('--wstack/--no-wstack', default=True)
@click.option('--double-accum/--no-double-accum', default=True)
@click.option('--use-beam/--no-use-beam', default=False)
@click.option('--use-psf/--no-use-psf', default=False)
@click.option('-cgtol', "--cg-tol", type=float, default=1e-5,
              help="Tolerance of conjugate gradient")
@click.option('-cgminit', "--cg-minit", type=int, default=10,
              help="Minimum number of iterations for conjugate gradient")
@click.option('-cgmaxit', "--cg-maxit", type=int, default=100,
              help="Maximum number of iterations for conjugate gradient")
@click.option('-cgverb', "--cg-verbose", type=int, default=0,
              help="Verbosity of conjugate gradient. "
              "Set to 2 for debugging or zero for silence.")
@click.option('-cgrf', "--cg-report-freq", type=int, default=10,
              help="Report freq for conjugate gradient.")
@click.option('--backtrack/--no-backtrack', default=True,
              help="Backtracking during cg iterations.")
@click.option('-ha', '--host-address',
              help='Address where the distributed client lives. '
              'Will use a local cluster if no address is provided')
@click.option('-nw', '--nworkers', type=int,
              help='Number of workers for the client.')
@click.option('-ntpw', '--nthreads-per-worker', type=int,
              help='Number of dask threads per worker.')
@click.option('-nvt', '--nvthreads', type=int,
              help="Total number of threads to use for vertical scaling (eg. gridder, fft's etc.)")
@click.option('-mem', '--mem-limit', type=float,
              help="Memory limit in GB. Default uses all available memory")
@click.option('-nthreads', '--nthreads', type=int,
              help="Total available threads. Default uses all available threads")
@click.option('-scheduler', '--scheduler', default='distributed',
              help="Total available threads. Default uses all available threads")
def forward(**kw):
    '''
    Extract flux at model locations.

    Will write out the result of solving

    x = (R.H W R + sigmainv**2 I)^{-1} ID

    assuming that R.H W R can be approximated as a convolution with the PSF.

    If a host address is provided the computation can be distributed
    over imaging band and row. When using a distributed scheduler both
    mem-limit and nthreads is per node and have to be specified.

    When using a local cluster, mem-limit and nthreads refer to the global
    memory and threads available, respectively. By default the gridder will
    use all available resources.

    On a local cluster, the default is to use:

        nworkers = nband
        nthreads-per-worker = 1

    They have to be specified in ~.config/dask/jobqueue.yaml in the
    distributed case.

    if LocalCluster:
        nvthreads = nthreads//(nworkers*nthreads_per_worker)
    else:
        nvthreads = nthreads//nthreads-per-worker

    where nvthreads refers to the number of threads used to scale vertically
    (eg. the number threads given to each gridder instance).

    '''
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')

    if args.nworkers is None:
        args.nworkers = args.nband

    OmegaConf.set_struct(args, True)

    with ExitStack() as stack:
        from pfb import set_client
        args = set_client(args, stack, log)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in args.keys():
            print('     %25s = %s' % (key, args[key]), file=log)

        return _forward(**args)

def _forward(**kw):
    args = OmegaConf.create(kw)
    OmegaConf.set_struct(args, True)

    import numpy as np
    import numexpr as ne
    import dask
    import dask.array as da
    from dask.distributed import performance_report
    from daskms.experimental.zarr import xds_from_zarr
    from pfb.utils.fits import load_fits, set_wcs, save_fits, data_from_header
    from pfb.opt.hogbom import hogbom
    from pfb.operators.psi import im2coef, coef2im
    from astropy.io import fits
    import pywt

    xds = xds_from_zarr(args.xds, chunks={'row':args.row_chunk})

    print("Loading residual", file=log)
    residual = xds[0].DIRTY.data
    nband, nx, ny = residual.shape

    wsums = xds[0].WSUM.data
    wsum = da.sum(wsums).compute()

    residual /= wsum

    residual_mfs = residual.sum(axis=0).compute()
    rms = da.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()

    print("Initial peak residual = %f, rms = %f" % (rmax, rms), file=log)

    # try getting interpolated beam model
    if args.use_beam:
        print("Initialising beam model", file=log)
        try:
            beam_image = xds.BEAM.data
            assert beam_image.shape == (nband, nx, ny)
        except Exception as e:
            print(f"Could not initialise beam model due to \n {e}. "
                  f"Using identity", file=log)
            beam_image = da.ones((nband, nx, ny), chunks=(1, -1, -1),
                              dtype=args.output_type)
    else:
        beam_image = da.ones((nband, nx, ny), chunks=(1, -1, -1),
                              dtype=args.output_type)

    if args.mask is not None:
        print("Initialising mask", file=log)
        mask = load_fits(args.mask).squeeze()
        assert mask.shape == (nx, ny)
        mask = da.from_array(mask, chunks=(-1, -1), name=False)
    else:
        mask = da.ones((nx, ny), chunks=(-1, -1), dtype=residual.dtype)

    # TODO - there must be a way to avoid multiplying by the beam
    # inside the hessian operator if neither mask nor beam is used.
    # incorporate mask in beam
    beam_image *= mask[None, :, :]

    if args.point_mask is not None:
        print("Initialising point source mask", file=log)
        pmask = load_fits(args.point_mask).squeeze()
        # passing model as mask
        if len(pmask.shape) == 3:
            print("Detected third axis on pmask. "
                  "Initialising pmask from model.", file=log)
            x0 = da.from_array(pmask, chunks=(1, -1, -1),
                               dtype=residual.dtype, name=False)
            pmask = np.any(pmask, axis=0)
        else:
            x0 = da.zeros((nband, nx, ny), dtype=residual.dtype)

        assert pmask.shape == (nx, ny)
    else:
        pmask = np.ones((nx, ny), dtype=residual.dtype)
        x0 = da.zeros((nband, nx, ny), dtype=residual.dtype)


    # wavelet setup
    print("Setting up wavelets", file=log)
    bases = args.bases.split('|')
    ntots = []
    iys = {}
    sys = {}
    tmp = x0[0].compute()
    for base in bases:
        if base == 'self':
            y, iy, sy = x0[0].ravel(), 0, 0
        else:
            alpha = pywt.wavedecn(tmp, base, mode='zero',
                                  level=args.nlevels)
            y, iy, sy = pywt.ravel_coeffs(alpha)
        iys[base] = iy
        sys[base] = sy
        ntots.append(y.size)

    # get padding info
    nmax = np.asarray(ntots).max()
    padding = []
    nbasis = len(ntots)
    for i in range(nbasis):
        padding.append(slice(0, ntots[i]))

    print("Initialising starting values", file=log)
    alpha0 = np.zeros((nband, nbasis, nmax), dtype=x0.dtype)
    alpha_resid = np.zeros((nband, nbasis, nmax), dtype=x0.dtype)
    for l in range(nband):
        alpha_resid[l] = im2coef((beam_image[l] * residual[l]).compute(),
                            pmask, bases, ntots, nmax, args.nlevels)
        alpha0[l] = im2coef(x0[l],
                            pmask, bases, ntots, nmax, args.nlevels)


    waveopts = {}
    waveopts['bases'] = bases
    waveopts['pmask'] = pmask
    waveopts['iy'] = iys
    waveopts['sy'] = sys
    waveopts['ntot'] = ntots
    waveopts['nmax'] = nmax
    waveopts['nlevels'] = args.nlevels
    waveopts['nx'] = nx
    waveopts['ny'] = ny
    waveopts['padding'] = padding

    cgopts = {}
    cgopts['tol'] = args.cg_tol
    cgopts['maxit'] = args.cg_maxit
    cgopts['minit'] = args.cg_minit
    cgopts['verbosity'] = args.cg_verbose
    cgopts['report_freq'] = args.cg_report_freq
    cgopts['backtrack'] = args.backtrack

    if args.use_psf:
        print("Solving for update using image space approximation", file=log)
        print("Initialising psf", file=log)
        psf = xds[0].PSF.data
        _, nx_psf, ny_psf = psf.shape
        wsum = np.sum(wsums)

        psf /= wsum
        psf_mfs = da.sum(psf, axis=0).compute()

        assert (psf_mfs.max() - 1.0) < 1e-4
        normfact = 1.0
        from pfb.operators.psf import hessian
        from ducc0.fft import r2c
        iFs = np.fft.ifftshift

        npad_xl = (nx_psf - nx)//2
        npad_xr = nx_psf - nx - npad_xl
        npad_yl = (ny_psf - ny)//2
        npad_yr = ny_psf - ny - npad_yl
        padding = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
        unpad_x = slice(npad_xl, -npad_xr)
        unpad_y = slice(npad_yl, -npad_yr)
        lastsize = ny + np.sum(padding[-1])
        psf_pad = iFs(psf.compute(), axes=(1, 2))
        psfhat = r2c(psf_pad, axes=(1, 2), forward=True,
                     nthreads=nthreads, inorm=0)

        psfhat = da.from_array(psfhat, chunks=(1, -1, -1), name=False)


        from pfb.opt.pcg import pcg_psf

        model = pcg_psf(psfhat,
                        residual,
                        x0,
                        beam_image,
                        args.sigmainv,
                        args.nvthreads,
                        padding,
                        unpad_x,
                        unpad_y,
                        lastsize,
                        args.cg_tol,
                        args.cg_maxit,
                        args.cg_minit,
                        args.cg_verbose,
                        args.cg_report_freq,
                        args.backtrack).compute()

        model_dask = da.from_array(model, chunks=(1, -1, -1), name=False)

        residual -= hessian(model,
                            psfhat,
                            padding,
                            args.nvthreads,
                            unpad_x,
                            unpad_y,
                            lastsize).compute()
    else:
        print("Solving for update using vis space approximation", file=log)
        from pfb.opt.pcg import pcg_wgt
        normfact = wsum
        freq = xds[0].FREQ.data
        nchan = freq.size
        freq_bin_idx = xds[0].FBIN_IDX.data
        freq_bin_counts = xds[0].FBIN_COUNTS.data
        cell_rad = xds[0].cell_rad

        uvw = xds[0].UVW.data
        weight = xds[0].WEIGHT.data
        alpha0 = da.from_array(alpha0, chunks=(1, -1, -1), name=False)
        alpha_resid = da.from_array(alpha_resid, chunks=(1, -1, -1),
                                    name=False)

        gridopts = {}
        gridopts['cell'] = cell_rad
        gridopts['wstack'] = args.wstack
        gridopts['epsilon'] = args.epsilon
        gridopts['double_accum'] = args.double_accum
        gridopts['nthreads'] = args.nvthreads
        gridopts['sigmainv'] = args.sigmainv
        gridopts['wsum'] = wsum

        alpha = pcg_wgt(uvw,
                        weight,
                        alpha_resid,
                        alpha0,
                        beam_image,
                        freq,
                        freq_bin_idx,
                        freq_bin_counts,
                        gridopts,
                        waveopts,
                        cgopts)

        from pfb.utils.misc import compute_context
        with compute_context(args.scheduler, args.output_filename + '_alpha'):
            alpha = dask.compute(alpha,
                                 optimize_graph=False,
                                 scheduler=args.scheduler)[0]

        print("Converting solution to model image", file=log)
        model = np.zeros((nband, nx, ny), dtype=args.output_type)
        for l in range(nband):
            model[l] = coef2im(alpha[l], pmask, bases, padding, iys, sys, nx, ny)

        from africanus.gridding.wgridder.dask import hessian

        model_dask = da.from_array(model, chunks=(1, -1, -1), name=False)
        print("Computing residual", file=log)
        residual -= hessian(uvw,
                            freq,
                            model_dask,
                            freq_bin_idx,
                            freq_bin_counts,
                            cell_rad,
                            weights=weight,
                            nthreads=args.nvthreads,
                            epsilon=args.epsilon,
                            do_wstacking=args.wstack,
                            double_accum=args.double_accum)/wsum

        with compute_context(args.scheduler, args.output_filename + '_residual'):
            residual = dask.compute(residual,
                                    optimize_graph=False,
                                    scheduler=args.scheduler)[0]


    # construct a header from xds attrs
    ra = xds[0].ra
    dec = xds[0].dec
    radec = [ra, dec]

    cell_rad = xds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)

    freq_out = xds[0].band.values
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)
    # TODO - add wsum info

    print("Saving results", file=log)
    save_fits(args.output_filename + '_update.fits', model, hdr)
    model_mfs = np.mean(model, axis=0)
    save_fits(args.output_filename + '_update_mfs.fits', model_mfs, hdr_mfs)
    save_fits(args.output_filename + '_residual.fits', residual, hdr)
    residual_mfs = np.sum(residual, axis=0)
    save_fits(args.output_filename + '_residual_mfs.fits', residual_mfs, hdr_mfs)

    print("All done here.", file=log)
