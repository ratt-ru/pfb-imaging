# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('BACKWARD')

@cli.command()
@click.option('-xds', '--xds', type=str, required=True,
              help="Path to xarray dataset containing data products")
@click.option('-m', '--model', type=str,
              help='Path to model.fits')
@click.option('-u', '--update', type=str,
              help='Path to update.fits')
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
@click.option('--no-use-beam/--use-beam', default=True)
@click.option('--no-use-psf/--use-psf', default=True)
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
def backward(**kw):
    '''
    Solves

    argmin_x r(x) + (v - x).H U (v - x)

    If a host address is provided the computation can be distributed
    over imaging band and row. When using a distributed scheduler both
    mem-limit and nthreads is per node and have to be specified.

    When using a local cluster, mem-limit and nthreads refer to the global
    memory and threads available, respectively. By default we will use all
    available resources.

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

        return _backward(**args)

def _backward(**kw):
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
    from africanus.gridding.wgridder.dask import hessian
    from astropy.io import fits
    import pywt
    from pfb.utils.misc import compute_context

    # TODO - how to deal with multiple datasets (eg. if we want to
    # partition by field, spw or scan)?
    xds = xds_from_zarr(args.xds, chunks={'row':args.row_chunk})[0]

    nband = xds.dims['nband']
    nx = xds.dims['nx']
    ny = xds.dims['ny']

    wsums = xds.WSUM.data
    wsum = da.sum(wsums).compute()

    # try getting interpolated beam model
    if not args.no_use_beam:
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
        alpha0[l] = im2coef(x0[l].compute(),
                            pmask, bases, ntots, nmax, args.nlevels)

    alpha0 = da.from_array(alpha0, chunks=(1, -1, -1), name=False)
    alpha_resid = da.from_array(alpha_resid, chunks=(1, -1, -1),
                                name=False)

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

    if not args.no_use_psf:
        print("Initialising psf", file=log)
        from ducc0.fft import r2c
        from pfb.opt.pcg import pcg_psf
        normfact = 1.0

        psf = xds.PSF.data
        _, nx_psf, ny_psf = psf.shape

        psf /= wsum
        psf_mfs = da.sum(psf, axis=0).compute()
        assert (psf_mfs.max() - 1.0) < args.epsilon

        iFs = np.fft.ifftshift

        npad_xl = (nx_psf - nx)//2
        npad_xr = nx_psf - nx - npad_xl
        npad_yl = (ny_psf - ny)//2
        npad_yr = ny_psf - ny - npad_yl
        padding_psf = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
        unpad_x = slice(npad_xl, -npad_xr)
        unpad_y = slice(npad_yl, -npad_yr)
        lastsize = ny + np.sum(padding_psf[-1])
        psf_pad = iFs(psf.compute(), axes=(1, 2))
        psfhat = r2c(psf_pad, axes=(1, 2), forward=True,
                     nthreads=args.nthreads, inorm=0)

        psfhat = da.from_array(psfhat, chunks=(1, -1, -1), name=False)

        hessopts = {}
        hessopts['padding_psf'] = padding_psf[1:]
        hessopts['unpad_x'] = unpad_x
        hessopts['unpad_y'] = unpad_y
        hessopts['lastsize'] = lastsize
        hessopts['nthreads'] = args.nvthreads
        hessopts['sigmainv'] = args.sigmainv

        def convolver(x):
            model = da.from_array(x,
                          chunks=(1, nx, ny), name=False)


            convolvedim = hessian(model,
                                  psfhat,
                                  padding_psf,
                                  args.nvthreads,
                                  unpad_x,
                                  unpad_y,
                                  lastsize)
            return convolvedim
    else:
        print("Solving for update using vis space approximation", file=log)
        from pfb.opt.pcg import pcg_wgt
        normfact = wsum

        hessopts = {}
        hessopts['cell'] = xds.cell_rad
        hessopts['wstack'] = args.wstack
        hessopts['epsilon'] = args.epsilon
        hessopts['double_accum'] = args.double_accum
        hessopts['nthreads'] = args.nvthreads
        hessopts['sigmainv'] = args.sigmainv
        hessopts['wsum'] = wsum

        def convolver(x):
            model = da.from_array(x,
                          chunks=(1, nx, ny), name=False)

            convolvedim = hessian(xds.UVW.data,
                                  freqs,
                                  model,
                                  freq_bin_idx,
                                  freq_bin_counts,
                                  cell_rad,
                                  weights=xds.WEIGHT.data.astype(args.output_type),
                                  nthreads=args.nvthreads,
                                  epsilon=args.epsilon,
                                  do_wstacking=args.wstack,
                                  double_accum=args.double_accum)
            return convolvedim

    with compute_context(args.scheduler, args.output_filename + '_alpha'):
        alpha = dask.compute(alpha,
                             optimize_graph=False,
                             scheduler=args.scheduler)[0]

    print("Converting solution to model image", file=log)
    model = np.zeros((nband, nx, ny), dtype=args.output_type)
    for l in range(nband):
        model[l] = coef2im(alpha[l], pmask, bases, padding, iys, sys, nx, ny)

    model_dask = da.from_array(model, chunks=(1, -1, -1), name=False)
    print("Computing residual", file=log)
    residual -= hessian(xds.UVW.data,
                        xds.FREQ.data,
                        model_dask,
                        xds.FBIN_IDX.data,
                        xds.FBIN_COUNTS.data,
                        xds.cell_rad,
                        weights=xds.WEIGHT.data,
                        nthreads=args.nvthreads,
                        epsilon=args.epsilon,
                        do_wstacking=args.wstack,
                        double_accum=args.double_accum)/wsum

    with compute_context(args.scheduler, args.output_filename + '_residual'):
        residual = dask.compute(residual,
                                optimize_graph=False,
                                scheduler=args.scheduler)[0]


    # construct a header from xds attrs
    ra = xds.ra
    dec = xds.dec
    radec = [ra, dec]

    cell_rad = xds.cell_rad
    cell_deg = np.rad2deg(cell_rad)

    freq_out = xds.band.values
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
