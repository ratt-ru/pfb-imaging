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
              help="Path to xarray dataset containing data products")
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
              help='Standard deviation of assumed GRF prior.'
              'Set it to rms/nband if uncertain')
@click.option('--wstack/--no-wstack', default=True)
@click.option('--double-accum/--no-double-accum', default=True)
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
def forward(**kw):
    '''
    Forward step aka flux mop.

    Solves

    x = (R.H W R + sigmainv**2 I)^{-1} ID

    with a suitable approximation to R.H W R (eg. convolution with the PSF,
    BDA'd weights or none).

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
    from daskms.experimental.zarr import xds_from_zarr
    from pfb.utils.fits import load_fits, set_wcs, save_fits
    from pfb.operators.psi import im2coef, coef2im
    from pfb.opt.pcg import pcg
    from astropy.io import fits
    import pywt

    xds = xds_from_zarr(args.xds, chunks={'row':args.row_chunk})
    nband = xds[0].nband
    nx = xds[0].nx
    ny = xds[0].ny

    # stitch residuals after beam application
    residual = []
    wsum = 0
    for ds in xds:
        d = ds.DIRTY.data
        b = ds.BEAM.data
        wsum += ds.WSUM.data.sum()
        residual.append(d * b)
    residual = da.stack(residual).sum(axis=0)/wsum

    if args.point_mask is not None:
        print("Initialising point source mask", file=log)
        pmask = load_fits(args.point_mask).squeeze()
        # passing model as mask
        if len(pmask.shape) == 3:
            print("Detected third axis on pmask. "
                  "Initialising pmask from model.", file=log)
            x0 = da.from_array(pmask, chunks=(1, -1, -1),
                               name=False)
            pmask = da.any(pmask, axis=0).astype(residual.dtype)
        else:
            x0 = da.zeros((nband, nx, ny), dtype=residual.dtype)

        assert pmask.shape == (nx, ny)
    else:
        pmask = da.ones((nx, ny), dtype=residual.dtype)
        x0 = da.zeros((nband, nx, ny), dtype=residual.dtype)

    # dictionary setup
    print("Setting up dictionary", file=log)
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

    waveopts = {}
    waveopts['bases'] = da.from_array(np.array(bases, dtype=object), chunks=-1)
    waveopts['pmask'] = pmask
    waveopts['iy'] = iys
    waveopts['sy'] = sys
    waveopts['ntot'] = da.from_array(np.array(ntots, dtype=object), chunks=-1)
    waveopts['nmax'] = nmax
    waveopts['nlevels'] = args.nlevels
    waveopts['nx'] = nx
    waveopts['ny'] = ny
    waveopts['padding'] = da.from_array(np.array(padding, dtype=object), chunks=-1)

    hessopts = {}
    hessopts['cell'] = xds[0].cell_rad
    hessopts['wstack'] = args.wstack
    hessopts['epsilon'] = args.epsilon
    hessopts['double_accum'] = args.double_accum
    hessopts['nthreads'] = args.nvthreads
    wsum = wsum.compute()

    if not args.no_use_psf:
        print("Initialising psf", file=log)
        from pfb.operators.psf import psf_convolve_alpha_xds
        from ducc0.fft import r2c
        normfact = 1.0

        psf = xds[0].PSF.data
        _, nx_psf, ny_psf = psf.shape

        iFs = np.fft.ifftshift

        npad_xl = (nx_psf - nx)//2
        npad_xr = nx_psf - nx - npad_xl
        npad_yl = (ny_psf - ny)//2
        npad_yr = ny_psf - ny - npad_yl
        padding_psf = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
        unpad_x = slice(npad_xl, -npad_xr)
        unpad_y = slice(npad_yl, -npad_yr)
        lastsize = ny + np.sum(padding_psf[-1])

        # add psfhat to Dataset
        for i, ds in enumerate(xds):
            psf_pad = iFs(ds.PSF.data.compute(), axes=(1, 2))
            psfhat = r2c(psf_pad, axes=(1, 2), forward=True,
                         nthreads=args.nthreads, inorm=0)

            psfhat = da.from_array(psfhat, chunks=(1, -1, -1), name=False)
            ds = ds.assign({'PSFHAT':(('band', 'x_psf', 'y_psfo2'), psfhat)})
            xds[i] = ds

        psfopts = {}
        psfopts['padding'] = padding_psf[1:]
        psfopts['unpad_x'] = unpad_x
        psfopts['unpad_y'] = unpad_y
        psfopts['lastsize'] = lastsize
        psfopts['nthreads'] = args.nvthreads

        hess = partial(psf_convolve_alpha_xds, xds=xds, waveopts=waveopts,
                       psfopts=psfopts, sigmainv=args.sigmainv, wsum=wsum, compute=True)

    else:
        print("Solving for update using vis space approximation", file=log)
        from pfb.operators.hessian import hessian_alpha_xds

        hess = partial(hessian_alpha_xds, xds=xds, waveopts=waveopts,
                       hessopts=hessopts, sigmainv=args.sigmainv, wsum=wsum, compute=True)

    # # import pdb; pdb.set_trace()
    # x = np.random.randn(nband, nbasis, nmax).astype(np.float32)
    # res = hess(x)
    # dask.visualize(res, color="order", cmap="autumn",
    #                node_attr={"penwidth": "4"},
    #                filename=args.output_filename + '_hess_I_ordered_graph.pdf',
    #                optimize_graph=False)
    # dask.visualize(res, filename=args.output_filename +
    #                '_hess_I_graph.pdf', optimize_graph=False)


    print("Initialising starting values", file=log)
    alpha_resid = im2coef(residual,  # beam alreadt applied
                          pmask, waveopts['bases'], waveopts['ntot'],
                          nmax, args.nlevels).compute()
    alpha0 = im2coef(x0, pmask, waveopts['bases'], waveopts['ntot'],
                     nmax, args.nlevels).compute()

    cgopts = {}
    cgopts['tol'] = args.cg_tol
    cgopts['maxit'] = args.cg_maxit
    cgopts['minit'] = args.cg_minit
    cgopts['verbosity'] = args.cg_verbose
    cgopts['report_freq'] = args.cg_report_freq
    cgopts['backtrack'] = args.backtrack

    # run pcg for update
    print("Solving for update", file=log)
    # import pdb; pdb.set_trace()
    alpha = pcg(hess, alpha_resid, alpha0, **cgopts)

    print("Getting residual", file=log)
    model = coef2im(da.from_array(alpha, chunks=(1, -1, -1), name=False),
                    pmask, waveopts['bases'], waveopts['padding'],
                    iys, sys, nx, ny)

    # from pfb.operators.hessian import hessian_xds
    # residual -= hessian_xds(model, xds, hessopts, wsum, 0.0, compute=True)
    from pfb.operators.psf import psf_convolve_xds
    residual -= psf_convolve_xds(model, xds, psfopts, wsum, 0.0, compute=True)

    # residual = dask.compute(residual,
    #                         optimize_graph=False)[0]

    # construct a header from xds attrs
    ra = xds[0].ra
    dec = xds[0].dec
    radec = [ra, dec]

    cell_rad = xds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)

    freq_out = np.unique(np.concatenate([ds.band.values for ds in xds], axis=0))
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
