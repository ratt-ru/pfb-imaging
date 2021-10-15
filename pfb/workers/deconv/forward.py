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
@click.option('-mask', '--mask',
              help="Path to mask.fits.")
@click.option('-pmask', '--point-mask',
              help="Path to point source mask.fits.")
@click.option('-bm', '--beam-model',
              help="Path to beam_model.fits or JimBeam.")
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

    xds = xds_from_zarr(args.weight_table)

    print("Loading residual", file=log)
    residual = load_fits(args.residual, dtype=args.output_type).squeeze()
    nband, nx, ny = residual.shape
    hdr = fits.getheader(args.residual)

    print("Loading psf", file=log)
    psf = load_fits(args.psf, dtype=args.output_type).squeeze()
    _, nx_psf, ny_psf = psf.shape
    hdr_psf = fits.getheader(args.psf)

    wsums = np.amax(psf.reshape(-1, nx_psf*ny_psf), axis=1)
    wsum = np.sum(wsums)

    psf /= wsum
    psf_mfs = np.sum(psf, axis=0)

    assert (psf_mfs.max() - 1.0) < 1e-4

    residual /= wsum
    residual_mfs = np.sum(residual, axis=0)

    # get info required to set WCS
    ra = np.deg2rad(hdr['CRVAL1'])
    dec = np.deg2rad(hdr['CRVAL2'])
    radec = [ra, dec]

    cell_deg = np.abs(hdr['CDELT1'])
    if cell_deg != np.abs(hdr['CDELT2']):
        raise NotImplementedError('cell sizes have to be equal')
    cell_rad = np.deg2rad(cell_deg)

    l_coord, ref_l = data_from_header(hdr, axis=1)
    l_coord -= ref_l
    m_coord, ref_m = data_from_header(hdr, axis=2)
    m_coord -= ref_m
    freq_out, ref_freq = data_from_header(hdr, axis=3)

    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()

    print("Initial peak residual = %f, rms = %f" % (rmax, rms), file=log)

    # load beam
    if args.beam_model is not None:
        if args.beam_model.endswith('.fits'):  # beam already interpolated
            bhdr = fits.getheader(args.beam_model)
            l_coord_beam, ref_lb = data_from_header(bhdr, axis=1)
            l_coord_beam -= ref_lb
            if not np.array_equal(l_coord_beam, l_coord):
                raise ValueError("l coordinates of beam model do not match those of image. Use power_beam_maker to interpolate to fits header.")

            m_coord_beam, ref_mb = data_from_header(bhdr, axis=2)
            m_coord_beam -= ref_mb
            if not np.array_equal(m_coord_beam, m_coord):
                raise ValueError("m coordinates of beam model do not match those of image. Use power_beam_maker to interpolate to fits header.")

            freq_beam, _ = data_from_header(bhdr, axis=freq_axis)
            if not np.array_equal(freq_out, freq_beam):
                raise ValueError("Freqs of beam model do not match those of image. Use power_beam_maker to interpolate to fits header.")

            beam_image = load_fits(args.beam_model, dtype=args.output_type).squeeze()
        elif args.beam_model.lower() == "jimbeam":
            from katbeam import JimBeam
            if args.band.lower() == 'l':
                beam = JimBeam('MKAT-AA-L-JIM-2020')
            elif args.band.lower() == 'uhf':
                beam = JimBeam('MKAT-AA-UHF-JIM-2020')
            else:
                raise ValueError("Unkown band %s"%args.band[i])

            xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')
            beam_image = np.zeros(residual.shape, dtype=args.output_type)
            for v in range(freq_out.size):
                # freq must be in MHz
                beam_image[v] = beam.I(xx, yy, freq_out[v]/1e6).astype(args.output_type)
    else:
        beam_image = np.ones((nband, nx, ny), dtype=args.output_type)

    if args.mask is not None:
        mask = load_fits(args.mask).squeeze()
        assert mask.shape == (nx, ny)
    else:
        mask = np.ones((nx, ny), dtype=residual.dtype)

    # incorporate mask in beam
    beam_image *= mask[None, :, :]

    if args.point_mask is not None:
        pmask = load_fits(args.point_mask).squeeze()
        # passing model as mask
        if len(pmask.shape) == 3:
            x0 = pmask
            pmask = np.any(pmask, axis=0)
        else:
            x0 = np.zeros((nband, nx, ny), dtype=residual.dtype)

        assert pmask.shape == (nx, ny)
    else:
        pmask = np.ones((nx, ny), dtype=residual.dtype)
        x0 = np.zeros((nband, nx, ny), dtype=residual.dtype)


    # wavelet setup
    bases = args.bases.split('|')
    ntots = []
    iys = {}
    sys = {}
    for base in bases:
        if base == 'self':
            y, iy, sy = x0[0].ravel(), 0, 0
        else:
            alpha = pywt.wavedecn(x0[0], base, mode='zero', level=args.nlevels)
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

    alpha0 = np.zeros((nband, nbasis, nmax), dtype=x0.dtype)
    alpha_resid = np.zeros((nband, nbasis, nmax), dtype=x0.dtype)
    for l in range(nband):
        alpha_resid[l] = im2coef(beam_image[l] * residual[l],
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

    # if weight table is provided we use the vis space Hessian approximation
    if args.weight_table is not None:
        print("Solving for update using vis space approximation", file=log)
        normfact = wsum


        xds = xds_from_zarr(args.weight_table)[0]
        nrow = xds.row.size
        freq = xds.chan.data
        nchan = freq.size

        # bin edges
        fmin = freq.min()
        fmax = freq.max()
        fbins = np.linspace(fmin, fmax, nband + 1)

        # chan <-> band mapping
        band_mapping = {}
        chan_chunks = {}
        freq_bin_idx = {}
        freq_bin_counts = {}
        band_map = np.zeros(freq.size, dtype=np.int32)
        for band in range(nband):
            indl = freq >= fbins[band]
            indu = freq < fbins[band + 1] + 1e-6
            band_map = np.where(indl & indu, band, band_map)

        # to dask arrays
        bands, bin_counts = np.unique(band_map, return_counts=True)
        band_mapping = tuple(bands)
        chan_chunks = {'chan': tuple(bin_counts)}
        freq = da.from_array(freq, chunks=tuple(bin_counts))
        bin_idx = np.append(np.array([0]), np.cumsum(bin_counts))[0:-1]
        freq_bin_idx = da.from_array(bin_idx, chunks=1)
        freq_bin_counts = da.from_array(bin_counts, chunks=1)

        max_chan_chunk = bin_counts.max()
        bin_counts = tuple(bin_counts)

        residual = da.from_array(residual, chunks=(1, -1, -1))

        xds = xds_from_zarr(args.weight_table, chunks={'row': -1,
                            'chan': bin_counts})[0]

        from pfb.opt.pcg import pcg_wgt

        uvw = dask.persist(xds.UVW.data)[0]
        weight = dask.persist(xds.WEIGHT.data.astype(args.output_type))[0]
        alpha0 = da.from_array(alpha0, chunks=(1, -1, -1))
        alpha_resid = da.from_array(alpha_resid, chunks=(1, -1, -1))
        beam_image = da.from_array(beam_image, chunks=(1, -1, -1))

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
                        cgopts).compute()

        model = np.zeros((nband, nx, ny), dtype=args.output_type)
        for l in range(nband):
            model[l] = coef2im(alpha[l], pmask, bases, padding, iys, sys, nx, ny)

        from africanus.gridding.wgridder.dask import hessian

        model_dask = da.from_array(model, chunks=(1, -1, -1))
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
                            double_accum=args.double_accum).compute()/wsum

    else:  # we use the image space approximation
        print("Solving for update using image space approximation", file=log)
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
        psf_pad = iFs(psf, axes=(1, 2))
        psfhat = r2c(psf_pad, axes=(1, 2), forward=True,
                     nthreads=nthreads, inorm=0)

        psfhat = da.from_array(psfhat, chunks=(1, -1, -1))
        residual = da.from_array(residual, chunks=(1, -1, -1))


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

        model_dask = da.from_array(model, chunks=(1, -1, -1))

        residual -= hessian(model,
                            psfhat,
                            padding,
                            args.nvthreads,
                            unpad_x,
                            unpad_y,
                            lastsize).compute()


    print("Saving results", file=log)
    save_fits(args.output_filename + '_update.fits', model, hdr)
    model_mfs = np.mean(model, axis=0)
    save_fits(args.output_filename + '_update_mfs.fits', model_mfs, hdr_mfs)
    save_fits(args.output_filename + '_residual.fits', residual, hdr)
    residual_mfs = np.sum(residual, axis=0)
    save_fits(args.output_filename + '_residual_mfs.fits', residual_mfs, hdr_mfs)

    print("All done here.", file=log)
