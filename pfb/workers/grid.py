# flake8: noqa
import os
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('GRID')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.grid)
def grid(**kw):
    '''
    Compute imaging weights and create a dirty image, psf etc.
    By default only the MFS images are converted to fits files.
    Set the --fits-cubes flag to also produce fits cubes.

    '''
    opts = OmegaConf.create(kw)
    from pfb.utils.naming import set_output_names
    opts, basedir, oname = set_output_names(opts)

    import psutil
    nthreads = psutil.cpu_count(logical=True)
    ncpu = psutil.cpu_count(logical=False)
    if opts.nthreads is None:
        if opts.nworkers > 1:
            ntpw = nthreads//opts.nworkers
            opts.nthreads = ntpw//2
            ncpu = ntpw//2
        else:
            opts.nthreads = nthreads//2
            ncpu = ncpu//2

    from daskms.fsspec_store import DaskMSStore
    basename = f'{basedir}/{oname}'
    fits_oname = f'{opts.fits_output_folder}/{oname}'
    dds_store = DaskMSStore(f'{basename}_{opts.suffix}.dds')

    if opts.xds is not None:
        xds_store = DaskMSStore(opts.xds.rstrip('/'))
        xds_name = opts.xds
    else:
        xds_store = DaskMSStore(f'{basename}.xds')
        xds_name = f'{basename}.xds'
    try:
        assert xds_store.exists()
    except Exception as e:
        raise ValueError(f"There must be an xds at {xds_name}. "
                            f"Original traceback {e}")
    opts.xds = xds_store.url
    OmegaConf.set_struct(opts, True)

    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(opts.log_directory)}/grid_{timestamp}.log'
    pyscilog.log_to_file(logname)
    print(f'Logs will be written to {logname}', file=log)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    from pfb import set_envs
    from ducc0.misc import resize_thread_pool, thread_pool_size
    resize_thread_pool(opts.nthreads)
    set_envs(opts.nthreads, ncpu)

    with ExitStack() as stack:
        from pfb import set_client
        if opts.nworkers > 1:
            client = set_client(opts.nworkers, log, stack)
            from distributed import as_completed
        else:
            print("Faking client", file=log)
            from pfb.utils.dist import fake_client
            client = fake_client()
            names = [0]
            as_completed = lambda x: x

        from pfb.utils.naming import xds_from_url
        from pfb.utils.fits import dds2fits

        ti = time.time()
        residual_mfs = _grid(**opts)

        dds_list = dds_store.fs.glob(f'{dds_store.url}/*.zarr')

        # convert to fits files
        futures = []
        if opts.fits_mfs or opts.fits_cubes:
            print(f"Writing fits files to {fits_oname}_{opts.suffix}", file=log)
            if opts.dirty:
                fut = client.submit(dds2fits,
                                    dds_list,
                                    'DIRTY',
                                    f'{fits_oname}_{opts.suffix}',
                                    norm_wsum=True,
                                    nthreads=opts.nthreads,
                                    do_mfs=opts.fits_mfs,
                                    do_cube=opts.fits_cubes)
                futures.append(fut)
            if opts.psf:
                fut = client.submit(dds2fits,
                                    dds_list,
                                    'PSF',
                                    f'{fits_oname}_{opts.suffix}',
                                    norm_wsum=True,
                                    nthreads=opts.nthreads,
                                    do_mfs=opts.fits_mfs,
                                    do_cube=opts.fits_cubes)
                futures.append(fut)
            if opts.residual:
                try:
                    fut = client.submit(dds2fits,
                                        dds_list,
                                        'MODEL',
                                        f'{fits_oname}_{opts.suffix}',
                                        norm_wsum=False,
                                        nthreads=opts.nthreads,
                                        do_mfs=opts.fits_mfs,
                                        do_cube=opts.fits_cubes)
                    futures.append(fut)

                    fut = client.submit(dds2fits,
                                        dds_list,
                                        'RESIDUAL',
                                        f'{fits_oname}_{opts.suffix}',
                                        norm_wsum=True,
                                        nthreads=opts.nthreads,
                                        do_mfs=opts.fits_mfs,
                                        do_cube=opts.fits_cubes)
                    futures.append(fut)
                except:
                    pass
            if opts.noise:
                fut = client.submit(dds2fits,
                                    dds_list,
                                    'NOISE',
                                    f'{fits_oname}_{opts.suffix}',
                                    norm_wsum=True,
                                    nthreads=opts.nthreads,
                                    do_mfs=opts.fits_mfs,
                                    do_cube=opts.fits_cubes)
                futures.append(fut)

            fut = client.submit(dds2fits,
                                dds_list,
                                'BEAM',
                                f'{fits_oname}_{opts.suffix}',
                                norm_wsum=False,
                                nthreads=opts.nthreads,
                                do_mfs=opts.fits_mfs,
                                do_cube=opts.fits_cubes)
            futures.append(fut)

            for fut in as_completed(futures):
                try:
                    column = fut.result()
                except:
                    continue
                print(f'Done writing {column}', file=log)

        print(f"All done after {time.time() - ti}s", file=log)

def _grid(xdsi=None, **kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    from distributed import as_completed, get_client
    from itertools import cycle
    import fsspec
    from daskms.fsspec_store import DaskMSStore
    from pfb.utils.misc import set_image_size
    from pfb.operators.gridder import image_data_products, wgridder_conventions
    import xarray as xr
    from pfb.utils.astrometry import get_coordinates
    from africanus.coordinates import radec_to_lm
    from pfb.utils.naming import xds_from_url, cache_opts, get_opts
    import sympy as sm
    from sympy.utilities.lambdify import lambdify
    from sympy.parsing.sympy_parser import parse_expr

    try:
        client = get_client()
        names = list(client.scheduler_info()['workers'].keys())
    except:
        from pfb.utils.dist import fake_client
        client = fake_client()
        names = [0]
        as_completed = lambda x: x

    basename = f'{opts.output_filename}'

    # xds contains vis products, no imaging weights applied
    xds_name = f'{basename}.xds' if opts.xds is None else opts.xds
    xds_store = DaskMSStore(xds_name)
    if xdsi is not None:
        xds = []
        for ds in xdsi:
            xds.append(ds.chunk({'row':-1,
                                 'chan': -1,
                                 'l_beam': -1,
                                 'm_beam': -1}))
    else:
        try:
            assert xds_store.exists()
        except Exception as e:
            raise ValueError(f"There must be a dataset at {xds_store.url}")

        xds = xds_from_url(xds_name)

    times_in = []
    freqs_in = []
    for ds in xds:
        times_in.append(ds.time_out)
        freqs_in.append(ds.freq_out)

    times_in = np.unique(times_in)
    freqs_out = np.unique(freqs_in)

    ntime_in = times_in.size
    nband = freqs_out.size

    real_type = xds[0].WEIGHT.dtype
    if real_type == np.float32:
        precision = 'single'
    else:
        precision = 'double'

    # max uv coords over all datasets
    uv_max = 0
    max_freq = 0
    for ds in xds:
        uv_max = np.maximum(uv_max, ds.uv_max)
        max_freq = np.maximum(max_freq, ds.max_freq)

    nx, ny, nx_psf, ny_psf, cell_N, cell_rad = set_image_size(
        uv_max,
        max_freq,
        opts.field_of_view,
        opts.super_resolution_factor,
        opts.cell_size,
        opts.nx,
        opts.ny,
        opts.psf_oversize
    )
    cell_deg = np.rad2deg(cell_rad)
    cell_size = cell_deg * 3600
    print(f"Super resolution factor = {cell_N/cell_rad}", file=log)
    print(f"Cell size set to {cell_size:.5e} arcseconds", file=log)
    print(f"Field of view is ({nx*cell_deg:.3e},{ny*cell_deg:.3e}) degrees",
          file=log)


    # TODO - how to glob with protocol in tact?
    ds_list = xds_store.fs.glob(f'{xds_store.url}/*')
    if '://' in xds_store.url:
        protocol = xds_store.url.split('://')[0]
    else:
        protocol = 'file'
    url_prepend = protocol + '://'
    ds_list = list(map(lambda x: url_prepend + x, ds_list))

    # create dds and cache
    dds_name = opts.output_filename + f'_{opts.suffix}' + '.dds'
    dds_store = DaskMSStore(dds_name)
    if '://' in dds_store.url:
        protocol = xds_store.url.split('://')[0]
    else:
        protocol = 'file'

    if dds_store.exists() and opts.overwrite:
        print(f"Removing {dds_store.url}", file=log)
        dds_store.rm(recursive=True)

    optsp_name = dds_store.url + '/opts.pkl'
    fs = fsspec.filesystem(protocol)
    if dds_store.exists() and not opts.overwrite:
        # get opts from previous run
        optsp = get_opts(dds_store.url,
                         protocol,
                         name='opts.pkl')

        # check if we need to remake the data products
        verify_attrs = ['epsilon',
                        'do_wgridding', 'double_accum',
                        'field_of_view', 'super_resolution_factor']
        try:
            for attr in verify_attrs:
                assert optsp[attr] == opts[attr]

            from_cache = True
            print("Initialising from cached data products", file=log)
        except Exception as e:
            print(f'Cache verification failed on {attr}. '
                  'Will remake image data products', file=log)
            dds_store.rm(recursive=True)
            fs.makedirs(dds_store.url, exist_ok=True)
            # dump opts to validate cache on rerun
            cache_opts(opts,
                       dds_store.url,
                       protocol,
                       name='opts.pkl')
            from_cache = False
    else:
        fs.makedirs(dds_store.url, exist_ok=True)
        cache_opts(opts,
                   dds_store.url,
                   protocol,
                   name='opts.pkl')
        from_cache = False
        print("Initialising from scratch.", file=log)

    print(f"Data products will be cached in {dds_store.url}", file=log)

    # filter datasets by time and band
    xds_dct = {}
    if opts.concat_row:
        ntime = 1
        times_out = np.mean(times_in, keepdims=True)
        for b in range(nband):
            for ds, ds_name in zip(xds, ds_list):
                if ds.bandid == b:
                    tbid = f'time0000_band{b:04d}'
                    xds_dct.setdefault(tbid, {})
                    xds_dct[tbid].setdefault('dsl', [])
                    xds_dct[tbid]['dsl'].append(ds_name)
                    xds_dct[tbid]['radec'] = (ds.ra, ds.dec)
                    xds_dct[tbid]['time_out'] = times_out[0]
                    xds_dct[tbid]['freq_out'] = freqs_out[b]
    else:
        ntime = ntime_in
        times_out = times_in
        for t in range(times_in.size):
            for b in range(nband):
                for ds, ds_name in zip(xds, ds_list):
                    if ds.time_out == times_in[t] and ds.bandid == b:
                        tbid = f'time{t:04d}_band{b:04d}'
                        xds_dct.setdefault(tbid, {})
                        xds_dct[tbid].setdefault('dsl', [])
                        xds_dct[tbid]['dsl'].append(ds_name)
                        xds_dct[tbid]['radec'] = (ds.ra, ds.dec)
                        xds_dct[tbid]['time_out'] = times_out[t]
                        xds_dct[tbid]['freq_out'] = freqs_out[b]

    if opts.dirty:
        print(f"Image size = (ntime={ntime}, nband={nband}, "
              f"nx={nx}, ny={ny})", file=log)

    if opts.psf:
        print(f"PSF size = (ntime={ntime}, nband={nband}, "
              f"nx={nx_psf}, ny={ny_psf})", file=log)

    # check if model exists
    if opts.transfer_model_from:
        try:
            mds = xr.open_zarr(opts.transfer_model_from, chunks=None)
        except Exception as e:
            raise ValueError(f"No dataset found at {opts.transfer_model_from}")

        # we only want to load these once
        model_coeffs = mds.coefficients.values
        locx = mds.location_x.values
        locy = mds.location_y.values
        params = mds.params.values

        print(f"Loading model from {opts.transfer_model_from}. ",
              file=log)

    futures = []
    for wname, (tbid, ds_dct) in zip(cycle(names), xds_dct.items()):
        bandid = tbid[-4:]
        timeid = tbid[4:8]
        ra = ds_dct['radec'][0]
        dec = ds_dct['radec'][1]
        dsl = ds_dct['dsl']
        time_out = ds_dct['time_out']
        freq_out = ds_dct['freq_out']

        if from_cache:
            out_ds = xr.open_zarr(f'{dds_store.url}/time{timeid}_band{bandid}.zarr',
                                  chunks=None)

        # compute lm coordinates of target
        if opts.target is not None:
            tmp = opts.target.split(',')
            if len(tmp) == 1 and tmp[0] == opts.target:
                tra, tdec = get_coordinates(time_out, target=opts.target)
            else:  # we assume a HH:MM:SS,DD:MM:SS format has been passed in
                from astropy import units as u
                from astropy.coordinates import SkyCoord
                c = SkyCoord(tmp[0], tmp[1], frame='fk5', unit=(u.hourangle, u.deg))
                tra = np.deg2rad(c.ra.value)
                tdec = np.deg2rad(c.dec.value)

            tcoords=np.zeros((1,2))
            tcoords[0,0] = tra
            tcoords[0,1] = tdec
            coords0 = np.array((ra, dec))
            lm0 = radec_to_lm(tcoords, coords0).squeeze()
            l0 = lm0[0]
            m0 = lm0[1]
        else:
            l0 = 0.0
            m0 = 0.0
            tra = ds.ra
            tdec = ds.dec

        attrs = {
            'ra': tra,
            'dec': tdec,
            'l0': l0,
            'm0': m0,
            'cell_rad': cell_rad,
            'bandid': bandid,
            'timeid': timeid,
            'freq_out': freq_out,
            'time_out': time_out,
            'robustness': opts.robustness,
            'super_resolution_factor': opts.super_resolution_factor,
            'field_of_view': opts.field_of_view,
            'product': opts.product
        }

        # get the model
        if opts.transfer_model_from:
            from pfb.utils.misc import eval_coeffs_to_slice
            _, _, _, x0, y0 = wgridder_conventions(l0, m0)
            model = eval_coeffs_to_slice(
                time_out,
                freq_out,
                model_coeffs,
                locx, locy,
                mds.parametrisation,
                params,
                mds.texpr,
                mds.fexpr,
                mds.npix_x, mds.npix_y,
                mds.cell_rad_x, mds.cell_rad_y,
                mds.center_x, mds.center_y,
                nx, ny,
                cell_rad, cell_rad,
                x0, y0
            )

        elif from_cache:
            if opts.use_best_model and 'BEST_MODEL' in out_ds:
                model = out_ds.MODEL_BEST.values
            elif 'MODEL' in out_ds:
                model = out_ds.MODEL.values
            else:
                model = None
        else:
            model = None

        fut = client.submit(image_data_products,
                            dsl,
                            None,  # counts
                            nx, ny,
                            nx_psf, ny_psf,
                            cell_rad, cell_rad,
                            dds_store.url + '/' + tbid + '.zarr',
                            attrs,
                            model=model,
                            robustness=opts.robustness,
                            l0=l0, m0=m0,
                            nthreads=opts.nthreads,
                            epsilon=opts.epsilon,
                            do_wgridding=opts.do_wgridding,
                            double_accum=opts.double_accum,
                            l2_reweight_dof=opts.l2_reweight_dof,
                            do_dirty=opts.dirty,
                            do_psf=opts.psf,
                            do_weight=opts.weight,
                            do_residual=opts.residual,
                            do_noise=opts.noise,
                            workers=wname)
        futures.append(fut)

    residual_mfs = np.zeros((nx, ny))
    wsum = 0.0
    nds = len(futures)
    n_launched = 1
    for fut in as_completed(futures):
        print(f"\rProcessing: {n_launched}/{nds}", end='', flush=True)
        residual, wsumb = fut.result()
        residual_mfs += residual
        wsum += wsumb
        n_launched += 1

    print("\n")  # after progressbar above

    residual_mfs /= wsum
    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()

    print(f"Initial max resid = {rmax:.3e}, rms resid = {rms:.3e}", file=log)
