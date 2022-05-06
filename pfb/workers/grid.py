# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('GRID')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.grid["inputs"].keys():
    defaults[key] = schema.grid["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.grid)
def grid(**kw):
    '''
    Compute imaging weights and create a dirty image, psf from xds.
    By default only the MFS images are converted to fits files.
    Set the --fits-cubes flag to also produce fits cubes.

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
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    pyscilog.log_to_file(f'{opts.output_filename}_{opts.product}{opts.postfix}.log')

    if opts.nworkers is None:
        if opts.scheduler=='distributed':
            opts.nworkers = opts.nband
        else:
            opts.nworkers = 1

    if opts.product.upper() not in ["I", "Q", "U", "V", "XX", "YX", "XY", "YY", "RR", "RL", "LR", "LL"]:
        raise NotImplementedError(f"Product {opts.product} not yet supported")

    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _grid(**opts)

def _grid(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import os
    import numpy as np
    import dask
    from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
    import dask.array as da
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from pfb.utils.fits import set_wcs, save_fits
    from pfb.utils.misc import compute_context
    from pfb.operators.gridder import vis2im
    from pfb.operators.fft import fft2d
    from pfb.utils.weighting import (compute_counts, counts_to_weights,
                                     filter_extreme_counts)
    from pfb.utils.beam import eval_beam
    import xarray as xr
    from uuid import uuid4
    from daskms.optimisation import inlined_array

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    xds_name = f'{basename}.xds.zarr'
    dds_name = f'{basename}{opts.postfix}.dds.zarr'
    mds_name = f'{basename}{opts.postfix}.mds.zarr'

    # necessary to exclude imaging weight column if changing from Briggs
    # to natural for example
    columns = ('UVW', 'WEIGHT', 'VIS', 'WSUM', 'MASK', 'FREQ', 'BEAM')
    if opts.robustness is not None:
        columns += (opts.imaging_weight_column,)

    xds = xds_from_zarr(xds_name, chunks={'row':opts.row_chunk},
                        columns=columns)

    # get max uv coords over all datasets
    uvw = []
    u_max = 0.0
    v_max = 0.0
    max_freq = 0.0
    for ds in xds:
        uvw = ds.UVW.data
        u_max = da.maximum(u_max, abs(uvw[:, 0]).max())
        v_max = da.maximum(v_max, abs(uvw[:, 1]).max())
        uv_max = da.maximum(u_max, v_max)
        max_freq = da.maximum(max_freq, ds.FREQ.data.max())

    uv_max = uv_max.compute()
    max_freq = max_freq.compute()

    # max cell size
    cell_N = 1.0 / (2 * uv_max * max_freq / lightspeed)

    if opts.cell_size is not None:
        cell_size = opts.cell_size
        cell_rad = cell_size * np.pi / 60 / 60 / 180
        if cell_N / cell_rad < 1:
            raise ValueError("Requested cell size too small. "
                             "Super resolution factor = ", cell_N / cell_rad)
        print(f"Super resolution factor = {cell_N/cell_rad}", file=log)
    else:
        cell_rad = cell_N / opts.super_resolution_factor
        cell_size = cell_rad * 60 * 60 * 180 / np.pi
        print(f"Cell size set to {cell_size} arcseconds", file=log)

    if opts.nx is None:
        fov = opts.field_of_view * 3600
        npix = int(fov / cell_size)
        npix = good_size(npix)
        while npix % 2:
            npix += 1
            npix = good_size(npix)
        nx = npix
        ny = npix
    else:
        nx = opts.nx
        ny = opts.ny if opts.ny is not None else nx

    nband = opts.nband
    if opts.dirty:
        print(f"Image size set to ({nband}, {nx}, {ny})", file=log)

    nx_psf = good_size(int(opts.psf_oversize * nx))
    while nx_psf % 2:
        nx_psf += 1
        nx_psf = good_size(nx_psf)

    ny_psf = good_size(int(opts.psf_oversize * ny))
    while ny_psf % 2:
        ny_psf += 1
        ny_psf = good_size(ny_psf)

    if opts.psf:
        print(f"PSF size set to ({nband}, {nx_psf}, {ny_psf})", file=log)

    if os.path.isdir(dds_name):
        if opts.overwrite:
            print(f'Removing {dds_name}', file=log)
            import shutil
            shutil.rmtree(dds_name)
        # else:
        #     raise RuntimeError(f'Not overwriting {dds_name}, directory exists. '
        #                        f'Set overwrite flag or specify a different '
        #                        'postfix to create a new data set')

    print(f'Data products will be stored in {dds_name}.', file=log)

    # LB - what is the point of specifying name here?
    if opts.robustness is not None:
        if os.path.isdir(f'{basename}_counts.zarr'):
            counts_ds = xr.open_dataset(f'{basename}_counts.zarr', chunks={'band':1, 'x':-1, 'y':-1}, engine='zarr')
            print(f'Found cached counts array at {basename}_counts.zarr',
                  file=log)
            counts = counts_ds.COUNTS.data
        else:
            counts = [da.zeros((nx, ny), chunks=(-1, -1),
                            name="zeros-"+uuid4().hex) for _ in range(nband)]
            # first loop over data to compute counts
            for ds in xds:
                uvw = ds.UVW.data
                freq = ds.FREQ.data
                mask = ds.MASK.data
                wgt = ds.WEIGHT.data
                bandid = ds.bandid
                count = compute_counts(
                            uvw,
                            freq,
                            mask,
                            nx,
                            ny,
                            cell_rad,
                            cell_rad,
                            wgt.dtype)

                counts[bandid] += count
            counts = da.stack(counts, axis=0)
            counts = counts.rechunk({0:1, 1:-1, 2:-1})
            # cache counts
            count_ds = xr.Dataset({'COUNTS': (('band', 'x', 'y'), counts)})
            count_ds.to_zarr(f'{basename}_counts.zarr', mode='w')

        # get rid of artificially high weights corresponding to nearly empty cells
        if opts.filter_extreme_counts:
            counts = filter_extreme_counts(counts, nbox=opts.filter_nbox,
                                           nlevel=opts.filter_level)

    # check if model exists
    try:
        mds = xds_from_zarr(mds_name, chunks={'band':1})[0]
        model = mds.get(opts.model_name).data
        print(f"Using {opts.model_name} for residual computation. ", file=log)
    except:
        if opts.residual:
            print("Cannot compute residual without a model. ", file=log)
        model = None



    writes = []
    freq_out = []
    for ds in xds:
        uvw = ds.UVW.data
        freq = ds.FREQ.data
        vis = ds.VIS.data
        wgt = ds.WEIGHT.data
        wsum = ds.WSUM.data
        if opts.robustness is not None:
            imwgt = counts_to_weights(counts[ds.bandid],
                                      uvw,
                                      freq,
                                      nx, ny,
                                      cell_rad, cell_rad,
                                      opts.robustness)
        else:
            imwgt = None
        mask = ds.MASK.data
        dvars = {}
        if opts.dirty:
            dirty = vis2im(uvw=uvw,
                           freq=freq,
                           vis=vis,
                           wgt=imwgt,
                           nx=nx,
                           ny=ny,
                           cellx=cell_rad,
                           celly=cell_rad,
                           nthreads=opts.nvthreads,
                           epsilon=opts.epsilon,
                           precision=opts.precision,
                           mask=mask,
                           do_wgridding=opts.wstack,
                           double_precision_accumulation=opts.double_accum)
            dirty = inlined_array(dirty, [uvw, freq])
            dvars['DIRTY'] = (('x', 'y'), dirty)

        if opts.psf:
            psf = vis2im(uvw=uvw,
                         freq=freq,
                         vis=wgt.astype(vis.dtype),
                         wgt=imwgt,
                         nx=nx_psf,
                         ny=ny_psf,
                         cellx=cell_rad,
                         celly=cell_rad,
                         nthreads=opts.nvthreads,
                         epsilon=opts.epsilon,
                         precision=opts.precision,
                         mask=mask,
                         do_wgridding=opts.wstack,
                         double_precision_accumulation=opts.double_accum)
            psf = inlined_array(psf, [uvw, freq])
            # get FT of psf
            psfhat = fft2d(psf, nthreads=opts.nvthreads)
            dvars['PSF'] = (('x_psf', 'y_psf'), psf)
            dvars['PSFHAT'] = (('x_psf', 'yo2'), psfhat)

        if imwgt is not None:
                wgt *= imwgt

        if opts.weight:
            # TODO - BDA
            # combine weights
            wgt = wgt.rechunk({0:100000, 1:-1})
            dvars['WEIGHT'] = (('row', 'chan'), wgt)

        dvars['FREQ'] = (('chan',), freq)
        dvars['UVW'] = (('row', 'three'), uvw.rechunk({0:100000, 1:-1}))
        mask = mask.rechunk({0:100000, 1:-1})
        dvars['MASK'] = (('row', 'chan'), mask)
        wsum = wgt[mask.astype(bool)].sum()
        dvars['WSUM'] = (('scalar',), da.atleast_1d(wsum))

        # evaluate beam at x and y coords
        cell_deg = np.rad2deg(cell_rad)
        l = (-(nx//2) + da.arange(nx)) * cell_deg
        m = (-(ny//2) + da.arange(ny)) * cell_deg
        ll, mm = da.meshgrid(l, m, indexing='ij')
        bvals = eval_beam(ds.BEAM.data, ll, mm)

        dvars['BEAM'] = (('x', 'y'), bvals)

        if opts.residual:
            if model is not None and model.any():
                from pfb.operators.hessian import hessian
                hessopts = {
                    'cell': cell_rad,
                    'wstack': opts.wstack,
                    'epsilon': opts.epsilon,
                    'double_accum': opts.double_accum,
                    'nthreads': opts.nvthreads
                }
                # we only want to apply the beam once here
                residual = dirty - hessian(bvals * model[ds.bandid], uvw, wgt,
                                           freq, None, hessopts)
                residual = inlined_array(residual, [uvw, freq])
                dvars['RESIDUAL'] = (('x', 'y'), residual)


        attrs = {
            'nx': nx,
            'ny': ny,
            'nx_psf': nx_psf,
            'ny_psf': ny_psf,
            'ra': xds[0].ra,
            'dec': xds[0].dec,
            'cell_rad': cell_rad,
            'bandid': ds.bandid,
            'freq_out': ds.freq_out,
            'robustness': opts.robustness
        }

        out_ds = xr.Dataset(dvars, attrs=attrs)
        writes.append(out_ds)
        freq_out.append(ds.freq_out)

    # dask.visualize(writes, color="order", cmap="autumn",
    #                node_attr={"penwidth": "4"},
    #                filename=f'{basename}_grid_ordered_graph.pdf',
    #                optimize_graph=False)
    # dask.visualize(writes, filename=f'{basename}_grid_graph.pdf',
    #                optimize_graph=False)

    print("Computing image space data products", file=log)
    dask.compute(xds_to_zarr(writes, dds_name, columns='ALL'))

    if model is None:
        print("Initialising model ds", file=log)
        # TODO - allow non-zero input model
        attrs = {'nband': nband,
                'nx': nx,
                'ny': ny,
                'ra': xds[0].ra,
                'dec': xds[0].dec,
                'cell_rad': cell_rad}
        freq_out = np.sort(np.stack(freq_out))
        coords = {'freq': freq_out}
        real_type = np.float64 if opts.precision=='double' else np.float32
        model = da.zeros((nband, nx, ny), chunks=(1, -1, -1), dtype=real_type)
        mask = da.zeros((nx, ny), chunks=(-1, -1), dtype=bool)
        data_vars = {'MODEL': (('band', 'x', 'y'), model),
                    'MASK': (('x', 'y'), mask)}
        mds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
        dask.compute(xds_to_zarr([mds], mds_name,columns='ALL'))

    # convert to fits files
    if opts.fits_mfs or opts.fits_cubes:
        xds = xds_from_zarr(dds_name)
        radec = (xds[0].ra, xds[0].dec)
        if opts.dirty:
            print("Saving dirty as fits", file=log)
            dirty = np.zeros((nband, nx, ny), dtype=np.float32)
            wsums = np.zeros(nband, dtype=np.float32)

            hdr = set_wcs(cell_size / 3600, cell_size / 3600,
                          nx, ny, radec, freq_out)
            hdr_mfs = set_wcs(cell_size / 3600, cell_size / 3600,
                              nx, ny, radec, np.mean(freq_out))

            for ds in xds:
                b = ds.bandid
                dirty[b] += ds.DIRTY.values
                wsums[b] += ds.WSUM.values

            for b, w in enumerate(wsums):
                hdr[f'WSUM{b}'] = w
            wsum = np.sum(wsums)
            hdr_mfs[f'WSUM'] = wsum

            dirty_mfs = np.sum(dirty, axis=0, keepdims=True)/wsum

            if opts.fits_mfs:
                save_fits(f'{basename}{opts.postfix}_dirty_mfs.fits',
                          dirty_mfs, hdr_mfs, dtype=np.float32)

            if opts.fits_cubes:
                fmask = wsums > 0
                dirty[fmask] /= wsums[fmask, None, None]
                save_fits(f'{basename}{opts.postfix}_dirty.fits', dirty, hdr,
                        dtype=np.float32)

        if opts.residual:
            print("Saving residual as fits", file=log)
            residual = np.zeros((nband, nx, ny), dtype=np.float32)
            wsums = np.zeros(nband, dtype=np.float32)

            hdr = set_wcs(cell_size / 3600, cell_size / 3600,
                          nx, ny, radec, freq_out)
            hdr_mfs = set_wcs(cell_size / 3600, cell_size / 3600,
                              nx, ny, radec, np.mean(freq_out))

            for ds in xds:
                b = ds.bandid
                residual[b] += ds.RESIDUAL.values
                wsums[b] += ds.WSUM.values

            for b, w in enumerate(wsums):
                hdr[f'WSUM{b}'] = w
            wsum = np.sum(wsums)
            hdr_mfs[f'WSUM'] = wsum

            residual_mfs = np.sum(residual, axis=0, keepdims=True)/wsum

            if opts.fits_mfs:
                save_fits(f'{basename}{opts.postfix}_residual_mfs.fits',
                          residual_mfs, hdr_mfs, dtype=np.float32)

            if opts.fits_cubes:
                fmask = wsums > 0
                dirty[fmask] /= wsums[fmask, None, None]
                save_fits(f'{basename}{opts.postfix}_residual.fits', residual,
                          hdr, dtype=np.float32)

        if opts.psf:
            print("Saving PSF as fits", file=log)
            psf = np.zeros((nband, nx_psf, ny_psf), dtype=np.float32)
            wsums = np.zeros(nband, dtype=np.float32)

            hdr_psf = set_wcs(cell_size / 3600, cell_size / 3600, nx_psf,
                              ny_psf, radec, freq_out)
            hdr_psf_mfs = set_wcs(cell_size / 3600, cell_size / 3600, nx_psf,
                                  ny_psf, radec, np.mean(freq_out))

            for ds in xds:
                b = ds.bandid
                psf[b] += ds.PSF.values
                wsums[b] += ds.WSUM.values

            for b, w in enumerate(wsums):
                hdr_psf[f'WSUM{b}'] = w
            wsum = np.sum(wsums)
            hdr_psf_mfs[f'WSUM'] = wsum

            psf_mfs = np.sum(psf, axis=0, keepdims=True)/wsum

            if opts.fits_mfs:
                save_fits(f'{basename}{opts.postfix}_psf_mfs.fits', psf_mfs,
                          hdr_psf_mfs, dtype=np.float32)

            if opts.fits_cubes:
                fmask = wsums > 0
                psf[fmask] /= wsums[fmask, None, None]
                save_fits(f'{basename}{opts.postfix}_psf.fits', psf, hdr_psf,
                        dtype=np.float32)

    print("All done here.", file=log)
