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
    Compute imaging weights and create a dirty image, psf etc.
    By default only the MFS images are converted to fits files.
    Set the --fits-cubes flag to also produce fits cubes.

    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'grid_{timestamp}.log')

    if opts.nworkers is None:
        if opts.scheduler=='distributed':
            opts.nworkers = opts.nband
        else:
            opts.nworkers = 1

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
    dask.config.set(**{'array.slicing.split_large_chunks': False})
    from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
    import dask.array as da
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from pfb.utils.fits import dds2fits, dds2fits_mfs
    from pfb.utils.misc import compute_context
    from pfb.operators.gridder import vis2im
    from pfb.operators.fft import fft2d
    from pfb.utils.weighting import (compute_counts, counts_to_weights,
                                     filter_extreme_counts)
    from pfb.utils.beam import eval_beam
    import xarray as xr
    from uuid import uuid4
    from daskms.optimisation import inlined_array
    from pfb.utils.astrometry import get_coordinates
    from africanus.coordinates import radec_to_lm

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    # xds contains vis products, no imaging weights applied
    xds_name = f'{basename}.xds.zarr'
    xdsp = xds_from_zarr(xds_name, chunks={'row': -1, 'chan': -1})
    # dds contains image space products including imaging weights and uvw
    dds_name = f'{basename}{opts.postfix}.dds.zarr'

    if opts.concat:
        # this is required because concat will try to mush different
        # imaging bands together if they are not split upfront
        print('Concatenating datasets along row dimension', file=log)
        xds = []
        for b in range(opts.nband):
            xdsb = []
            times = []
            timeids = []
            for ds in xdsp:
                if ds.bandid == b:
                    xdsb.append(ds)
                    times.append(ds.time_out)
                    timeids.append(ds.timeid)
            xdso = xr.concat(xdsb, dim='row',
                             data_vars='minimal',
                             coords='minimal').chunk({'row':-1})
            tid = np.array(timeids).min()
            tout = np.mean(np.array(times))
            xdso.assign_attrs(
                {'time_out': tout, 'timeid': tid}
            )
            xds.append(xdso)
        try:
            assert len(xds) == opts.nband
        except Exception as e:
            raise RuntimeError('Something went wrong during concatenation.'
                               'This is probably a bug.')
    else:
        xds = xdsp

    real_type = xds[0].WEIGHT.dtype
    if real_type == np.float32:
        precision = 'single'
    else:
        precision = 'double'

    # max uv coords over all datasets
    uv_maxs = []
    max_freqs = []
    for ds in xds:
        uvw = ds.UVW.data
        u_max = abs(uvw[:, 0]).max()
        v_max = abs(uvw[:, 1]).max()
        uv_maxs.append(da.maximum(u_max, v_max))
        max_freqs.append(ds.FREQ.data.max())

    # early compute necessary to set image size
    uv_maxs, max_freqs = dask.compute(uv_maxs, max_freqs)
    uv_max = max(uv_maxs)
    max_freq = max(max_freqs)

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
        print(f"Image size = ({nband}, {nx}, {ny})", file=log)

    nx_psf = good_size(int(opts.psf_oversize * nx))
    while nx_psf % 2:
        nx_psf += 1
        nx_psf = good_size(nx_psf)

    ny_psf = good_size(int(opts.psf_oversize * ny))
    while ny_psf % 2:
        ny_psf += 1
        ny_psf = good_size(ny_psf)

    if opts.psf:
        print(f"PSF size = ({nband}, {nx_psf}, {ny_psf})", file=log)

    if os.path.isdir(dds_name):
        if opts.overwrite:
            print(f'Removing {dds_name}', file=log)
            import shutil
            shutil.rmtree(dds_name)
        else:
            raise RuntimeError(f'Not overwriting {dds_name}, directory exists. '
                               f'Set overwrite flag if you mean to overwrite.')

    print(f'Data products will be stored in {dds_name}.', file=log)

    if opts.robustness is not None:
        try:
            counts_ds = xds_from_zarr(f'{basename}.counts.zarr',
                                      chunks={'band':1, 'x':-1, 'y':-1})
            assert counts_ds[0].bands.size == nband
            assert counts_ds[0].x.size == nx
            assert counts_ds[0].y.size == ny
            assert counts_ds[0].bands.size == nband
            assert counts_ds[0].cell_rad == cell_rad
            print(f'Found cached gridded weights at {basename}.counts.zarr. '
                  f'Coords and cell sizes indicate that it can be reused',
                  file=log)
            counts = counts_ds.COUNTS.data
        except:
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
            counts = da.stack(counts)
            counts = counts.rechunk({0:1, 1:-1, 2:-1})
            # cache counts
            dvars = {'COUNTS': (('band', 'x', 'y'), counts)}
            attrs = {
                'cell_rad': cell_rad
            }
            counts_ds = xr.Dataset(dvars, attrs=attrs)
            # we need to rechunk for zarr codec but subsequent usage of counts
            # array needs to be chunks as (1, -1, -1)
            counts_ds = counts_ds.chunk({'x':'auto', 'y':'auto'})
            writes = xds_to_zarr(counts_ds, f'{basename}.counts.zarr',
                                 columns='ALL')
            print("Computing gridded weights", file=log)
            counts = dask.compute(counts, writes)[0]
            counts = da.from_array(counts, chunks=(1, -1, -1))

        # get rid of artificially high weights corresponding to
        # nearly empty cells
        if opts.filter_extreme_counts:
            counts = filter_extreme_counts(counts, nbox=opts.filter_nbox,
                                           nlevel=opts.filter_level)

    # check if model exists
    if opts.transfer_model_from is not None:
        try:
            mds = xds_from_zarr(opts.transfer_model_from,
                                chunks={'x':-1, 'y':-1})
        except Exception as e:
            raise ValueError(f"No dataset found at {opts.transfer_model_from}")
        try:
            assert len(mds) == len(xds)
            for ms, ds in zip(mds, xds):
                assert ms.bandid == ds.bandid
        except Exception as e:
            raise ValueError("Transfer from dataset mismatched. "
                             "This is not currently supported.")
        try:
            assert 'MODEL' in mds[0]
        except Exception as e:
            raise ValueError(f"No MODEL variable in {opts.transfer_model_from}")

        print(f"Found MODEL in {opts.transfer_model_from}. ",
              file=log)
        has_model = True
    else:
        has_model = False

    dds_out = []
    for i, ds in enumerate(xds):
        uvw = ds.UVW.data
        freq = ds.FREQ.data
        vis = ds.VIS.data
        wgt = ds.WEIGHT.data
        if opts.robustness is not None:
            imwgt = counts_to_weights(counts[ds.bandid],
                                      uvw,
                                      freq,
                                      nx, ny,
                                      cell_rad, cell_rad,
                                      opts.robustness)
            wgt *= imwgt
        # This is a vis space mask (see wgridder convention)
        mask = ds.MASK.data
        if opts.target is not None:
            obs_time = ds.time_out
            tra, tdec = get_coordinates(obs_time)
            tcoords=np.zeros((1,2))
            tcoords[0,0] = tra
            tcoords[0,1] = tdec
            coords0 = np.array((ds.ra, ds.dec))
            l0, m0 = radec_to_lm(tcoords, coords0)
        else:
            l0 = None
            m0 = None

        dvars = {}
        if opts.dirty:
            dirty = vis2im(uvw=uvw,
                           freq=freq,
                           vis=vis,
                           wgt=wgt,
                           nx=nx,
                           ny=ny,
                           cellx=cell_rad,
                           celly=cell_rad,
                           x0=l0, y0=m0,
                           nthreads=opts.nvthreads,
                           epsilon=opts.epsilon,
                           precision=precision,
                           mask=mask,
                           do_wgridding=opts.wstack,
                           double_precision_accumulation=opts.double_accum)
            dirty = inlined_array(dirty, [uvw, freq])
            dvars['DIRTY'] = (('x', 'y'), dirty)

        if opts.psf:
            psf = vis2im(uvw=uvw,
                         freq=freq,
                         vis=wgt.astype(vis.dtype),
                         nx=nx_psf,
                         ny=ny_psf,
                         cellx=cell_rad,
                         celly=cell_rad,
                         x0=l0, y0=m0,
                         nthreads=opts.nvthreads,
                         epsilon=opts.epsilon,
                         precision=precision,
                         mask=mask,
                         do_wgridding=opts.wstack,
                         double_precision_accumulation=opts.double_accum)
            psf = inlined_array(psf, [uvw, freq])
            # get FT of psf
            psfhat = fft2d(psf, nthreads=opts.nvthreads)
            dvars['PSF'] = (('x_psf', 'y_psf'), psf)
            dvars['PSFHAT'] = (('x_psf', 'yo2'), psfhat)

        if opts.weight:
            dvars['WEIGHT'] = (('row', 'chan'), wgt)

        dvars['FREQ'] = (('chan',), freq)
        dvars['UVW'] = (('row', 'three'), uvw)
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

        # only make residual if model exists
        if has_model:
            model = mds[i].MODEL.data
            dvars['MODEL'] = (('x', 'y'), model)
            from pfb.operators.hessian import hessian
            hessopts = {
                'cell': cell_rad,
                'wstack': opts.wstack,
                'epsilon': opts.epsilon,
                'double_accum': opts.double_accum,
                'nthreads': opts.nvthreads
            }
            # we only want to apply the beam once here
            residual = dirty - hessian(bvals * model, uvw, wgt,
                                       mask, freq, None, hessopts)
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
            'timeid': ds.timeid,
            'freq_out': ds.freq_out,
            'time_out': ds.time_out,
            'robustness': opts.robustness
        }

        out_ds = xr.Dataset(dvars, attrs=attrs).chunk({'row':100000,
                                                       'chan':128,
                                                       'x':4096,
                                                       'y':4096,
                                                       'x_psf': 4096,
                                                       'y_psf':4096,
                                                       'yo2': 2048})
        dds_out.append(out_ds.unify_chunks())

    writes = xds_to_zarr(dds_out, dds_name, columns='ALL')

    # dask.visualize(writes, color="order", cmap="autumn",
    #                node_attr={"penwidth": "4"},
    #                filename=f'{basename}_grid_ordered_graph.pdf',
    #                optimize_graph=False)
    # dask.visualize(writes, filename=f'{basename}_grid_graph.pdf',
    #                optimize_graph=False)

    print("Computing image space data products", file=log)
    with compute_context(opts.scheduler, basename+'_grid'):
        dask.compute(writes)

    dds = xds_from_zarr(dds_name, chunks={'x': -1, 'y': -1})

    # convert to fits files
    fitsout = []
    if opts.fits_mfs:
        if opts.dirty:
            fitsout.append(dds2fits_mfs(dds, 'DIRTY', basename, norm_wsum=True))
        if opts.psf:
            fitsout.append(dds2fits_mfs(dds, 'PSF', basename, norm_wsum=True))
        if has_model:
            fitsout.append(dds2fits_mfs(dds, 'RESIDUAL', basename, norm_wsum=True))
            fitsout.append(dds2fits_mfs(dds, 'MODEL', basename, norm_wsum=False))

    if opts.fits_cubes:
        if opts.dirty:
            fitsout.append(dds2fits(dds, 'DIRTY', basename, norm_wsum=True))
        if opts.psf:
            fitsout.append(dds2fits(dds, 'PSF', basename, norm_wsum=True))
        if has_model:
            fitsout.append(dds2fits(dds, 'RESIDUAL', basename, norm_wsum=True))
            fitsout.append(dds2fits(dds, 'MODEL', basename, norm_wsum=False))

    if len(fitsout):
        print("Writing fits", file=log)
        dask.compute(fitsout)

    if opts.scheduler=='distributed':
        from distributed import get_client
        client = get_client()
        client.close()

    print("All done here.", file=log)
