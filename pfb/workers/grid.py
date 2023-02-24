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
    from pfb.operators.gridder import vis2im, loc2psf_vis
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
    dds_name = f'{basename}_{opts.postfix}.dds.zarr'

    if os.path.isdir(dds_name):
        dds_exists = True
        if opts.overwrite:
            print(f'Removing {dds_name}', file=log)
            import shutil
            shutil.rmtree(dds_name)
            dds_exists = False
    else:
        dds_exists = False

    if opts.concat and len(xdsp) > opts.nband:
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
        ntime = 1
    else:
        xds = xdsp
        ntime = len(xds) // opts.nband
        if len(xds) < opts.nband:
            raise RuntimeError("len(xds) < nband. Run with the same nband as init worker.")
        elif len(xds) % ntime:
            raise RuntimeError("len(xds) != ntime*nband. This is a bug.")

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

    # print(f'Not overwriting {dds_name}, directory exists. '
    #               f'Set overwrite flag if you mean to overwrite.', file=log)
    # if dds exists, check that existing dds is compatible with input
    if dds_exists:
        dds = xds_from_zarr(dds_name)
        # these need to be aligned at this stage
        for ds, out_ds in zip(xds, dds):
            assert ds.bandid == out_ds.bandid
            assert ds.freq_out == out_ds.freq_out
            assert ds.timeid == out_ds.timeid
            assert ds.time_out == out_ds.time_out
            assert ds.ra == out_ds.ra
            assert ds.dec == out_ds.dec
            assert out_ds.x.size == nx
            assert out_ds.y.size == ny
            assert out_ds.cell_rad == cell_rad
        print(f'As far as we can tell {dds_name} can be reused/updated.',
              file=log)
    else:
        print(f'Image space data products will be stored in {dds_name}.', file=log)

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
        if dds_exists:
            out_ds = dds[i]
        else:
            out_ds = xr.Dataset()
        uvw = ds.UVW.data
        freq = ds.FREQ.data
        vis = ds.VIS.data
        wgt = ds.WEIGHT.data
        # This is a vis space mask (see wgridder convention)
        mask = ds.MASK.data
        if opts.robustness is not None:
            # we'll skip this if counts already exists
            # what to do if flags have changed?
            if 'COUNTS' not in out_ds:
                counts = compute_counts(
                        uvw,
                        freq,
                        mask,
                        nx,
                        ny,
                        cell_rad,
                        cell_rad,
                        wgt.dtype)
                # get rid of artificially high weights corresponding to
                # nearly empty cells
                if opts.filter_extreme_counts:
                    counts = filter_extreme_counts(counts, nbox=opts.filter_nbox,
                                                   nlevel=opts.filter_level)

                # do we want the coordinates to be ug, vg rather?
                out_ds = ds.assign(**{'COUNTS': (('x', 'y'), counts)})

            # we usually want to re-evaluate this since the robustness may change
            imwgt = counts_to_weights(out_ds.COUNTS.data,
                                      uvw,
                                      freq,
                                      nx, ny,
                                      cell_rad, cell_rad,
                                      opts.robustness)

            wgt *= imwgt


        # compute lm coordinates of target
        if opts.target is not None:
            obs_time = ds.time_out
            tra, tdec = get_coordinates(obs_time)
            tcoords=np.zeros((1,2))
            tcoords[0,0] = tra
            tcoords[0,1] = tdec
            coords0 = np.array((ds.ra, ds.dec))
            lm0 = radec_to_lm(tcoords, coords0).squeeze()
            # LB - why the negative?
            x0 = -lm0[0]
            y0 = -lm0[1]
        else:
            x0 = 0.0
            y0 = 0.0
            tra = ds.ra
            tdec = ds.dec

        # TODO - assign coordinates
        if not dds_exists:
            out_ds = out_ds.assign_attrs(**{
                'ra': tra,
                'dec': tdec,
                'x0': x0,
                'y0': y0,
                'cell_rad': cell_rad,
                'bandid': ds.bandid,
                'timeid': ds.timeid,
                'freq_out': ds.freq_out,
                'time_out': ds.time_out,
            })

        if opts.dirty:
            dirty = vis2im(uvw=uvw,
                           freq=freq,
                           vis=vis,
                           wgt=wgt,
                           nx=nx,
                           ny=ny,
                           cellx=cell_rad,
                           celly=cell_rad,
                           x0=x0, y0=y0,
                           nthreads=opts.nvthreads,
                           epsilon=opts.epsilon,
                           precision=precision,
                           mask=mask,
                           do_wgridding=opts.wstack,
                           double_precision_accumulation=opts.double_accum)
            dirty = inlined_array(dirty, [uvw, freq])
            out_ds = out_ds.assign(**{'DIRTY': (('x', 'y'), dirty)})

        if opts.psf:
            psf_vis = loc2psf_vis(uvw,
                                  freq,
                                  cell_rad,
                                  x0,
                                  y0,
                                  wstack=opts.wstack,
                                  epsilon=opts.epsilon,
                                  nthreads=opts.nvthreads,
                                  precision=precision)
            psf = vis2im(uvw=uvw,
                         freq=freq,
                         vis=psf_vis,
                         wgt=wgt,
                         nx=nx_psf,
                         ny=ny_psf,
                         cellx=cell_rad,
                         celly=cell_rad,
                         x0=x0, y0=y0,
                         nthreads=opts.nvthreads,
                         epsilon=opts.epsilon,
                         precision=precision,
                         mask=mask,
                         do_wgridding=opts.wstack,
                         double_precision_accumulation=opts.double_accum)
            psf = inlined_array(psf, [uvw, freq])
            # get FT of psf
            psfhat = fft2d(psf, nthreads=opts.nvthreads)
            out_ds = out_ds.assign(**{'PSF': (('x_psf', 'y_psf'), psf),
                                      'PSFHAT': (('x_psf', 'yo2'), psfhat)})

        # TODO - don't put vis space products in dds
        if opts.weight:
            out_ds = out_ds.assign(**{'WEIGHT': (('row', 'chan'), wgt)})

        wsum = wgt[mask.astype(bool)].sum()

        # evaluate beam at x and y coords
        cell_deg = np.rad2deg(cell_rad)
        l = (-(nx//2) + da.arange(nx)) * cell_deg + np.deg2rad(x0)
        m = (-(ny//2) + da.arange(ny)) * cell_deg + np.deg2rad(y0)
        ll, mm = da.meshgrid(l, m, indexing='ij')
        bvals = eval_beam(ds.BEAM.data, ll, mm)

        if opts.residual and 'MODEL' in out_ds:
            model = out_ds.MODEL.data
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
            out_ds = out_ds.assign(**{'RESIDUAL': (('x', 'y'), residual)})


        out_ds = out_ds.assign(**{'FREQ': (('chan',), freq),
                                  'UVW': (('row', 'three'), uvw),
                                  'MASK': (('row', 'chan'), mask),
                                  'WSUM': (('scalar',), da.atleast_1d(wsum)),
                                  'BEAM': (('x', 'y'), bvals)})



        out_ds = out_ds.chunk({'row':100000,
                               'chan':128,
                               'x':4096,
                               'y':4096})
        # necessary to make psf optional
        if 'x_psf' in out_ds.dims:
            out_ds = out_ds.chunk({'x_psf': 4096,
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
            fitsout.append(dds2fits_mfs(dds, 'DIRTY', f'{basename}_{opts.postfix}', norm_wsum=True))
        if opts.psf:
            fitsout.append(dds2fits_mfs(dds, 'PSF', f'{basename}_{opts.postfix}', norm_wsum=True))
        if has_model:
            fitsout.append(dds2fits_mfs(dds, 'RESIDUAL', f'{basename}_{opts.postfix}', norm_wsum=True))
            fitsout.append(dds2fits_mfs(dds, 'MODEL', f'{basename}_{opts.postfix}', norm_wsum=False))

    if opts.fits_cubes:
        if opts.dirty:
            fitsout.append(dds2fits(dds, 'DIRTY', f'{basename}_{opts.postfix}', norm_wsum=True))
        if opts.psf:
            fitsout.append(dds2fits(dds, 'PSF', f'{basename}_{opts.postfix}', norm_wsum=True))
        if has_model:
            fitsout.append(dds2fits(dds, 'RESIDUAL', f'{basename}_{opts.postfix}', norm_wsum=True))
            fitsout.append(dds2fits(dds, 'MODEL', f'{basename}_{opts.postfix}', norm_wsum=False))

    if len(fitsout):
        print("Writing fits", file=log)
        dask.compute(fitsout)

    if opts.scheduler=='distributed':
        from distributed import get_client
        client = get_client()
        client.close()

    print("All done here.", file=log)
