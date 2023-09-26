# flake8: noqa
import os
import dask
dask.config.set(**{'array.slicing.split_large_chunks': False})
from pathlib import Path
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
    OmegaConf.set_struct(opts, True)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ldir = Path(opts.log_directory).resolve()
    ldir.mkdir(parents=True, exist_ok=True)
    pyscilog.log_to_file(f'{str(ldir)}/grid_{timestamp}.log')
    print(f'Logs will be written to {str(ldir)}/grid_{timestamp}.log', file=log)
    basedir = Path(opts.output_filename).resolve().parent
    basedir.mkdir(parents=True, exist_ok=True)
    basename = f'{opts.output_filename}_{opts.product.upper()}'
    dds_name = f'{basename}_{opts.postfix}.dds'

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)
        import dask
        dask.config.set(**{'array.slicing.split_large_chunks': False})
        from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
        from pfb.utils.misc import compute_context
        from pfb.utils.fits import dds2fits, dds2fits_mfs

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        dds_out = _grid(**opts)

        writes = xds_to_zarr(dds_out, dds_name, columns='ALL')

        # dask.visualize(writes, color="order", cmap="autumn",
        #                node_attr={"penwidth": "4"},
        #                filename=f'{basename}_grid_ordered_graph.pdf',
        #                optimize_graph=False)
        # dask.visualize(writes, filename=f'{basename}_grid_graph.pdf',
        #                optimize_graph=False)

        print("Computing image space data products", file=log)
        with compute_context(opts.scheduler, f'{str(ldir)}/grid_{timestamp}'):
            dask.compute(writes, optimize_graph=False)

        dds = xds_from_zarr(dds_name, chunks={'x': -1, 'y': -1})
        if 'PSF' in dds[0]:
            for i, ds in enumerate(dds):
                dds[i] = ds.chunk({'x_psf': -1, 'y_psf': -1})

        # convert to fits files
        fitsout = []
        if opts.fits_mfs:
            if opts.dirty:
                fitsout.append(dds2fits_mfs(dds, 'DIRTY', f'{basename}_{opts.postfix}', norm_wsum=True))
            if opts.psf:
                fitsout.append(dds2fits_mfs(dds, 'PSF', f'{basename}_{opts.postfix}', norm_wsum=True))
            if opts.residual and 'MODEL' in dds[0]:
                fitsout.append(dds2fits_mfs(dds, 'RESIDUAL', f'{basename}_{opts.postfix}', norm_wsum=True))
                fitsout.append(dds2fits_mfs(dds, 'MODEL', f'{basename}_{opts.postfix}', norm_wsum=False))

        if opts.fits_cubes:
            if opts.dirty:
                fitsout.append(dds2fits(dds, 'DIRTY', f'{basename}_{opts.postfix}', norm_wsum=True))
            if opts.psf:
                fitsout.append(dds2fits(dds, 'PSF', f'{basename}_{opts.postfix}', norm_wsum=True))
            if opts.residual and 'MODEL' in dds[0]:
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

def _grid(xdsi=None, **kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)


    import numpy as np
    import dask
    from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
    import dask.array as da
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from pfb.utils.misc import compute_context
    from pfb.operators.gridder import image_data_products
    from pfb.operators.fft import fft2d
    from pfb.utils.weighting import (compute_counts,
                                     filter_extreme_counts)
    from pfb.utils.beam import eval_beam
    import xarray as xr
    from uuid import uuid4
    from dask.graph_manipulation import clone
    from pfb.utils.astrometry import get_coordinates
    from africanus.coordinates import radec_to_lm
    from pfb.utils.misc import concat_chan, concat_row
    import sympy as sm
    from sympy.utilities.lambdify import lambdify
    from sympy.parsing.sympy_parser import parse_expr
    from quartical.utils.dask import Blocker


    basename = f'{opts.output_filename}_{opts.product.upper()}'

    # xds contains vis products, no imaging weights applied
    xds_name = f'{basename}.xds'
    if xdsi is not None:
        xds = []
        for ds in xdsi:
            xds.append(ds.chunk({'row':-1}))
    else:
        xds = xds_from_zarr(xds_name, chunks={'row': -1})
    # dds contains image space products including imaging weights and uvw
    dds_name = f'{basename}_{opts.postfix}.dds'

    if os.path.isdir(dds_name):
        dds_exists = True
        if opts.overwrite:
            print(f'Removing {dds_name}', file=log)
            import shutil
            shutil.rmtree(dds_name)
            dds_exists = False
    else:
        dds_exists = False

    times_in = []
    freqs_in = []
    for ds in xds:
        times_in.append(ds.time_out)
        freqs_in.append(ds.freq_out)

    times_in = np.unique(times_in)
    freqs_in = np.unique(freqs_in)

    ntime_in = times_in.size
    nband_in = freqs_in.size

    if opts.concat_row and len(xds) > nband_in:
        print('Concatenating datasets along row dimension', file=log)
        xds = concat_row(xds)
        ntime = 1
        times_out = np.array((xds[0].time_out,))
    else:
        ntime = ntime_in
        times_out = times_in

    if opts.nband != nband_in and len(xds) > ntime:
        print('Concatenating datasets along chan dimension. '
              f'Mapping {nband_in} datasets to {opts.nband} bands', file=log)
        xds = concat_chan(xds, nband_out=opts.nband)
        nband = opts.nband
        freqs_out = []
        for ds in xds:
            freqs_out.append(ds.freq_out)
        freqs_out = np.unique(freqs_out)
    else:
        nband = nband_in
        freqs_out = freqs_in

    # do this after concatenation (to check)
    for i, ds in enumerate(xds):
        xds[i] = ds.chunk({'chan': -1})

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
            raise ValueError("Requested cell size too large. "
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

    if opts.dirty:
        print(f"Image size = (ntime={ntime}, nband={nband}, nx={nx}, ny={ny})", file=log)

    nx_psf = good_size(int(opts.psf_oversize * nx))
    while nx_psf % 2:
        nx_psf += 1
        nx_psf = good_size(nx_psf)

    ny_psf = good_size(int(opts.psf_oversize * ny))
    while ny_psf % 2:
        ny_psf += 1
        ny_psf = good_size(ny_psf)
    nyo2 = ny_psf//2 + 1

    if opts.psf:
        print(f"PSF size = (ntime={ntime}, nband={nband}, nx={nx_psf}, ny={ny_psf})", file=log)

    # if dds exists, check that existing dds is compatible with input
    if dds_exists:
        dds = xds_from_zarr(dds_name)
        # these need to be aligned at this stage
        for ds, out_ds in zip(xds, dds):
            assert ds.freq_out == out_ds.freq_out
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
                                chunks={'params':-1, 'comps':-1})[0]
        except Exception as e:
            raise ValueError(f"No dataset found at {opts.transfer_model_from}")

        # should we construct the model func outside
        # of eval_coeffs_to_slice?
        # params = sm.symbols(('t','f'))
        # params += sm.symbols(tuple(mds.params.values))
        # symexpr = parse_expr(mds.parametrisation)
        # model_func = lambdify(params, symexpr)
        # texpr = parse_expr(mds.texpr)
        # tfunc = lambdify(params[0], texpr)
        # fexpr = parse_expr(mds.fexpr)
        # ffunc = lambdify(params[1], fexpr)


        # we only want to load these once
        model_coeffs = mds.coefficients.values
        locx = mds.location_x.values
        locy = mds.location_y.values
        params = mds.params.values
        coeffs = mds.coefficients.values

        print(f"Loading model from {opts.transfer_model_from}. ",
              file=log)

    dds_out = []
    for i, ds in enumerate(xds):
        if dds_exists:
            out_ds = dds[i].chunk({'row':-1,
                                   'chan':-1,
                                   'x':-1,
                                   'y':-1})
        else:
            out_ds = xr.Dataset()
        uvw = ds.UVW.data
        freq = ds.FREQ.data
        vis = ds.VIS.data
        wgt = ds.WEIGHT.data
        # This is a vis space mask (see wgridder convention)
        mask = ds.MASK.data
        bandid = np.where(freqs_out == ds.freq_out)[0][0]
        timeid = np.where(times_out == ds.time_out)[0][0]

        # compute lm coordinates of target
        if opts.target is not None:
            tmp = opts.target.split(',')
            if len(tmp) == 1 and tmp[0] == opts.target:
                obs_time = ds.time_out
                tra, tdec = get_coordinates(obs_time, target=opts.target)
            else:  # we assume a HH:MM:SS,DD:MM:SS format has been passed in
                from astropy import units as u
                from astropy.coordinates import SkyCoord
                c = SkyCoord(tmp[0], tmp[1], frame='fk5', unit=(u.hourangle, u.deg))
                tra = np.deg2rad(c.ra.value)
                tdec = np.deg2rad(c.dec.value)

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

        out_ds = out_ds.assign_attrs(**{
            'ra': tra,
            'dec': tdec,
            'x0': x0,
            'y0': y0,
            'cell_rad': cell_rad,
            'bandid': bandid,
            'timeid': timeid,
            'freq_out': ds.freq_out,
            'time_out': ds.time_out,
            'robustness': opts.robustness,
            'super_resolution_factor': opts.super_resolution_factor,
            'field_of_view': opts.field_of_view,
            'product': opts.product
        })
        # TODO - assign ug,vg-coordinates
        x = (-nx/2 + np.arange(nx)) * cell_rad + x0
        y = (-ny/2 + np.arange(ny)) * cell_rad + y0
        out_ds = out_ds.assign_coords(**{
           'x': x,
           'y': y
        })

        # evaluate beam at x and y coords
        cell_deg = np.rad2deg(cell_rad)
        l = (-(nx//2) + da.arange(nx)) * cell_deg + np.deg2rad(x0)
        m = (-(ny//2) + da.arange(ny)) * cell_deg + np.deg2rad(y0)
        ll, mm = da.meshgrid(l, m, indexing='ij')
        bvals = eval_beam(ds.BEAM.data, ll, mm)
        out_ds = out_ds.assign(**{'BEAM': (('x', 'y'), bvals)})

        # get the model
        if opts.transfer_model_from is not None:
            from pfb.utils.misc import eval_coeffs_to_slice
            model = eval_coeffs_to_slice(
                ds.time_out,
                ds.freq_out,
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
            model = da.from_array(model, chunks=(-1,-1))
            out_ds = out_ds.assign(**{'MODEL': (('x', 'y'), model)})

        elif 'MODEL' in out_ds:
            model = out_ds.MODEL.data
        else:
            model = None


        if opts.robustness is not None:
            # we'll skip this if counts already exists
            # what to do if flags have changed?
            if 'COUNTS' not in out_ds:
                counts = compute_counts(
                        clone(uvw),
                        freq,
                        mask,
                        nx,
                        ny,
                        cell_rad,
                        cell_rad,
                        real_type,
                        ngrid=opts.nvthreads)
                # get rid of artificially high weights corresponding to
                # nearly empty cells
                if opts.filter_extreme_counts:
                    counts = filter_extreme_counts(counts, nbox=opts.filter_nbox,
                                                   nlevel=opts.filter_level)

                # do we want the coordinates to be ug, vg rather?
                out_ds = out_ds.assign(**{'COUNTS': (('x', 'y'), counts)})

            else:
                counts = out_ds.COUNTS.data
        else:
            counts = None

        # we might want to chunk over row chan in the future but
        # for now these will always have chunks=(-1,-1)
        blocker = Blocker(image_data_products, ('row', 'chan'))
        blocker.add_input('uvw', uvw, ('row','three'))
        blocker.add_input('freq', freq, ('chan',))
        blocker.add_input('vis', vis, ('row','chan'))
        blocker.add_input('wgt', wgt, ('row','chan'))
        blocker.add_input('mask', mask, ('row','chan'))
        if counts is not None:
            blocker.add_input('counts', counts, ('x','y'))
        else:
            blocker.add_input('counts', None)
        blocker.add_input('nx', nx)
        blocker.add_input('ny', ny)
        blocker.add_input('nx_psf', nx_psf)
        blocker.add_input('ny_psf', ny_psf)
        blocker.add_input('cellx', cell_rad)
        blocker.add_input('celly', cell_rad)
        if model is not None:
            blocker.add_input('model', model, ('x', 'y'))
        else:
            blocker.add_input('model', None)
        blocker.add_input('robustness', opts.robustness)
        blocker.add_input('x0', x0)
        blocker.add_input('y0', y0)
        blocker.add_input('nthreads', opts.nvthreads)
        blocker.add_input('epsilon', opts.epsilon)
        blocker.add_input('do_wgridding', opts.do_wgridding)
        blocker.add_input('double_accum', opts.double_accum)
        blocker.add_input('l2reweight_dof', opts.l2reweight_dof)
        blocker.add_input('do_psf', opts.psf)
        blocker.add_input('do_weight', opts.weight)
        blocker.add_input('do_residual', opts.residual)

        blocker.add_output(
            'DIRTY',
            ('x', 'y'),
            ((nx,), (ny,)),
            wgt.dtype)

        blocker.add_output(
            'WSUM',
            ('scalar',),
            ((1,),),
            wgt.dtype)

        if opts.residual:
            blocker.add_output(
                'RESIDUAL',
                ('x', 'y'),
                ((nx,), (ny,)),
                wgt.dtype)

        if opts.psf:
            blocker.add_output(
                'PSF',
                ('x_psf', 'y_psf'),
                ((nx_psf,), (ny_psf,)),
                wgt.dtype)
            blocker.add_output(
                'PSFHAT',
                ('x_psf', 'yo2'),
                ((nx_psf,), (nyo2,)),
                vis.dtype)

        if opts.weight:
            blocker.add_output(
                'WEIGHT',
                ('row', 'chan'),
                wgt.chunks,
                wgt.dtype)

        output_dict = blocker.get_dask_outputs()
        out_ds = out_ds.assign(**{
            'DIRTY': (('x', 'y'), output_dict['DIRTY'])
            })

        if opts.psf:
            out_ds = out_ds.assign(**{
                'PSF': (('x_psf', 'y_psf'), output_dict['PSF']),
                'PSFHAT': (('x_psf', 'yo2'), output_dict['PSFHAT'])
                })

        # TODO - don't put vis space products in dds
        # but how apply imaging weights in that case?
        if opts.weight:
            out_ds = out_ds.assign(**{
                'WEIGHT': (('row', 'chan'), output_dict['WEIGHT'])
                })

        if opts.residual:
            out_ds = out_ds.assign(**{
                'RESIDUAL': (('x', 'y'), output_dict['RESIDUAL'])
                })


        out_ds = out_ds.assign(**{
            'FREQ': (('chan',), freq),
            'UVW': (('row', 'three'), uvw),
            'MASK': (('row', 'chan'), mask),
            'WSUM': (('scalar',), output_dict['WSUM'])
            })


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

    return dds_out
