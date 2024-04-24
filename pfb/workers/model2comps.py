# flake8: noqa
from pathlib import Path
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('MODEL2COMPS')


from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.model2comps["inputs"].keys():
    defaults[key.replace("-", "_")] = schema.model2comps["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.model2comps)
def model2comps(**kw):
    '''
    Convert model in dds to components.
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ldir = Path(opts.log_directory).resolve()
    ldir.mkdir(parents=True, exist_ok=True)
    pyscilog.log_to_file(f'{ldir}/model2comps_{timestamp}.log')
    print(f'Logs will be written to {str(ldir)}/model2comps_{timestamp}.log', file=log)

    OmegaConf.set_struct(opts, True)

    if opts.product.upper() not in ["I"]:
                                    # , "Q", "U", "V", "XX", "YX", "XY",
                                    # "YY", "RR", "RL", "LR", "LL"]:
        raise NotImplementedError(f"Product {opts.product} not yet supported")

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _model2comps(**opts)

def _model2comps(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import os
    import numpy as np
    from pfb.utils.misc import construct_mappings
    import dask
    from dask.distributed import performance_report
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    import dask.array as da
    from africanus.constants import c as lightspeed
    from pfb.utils.fits import load_fits, data_from_header
    from astropy.io import fits
    from pfb.utils.misc import compute_context, fit_image_cube
    import xarray as xr
    import fsspec as fs
    from daskms.fsspec_store import DaskMSStore
    import json

    basename = f'{opts.output_filename}_{opts.product.upper()}'
    dds_name = f'{basename}_{opts.suffix}.dds'

    if opts.model_out is not None:
        coeff_name = opts.model_out
    else:
        coeff_name = f'{basename}_{opts.suffix}_{opts.model_name.lower()}.mds'

    mdsstore = DaskMSStore(coeff_name)
    if mdsstore.exists():
        if opts.overwrite:
            print(f"Overwriting {coeff_name}", file=log)
            mdsstore.rm(recursive=True)
        else:
            raise ValueError(f"{coeff_name} exists. "
                             "Set --overwrite to overwrite it. ")

    dds = xds_from_zarr(dds_name,
                        chunks={'x':-1,
                                'y':-1})
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)

    # get cube info
    nband = 0
    ntime = 0
    mfreqs = []
    mtimes = []
    for ds in dds:
        nband = np.maximum(ds.bandid+1, nband)
        ntime = np.maximum(ds.timeid+1, ntime)
        mfreqs.append(ds.freq_out)
        mtimes.append(ds.time_out)
    mfreqs = np.unique(np.array(mfreqs))
    mtimes = np.unique(np.array(mtimes))
    assert mfreqs.size == nband
    assert mtimes.size == ntime
    assert len(dds) == nband*ntime

    # stack cube
    nx = dds[0].x.size
    ny = dds[0].y.size
    x0 = dds[0].x0
    y0 = dds[0].y0

    # TODO - do this as a dask graph
    # model = [da.zeros((nx, ny)) for _ in range(opts.nband)]
    # wsums = [da.zeros(1) for _ in range(opts.nband)]
    model = np.zeros((ntime, nband, nx, ny), dtype=np.float64)
    wsums = np.zeros((ntime, nband), dtype=np.float64)
    for ds in dds:
        b = ds.bandid
        t = ds.timeid
        model[t, b] = getattr(ds, opts.model_name).values
        wsums[t, b] += ds.WSUM.values[0]

    if not opts.use_wsum:
        wsums[...] = 1.0

    if opts.min_val is not None:
        mmfs = model[wsums>0]
        if mmfs.ndim==3:
            mmfs = np.mean(mmfs, axis=0)
        elif mmfs.ndim==4:
            mmfs = np.mean(mmfs, axis=(0,1))
        model = np.where(np.abs(mmfs)[None, None] >= opts.min_val, model, 0.0)

    if not np.any(model):
        raise ValueError(f'Model is empty or has no components above {opts.min_val}')
    radec = (dds[0].ra, dds[0].dec)

    try:
        coeffs, Ix, Iy, expr, params, texpr, fexpr = \
            fit_image_cube(mtimes, mfreqs, model,
                           wgt=wsums,
                           nbasisf=opts.nbasisf,
                           method=opts.fit_mode,
                           sigmasq=opts.sigmasq)
    except np.linalg.LinAlgError as e:
        print(f"Exception {e} raised during fit ."
              f"Do you perhaps have empty sub-bands?"
              f"Decreasing nbasisf", file=log)
        quit()

    # save interpolated dataset
    data_vars = {
        'coefficients': (('par', 'comps'), coeffs),
    }
    coords = {
        'location_x': (('x',), Ix),
        'location_y': (('y',), Iy),
        # 'shape_x':,
        'params': (('par',), params),  # already converted to list
        'times': (('t',), mtimes),  # to allow rendering to original grid
        'freqs': (('f',), mfreqs)
    }
    attrs = {
        'spec': 'genesis',
        'cell_rad_x': cell_rad,
        'cell_rad_y': cell_rad,
        'npix_x': nx,
        'npix_y': ny,
        'texpr': texpr,
        'fexpr': fexpr,
        'center_x': x0,
        'center_y': y0,
        'ra': dds[0].ra,
        'dec': dds[0].dec,
        'stokes': opts.product,  # I,Q,U,V, IQ/IV, IQUV
        'parametrisation': expr  # already converted to str
    }


    coeff_dataset = xr.Dataset(data_vars=data_vars,
                               coords=coords,
                               attrs=attrs)
    print(f'Writing interpolated model to {coeff_name}',
          file=log)

    if opts.out_format == 'zarr':
        coeff_dataset.to_zarr(mdsstore.url)
    elif opts.out_format == 'json':
        coeff_dict = coeff_dataset.to_dict()
        with fs.open(mdsstore.url, 'w') as f:
            json.dump(coeff_dict, f)


    print("All done here.", file=log)
