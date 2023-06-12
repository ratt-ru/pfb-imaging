# flake8: noqa
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
    defaults[key] = schema.model2comps["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.model2comps)
def model2comps(**kw):
    '''
    Convert model to parameters to make it possible to evaluate it at
    an arbitrary (t, nu)
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'model2comps_{timestamp}.log')

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

    import numpy as np
    from pfb.utils.misc import construct_mappings
    import dask
    from dask.distributed import performance_report
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    import dask.array as da
    from africanus.constants import c as lightspeed
    from pfb.utils.fits import load_fits, data_from_header
    from pfb.utils.misc import restore_corrs, model_from_comps
    from astropy.io import fits
    from pfb.utils.misc import compute_context
    import xarray as xr
    from sympy.utilities.lambdify import lambdify
    from sympy.parsing.sympy_parser import parse_expr

    basename = f'{opts.output_filename}_{opts.product.upper()}'
    dds_name = f'{basename}_{opts.postfix}.dds.zarr'
    coeff_name = f'{basename}_{opts.postfix}.coeffs.zarr'
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
    mask = np.zeros((nx, ny), dtype=bool)
    for ds in dds:
        b = ds.bandid
        t = ds.timeid
        model[t, b] = getattr(ds, opts.model_name).values
        wsums[t, b] += ds.WSUM.values[0]

    # model = da.stack(model)
    # wsums = da.stack(wsums).squeeze()
    # model, wsums = dask.compute(model, wsums)
    if not np.any(model):
        raise ValueError('Model is empty')
    radec = (dds[0].ra, dds[0].dec)

    ref_freq = mfreqs[0]
    ref_time = mtimes[0]

    # interpolate model in frequency
    mask = np.any(model, axis=(0,1))
    Ix, Iy = np.where(mask)
    ncomps = Ix.size

    # components excluding zeros
    beta = model[:, :, Ix, Iy]
    if opts.spectral_poly_order is not None and nband > 1:
        orderf = opts.spectral_poly_order
        if orderf > nband:
            raise ValueError("spectral-poly-order can't be larger than nband")
        print(f"Fitting freq axis with polynomial of order {orderf}", file=log)
        # we are given frequencies at bin centers, convert to bin edges
        delta_freq = mfreqs[1] - mfreqs[0]
        wlow = (mfreqs - delta_freq/2.0)/ref_freq
        whigh = (mfreqs + delta_freq/2.0)/ref_freq
        wdiff = whigh - wlow

        # set design matrix for each component
        Xfit = np.zeros([mfreqs.size, orderf])
        for i in range(1, orderf+1):
            Xfit[:, i-1] = (whigh**i - wlow**i)/(i*wdiff)

        comps = np.zeros((ntime, orderf, Ix.size))
        freq_fitted = True
        for t in range(ntime):
            dirty_comps = Xfit.T.dot(wsums[t, :, None]*beta[t])

            hess_comps = Xfit.T.dot(wsums[t, :, None]*Xfit)

            comps[t] = np.linalg.solve(hess_comps, dirty_comps)

    else:
        raise NotImplementedError("Interpolation is currently mandatory")
        print("Not fitting frequency axis", file=log)
        comps = beta
        freq_fitted = False
        orderf = mfreqs.size

    if opts.temporal_poly_order is not None and ntime > 1:
        ordert = opts.temporal_poly_order
        if order > ntime:
            raise ValueError("temporal-poly-order can't be larger than ntime")
        print(f"Fitting time axis with polynomial of order {orderf}", file=log)
        # we are given times at bin centers, convert to bin edges
        delta_time = mtimes[1] - mtimes[0]
        wlow = (mtimes - delta_times/2.0)/ref_time
        whigh = (mfreqs + delta_freq/2.0)/ref_time
        wdiff = whigh - wlow

        # set design matrix for each component
        Xfit = np.zeros([mtimes.size, ordert])
        for i in range(1, ordert+1):
            Xfit[:, i-1] = (whigh**i - wlow**i)/(i*wdiff)

        if freq_fitted:
            compsnu = comps.copy()
        comps = np.zeros((ordert, opts.spectral_poly_order, Ix.size))
        time_fitted = True
        for b in range(orderf):
            dirty_comps = Xfit.T.dot(wsums[:, b, None]*compsnu[b])

            hess_comps = Xfit.T.dot(wsums[:, b, None]*Xfit)

            comps[:, b] = np.linalg.solve(hess_comps, dirty_comps)
    else:
        raise NotImplementedError("Interpolation is currently mandatory")
        print("Not fitting time axis", file=log)
        comps = comps
        time_fitted = False
        ordert = mtimes.size


    # construct symbolic expression
    from sympy.abc import t, f
    from sympy import symbols

    thetasf = []
    params = ()
    for i in range(orderf):
        coefft = symbols(f't(0:{ordert})_f{i}')
        # the reshape on comps needs to be consistent with the ordering in params
        params += coefft
        thetaf = sum(co*t**j for j, co in enumerate(coefft))
        thetasf.append(thetaf)

    polysym = sum(co*f**j for j, co in enumerate(thetasf))
    # polyfunc = lambdify(params, polysym)
    # import ipdb; ipdb.set_trace()

    # save interpolated dataset
    data_vars = {
        'coefficients': (('params', 'comps'), comps.reshape(ordert*orderf, ncomps))
    }
    coords = {
        'location_x': (('location_x',), Ix),
        'location_y': (('location_y',), Iy),
        'params': (('params',), list(map(str, params)))
    }
    attrs = {
        'cell_rad_x': cell_rad,
        'cell_rad_y': cell_rad,
        'npix_x': nx,
        'npix_y': ny,
        'ref_freq': ref_freq,
        'ref_time': ref_time,
        'center_x': x0,
        'center_y': y0,
        'ra': dds[0].ra,
        'dec': dds[0].dec,
        'stokes': opts.product,
        'parametrisation': str(polysym)
    }


    coeff_dataset = xr.Dataset(data_vars=data_vars,
                               coords=coords,
                               attrs=attrs)
    writes = xds_to_zarr(coeff_dataset,
                         f'{basename}_{opts.postfix}.coeffs.zarr',
                         columns='ALL')
    print(f'Writing interpolated model to {basename}_{opts.postfix}.coeffs.zarr',
          file=log)
    dask.compute(writes)


    print("All done here.", file=log)
