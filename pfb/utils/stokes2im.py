import numpy as np
import numexpr as ne
from numba import njit, prange, literally
from numba.extending import overload
from dask.graph_manipulation import clone
import dask.array as da
from xarray import Dataset
from pfb.operators.gridder import vis2im
# from quartical.utils.numba import coerce_literal
from operator import getitem
from pfb.utils.beam import interp_beam
from pfb.utils.misc import JIT_OPTIONS, weight_from_sigma
import dask
from quartical.utils.dask import Blocker
from pfb.utils.stokes import stokes_funcs
from pfb.utils.stokes2vis import _weight_data
from pfb.operators.gridder import image_data_products
from pfb.utils.weighting import _compute_counts, _counts_to_weights
from pfb.utils.misc import eval_coeffs_to_slice
from ducc0.wgridder.experimental import vis2dirty, dirty2vis
from daskms.experimental.zarr import xds_from_zarr

# for old style vs new style warnings
from numba.core.errors import NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def single_stokes_image(ds=None,
                        mds=None,
                  jones=None,
                  opts=None,
                  nx=None,
                  ny=None,
                  freq=None,
                  chan_width=None,
                  cell_rad=None,
                  utime=None,
                  tbin_idx=None,
                  tbin_counts=None,
                  radec=None,
                  antpos=None,
                  poltype=None):

    if opts.precision.lower() == 'single':
        real_type = np.float32
        complex_type = np.complex64
    elif opts.precision.lower() == 'double':
        real_type = np.float64
        complex_type = np.complex128

    data = getattr(ds, opts.data_column).data
    nrow, nchan, ncorr = data.shape
    ntime = utime.size
    nant = antpos.shape[0]
    time_out = np.mean(utime)

    freq_out = np.mean(freq)
    freq_min = freq.min()
    freq_max = freq.max()

    # clone shared nodes
    ant1 = clone(ds.ANTENNA1.data)
    ant2 = clone(ds.ANTENNA2.data)
    uvw = clone(ds.UVW.data)

    # MS may contain auto-correlations
    if 'FLAG_ROW' in ds:
        frow = clone(ds.FLAG_ROW.data) | (ant1 == ant2)
    else:
        frow = (ant1 == ant2)

    if opts.flag_column is not None:
        flag = getattr(ds, opts.flag_column).data
        flag = da.any(flag, axis=2)
        flag = da.logical_or(flag, frow[:, None])
    else:
        flag = da.broadcast_to(frow[:, None], (nrow, nchan))

    if opts.sigma_column is not None:
        sigma = getattr(ds, opts.sigma_column).data
        # weight = 1.0/sigma**2
        weight = da.map_blocks(weight_from_sigma,
                               sigma,
                               chunks=sigma.chunks)
    elif opts.weight_column is not None:
        weight = getattr(ds, opts.weight_column).data
        if opts.weight_column=='WEIGHT':
            weight = da.broadcast_to(weight[:, None, :],
                                     (nrow, nchan, ncorr),
                                     chunks=data.chunks)
    else:
        # weight = da.ones_like(data, dtype=real_type)
        weight = da.ones((nrow, nchan, ncorr),
                         chunks=data.chunks,
                         dtype=real_type)

    if data.dtype != complex_type:
        data = data.astype(complex_type)

    if weight.dtype != real_type:
        weight = weight.astype(real_type)

    if jones is not None:
        if jones.dtype != complex_type:
            jones = jones.astype(complex_type)
        # qcal has chan and ant axes reversed compared to pfb implementation
        jones = da.swapaxes(jones, 1, 2)
        # data are not 2x2 so we need separate labels
        # for jones correlations and data/weight correlations
        # reshape to dispatch with generated_jit
        jones_ncorr = jones.shape[-1]
        if jones_ncorr == 2:
            jout = 'rafdx'
        elif jones_ncorr == 4:
            jout = 'rafdxx'
            jones = jones.reshape(ntime, nant, nchan, 1, 2, 2)
        else:
            raise ValueError("Incorrect number of correlations of "
                             f"{jones_ncorr} for product {opts.product}")
    else:
        jones = da.ones((ntime, nant, nchan, 1, 2),
                        chunks=(-1,)*5,
                        dtype=complex_type)
        jout = 'rafdx'

    # Note we do not chunk at this level since all the chunking happens upfront
    # we cast to dask arrays simply to defer the compute
    tbin_idx = da.from_array(tbin_idx, chunks=(-1))
    tbin_counts = da.from_array(tbin_counts, chunks=(-1))

    # compute lm coordinates of target
    if opts.target is not None:
        tmp = opts.target.split(',')
        if len(tmp) == 1 and tmp[0] == opts.target:
            obs_time = time_out
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
        tra = radec[0]
        tdec = radec[1]


    # print(model.max(), model.min())

    # import ipdb; ipdb.set_trace()
    # quit()
    # compute Stokes data and weights
    blocker = Blocker(image_data, 'rf')
    blocker.add_input("data", data, 'rfc')
    blocker.add_input('weight', weight, 'rfc')
    blocker.add_input('flag', flag, 'rf')
    blocker.add_input('uvw', uvw, ('row','three'))
    blocker.add_input('freq', da.from_array(freq, chunks=(-1,)), ('chan',))
    blocker.add_input('jones', jones, jout)
    blocker.add_input('tbin_idx', tbin_idx, 'r')
    blocker.add_input('tbin_counts', tbin_counts, 'r')
    blocker.add_input('ant1', ant1, 'r')
    blocker.add_input('ant2', ant2, 'r')
    blocker.add_input('pol', poltype)
    blocker.add_input('product', opts.product)
    blocker.add_input('nc', str(ncorr))  # dispatch based on ncorr to deal with dual pol
    blocker.add_input('nx', nx)
    blocker.add_input('ny', ny)
    blocker.add_input('cellx', cell_rad)
    blocker.add_input('celly', cell_rad)
    blocker.add_input('mds', mds)
    # if model is not None:
    #     blocker.add_input('model', model, ('x', 'y'))
    # else:
    #     blocker.add_input('model', None)
    blocker.add_input('robustness', opts.robustness)
    blocker.add_input('x0', x0)
    blocker.add_input('y0', y0)
    blocker.add_input('nthreads', opts.nvthreads)
    blocker.add_input('epsilon', opts.epsilon)
    blocker.add_input('do_wgridding', opts.do_wgridding)
    blocker.add_input('double_accum', opts.double_accum)
    blocker.add_input('l2reweight_dof', opts.l2reweight_dof)
    blocker.add_input('do_dirty', opts.dirty)
    blocker.add_input('do_residual', opts.residual)
    blocker.add_input('time_out', time_out)
    blocker.add_input('freq_out', freq_out)

    blocker.add_output(
        'WSUM',
        ('scalar',),
        ((1,),),
        weight.dtype)

    if opts.dirty:
        blocker.add_output(
            'DIRTY',
            ('x', 'y'),
            ((nx,), (ny,)),
            weight.dtype)

    if opts.residual:
        blocker.add_output(
            'RESIDUAL',
            ('x', 'y'),
            ((nx,), (ny,)),
            weight.dtype)

    output_dict = blocker.get_dask_outputs()

    data_vars = {}
    data_vars['WSUM'] = (('scalar',), output_dict['WSUM'])

    coords = {'chan': (('chan',), freq),
              'time': (('time',), utime),
    }

    # TODO - provide time and freq centroids
    attrs = {
        'ra' : tra,
        'dec': tdec,
        'x0': x0,
        'y0': y0,
        'cell_rad': cell_rad,
        'fieldid': ds.FIELD_ID,
        'ddid': ds.DATA_DESC_ID,
        'scanid': ds.SCAN_NUMBER,
        'freq_out': freq_out,
        'freq_min': freq_min,
        'freq_max': freq_max,
        'time_out': time_out,
        'time_min': utime.min(),
        'time_max': utime.max(),
        'product': opts.product
    }

    out_ds = Dataset(data_vars, coords=coords,
                     attrs=attrs)

    if opts.dirty:
        out_ds = out_ds.assign(**{
            'DIRTY': (('x', 'y'), output_dict['DIRTY'])
            })

    if opts.residual:
        out_ds = out_ds.assign(**{
            'RESIDUAL': (('x', 'y'), output_dict['RESIDUAL'])
            })

    out_ds = out_ds.chunk({'x':4096,
                           'y':4096})

    return out_ds.unify_chunks()


def image_data(data,
               weight,
               flag,
               uvw,
               freq,
               jones,
               tbin_idx,
               tbin_counts,
               ant1, ant2,
               pol, product, nc,
               nx, ny,
               cellx, celly,
               mds,
            #    model,
               robustness,
               x0, y0,
               nthreads,
               epsilon,
               do_wgridding,
               double_accum,
               l2reweight_dof,
               do_dirty,
               do_residual,
               time_out,
               freq_out):

    # we currently need this extra loop through the data because
    # we don't have access to the grid
    vis, wgt = _weight_data(data, weight, flag, jones,
                            tbin_idx, tbin_counts,
                            ant1, ant2,
                            literally(pol),
                            literally(product),
                            literally(nc))

    mask = (~flag).astype(np.uint8)
    # apply weighting
    if robustness is not None:
        counts = compute_counts(
                clone(uvw),
                freq,
                mask,
                nx,
                ny,
                cellx,
                celly,
                real_type,
                ngrid=opts.nvthreads)
        # get rid of artificially high weights corresponding to
        # nearly empty cells
        if opts.filter_extreme_counts:
            counts = filter_extreme_counts(counts, nbox=opts.filter_nbox,
                                            nlevel=opts.filter_level)


    if mds is None:
        if l2reweight_dof:
            raise ValueError('Requested l2 reweight but no model passed in. '
                             'Perhaps transfer model from somewhere?')
    else:
        # we only want to load these once
        model_coeffs = mds.coefficients.values
        locx = mds.location_x.values
        locy = mds.location_y.values
        params = mds.params.values
        coeffs = mds.coefficients.values


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
            cellx, celly,
            x0, y0
        )

        # do not apply weights in this direction
        model_vis = dirty2vis(
            uvw=uvw,
            freq=freq,
            dirty=model,
            pixsize_x=cellx,
            pixsize_y=celly,
            center_x=x0,
            center_y=y0,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            flip_v=False,
            nthreads=nthreads,
            divide_by_n=False,
            sigma_min=1.1, sigma_max=3.0)

        residual_vis = vis - model_vis
        # apply mask to both
        residual_vis *= mask

    if l2reweight_dof:
        ressq = (residual_vis*residual_vis.conj()).real
        wcount = mask.sum()
        if wcount:
            ovar = ressq.sum()/wcount  # use 67% quantile?
            wgt = (l2reweight_dof + 1)/(l2reweight_dof + ressq/ovar)/ovar
        else:
            wgt = None


    # we usually want to re-evaluate this since the robustness may change
    if robustness is not None:
        if counts is None:
            raise ValueError('counts are None but robustness specified. '
                             'This is probably a bug!')
        imwgt = _counts_to_weights(
            counts,
            uvw,
            freq,
            nx, ny,
            cellx, celly,
            robustness)
        if wgt is not None:
            wgt *= imwgt
        else:
            wgt = imwgt

    wsum = wgt[mask.astype(bool)].sum()

    if do_dirty:
        dirty = vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=vis,
            wgt=wgt,
            mask=mask,
            npix_x=nx, npix_y=ny,
            pixsize_x=cellx, pixsize_y=celly,
            center_x=x0, center_y=y0,
            epsilon=epsilon,
            flip_v=False,  # hardcoded for now
            do_wgridding=do_wgridding,
            divide_by_n=False,  # hardcoded for now
            nthreads=nthreads,
            sigma_min=1.1, sigma_max=3.0,
            double_precision_accumulation=double_accum)
    else:
        dirty = None

    if do_residual and model is not None:
        residual = vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=residual_vis,
            wgt=wgt,
            mask=mask,
            npix_x=nx, npix_y=ny,
            pixsize_x=cellx, pixsize_y=celly,
            center_x=x0, center_y=y0,
            epsilon=epsilon,
            flip_v=False,  # hardcoded for now
            do_wgridding=do_wgridding,
            divide_by_n=False,  # hardcoded for now
            nthreads=nthreads,
            sigma_min=1.1, sigma_max=3.0,
            double_precision_accumulation=double_accum)

    else:
        residual = None


    out_dict = {}
    out_dict['WSUM'] = wsum
    out_dict['DIRTY'] = dirty
    out_dict['RESIDUAL'] = residual


    return out_dict
