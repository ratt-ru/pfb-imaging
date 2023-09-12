import packratt
import pytest
from pathlib import Path
from xarray import Dataset
from collections import namedtuple
import dask
import dask.array as da
from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
pmp = pytest.mark.parametrize

# @pmp('beam_model', (None, 'kbl',))
# @pmp('do_gains', (False, True))
def test_forwardmodel(gain_name, ms_name):
    '''
    This test is based on the observation that imaging all of corrected
    data at natural weighting should be the same applying the gains per
    band and time on the fly and computing the sum afterwards.
    '''
    import numpy as np
    np.random.seed(420)
    from numpy.testing import assert_allclose
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
    from africanus.constants import c as lightspeed
    from ducc0.fft import good_size
    from ducc0.wgridder.experimental import vis2dirty
    from africanus.calibration.utils import corrupt_vis, correct_vis, chunkify_rows

    test_dir = Path(ms_name).resolve().parent
    xds = xds_from_ms(ms_name,
                      chunks={'row': -1, 'chan': -1})
    xds = xr.concat(xds, 'row')
    spw = xds_from_table(f'{ms_name}::SPECTRAL_WINDOW')[0]

    utime = np.unique(xds.TIME.values)
    freq = spw.CHAN_FREQ.values.squeeze()
    freq0 = np.mean(freq)

    ntime = utime.size
    nchan = freq.size
    nant = np.maximum(xds.ANTENNA1.values.max(), xds.ANTENNA2.values.max()) + 1

    ncorr = xds.corr.size

    uvw = xds.UVW.values
    nrow = uvw.shape[0]
    u_max = abs(uvw[:, 0]).max()
    v_max = abs(uvw[:, 1]).max()
    uv_max = np.maximum(u_max, v_max)

    # image size
    cell_N = 1.0 / (2 * uv_max * freq.max() / lightspeed)

    srf = 2.0
    cell_rad = cell_N / srf
    cell_deg = cell_rad * 180 / np.pi
    cell_size = cell_deg * 3600
    print("Cell size set to %5.5e arcseconds" % cell_size)


    # the test will fail in intrinsic if sources fall near beam sidelobes
    fov = 1.0
    npix = good_size(int(fov / cell_deg))
    while npix % 2:
        npix += 1
        npix = good_size(npix)

    nx = npix
    ny = npix

    print("Image size set to (%i, %i, %i)" % (nchan, nx, ny))

    # first make full image
    gds = xds_from_zarr(f"{gain_name.rstrip('/')}::GK-net")
    gds = xr.concat(gds, 'gain_time')
    jones = gds.GAIN.values
    data = xds.DATA.values.astype(np.complex128)
    ssp = xds.SIGMA_SPECTRUM.values.astype(np.float64)
    wgt  = 1.0/ssp**2
    flag = xds.FLAG.values
    ant1 = xds.ANTENNA1.values
    ant2 = xds.ANTENNA2.values

    row_chunks, tbin_idx, tbin_counts = chunkify_rows(xds.TIME.values, -1)

    # corrupt twice for cwgt
    if ncorr == 4:
        mwgt = wgt.reshape(nrow, nchan, 1, 2, 2)
    else:
        mwgt = wgt.reshape(nrow, nchan, 1, ncorr)
    cwgt = corrupt_vis(tbin_idx,
                       tbin_counts,
                       ant1,
                       ant2,
                       jones,
                       mwgt)
    cwgt = corrupt_vis(tbin_idx,
                       tbin_counts,
                       ant1,
                       ant2,
                       jones.conj(),
                       cwgt).reshape(nrow, nchan, ncorr).real

    # correct data
    if ncorr == 4:
        vis = data.reshape(nrow, nchan, 2, 2)
        flag = flag.reshape(nrow, nchan, 2, 2)
    cdata = correct_vis(tbin_idx,
                        tbin_counts,
                        ant1,
                        ant2,
                        jones,
                        vis,
                        flag).reshape(nrow, nchan, ncorr)

    # take weighted sum of correlations to get Stokes I vis
    visI = cwgt[:, :, 0] * cdata[:, :, 0] + cwgt[:, :, -1] * cdata[:, :, -1]
    wgtI = cwgt[:, :, 0] + cwgt[:, :, -1]
    msk = wgtI > 0
    visI[msk] = visI[msk]/wgtI[msk]
    mask = (~(flag[:, :, 0] | flag[:, :, -1])).astype(np.uint8)

    # image Stokes I data
    dirtyI = vis2dirty(uvw=xds.UVW.values,
                       freq=freq,
                       vis=visI,
                       wgt=wgtI,
                       mask=mask,
                       npix_x=nx, npix_y=ny,
                       pixsize_x=cell_rad, pixsize_y=cell_rad,
                       epsilon=1e-7,
                       do_wgridding=True,
                       divide_by_n=False,
                       nthreads=8)

    psfI = vis2dirty(uvw=xds.UVW.values,
                     freq=freq,
                     vis=wgtI.astype(np.complex128),
                     mask=mask,
                     npix_x=nx, npix_y=ny,
                     pixsize_x=cell_rad, pixsize_y=cell_rad,
                     epsilon=1e-7,
                     do_wgridding=True,
                     divide_by_n=False,
                     nthreads=8)


    # next compute via pfb workers

    postfix = "main"
    outname = str(test_dir / 'test')
    basename = f'{outname}_I'
    dds_name = f'{basename}_{postfix}.dds'
    # set defaults from schema
    from pfb.parser.schemas import schema
    init_args = {}
    for key in schema.init["inputs"].keys():
        init_args[key] = schema.init["inputs"][key]["default"]
    # overwrite defaults
    init_args["ms"] = str(test_dir / 'test_ascii_1h60.0s.MS')
    init_args["output_filename"] = outname
    init_args["data_column"] = "DATA"
    init_args["flag_column"] = 'FLAG'
    init_args["gain_table"] = gain_name
    init_args['gain_term'] = 'GK-net'
    init_args["max_field_of_view"] = fov
    init_args["overwrite"] = True
    init_args["channels_per_image"] = 1
    from pfb.workers.init import _init
    xds = _init(**init_args)

    # grid data to produce dirty image
    grid_args = {}
    for key in schema.grid["inputs"].keys():
        grid_args[key] = schema.grid["inputs"][key]["default"]
    # overwrite defaults
    grid_args["output_filename"] = outname
    grid_args["postfix"] = postfix
    grid_args["nband"] = nchan
    grid_args["field_of_view"] = fov
    grid_args["fits_mfs"] = True
    grid_args["psf"] = True
    grid_args["residual"] = False
    grid_args["nthreads"] = 8  # has to be set when calling _grid
    grid_args["nvthreads"] = 8
    grid_args["overwrite"] = True
    grid_args["robustness"] = 0.0
    grid_args["do_wgridding"] = True
    from pfb.workers.grid import _grid
    dds = _grid(xdsi=xds, **grid_args)
